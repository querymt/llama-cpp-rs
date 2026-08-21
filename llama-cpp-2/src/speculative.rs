//! Experimental wrappers for llama.cpp speculative decoding helpers.

use std::ptr::NonNull;

use crate::context::LlamaContext;
use crate::llama_batch::LlamaBatch;
use crate::status_is_ok;
use crate::token::LlamaToken;

/// Speculative decoding implementation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpeculativeType {
    /// Multi-token prediction using bundled tensors or a sidecar model.
    Mtp,
    /// DFlash, including DFlash2 models detected through GGUF metadata.
    DraftDflash,
}

/// Parameters for model-based speculative decoding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpeculativeParams {
    /// Speculative decoding implementation.
    pub kind: SpeculativeType,
    /// Maximum number of draft tokens to propose.
    pub n_max: i32,
    /// Minimum number of draft tokens required before returning a draft.
    pub n_min: i32,
    /// Minimum draft probability accepted by llama.cpp's MTP drafter.
    pub p_min: f32,
}

impl Default for SpeculativeParams {
    fn default() -> Self {
        Self {
            kind: SpeculativeType::Mtp,
            n_max: 3,
            n_min: 0,
            p_min: 0.0,
        }
    }
}

/// Errors returned by the MTP speculative wrapper.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum SpeculativeError {
    /// Invalid parameters were provided.
    #[error("invalid MTP speculative parameters")]
    InvalidParams,
    /// llama.cpp returned a null speculative handle.
    #[error("llama.cpp failed to initialize MTP speculative decoding")]
    InitFailed,
    /// llama.cpp rejected a wrapper call.
    #[error("llama.cpp MTP speculative call failed with status {0}")]
    Status(i32),
    /// The draft output exceeded the caller-provided bound.
    #[error("llama.cpp speculative draft exceeded configured maximum")]
    DraftOverflow,
    /// Verification output exceeded the caller-provided bound.
    #[error("llama.cpp speculative verification exceeded configured maximum")]
    VerificationOverflow,
}

/// Target-approved tokens from one speculative verification round.
#[derive(Debug, Eq, PartialEq)]
pub struct SpeculativeVerification {
    /// Accepted draft prefix followed by one target continuation token.
    pub tokens: Vec<LlamaToken>,
    /// Number of draft tokens accepted before the continuation token.
    pub accepted: usize,
}

/// RAII owner for a same-model MTP speculative context.
///
/// This wrapper currently binds llama.cpp's speculative state to sequence 0.
/// Batches passed to [`Self::process`] must therefore contain only sequence 0.
#[derive(Debug)]
pub struct Speculative<'model> {
    raw: NonNull<llama_cpp_sys_2::llama_rs_speculative>,
    target_context: LlamaContext<'model>,
    draft_context: LlamaContext<'model>,
    n_max: usize,
}

impl<'model> Speculative<'model> {
    /// Create a new MTP speculative helper from a target context and an MTP
    /// draft context.
    ///
    /// # Errors
    ///
    /// Returns an error if parameters are invalid or llama.cpp cannot
    /// initialize the speculative implementation for the loaded model.
    pub fn new(
        target_context: LlamaContext<'model>,
        draft_context: LlamaContext<'model>,
        params: SpeculativeParams,
    ) -> Result<Self, SpeculativeError> {
        if params.n_max <= 0 || params.n_min < 0 || params.n_min > params.n_max {
            return Err(SpeculativeError::InvalidParams);
        }
        let n_max = usize::try_from(params.n_max).map_err(|_| SpeculativeError::InvalidParams)?;

        if !params.p_min.is_finite() || !(0.0..=1.0).contains(&params.p_min) {
            return Err(SpeculativeError::InvalidParams);
        }
        let kind = match params.kind {
            SpeculativeType::Mtp => llama_cpp_sys_2::LLAMA_RS_SPECULATIVE_TYPE_MTP,
            SpeculativeType::DraftDflash => llama_cpp_sys_2::LLAMA_RS_SPECULATIVE_TYPE_DRAFT_DFLASH,
        };
        let raw = unsafe {
            llama_cpp_sys_2::llama_rs_speculative_init(
                kind,
                target_context.context.as_ptr(),
                draft_context.context.as_ptr(),
                params.n_max,
                params.n_min,
                params.p_min,
            )
        };
        let raw = NonNull::new(raw).ok_or(SpeculativeError::InitFailed)?;

        Ok(Self {
            raw,
            target_context,
            draft_context,
            n_max,
        })
    }

    /// Access the target context.
    #[must_use]
    pub fn target_context(&self) -> &LlamaContext<'model> {
        &self.target_context
    }

    /// Access the target context for decode and cache rollback operations.
    pub fn target_context_mut(&mut self) -> &mut LlamaContext<'model> {
        &mut self.target_context
    }

    /// Access the draft context for cache rollback operations.
    pub fn draft_context_mut(&mut self) -> &mut LlamaContext<'model> {
        &mut self.draft_context
    }

    /// Begin a new generation from the given prompt tokens.
    ///
    /// # Errors
    ///
    /// Returns an error if llama.cpp rejects the call.
    pub fn begin(&mut self, prompt_tokens: &[LlamaToken]) -> Result<(), SpeculativeError> {
        let prompt = tokens_to_raw(prompt_tokens);
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_speculative_begin(
                self.raw.as_ptr(),
                prompt.as_ptr(),
                prompt.len(),
            )
        };
        status_to_result(status)
    }

    /// Process a batch that was just decoded by the target context.
    ///
    /// The batch must contain token input for sequence 0 only.
    ///
    /// # Errors
    ///
    /// Returns an error if llama.cpp cannot update the MTP draft context.
    pub fn process(&mut self, batch: &LlamaBatch<'_>) -> Result<(), SpeculativeError> {
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_speculative_process(
                self.raw.as_ptr(),
                std::ptr::from_ref(&batch.llama_batch),
            )
        };
        status_to_result(status)
    }

    /// Generate draft tokens after `id_last`.
    ///
    /// # Errors
    ///
    /// Returns an error if llama.cpp rejects the draft operation or emits more
    /// draft tokens than requested.
    pub fn draft(
        &mut self,
        n_past: i32,
        id_last: LlamaToken,
        prompt_tokens: &[LlamaToken],
        temperature: f32,
        seed: u32,
    ) -> Result<Vec<LlamaToken>, SpeculativeError> {
        if n_past < 0 {
            return Err(SpeculativeError::InvalidParams);
        }

        let prompt = tokens_to_raw(prompt_tokens);
        let mut raw_out = vec![0; self.n_max];
        let mut out_len = 0_usize;
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_speculative_draft(
                self.raw.as_ptr(),
                n_past,
                id_last.0,
                prompt.as_ptr(),
                prompt.len(),
                temperature,
                seed,
                raw_out.as_mut_ptr(),
                raw_out.len(),
                &raw mut out_len,
            )
        };
        if status == llama_cpp_sys_2::LLAMA_RS_STATUS_ALLOCATION_FAILED {
            return Err(SpeculativeError::DraftOverflow);
        }
        status_to_result(status)?;
        raw_out.truncate(out_len);
        Ok(raw_out.into_iter().map(LlamaToken).collect())
    }

    /// Verify the pending draft against target logits and update the sampler.
    pub fn verify(
        &mut self,
        sampler: &mut crate::sampling::LlamaSampler,
    ) -> Result<SpeculativeVerification, SpeculativeError> {
        let mut raw_out = vec![0; self.n_max + 1];
        let mut out_len = 0_usize;
        let mut accepted = 0_usize;
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_speculative_verify(
                self.raw.as_ptr(),
                sampler.sampler,
                raw_out.as_mut_ptr(),
                raw_out.len(),
                &raw mut out_len,
                &raw mut accepted,
            )
        };
        if status == llama_cpp_sys_2::LLAMA_RS_STATUS_ALLOCATION_FAILED {
            return Err(SpeculativeError::VerificationOverflow);
        }
        status_to_result(status)?;
        raw_out.truncate(out_len);
        Ok(SpeculativeVerification {
            tokens: raw_out.into_iter().map(LlamaToken).collect(),
            accepted,
        })
    }

    /// Notify llama.cpp how many draft tokens the target context accepted.
    ///
    /// # Errors
    ///
    /// Returns an error if llama.cpp rejects the call.
    pub fn accept(&mut self, n_accepted: u16) -> Result<(), SpeculativeError> {
        let status =
            unsafe { llama_cpp_sys_2::llama_rs_speculative_accept(self.raw.as_ptr(), n_accepted) };
        status_to_result(status)
    }
}

impl Drop for Speculative<'_> {
    fn drop(&mut self) {
        unsafe {
            llama_cpp_sys_2::llama_rs_speculative_free(self.raw.as_ptr());
        }
    }
}

fn tokens_to_raw(tokens: &[LlamaToken]) -> Vec<llama_cpp_sys_2::llama_token> {
    tokens.iter().map(|token| token.0).collect()
}

fn status_to_result(status: llama_cpp_sys_2::llama_rs_status) -> Result<(), SpeculativeError> {
    if status_is_ok(status) {
        Ok(())
    } else {
        Err(SpeculativeError::Status(status as i32))
    }
}
