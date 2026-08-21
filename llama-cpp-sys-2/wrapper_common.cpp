#include "wrapper_common.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <random>
#include <string>
#include <stdint.h>
#include <unordered_map>
#include <vector>

#include "llama.cpp/common/common.h"
#include "llama.cpp/common/fit.h"
#include "llama.cpp/common/json-schema-to-grammar.h"
#include "llama.cpp/common/speculative.h"
#include "llama.cpp/include/llama.h"
#include "llama.cpp/src/llama-ext.h"
#include "wrapper_utils.h"

#include <nlohmann/json.hpp>

extern "C" llama_rs_status llama_rs_json_schema_to_grammar(
    const char * schema_json,
    bool force_gbnf,
    char ** out_grammar) {
    if (!schema_json || !out_grammar) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    *out_grammar = nullptr;
    try {
        const auto schema = nlohmann::ordered_json::parse(schema_json);
        const auto grammar = json_schema_to_grammar(schema, force_gbnf);
        *out_grammar = llama_rs_dup_string(grammar);
        return *out_grammar ? LLAMA_RS_STATUS_OK : LLAMA_RS_STATUS_ALLOCATION_FAILED;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" void llama_rs_string_free(char * ptr) {
    if (ptr) {
        std::free(ptr);
    }
}

extern "C" struct llama_sampler * llama_rs_sampler_init_grammar(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root) {
    try {
        return llama_sampler_init_grammar(vocab, grammar_str, grammar_root);
    } catch (...) {
        return nullptr;
    }
}

extern "C" struct llama_sampler * llama_rs_sampler_init_grammar_lazy(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_words,
    size_t num_trigger_words,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens) {
    try {
        std::vector<std::string> trigger_patterns;
        trigger_patterns.reserve(num_trigger_words);
        for (size_t i = 0; i < num_trigger_words; ++i) {
            const char * word = trigger_words ? trigger_words[i] : nullptr;
            if (word && word[0] != '\0') {
                trigger_patterns.push_back(regex_escape(word));
            }
        }
        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & pattern : trigger_patterns) {
            trigger_patterns_c.push_back(pattern.c_str());
        }
        return llama_sampler_init_grammar_lazy_patterns(
            vocab,
            grammar_str,
            grammar_root,
            trigger_patterns_c.data(),
            trigger_patterns_c.size(),
            trigger_tokens,
            num_trigger_tokens);
    } catch (...) {
        return nullptr;
    }
}

extern "C" struct llama_sampler * llama_rs_sampler_init_grammar_lazy_patterns(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_patterns,
    size_t num_trigger_patterns,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens) {
    try {
        return llama_sampler_init_grammar_lazy_patterns(
            vocab,
            grammar_str,
            grammar_root,
            trigger_patterns,
            num_trigger_patterns,
            trigger_tokens,
            num_trigger_tokens);
    } catch (...) {
        return nullptr;
    }
}

extern "C" llama_rs_status llama_rs_sampler_accept(struct llama_sampler * sampler, llama_token token) {
    if (!sampler) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    try {
        llama_sampler_accept(sampler, token);
        return LLAMA_RS_STATUS_OK;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

// Thin pass-through to llama.cpp's common_fit_params (a C++ symbol in libcommon).
// Returns common_params_fit_status as an int: 0 = success, 1 = failure, 2 = error.
extern "C" int llama_rs_fit_params(
    const char * path_model,
    struct llama_model_params * mparams,
    struct llama_context_params * cparams,
    float * tensor_split,
    struct llama_model_tensor_buft_override * tensor_buft_overrides,
    size_t * margins,
    uint32_t n_ctx_min,
    enum ggml_log_level log_level) {
    return static_cast<int>(common_fit_params(
        path_model,
        mparams,
        cparams,
        tensor_split,
        tensor_buft_overrides,
        margins,
        n_ctx_min,
        log_level));
}

extern "C" void llama_rs_memory_breakdown_print(const struct llama_context * ctx) {
    common_memory_breakdown_print(ctx);
}

struct llama_rs_speculative {
    common_params_speculative params;
    common_speculative * spec = nullptr;
    llama_context * ctx_tgt = nullptr;
    std::vector<llama_token> prompt;
    std::vector<llama_token> draft;
    std::vector<common_speculative_token_dist> dists;
    std::mt19937 verifier_rng;
    uint32_t verifier_seed = LLAMA_DEFAULT_SEED;
    size_t last_draft_len = 0;
    bool draft_pending = false;
};

static constexpr llama_seq_id LLAMA_RS_SPEC_SEQ_ID = 0;

static bool llama_rs_spec_batch_compatible(const struct llama_batch & batch) {
    if (batch.n_tokens <= 0 || !batch.token || batch.embd || !batch.pos || !batch.n_seq_id ||
        !batch.seq_id) {
        return false;
    }
    for (int32_t k = 0; k < batch.n_tokens; ++k) {
        if (batch.n_seq_id[k] != 1 || !batch.seq_id[k] ||
            batch.seq_id[k][0] != LLAMA_RS_SPEC_SEQ_ID) {
            return false;
        }
    }
    return true;
}

static void llama_rs_assign_tokens(
    std::vector<llama_token> & dst,
    const llama_token * tokens,
    size_t count) {
    if (count == 0) {
        dst.clear();
        return;
    }
    dst.assign(tokens, tokens + count);
}

static bool llama_rs_dflash_models_compatible(llama_context * ctx_tgt, llama_context * ctx_dft) {
    const llama_model * model_tgt = llama_get_model(ctx_tgt);
    const llama_model * model_dft = llama_get_model(ctx_dft);
    if (!model_tgt || !model_dft) {
        return false;
    }
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);
    if (!vocab_tgt || !vocab_dft || llama_vocab_n_tokens(vocab_tgt) != llama_vocab_n_tokens(vocab_dft)) {
        return false;
    }
    const uint32_t n_target_layers = llama_model_target_layer_ids_n(model_dft);
    const int32_t * target_layers = llama_model_target_layer_ids(model_dft);
    if (n_target_layers == 0 || !target_layers) {
        return false;
    }
    for (uint32_t i = 0; i < n_target_layers; ++i) {
        if (target_layers[i] < 0 || target_layers[i] >= llama_model_n_layer(model_tgt)) {
            return false;
        }
    }
    return true;
}

extern "C" struct llama_rs_speculative * llama_rs_speculative_init(
    enum llama_rs_speculative_type type,
    struct llama_context * ctx_tgt,
    struct llama_context * ctx_dft,
    int32_t n_max,
    int32_t n_min,
    float p_min) {
    if (!ctx_tgt || !ctx_dft || n_max <= 0 || n_min < 0 || n_min > n_max ||
        !std::isfinite(p_min) || p_min < 0.0f || p_min > 1.0f) {
        return nullptr;
    }

    try {
        auto wrapper = std::make_unique<llama_rs_speculative>();
        switch (type) {
            case LLAMA_RS_SPECULATIVE_TYPE_MTP:
                wrapper->params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
                break;
            case LLAMA_RS_SPECULATIVE_TYPE_DRAFT_DFLASH:
                if (!llama_rs_dflash_models_compatible(ctx_tgt, ctx_dft)) {
                    return nullptr;
                }
                wrapper->params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH };
                break;
            default:
                return nullptr;
        }
        wrapper->ctx_tgt = ctx_tgt;
        wrapper->params.draft.ctx_tgt = ctx_tgt;
        wrapper->params.draft.ctx_dft = ctx_dft;
        wrapper->params.draft.n_max = n_max;
        wrapper->params.draft.n_min = n_min;
        wrapper->params.draft.p_min = p_min;

        wrapper->spec = common_speculative_init(wrapper->params, 1);
        if (!wrapper->spec) {
            return nullptr;
        }

        return wrapper.release();
    } catch (...) {
        return nullptr;
    }
}

extern "C" void llama_rs_speculative_free(struct llama_rs_speculative * spec) {
    if (!spec) {
        return;
    }
    if (spec->spec) {
        common_speculative_free(spec->spec);
        spec->spec = nullptr;
    }
    delete spec;
}

extern "C" llama_rs_status llama_rs_speculative_begin(
    struct llama_rs_speculative * spec,
    const llama_token * prompt_tokens,
    size_t prompt_tokens_count) {
    if (!spec || !spec->spec || (!prompt_tokens && prompt_tokens_count > 0)) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    try {
        llama_rs_assign_tokens(spec->prompt, prompt_tokens, prompt_tokens_count);
        spec->last_draft_len = 0;
        spec->draft_pending = false;
        spec->dists.clear();
        common_speculative_begin(spec->spec, LLAMA_RS_SPEC_SEQ_ID, spec->prompt);
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_speculative_process(
    struct llama_rs_speculative * spec,
    const struct llama_batch * batch) {
    if (!spec || !spec->spec || !batch || !llama_rs_spec_batch_compatible(*batch)) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    try {
        return common_speculative_process(spec->spec, *batch)
            ? LLAMA_RS_STATUS_OK
            : LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_speculative_draft(
    struct llama_rs_speculative * spec,
    llama_pos n_past,
    llama_token id_last,
    const llama_token * prompt_tokens,
    size_t prompt_tokens_count,
    float temperature,
    uint32_t seed,
    llama_token * out_tokens,
    size_t out_tokens_capacity,
    size_t * out_tokens_count) {
    if (!spec || !spec->spec || (!prompt_tokens && prompt_tokens_count > 0) ||
        !out_tokens_count || n_past < 0 || !std::isfinite(temperature)) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    try {
        if (spec->draft_pending) {
            return LLAMA_RS_STATUS_INVALID_ARGUMENT;
        }
        llama_rs_assign_tokens(spec->prompt, prompt_tokens, prompt_tokens_count);
        spec->draft.clear();
        spec->dists.clear();
        spec->last_draft_len = 0;
        if (spec->verifier_seed != seed) {
            spec->verifier_seed = seed;
            spec->verifier_rng.seed(seed ^ 0x9E3779B9u);
        }

        auto & params = common_speculative_get_draft_params(spec->spec, LLAMA_RS_SPEC_SEQ_ID);
        params.drafting = true;
        params.n_max = spec->params.draft.n_max;
        params.n_past = n_past;
        params.id_last = id_last;
        params.prompt = &spec->prompt;
        params.result = &spec->draft;
        params.dists = &spec->dists;
        params.temperature = temperature;
        params.seed = seed;

        common_speculative_draft(spec->spec);

        *out_tokens_count = spec->draft.size();
        if (spec->draft.size() > out_tokens_capacity) {
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
        if (!spec->draft.empty() && !out_tokens) {
            return LLAMA_RS_STATUS_INVALID_ARGUMENT;
        }
        if (!spec->draft.empty()) {
            std::memcpy(out_tokens, spec->draft.data(), spec->draft.size() * sizeof(llama_token));
        }
        spec->last_draft_len = spec->draft.size();
        spec->draft_pending = !spec->draft.empty();
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

static llama_token_data_array llama_rs_apply_sampler(
    llama_context * ctx,
    llama_sampler * sampler,
    int32_t idx,
    std::vector<llama_token_data> & candidates) {
    const llama_model * model = llama_get_model(ctx);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * logits = llama_get_logits_ith(ctx, idx);
    if (!logits) {
        return { nullptr, 0, -1, false };
    }
    candidates.resize(n_vocab);
    for (llama_token id = 0; id < n_vocab; ++id) {
        candidates[id] = { id, logits[id], 0.0f };
    }
    llama_token_data_array data = { candidates.data(), candidates.size(), -1, false };
    llama_sampler_apply(sampler, &data);
    return data;
}

extern "C" llama_rs_status llama_rs_speculative_verify(
    struct llama_rs_speculative * spec,
    struct llama_sampler * sampler,
    llama_token * out_tokens,
    size_t out_tokens_capacity,
    size_t * out_tokens_count,
    size_t * out_accepted) {
    if (!spec || !spec->spec || !spec->draft_pending || !sampler || !out_tokens_count || !out_accepted) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    try {
        std::vector<llama_token> result;
        result.reserve(spec->draft.size() + 1);
        size_t accepted = 0;
        size_t i = 0;
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        for (; i < spec->draft.size(); ++i) {
            std::vector<llama_token_data> candidates;
            auto p = llama_rs_apply_sampler(spec->ctx_tgt, sampler, static_cast<int32_t>(i), candidates);
            if (!p.data || p.selected < 0 || static_cast<size_t>(p.selected) >= p.size) {
                return LLAMA_RS_STATUS_EXCEPTION;
            }
            llama_token selected = p.data[p.selected].id;
            bool accept_draft = false;
            if (spec->dists.size() == spec->draft.size()) {
                const auto & q = spec->dists[i];
                std::unordered_map<llama_token, float> q_probs;
                for (size_t j = 0; j < q.ids.size(); ++j) {
                    q_probs[q.ids[j]] += q.probs[j];
                }
                const auto q_prob = [&](llama_token id) {
                    const auto it = q_probs.find(id);
                    return it == q_probs.end() ? 0.0f : it->second;
                };
                float p_draft = 0.0f;
                for (size_t j = 0; j < p.size; ++j) {
                    if (p.data[j].id == spec->draft[i]) {
                        p_draft = p.data[j].p;
                        break;
                    }
                }
                const float q_draft = q_prob(spec->draft[i]);
                accept_draft = q_draft > 0.0f && uniform(spec->verifier_rng) * q_draft <= p_draft;
                if (!accept_draft) {
                    std::vector<float> residual(p.size);
                    float sum = 0.0f;
                    for (size_t j = 0; j < p.size; ++j) {
                        residual[j] = std::max(0.0f, p.data[j].p - q_prob(p.data[j].id));
                        sum += residual[j];
                    }
                    if (sum > 0.0f) {
                        std::discrete_distribution<size_t> sample(residual.begin(), residual.end());
                        selected = p.data[sample(spec->verifier_rng)].id;
                    }
                }
            } else {
                accept_draft = selected == spec->draft[i];
            }
            if (accept_draft) {
                selected = spec->draft[i];
                ++accepted;
            }
            llama_sampler_accept(sampler, selected);
            result.push_back(selected);
            if (!accept_draft) {
                break;
            }
        }
        if (i == spec->draft.size()) {
            std::vector<llama_token_data> candidates;
            auto p = llama_rs_apply_sampler(spec->ctx_tgt, sampler, static_cast<int32_t>(i), candidates);
            if (!p.data || p.selected < 0 || static_cast<size_t>(p.selected) >= p.size) {
                return LLAMA_RS_STATUS_EXCEPTION;
            }
            const llama_token selected = p.data[p.selected].id;
            llama_sampler_accept(sampler, selected);
            result.push_back(selected);
        }
        *out_tokens_count = result.size();
        *out_accepted = accepted;
        if (result.size() > out_tokens_capacity) {
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
        if (!result.empty() && !out_tokens) {
            return LLAMA_RS_STATUS_INVALID_ARGUMENT;
        }
        if (!result.empty()) {
            std::memcpy(out_tokens, result.data(), result.size() * sizeof(llama_token));
        }
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_speculative_accept(
    struct llama_rs_speculative * spec,
    uint16_t n_accepted) {
    if (!spec || !spec->spec || !spec->draft_pending || n_accepted > spec->last_draft_len) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    try {
        common_speculative_accept(spec->spec, LLAMA_RS_SPEC_SEQ_ID, n_accepted);
        spec->last_draft_len = 0;
        spec->draft_pending = false;
        spec->dists.clear();
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}
