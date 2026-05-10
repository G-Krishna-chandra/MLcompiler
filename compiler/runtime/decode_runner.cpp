#include "runtime/decode_runner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "runtime/kernel_registry.hpp"
#include "runtime/quant_utils.hpp"
#include "frontends/ggml_types.hpp"

namespace mlc {
namespace runtime {

DecodeRunner::DecodeRunner(const std::string& gguf_path)
    : session_(gguf_path) {
}

namespace {
CacheReportEntry makeEntry(const ExecutionTensor& info,
                           const TensorStorage* storage,
                           const frontend::GGUFLoader& loader,
                           size_t head_dim,
                           size_t element_count) {
    CacheReportEntry entry;
    entry.name = info.name;
    entry.dtype = storage ? storage->dtype
                          : (info.has_ggml_dtype ? info.ggml_dtype : frontend::GGML_TYPE_F32);
    entry.quant_version = storage ? storage->quant_version
                                  : (info.quant_version != 0 ? info.quant_version
                                                             : loader.quantizationVersion());

    auto resolveStride = [&]() -> size_t {
        if (storage && storage->row_stride_bytes > 0) return storage->row_stride_bytes;
        if (entry.dtype == frontend::GGML_TYPE_F32) return head_dim * sizeof(float);
        if (head_dim == 0) return 0;
        return ggmlRowSizeBytes(entry.dtype, head_dim, entry.quant_version);
    };
    entry.row_stride_bytes = resolveStride();

    if (storage) {
        if (!storage->raw_data.empty()) {
            entry.byte_size = storage->raw_data.size();
        } else if (!storage->float_data.empty()) {
            entry.byte_size = storage->float_data.size() * sizeof(float);
        } else if (entry.row_stride_bytes > 0 && element_count > 0) {
            entry.byte_size = element_count * entry.row_stride_bytes;
        }
    } else if (entry.row_stride_bytes > 0 && element_count > 0) {
        entry.byte_size = element_count * entry.row_stride_bytes;
    }
    return entry;
}

// Returns top_k indices/probabilities after softmax. If top_k==0, returns empty.
std::pair<std::vector<uint64_t>, std::vector<float>>
softmaxTopK(const std::vector<float>& logits, size_t top_k) {
    if (top_k == 0 || logits.empty()) return {};
    top_k = std::min(top_k, logits.size());

    float max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<std::pair<float, size_t>> scored;
    scored.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        scored.push_back({logits[i] - max_logit, i});
    }
    std::nth_element(scored.begin(), scored.begin() + top_k, scored.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    scored.resize(top_k);
    float sum = 0.f;
    for (auto& p : scored) {
        p.first = std::exp(p.first);
        sum += p.first;
    }
    std::vector<uint64_t> indices;
    std::vector<float> probs;
    indices.reserve(top_k);
    probs.reserve(top_k);
    for (const auto& p : scored) {
        indices.push_back(static_cast<uint64_t>(p.second));
        probs.push_back(p.first / sum);
    }
    return {std::move(indices), std::move(probs)};
}

struct LogitsTraceConfig {
    FILE* file = nullptr;
    size_t top_k_override = 0;
};

bool parseSizeTEnv(const char* text, size_t& out) {
    if (!text || !*text) return false;
    char* end = nullptr;
    unsigned long value = std::strtoul(text, &end, 10);
    if (!end || *end != '\0') return false;
    out = static_cast<size_t>(value);
    return true;
}

const LogitsTraceConfig& logitsTraceConfig() {
    static LogitsTraceConfig cfg = []() {
        LogitsTraceConfig out;
        const char* path = std::getenv("MLC_LOGITS_FILE");
        if (path && *path) {
            out.file = std::fopen(path, "a");
            if (out.file) {
                std::setvbuf(out.file, nullptr, _IOLBF, 0);
            }
        }
        size_t top_k = 0;
        const char* env_topk = std::getenv("MLC_LOGITS_TOPK");
        if (parseSizeTEnv(env_topk, top_k)) {
            out.top_k_override = top_k;
        }
        return out;
    }();
    return cfg;
}

size_t resolveTopK(const LogitsTraceConfig& cfg, size_t default_top_k) {
    if (cfg.top_k_override > 0) return cfg.top_k_override;
    if (default_top_k > 0) return default_top_k;
    return 5;
}

std::vector<std::pair<uint64_t, float>>
topKLogits(const std::vector<float>& logits, size_t top_k) {
    std::vector<std::pair<uint64_t, float>> out;
    if (logits.empty() || top_k == 0) return out;
    top_k = std::min(top_k, logits.size());
    std::vector<std::pair<float, uint64_t>> scored;
    scored.reserve(logits.size());
    for (uint64_t i = 0; i < logits.size(); ++i) {
        scored.push_back({logits[i], i});
    }
    if (top_k < scored.size()) {
        std::nth_element(scored.begin(),
                         scored.begin() + top_k,
                         scored.end(),
                         [](const auto& a, const auto& b) {
                             return a.first > b.first;
                         });
        scored.resize(top_k);
    }
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    out.reserve(top_k);
    for (const auto& p : scored) {
        out.push_back({p.second, p.first});
    }
    return out;
}

void logTokenList(const LogitsTraceConfig& cfg,
                  const char* label,
                  size_t seq,
                  const std::vector<uint64_t>& tokens) {
    if (!cfg.file) return;
    std::fprintf(cfg.file, "[Tokens] seq=%zu label=%s count=%zu ids=",
                 seq,
                 label ? label : "tokens",
                 tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) std::fprintf(cfg.file, ",");
        std::fprintf(cfg.file, "%llu", static_cast<unsigned long long>(tokens[i]));
    }
    std::fprintf(cfg.file, "\n");
    std::fflush(cfg.file);
}

void logLogits(const LogitsTraceConfig& cfg,
               const char* phase,
               size_t seq,
               size_t step,
               uint64_t token,
               size_t pos,
               const std::vector<float>& logits,
               size_t default_top_k) {
    if (!cfg.file || logits.empty()) return;
    size_t top_k = resolveTopK(cfg, default_top_k);
    if (top_k == 0) return;
    float min_val = logits.front();
    float max_val = logits.front();
    double sum = 0.0;
    for (float v : logits) {
        min_val = std::min(min_val, v);
        max_val = std::max(max_val, v);
        sum += static_cast<double>(v);
    }
    double mean = sum / static_cast<double>(logits.size());
    auto top = topKLogits(logits, top_k);
    std::fprintf(cfg.file,
                 "[Logits] seq=%zu phase=%s step=%zu token=%llu pos=%zu count=%zu min=%.6g max=%.6g mean=%.6g topk=%zu entries=",
                 seq,
                 phase ? phase : "step",
                 step,
                 static_cast<unsigned long long>(token),
                 pos,
                 logits.size(),
                 min_val,
                 max_val,
                 mean,
                 top.size());
    for (size_t i = 0; i < top.size(); ++i) {
        if (i > 0) std::fprintf(cfg.file, ",");
        std::fprintf(cfg.file,
                     "%llu:%.6g",
                     static_cast<unsigned long long>(top[i].first),
                     top[i].second);
    }
    std::fprintf(cfg.file, "\n");
    std::fflush(cfg.file);
}
} // namespace

std::vector<CacheReportEntry> BuildCacheReport(const ExecutionGraph& graph,
                                               const ExecutionContext& context,
                                               const frontend::GGUFLoader& loader) {
    std::vector<CacheReportEntry> report;
    for (const auto& [name, info] : graph.tensors()) {
        if (!info.is_state) continue;
        const TensorStorage* storage = context.tensorStorage(name);
        size_t head_dim = 0;
        if (!info.shape.empty()) {
            head_dim = static_cast<size_t>(std::max<int64_t>(1, info.shape.back()));
        }
        size_t elems = context.tensorElementCount(info);
        report.push_back(makeEntry(info, storage, loader, head_dim, elems));
    }
    return report;
}

DecodeResult DecodeRunner::run(const DecodeOptions& options) {
    if (options.tokens.empty()) {
        throw std::runtime_error("DecodeRunner: token list is empty");
    }

    DecodeResult result;
    const auto& log_cfg = logitsTraceConfig();
    logTokenList(log_cfg, "input_tokens", 0, options.tokens);

    // Build execution graph once.
    auto graph = ExecutionPlanBuilder::BuildFromLoader(session_.loader());
    const size_t context_len = std::max<size_t>(1, graph.modelConfig().context_length);

    ExecutionContext context(session_, &graph);
    ExecutionExecutor executor(graph, &BackendRegistry::Default(), &context);

    const size_t steps = (options.max_steps > 0)
                             ? std::min(options.tokens.size(), options.max_steps)
                             : options.tokens.size();
    size_t cursor = options.start_position;

    for (size_t i = 0; i < steps; ++i) {
        DecodeStep step;
        step.token = options.tokens[i];
        size_t pos = cursor;
        if (pos >= context_len) {
            if (options.evict_on_full) {
                context.clearStateTensors();
                pos = 0;
                cursor = 0;
                step.cache_evicted = true;
            } else {
                step.success = false;
                step.position = context_len - 1;
                step.error = "Decode aborted: context length exceeded and eviction disabled";
                result.steps.push_back(std::move(step));
                result.success = false;
                break;
            }
        }
        step.position = pos;

        context.setToken(step.token);
        context.setSequencePosition(pos);

        // Propagate token into known token tensor inputs if present.
        static const std::vector<std::string> kTokenInputs = {"tokens", "token_ids"};
        for (const auto& name : kTokenInputs) {
            if (graph.tensors().count(name)) {
                context.setTensor(name, {static_cast<float>(step.token)});
            }
        }

        auto exec_res = executor.run();
        step.success = exec_res.success;
        for (const auto& entry : exec_res.trace) {
            step.trace.push_back(formatTraceEntry(entry));
        }

        if (!exec_res.success) {
            if (!exec_res.trace.empty()) {
                step.error = formatTraceEntry(exec_res.trace.back());
            } else {
                step.error = "execution failed";
            }
            result.steps.push_back(std::move(step));
            result.success = false;
            break;
        }

        // Collect logits if produced.
        if (const auto* logits = context.getTensor("logits")) {
            step.logits = *logits;
            if (options.top_k > 0) {
                auto top = softmaxTopK(step.logits, options.top_k);
                step.top_indices = std::move(top.first);
                step.top_probs = std::move(top.second);
            }
            logLogits(log_cfg, "input", 0, i, step.token, pos, step.logits, options.top_k);
        }

        result.steps.push_back(std::move(step));
        cursor = pos + 1;
    }

    // Success is true only if every step succeeded.
    result.success = !result.steps.empty();
    for (const auto& s : result.steps) {
        if (!s.success) {
            result.success = false;
            break;
        }
    }
    if (options.cache_report) {
        result.cache_report = BuildCacheReport(graph, context, session_.loader());
    }
    return result;
}

DecodeBatchResult DecodeRunner::runBatch(const DecodeBatchOptions& options) {
    if (options.sequences.empty()) {
        throw std::runtime_error("DecodeRunner: no sequences provided for batch decode");
    }

    DecodeBatchResult batch;
    batch.success = true;

    auto graph = ExecutionPlanBuilder::BuildFromLoader(session_.loader());
    const size_t context_len = std::max<size_t>(1, graph.modelConfig().context_length);

    const auto& log_cfg = logitsTraceConfig();
    auto run_one = [&](const std::vector<uint64_t>& tokens, size_t seq_index) -> DecodeResult {
        DecodeResult result;
        if (tokens.empty()) {
            result.success = false;
            return result;
        }
        logTokenList(log_cfg, "input_tokens", seq_index, tokens);

        ExecutionContext context(session_, &graph);
        ExecutionExecutor executor(graph, &BackendRegistry::Default(), &context);

        const size_t steps = (options.max_steps > 0)
                                 ? std::min(tokens.size(), options.max_steps)
                                 : tokens.size();
        size_t cursor = options.start_position;

        for (size_t i = 0; i < steps; ++i) {
            DecodeStep step;
            step.token = tokens[i];
            size_t pos = cursor;
            if (pos >= context_len) {
                if (options.evict_on_full) {
                    context.clearStateTensors();
                    pos = 0;
                    cursor = 0;
                    step.cache_evicted = true;
                } else {
                    step.success = false;
                    step.position = context_len - 1;
                    step.error = "Decode aborted: context length exceeded and eviction disabled";
                    result.steps.push_back(std::move(step));
                    break;
                }
            }
            step.position = pos;

            context.setToken(step.token);
            context.setSequencePosition(pos);

            static const std::vector<std::string> kTokenInputs = {"tokens", "token_ids"};
            for (const auto& name : kTokenInputs) {
                if (graph.tensors().count(name)) {
                    context.setTensor(name, {static_cast<float>(step.token)});
                }
            }

            auto exec_res = executor.run();
            step.success = exec_res.success;
            for (const auto& entry : exec_res.trace) {
                step.trace.push_back(formatTraceEntry(entry));
            }

            if (!exec_res.success) {
                if (!exec_res.trace.empty()) {
                    step.error = formatTraceEntry(exec_res.trace.back());
                } else {
                    step.error = "execution failed";
                }
                result.steps.push_back(std::move(step));
                break;
            }

            if (const auto* logits = context.getTensor("logits")) {
                step.logits = *logits;
                if (options.top_k > 0) {
                    auto top = softmaxTopK(step.logits, options.top_k);
                    step.top_indices = std::move(top.first);
                    step.top_probs = std::move(top.second);
                }
                logLogits(log_cfg, "input", seq_index, i, step.token, pos, step.logits, options.top_k);
            }

            result.steps.push_back(std::move(step));
            cursor = pos + 1;
        }

        result.success = !result.steps.empty();
        for (const auto& s : result.steps) {
            if (!s.success) {
                result.success = false;
                break;
            }
        }
        if (options.cache_report) {
            result.cache_report = BuildCacheReport(graph, context, session_.loader());
        }
        return result;
    };

    for (size_t i = 0; i < options.sequences.size(); ++i) {
        auto res = run_one(options.sequences[i], i);
        if (!res.success) {
            batch.success = false;
        }
        batch.results.push_back(std::move(res));
    }

    return batch;
}

} // namespace runtime
} // namespace mlc
