#include "runtime/execution_executor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "runtime/kernel_registry.hpp"

namespace mlc {
namespace runtime {

namespace {
size_t elementCount(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 0;
    size_t total = 1;
    for (int64_t dim : shape) {
        total *= static_cast<size_t>(std::max<int64_t>(1, dim));
    }
    return total;
}

struct TraceConfig {
    bool enabled = false;
    std::vector<std::string> filters;
};

struct TracePreviewConfig {
    size_t count = 0;
};

FILE* traceFileHandle() {
    static FILE* file = []() -> FILE* {
        const char* path = std::getenv("MLC_TRACE_TENSORS_FILE");
        if (!path || !*path) return nullptr;
        FILE* handle = std::fopen(path, "a");
        if (!handle) return nullptr;
        std::setvbuf(handle, nullptr, _IOLBF, 0);
        return handle;
    }();
    return file;
}

const TraceConfig& traceConfig() {
    static TraceConfig cfg = []() {
        TraceConfig out;
        const char* env = std::getenv("MLC_TRACE_TENSORS");
        const char* file_env = std::getenv("MLC_TRACE_TENSORS_FILE");
        if ((!env || !*env) && (!file_env || !*file_env)) return out;
        out.enabled = true;
        if (!env || !*env) return out;
        std::string value(env);
        if (value == "1" || value == "all") return out;
        size_t start = 0;
        while (start < value.size()) {
            size_t end = value.find(',', start);
            std::string token = value.substr(start, end - start);
            size_t first = token.find_first_not_of(" \t");
            size_t last = token.find_last_not_of(" \t");
            if (first != std::string::npos && last != std::string::npos) {
                token = token.substr(first, last - first + 1);
                if (!token.empty()) {
                    out.filters.push_back(token);
                }
            }
            if (end == std::string::npos) break;
            start = end + 1;
        }
        return out;
    }();
    return cfg;
}

const TracePreviewConfig& tracePreviewConfig() {
    static TracePreviewConfig cfg = []() {
        TracePreviewConfig out;
        const char* env = std::getenv("MLC_TRACE_TENSORS_PREVIEW");
        if (!env || !*env) return out;
        char* end = nullptr;
        unsigned long val = std::strtoul(env, &end, 10);
        if (end && *end == '\0' && val > 0) {
            out.count = static_cast<size_t>(val);
        }
        return out;
    }();
    return cfg;
}

bool traceMatch(const TraceConfig& cfg,
                const std::string& node_name,
                const std::string& tensor_name) {
    if (!cfg.enabled) return false;
    if (cfg.filters.empty()) return true;
    for (const auto& token : cfg.filters) {
        if (node_name.find(token) != std::string::npos ||
            tensor_name.find(token) != std::string::npos) {
            return true;
        }
    }
    return false;
}

struct TensorStats {
    size_t count = 0;
    size_t finite = 0;
    size_t zeros = 0;
    size_t nans = 0;
    size_t infs = 0;
    float min = 0.0f;
    float max = 0.0f;
    double mean = 0.0;
};

TensorStats computeStats(const std::vector<float>& data) {
    TensorStats stats;
    stats.count = data.size();
    if (data.empty()) return stats;
    bool have_finite = false;
    double sum = 0.0;
    for (float v : data) {
        if (v == 0.0f) {
            ++stats.zeros;
        }
        if (std::isnan(v)) {
            ++stats.nans;
            continue;
        }
        if (std::isinf(v)) {
            ++stats.infs;
            continue;
        }
        if (!have_finite) {
            stats.min = v;
            stats.max = v;
            have_finite = true;
        } else {
            stats.min = std::min(stats.min, v);
            stats.max = std::max(stats.max, v);
        }
        sum += static_cast<double>(v);
        ++stats.finite;
    }
    if (stats.finite > 0) {
        stats.mean = sum / static_cast<double>(stats.finite);
    }
    return stats;
}
}

ExecutionExecutor::ExecutionExecutor(const ExecutionGraph& graph,
                                     const BackendRegistry* registry,
                                     ExecutionContext* context)
    : graph_(graph),
      registry_(registry ? registry : &BackendRegistry::Default()),
      context_(context) {
    if (context_) {
        context_->setGraph(&graph_);
        for (const auto& [name, tensor] : graph_.tensors()) {
            if (!tensor.is_state) continue;
            context_->ensureStateTensor(tensor);
        }
    }
}

ExecutionExecutor::Result ExecutionExecutor::run(size_t max_nodes) const {
    Result result;
    const bool verbose = (std::getenv("MLC_VERBOSE") != nullptr);
        const TraceConfig& trace_cfg = traceConfig();
    const TracePreviewConfig& preview_cfg = tracePreviewConfig();
    auto order = graph_.topologicalOrder();
    if (max_nodes > 0 && max_nodes < order.size()) {
        order.resize(max_nodes);
    }

    std::unordered_map<std::string, const ExecutionNode*> node_lookup;
    for (const auto& node : graph_.nodes()) {
        node_lookup[node.name] = &node;
    }

    std::unordered_map<std::string, int> producer_count;
    std::unordered_set<std::string> available;
    for (const auto& [name, tensor] : graph_.tensors()) {
        producer_count[name] = 0;
        if (tensor.is_state) {
            available.insert(name); // states can be treated as persistent inputs
        }
    }
    for (const auto& node : graph_.nodes()) {
        for (const auto& output : node.outputs) {
            producer_count[output]++;
        }
    }
    for (const auto& [name, count] : producer_count) {
        if (count == 0) {
            available.insert(name);
        }
    }
    if (context_) {
        for (const auto& name : context_->tensorNames()) {
            available.insert(name);
        }
    }

    for (const auto& node_name : order) {
        auto it = node_lookup.find(node_name);
        if (it == node_lookup.end()) continue;
        const ExecutionNode* node = it->second;
        ExecutionTraceEntry entry;
        entry.node = node->name;
        entry.op = node->op;
        entry.backend = node->backend;

        for (const auto& input : node->inputs) {
            if (input.empty()) continue;
            if (!available.count(input)) {
                entry.success = false;
                entry.missing_inputs.push_back(input);
            }
        }

        const KernelDescriptor* descriptor = nullptr;
        if (!node->kernel_id.empty()) {
            descriptor = KernelDescriptorRegistry::Instance().findById(node->kernel_id);
        }
        const auto& backend = registry_->backendFor(node->backend);
        auto backend_result = backend.execute(*node, context_, descriptor);
        if (context_) {
            context_->recordDispatch(node->name, backend_result.actual_backend);
        }
        if (!backend_result.message.empty()) {
            entry.notes.push_back(backend_result.message);
        }
        if (!backend_result.kernel_id.empty()) {
            entry.notes.push_back("kernel=" + backend_result.kernel_id);
        }
        entry.success &= backend_result.success;

        if (!entry.success) {
            result.success = false;
        }

        if (context_ && trace_cfg.enabled) {
            for (const auto& output : node->outputs) {
                if (!traceMatch(trace_cfg, node->name, output)) continue;
                const auto* tensor = context_->getTensor(output);
                if (!tensor) continue;
                TensorStats stats = computeStats(*tensor);
                if (FILE* trace_file = traceFileHandle()) {
                    uint64_t token_id = context_ ? context_->token() : 0;
                    size_t pos = context_ ? context_->sequencePosition() : 0;
                    std::fprintf(trace_file,
                                 "[Trace] token=%llu pos=%zu node=%s output=%s size=%zu finite=%zu zeros=%zu nan=%zu inf=%zu min=%.6g max=%.6g mean=%.6g\n",
                                 static_cast<unsigned long long>(token_id),
                                 pos,
                                 node->name.c_str(),
                                 output.c_str(),
                                 stats.count,
                                 stats.finite,
                                 stats.zeros,
                                 stats.nans,
                                 stats.infs,
                                 stats.min,
                                 stats.max,
                                 stats.mean);
                    std::fflush(trace_file);
                }
                std::fprintf(stderr,
                             "[Trace] node=%s output=%s size=%zu finite=%zu zeros=%zu nan=%zu inf=%zu min=%.6g max=%.6g mean=%.6g\n",
                             node->name.c_str(),
                             output.c_str(),
                             stats.count,
                             stats.finite,
                             stats.zeros,
                             stats.nans,
                             stats.infs,
                             stats.min,
                             stats.max,
                             stats.mean);
                if (preview_cfg.count > 0) {
                    size_t preview = std::min(preview_cfg.count, tensor->size());
                    if (preview > 0) {
                        uint64_t token_id = context_ ? context_->token() : 0;
                        size_t pos = context_ ? context_->sequencePosition() : 0;
                        if (FILE* trace_file = traceFileHandle()) {
                            std::fprintf(trace_file,
                                         "[TraceValues] token=%llu pos=%zu node=%s output=%s count=%zu values=",
                                         static_cast<unsigned long long>(token_id),
                                         pos,
                                         node->name.c_str(),
                                         output.c_str(),
                                         preview);
                            for (size_t i = 0; i < preview; ++i) {
                                if (i > 0) std::fprintf(trace_file, ",");
                                std::fprintf(trace_file, "%.6g", (*tensor)[i]);
                            }
                            std::fprintf(trace_file, "\n");
                            std::fflush(trace_file);
                        }
                        std::fprintf(stderr,
                                     "[TraceValues] node=%s output=%s count=%zu values=",
                                     node->name.c_str(),
                                     output.c_str(),
                                     preview);
                        for (size_t i = 0; i < preview; ++i) {
                            if (i > 0) std::fprintf(stderr, ",");
                            std::fprintf(stderr, "%.6g", (*tensor)[i]);
                        }
                        std::fprintf(stderr, "\n");
                    }
                }
            }
            auto trace_state_tensor = [&](const std::string& tensor_name) {
                if (!traceMatch(trace_cfg, node->name, tensor_name)) return;
                const auto* tensor = context_->getTensor(tensor_name);
                if (!tensor) return;
                TensorStats stats = computeStats(*tensor);
                if (FILE* trace_file = traceFileHandle()) {
                    uint64_t token_id = context_ ? context_->token() : 0;
                    size_t pos = context_ ? context_->sequencePosition() : 0;
                    std::fprintf(trace_file,
                                 "[Trace] token=%llu pos=%zu node=%s output=%s size=%zu finite=%zu zeros=%zu nan=%zu inf=%zu min=%.6g max=%.6g mean=%.6g\n",
                                 static_cast<unsigned long long>(token_id),
                                 pos,
                                 node->name.c_str(),
                                 tensor_name.c_str(),
                                 stats.count,
                                 stats.finite,
                                 stats.zeros,
                                 stats.nans,
                                 stats.infs,
                                 stats.min,
                                 stats.max,
                                 stats.mean);
                    std::fflush(trace_file);
                }
                std::fprintf(stderr,
                             "[Trace] node=%s output=%s size=%zu finite=%zu zeros=%zu nan=%zu inf=%zu min=%.6g max=%.6g mean=%.6g\n",
                             node->name.c_str(),
                             tensor_name.c_str(),
                             stats.count,
                             stats.finite,
                             stats.zeros,
                             stats.nans,
                             stats.infs,
                             stats.min,
                             stats.max,
                             stats.mean);
                if (preview_cfg.count > 0) {
                    size_t preview = std::min(preview_cfg.count, tensor->size());
                    if (preview > 0) {
                        uint64_t token_id = context_ ? context_->token() : 0;
                        size_t pos = context_ ? context_->sequencePosition() : 0;
                        if (FILE* trace_file = traceFileHandle()) {
                            std::fprintf(trace_file,
                                         "[TraceValues] token=%llu pos=%zu node=%s output=%s count=%zu values=",
                                         static_cast<unsigned long long>(token_id),
                                         pos,
                                         node->name.c_str(),
                                         tensor_name.c_str(),
                                         preview);
                            for (size_t i = 0; i < preview; ++i) {
                                if (i > 0) std::fprintf(trace_file, ",");
                                std::fprintf(trace_file, "%.6g", (*tensor)[i]);
                            }
                            std::fprintf(trace_file, "\n");
                            std::fflush(trace_file);
                        }
                        std::fprintf(stderr,
                                     "[TraceValues] node=%s output=%s count=%zu values=",
                                     node->name.c_str(),
                                     tensor_name.c_str(),
                                     preview);
                        for (size_t i = 0; i < preview; ++i) {
                            if (i > 0) std::fprintf(stderr, ",");
                            std::fprintf(stderr, "%.6g", (*tensor)[i]);
                        }
                        std::fprintf(stderr, "\n");
                    }
                }
            };
            auto kv_it = node->annotations.find("kv_cache_k");
            if (kv_it != node->annotations.end()) {
                trace_state_tensor(kv_it->second);
            }
            kv_it = node->annotations.find("kv_cache_v");
            if (kv_it != node->annotations.end()) {
                trace_state_tensor(kv_it->second);
            }
        }

        for (const auto& output : node->outputs) {
            available.insert(output);
            auto tensor_it = graph_.tensors().find(output);
            if (tensor_it != graph_.tensors().end() && tensor_it->second.is_state) {
                std::ostringstream oss;
                oss << "updates state tensor '" << output << "'";
                entry.notes.push_back(oss.str());
            }
        }

        // Tap: capture per-tensor host snapshots if registered. Cheap when no taps.
        if (context_ && !context_->tapsEmpty()) {
            for (const auto& output : node->outputs) {
                context_->captureTapIfRegistered(output);
            }
        }

        if (node->annotations.count("kv_cache_k") || node->annotations.count("kv_cache_v")) {
            std::ostringstream oss;
            oss << "kv-cache";
            if (node->annotations.count("kv_cache_k")) {
                oss << " K=" << node->annotations.at("kv_cache_k");
            }
            if (node->annotations.count("kv_cache_v")) {
                oss << " V=" << node->annotations.at("kv_cache_v");
            }
            entry.notes.push_back(oss.str());
        }
        entry.notes.push_back("backend=" + toString(node->backend));

        if (verbose) {
            std::ostringstream oss;
            oss << "[Exec] " << node->name << " op=" << toString(node->op)
                << " backend=" << toString(node->backend)
                << " success=" << (entry.success ? "1" : "0");
            if (!entry.missing_inputs.empty()) {
                oss << " missing=";
                for (size_t i = 0; i < entry.missing_inputs.size(); ++i) {
                    if (i) oss << ",";
                    oss << entry.missing_inputs[i];
                }
            }
            if (!backend_result.message.empty()) {
                oss << " note=" << backend_result.message;
            }
            if (!backend_result.kernel_id.empty()) {
                oss << " kernel=" << backend_result.kernel_id;
            }
            fprintf(stderr, "%s\n", oss.str().c_str());
        }

        result.trace.push_back(entry);
    }

    result.executed_nodes = result.trace.size();
    return result;
}

std::string formatTraceEntry(const ExecutionTraceEntry& entry) {
    std::ostringstream oss;
    oss << entry.node << " (" << toString(entry.op) << ")";
    if (!entry.success) {
        oss << " [missing inputs: ";
        for (size_t i = 0; i < entry.missing_inputs.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << entry.missing_inputs[i];
        }
        oss << "]";
    }
    if (!entry.notes.empty()) {
        oss << " - ";
        for (size_t i = 0; i < entry.notes.size(); ++i) {
            if (i > 0) oss << "; ";
            oss << entry.notes[i];
        }
    }
    return oss.str();
}

} // namespace runtime
} // namespace mlc
