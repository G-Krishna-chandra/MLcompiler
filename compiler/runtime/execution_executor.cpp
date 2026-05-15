#include "runtime/execution_executor.hpp"

#include "frontends/ggml_types.hpp"

#include <algorithm>
#include <chrono>
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
bool nodeProfileEnabled() {
    static const bool enabled = (std::getenv("MLC_PROFILE_NODES") != nullptr);
    return enabled;
}
std::unordered_map<ExecOpType, ExecutionExecutor::OpProfileEntry>& mutableNodeProfile() {
    static std::unordered_map<ExecOpType, ExecutionExecutor::OpProfileEntry> table;
    return table;
}

bool fuseLayerEnabled() {
    static const bool enabled = (std::getenv("MLC_FUSE_LAYER") != nullptr);
    return enabled;
}

bool isFusableOp(ExecOpType op) {
    switch (op) {
        case ExecOpType::MatMul:
        case ExecOpType::Linear:
        case ExecOpType::Norm:
        case ExecOpType::Add:
        case ExecOpType::Slice:
            return true;
        default:
            return false;
    }
}
} // namespace

const std::unordered_map<ExecOpType, ExecutionExecutor::OpProfileEntry>&
ExecutionExecutor::nodeProfile() {
    return mutableNodeProfile();
}

void ExecutionExecutor::clearNodeProfile() {
    mutableNodeProfile().clear();
}

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

    // === Forward-pass CB + GPU-residency state ===
    // Open iff MetalExecutor::Instance().hasForwardPassCB() returns true.
    // (Naming: this used to be a per-fusion-window CB; commit-A renamed it
    // to "forward pass" to match the eventual scope of one CB per
    // executor.run() call. Today the CB still flushes on the first
    // non-fusable op encountered; later commits add encode entry points
    // for those ops so the CB stays open for the entire forward pass.)
    // pass_outputs maps output tensor name -> {pool buffer, host_dst,
    // element_count, needs_host}. A node whose input is in pass_outputs
    // can read directly from the GPU buffer (FromBuffer variant);
    // otherwise reads via host vec (FromHost variant).
    const bool fuse_layer = fuseLayerEnabled();
    struct PassOutput {
        void* buffer = nullptr;          // id<MTLBuffer>, opaque
        std::vector<float>* host_dst = nullptr;
        size_t element_count = 0;
        bool needs_host = false;
    };
    std::unordered_map<std::string, PassOutput> pass_outputs;

    // Pre-analysis: compute (a) per-tensor element count for fusable
    // producers and (b) per-tensor needs_host_output flag. needs_host=true
    // forces a host memcpy at flush; needs_host=false lets the buffer stay
    // in the pool and be consumed by a later FromBuffer encode.
    //
    // needs_host rule: a tensor needs host materialization if any consumer
    // is non-fusable (CPU op, non-fusable Metal op, or fusable op that
    // happens AFTER a non-fusable op in topo order — that mid-block flush
    // would have already returned the buffer to the pool), or if it's a
    // tap target, or if it has no consumers (final output).
    std::unordered_map<std::string, size_t> tensor_count;
    std::unordered_map<std::string, bool> tensor_needs_host;
    // Predicate must MIRROR what backend.encode() actually accepts. If
    // pre-analysis thinks an op is fusable but encode() rejects it at
    // runtime, the executor falls back to execute() — and execute()
    // reads inputs from context host vecs. If the predecessor was marked
    // needs_host=false because its consumer was thought fusable, the
    // host vec stays at the allocateTensor zero placeholder and execute()
    // reads zeros. Caught with lm_head (Q6_K matmul rejected at the
    // dtype gate in encode()).
    auto encodeWillAccept = [&](const ExecutionNode* n) -> bool {
        if (!n) return false;
        switch (n->op) {
            case ExecOpType::MatMul:
            case ExecOpType::Linear: {
                // Encode requires Q4_0 weight, no bias, non-transposed shape.
                auto wit = n->annotations.find("weight");
                if (wit == n->annotations.end()) return false;
                if (!context_) return false;
                const auto& tensors = context_->session().loader().tensors();
                auto t = tensors.find(wit->second);
                if (t == tensors.end() || t->second.shape.size() != 2) return false;
                if (t->second.dtype != frontend::GGML_TYPE_Q4_0) return false;
                if (n->annotations.count("bias")) return false;
                return true;
            }
            case ExecOpType::Norm: {
                // Encode requires RMS variety, no bias.
                auto wit = n->annotations.find("weight");
                if (wit == n->annotations.end()) return false;
                auto nk = n->annotations.find("norm_kind");
                if (nk != n->annotations.end() && nk->second == "layer") return false;
                if (n->annotations.count("bias")) return false;
                return true;
            }
            case ExecOpType::Add: {
                if (n->annotations.count("bias")) return false;
                return n->inputs.size() >= 2;
            }
            case ExecOpType::Slice: {
                // Lever 5: Slice over a single fp32 input with constant offset/length.
                auto off = n->attributes.find("slice_offset");
                auto len = n->attributes.find("slice_length");
                if (off == n->attributes.end() || len == n->attributes.end()) return false;
                if (n->inputs.size() < 1) return false;
                return true;
            }
            default: return false;
        }
    };
    const size_t preanalysis_seq_len = context_ ? context_->seqLen() : 1;
    std::vector<bool> is_nonfusable;
    is_nonfusable.reserve(order.size());
    std::unordered_map<std::string, size_t> node_topo_pos;
    for (size_t i = 0; i < order.size(); ++i) {
        node_topo_pos[order[i]] = i;
        auto it = node_lookup.find(order[i]);
        if (it == node_lookup.end()) { is_nonfusable.push_back(true); continue; }
        const auto* n = it->second;
        bool fus = fuse_layer
                && preanalysis_seq_len <= 1
                && isFusableOp(n->op)
                && n->backend == BackendKind::Metal
                && MetalExecutor::Instance().shouldUseFor(*n)
                && encodeWillAccept(n);
        is_nonfusable.push_back(!fus);
    }
    // Prefix sum of is_nonfusable: nonfusable_prefix[i] = count in [0, i).
    std::vector<size_t> nonfusable_prefix(order.size() + 1, 0);
    for (size_t i = 0; i < order.size(); ++i) {
        nonfusable_prefix[i+1] = nonfusable_prefix[i] + (is_nonfusable[i] ? 1 : 0);
    }
    // Per-tensor consumer positions.
    std::unordered_map<std::string, std::vector<size_t>> tensor_consumers;
    for (size_t ci = 0; ci < order.size(); ++ci) {
        auto it = node_lookup.find(order[ci]);
        if (it == node_lookup.end()) continue;
        for (const auto& in : it->second->inputs) {
            if (!in.empty()) tensor_consumers[in].push_back(ci);
        }
    }
    // Fill tensor_count for fusable ops.
    //
    // For MatMul/Linear we read shape[1] (rows) from the weight tensor in
    // the GGUF loader — this is the authoritative output dim, and notably
    // is the only correct source for GQA outputs (graph_.tensors() stores a
    // placeholder hidden_size for attn_k.out/attn_v.out, not the real
    // [kv_heads * head_dim] count).
    //
    // For Norm/Add we read the graph tensor's stored shape, which IS
    // reliable for these element-wise ops (single dim = hidden_size).
    // We do NOT chain through n->inputs[0]'s tensor_count: residual_add
    // nodes order their inputs as [residual_stream, new_addend] so input[0]
    // is the stream — which for block 0 has no tensor_count (it's the
    // embedding output, produced by a non-fusable op). Reading the graph
    // shape avoids that input-order trap.
    const auto& graph_tensors = graph_.tensors();
    auto elementCountFromGraphShape = [&](const std::string& name) -> size_t {
        auto git = graph_tensors.find(name);
        if (git == graph_tensors.end()) return 0;
        const auto& sh = git->second.shape;
        if (sh.empty()) return 0;
        size_t count = 1;
        for (int64_t d : sh) count *= static_cast<size_t>(std::max<int64_t>(1, d));
        return count;
    };
    for (size_t pi = 0; pi < order.size(); ++pi) {
        auto it = node_lookup.find(order[pi]);
        if (it == node_lookup.end()) continue;
        const auto* n = it->second;
        if (is_nonfusable[pi]) continue;
        if (n->outputs.empty()) continue;
        const auto& out = n->outputs[0];
        if (out.empty()) continue;
        size_t out_count = 0;
        switch (n->op) {
            case ExecOpType::MatMul:
            case ExecOpType::Linear: {
                auto wit = n->annotations.find("weight");
                if (wit == n->annotations.end()) break;
                if (!context_) break;
                const auto& tensors = context_->session().loader().tensors();
                auto t = tensors.find(wit->second);
                if (t == tensors.end() || t->second.shape.size() != 2) break;
                out_count = static_cast<size_t>(t->second.shape[1]);
                break;
            }
            case ExecOpType::Norm:
            case ExecOpType::Add: {
                out_count = elementCountFromGraphShape(out);
                if (out_count == 0) {
                    // Belt + suspenders: if graph shape is missing, fall
                    // back to searching ALL inputs for a known tensor_count.
                    for (const auto& in : n->inputs) {
                        if (in.empty()) continue;
                        auto cit = tensor_count.find(in);
                        if (cit != tensor_count.end()) { out_count = cit->second; break; }
                    }
                }
                break;
            }
            default: break;
        }
        if (out_count > 0) tensor_count[out] = out_count;
    }
    // Compute needs_host per tensor produced by a fusable op.
    for (size_t pi = 0; pi < order.size(); ++pi) {
        auto it = node_lookup.find(order[pi]);
        if (it == node_lookup.end()) continue;
        const auto* n = it->second;
        if (is_nonfusable[pi]) continue;
        for (const auto& out : n->outputs) {
            if (out.empty()) continue;
            bool host = false;
            auto cit = tensor_consumers.find(out);
            if (cit == tensor_consumers.end() || cit->second.empty()) {
                host = true;   // no consumers — final output, must materialize
            } else {
                for (size_t ci : cit->second) {
                    // Any non-fusable at positions in (pi, ci] forces flush.
                    if (ci > pi && nonfusable_prefix[ci+1] - nonfusable_prefix[pi+1] > 0) {
                        host = true; break;
                    }
                }
            }
            if (!host && context_ && context_->isTapped(out)) host = true;
            tensor_needs_host[out] = host;
        }
    }

    // Pre-analysis dump for needs_host debugging. Writes one line per
    // tensor in tensor_needs_host to stderr (or MLC_DUMP_PREANALYSIS=
    // <path>) when set. Columns: name | needs_host | n_consumers |
    // first_consumer_node | first_consumer_op | first_consumer_backend |
    // first_consumer_fusable. Used to find pre-analysis bugs where a
    // tensor classified needs_host=false actually has a host-vec
    // reader downstream.
    if (const char* env = std::getenv("MLC_DUMP_PREANALYSIS")) {
        FILE* out = stderr;
        FILE* opened = nullptr;
        if (env[0] && env[0] != '1') {
            opened = std::fopen(env, "w");
            if (opened) out = opened;
        }
        std::fprintf(out, "# pre-analysis dump: tensor | needs_host | n_consumers | first_consumer (node, op, backend, fusable)\n");
        // Sort by topo position of producer so the dump reads in graph
        // order — easier to find the breaking tensor.
        std::vector<std::pair<size_t, std::string>> sorted;
        sorted.reserve(tensor_needs_host.size());
        for (const auto& kv : tensor_needs_host) {
            size_t producer_pos = order.size();  // unknown -> sort at end
            for (size_t pi = 0; pi < order.size(); ++pi) {
                auto nit = node_lookup.find(order[pi]);
                if (nit == node_lookup.end()) continue;
                bool found = false;
                for (const auto& o : nit->second->outputs) {
                    if (o == kv.first) { found = true; break; }
                }
                if (found) { producer_pos = pi; break; }
            }
            sorted.emplace_back(producer_pos, kv.first);
        }
        std::sort(sorted.begin(), sorted.end());
        for (const auto& [pos, name] : sorted) {
            bool needs_host = tensor_needs_host[name];
            const auto& consumers = tensor_consumers[name];
            std::string first_node = "(none)";
            std::string first_op = "-";
            std::string first_backend = "-";
            std::string first_fusable = "-";
            if (!consumers.empty()) {
                size_t ci = consumers.front();
                if (ci < order.size()) {
                    first_node = order[ci];
                    auto cnit = node_lookup.find(order[ci]);
                    if (cnit != node_lookup.end()) {
                        first_op = toString(cnit->second->op);
                        first_backend = toString(cnit->second->backend);
                        first_fusable = is_nonfusable[ci] ? "no" : "yes";
                    }
                }
            }
            std::fprintf(out, "%-40s | needs_host=%-5s | n_cons=%zu | first=%s op=%s backend=%s fusable=%s",
                         name.c_str(),
                         needs_host ? "true" : "false",
                         consumers.size(),
                         first_node.c_str(),
                         first_op.c_str(),
                         first_backend.c_str(),
                         first_fusable.c_str());
            // For needs_host=false with multiple consumers, list them all
            // so we can spot any consumer that's actually after a flush boundary.
            if (!needs_host && consumers.size() > 1) {
                std::fprintf(out, " | all_cons=[");
                for (size_t ix = 0; ix < consumers.size(); ++ix) {
                    size_t cj = consumers[ix];
                    if (ix > 0) std::fprintf(out, ", ");
                    if (cj < order.size()) {
                        std::fprintf(out, "%s@pi=%zu", order[cj].c_str(), cj);
                        auto cnit2 = node_lookup.find(order[cj]);
                        if (cnit2 != node_lookup.end()) {
                            std::fprintf(out, "(%s,%s)",
                                         is_nonfusable[cj] ? "nonfusable" : "fusable",
                                         toString(cnit2->second->op).c_str());
                        }
                    }
                }
                std::fprintf(out, "]");
            }
            std::fprintf(out, "\n");
        }
        std::fflush(out);
        if (opened) std::fclose(opened);
    }

    auto flushWindow = [&]() {
        if (MetalExecutor::Instance().hasForwardPassCB()) {
            // Resolver re-looks up host_dst by tensor name at flush time.
            // Survives unordered_map rehashes between encode and flush —
            // matters once a single forward-pass CB stays open across
            // many ops (commit B4). Legacy raw-pointer readbacks (empty
            // tensor_name) keep working via the captured pointer.
            auto resolver = [&](const std::string& name) -> std::vector<float>* {
                return context_ ? context_->mutableTensor(name) : nullptr;
            };
            MetalExecutor::Instance().flushForwardPassCB(resolver);
        }
        pass_outputs.clear();
    };
    // RAII discard on abnormal exit (exception): drop the window without
    // committing. Normal close path is flushWindow() called explicitly
    // below before each non-fusable node and once after the loop.
    struct WindowDiscardGuard {
        ~WindowDiscardGuard() {
            if (MetalExecutor::Instance().hasForwardPassCB()) {
                MetalExecutor::Instance().discardForwardPassCB();
            }
        }
    } discard_guard;
    (void)discard_guard;

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

        // Fusion decision. A node is "fusable" if MLC_FUSE_LAYER is set,
        // it's a supported op type, and it dispatches to Metal. For
        // fusable nodes we open a window, checkout an output buffer from
        // the pool, look up any GPU-resident inputs from pass_outputs,
        // and call backend.encode() with FusionInputs. Non-fusable nodes
        // force a window flush (drains any pending readbacks) then run
        // synchronously via execute().
        // Multi-token forward pass (item 3 prefill batching) disables fusion:
        // encode() expects single-token tensors and the pool buffers are
        // single-output-sized. Multi-token traffic flows through execute(),
        // which falls into per-token loops in each handler.
        const size_t exec_seq_len = context_ ? context_->seqLen() : 1;
        bool fusable = fuse_layer
                    && exec_seq_len <= 1
                    && isFusableOp(node->op)
                    && node->backend == BackendKind::Metal
                    && MetalExecutor::Instance().shouldUseFor(*node);

        // Need the output count to checkout a pool buffer. If unknown,
        // skip fusion for this node.
        size_t output_count = 0;
        if (fusable && !node->outputs.empty() && !node->outputs[0].empty()) {
            auto cit = tensor_count.find(node->outputs[0]);
            if (cit != tensor_count.end()) output_count = cit->second;
        }
        if (fusable && output_count == 0) {
            fusable = false;
        }

        // CB lifetime extension (commit B4): keep the forward-pass CB
        // open across single-token Metal-eligible non-fusable ops too
        // (Attention, FeedForward, Linear matmul). Their dual-mode run*
        // detects the open CB and defers their result download. CPU ops
        // and multi-token recursion still flush before running.
        //
        // Only engage when fuse_layer is on. With fusion off, the
        // surrounding fusable ops (Norm, MatMul, Add) all run via
        // execute() instead of encode(), each with its own CB; if
        // attention deferred onto a CB those ops weren't writing to,
        // the deferred attention output would never become host-visible
        // before the next fusable consumer reads it.
        const bool metal_eligible_node = fuse_layer
                                       && (exec_seq_len <= 1)
                                       && node->backend == BackendKind::Metal
                                       && MetalExecutor::Instance().shouldUseFor(*node);
        if (!fusable && !metal_eligible_node) {
            flushWindow();
        }
        if ((fusable || metal_eligible_node) &&
            !MetalExecutor::Instance().hasForwardPassCB()) {
            if (!MetalExecutor::Instance().beginForwardPassCB()) {
                fusable = false;
            }
        }

        // Decision 4: lm_head computes for final token only by default during
        // multi-token prefill. MLC_PREFILL_ALL_LOGITS=1 keeps all-tokens for
        // harness compatibility. Identified by output tensor name "logits"
        // (lm_head is the last node and the only producer of that tensor).
        const bool lm_head_truncate = context_ &&
            context_->seqLen() > 1 &&
            !node->outputs.empty() && node->outputs[0] == "logits" &&
            (std::getenv("MLC_PREFILL_ALL_LOGITS") == nullptr);
        if (lm_head_truncate && !node->inputs.empty()) {
            const auto* hidden = context_->getTensor(node->inputs[0]);
            if (hidden && hidden->size() % context_->seqLen() == 0) {
                size_t per = hidden->size() / context_->seqLen();
                std::vector<float> last(hidden->end() - per, hidden->end());
                context_->setTensor(node->inputs[0], std::move(last));
                context_->setSeqLen(1);
            }
        }

        auto t_op_begin = std::chrono::steady_clock::now();
        BackendExecutionResult backend_result;
        if (fusable) {
            // Look up input residency. For Add we may have two inputs that
            // are independently GPU-or-host. For matmul/norm we only check
            // input 0 (input 1 is weight, fetched from session).
            FusionInputs fi;
            if (!node->inputs.empty() && !node->inputs[0].empty()) {
                auto wit = pass_outputs.find(node->inputs[0]);
                if (wit != pass_outputs.end()) {
                    fi.primary_input_buffer = wit->second.buffer;
                    fi.primary_input_count = wit->second.element_count;
                }
            }
            if (node->op == ExecOpType::Add && node->inputs.size() >= 2 && !node->inputs[1].empty()) {
                auto wit = pass_outputs.find(node->inputs[1]);
                if (wit != pass_outputs.end()) {
                    fi.secondary_input_buffer = wit->second.buffer;
                    fi.secondary_input_count = wit->second.element_count;
                }
            }
            // Checkout output buffer and track for window-end return.
            void* out_buf = MetalExecutor::Instance().checkoutPoolBuffer(output_count * sizeof(float));
            if (!out_buf) {
                // Pool checkout failed; fall back.
                flushWindow();
                backend_result = backend.execute(*node, context_, descriptor);
                fusable = false;
            } else {
                MetalExecutor::Instance().trackWindowBuffer(out_buf);
                // Stable host destination for drain.
                std::vector<float>& host_dst =
                    context_->allocateTensor(node->outputs[0], output_count, /*zero_initialize=*/false);
                auto nh_it = tensor_needs_host.find(node->outputs[0]);
                bool needs_host = (nh_it == tensor_needs_host.end()) ? true : nh_it->second;
                fi.output_buffer = out_buf;
                fi.host_dst = &host_dst;
                fi.needs_host_output = needs_host;
                backend_result = backend.encode(*node, context_, descriptor, fi);
                if (!backend_result.success) {
                    // Encode rejected. The output buffer is still in
                    // pass_checked_out_ and will be returned at flush;
                    // host_dst is allocated and execute() will repopulate
                    // via context->setTensor (which replaces the entry).
                    flushWindow();
                    backend_result = backend.execute(*node, context_, descriptor);
                } else {
                    PassOutput wo;
                    wo.buffer = out_buf;
                    wo.host_dst = &host_dst;
                    wo.element_count = output_count;
                    wo.needs_host = needs_host;
                    pass_outputs[node->outputs[0]] = wo;
                }
            }
        } else {
            backend_result = backend.execute(*node, context_, descriptor);
        }
        if (nodeProfileEnabled()) {
            auto t_op_end = std::chrono::steady_clock::now();
            auto& entry = mutableNodeProfile()[node->op];
            entry.total_ms += std::chrono::duration<double, std::milli>(t_op_end - t_op_begin).count();
            entry.calls += 1;
        }
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

        // CB-batching (B4): if a non-fusable Metal op deferred onto the
        // open forward-pass CB and exposed a (fp32) GPU result buffer,
        // register it in pass_outputs so the next FromBuffer-capable
        // consumer (residual_1 Add, etc.) reads from GPU memory instead
        // of the empty host slot. The deferred readback in
        // flushForwardPassCB still populates the host slot for any
        // non-FromBuffer consumer or for tap captures.
        if (backend_result.gpu_output_buffer && backend_result.gpu_output_element_count > 0
            && !node->outputs.empty() && !node->outputs[0].empty()) {
            const std::string& out_name = node->outputs[0];
            std::vector<float>* host_dst = context_ ? context_->mutableTensor(out_name) : nullptr;
            if (pass_outputs.find(out_name) == pass_outputs.end() && host_dst) {
                PassOutput po;
                po.buffer = backend_result.gpu_output_buffer;
                po.host_dst = host_dst;
                po.element_count = backend_result.gpu_output_element_count;
                po.needs_host = true;
                pass_outputs[out_name] = po;
            }
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
        //
        // Taps require committed data. When a fused-and-still-pending output is
        // tapped, force a flush so the tap snapshot reads real GPU output instead
        // of allocateTensor's zero-initialized placeholder. Consequence: any run
        // with taps registered (e.g. `mlc compare --metal-vs-cpu`) collapses
        // fusion windows wherever tapped outputs land, so the per-op profile and
        // turn timings measured in compare are NOT representative of production
        // fusion behavior. For perf measurement, use `mlc chat` or `mlc chat-repl`,
        // not the parity compare path.
        if (context_ && !context_->tapsEmpty()) {
            bool needs_flush_for_tap = false;
            for (const auto& output : node->outputs) {
                if (output.empty() || !context_->isTapped(output)) continue;
                // Two ways the data may not yet be on host:
                //   (a) fusable encode put the output in pass_outputs;
                //   (b) non-fusable Metal op deferred its readback (B4).
                // Either way, flush before captureTap reads stale host slot.
                if (pass_outputs.count(output) ||
                    MetalExecutor::Instance().hasDeferredReadback(output)) {
                    needs_flush_for_tap = true;
                    break;
                }
            }
            if (needs_flush_for_tap) flushWindow();
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

    // End-of-graph flush: any remaining encoded work commits + drains here.
    flushWindow();

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
