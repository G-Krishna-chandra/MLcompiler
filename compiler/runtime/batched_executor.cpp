#include "runtime/batched_executor.hpp"

#include "runtime/batched_walker.hpp"
#include "runtime/execution_plan_builder.hpp"
#include "runtime/kernel_registry.hpp"
#include "runtime/operator_backend.hpp"
#include "runtime/quantization.hpp"
#include "runtime/float_convert.hpp"
#include "runtime/metal_runtime.hpp"

#include <cstring>
#include <stdexcept>

namespace mlc {
namespace runtime {

namespace {
constexpr const char* kLogitsTensorName = "logits";
const std::vector<std::string>& tokenInputNames() {
    static const std::vector<std::string> names = {"tokens", "token_ids"};
    return names;
}
}

BatchedExecutor::BatchedExecutor(const Session& session)
    : session_(session),
      graph_(ExecutionPlanBuilder::BuildFromLoader(session.loader())) {}

void BatchedExecutor::reset() {
    if (page_pool_) {
        for (auto& kv : requests_) {
            if (kv.second.page_state) kv.second.page_state->release_all(*page_pool_);
        }
    }
    requests_.clear();
}

void BatchedExecutor::release_request(uint32_t request_id) {
    auto it = requests_.find(request_id);
    if (it == requests_.end()) return;
    if (page_pool_ && it->second.page_state) {
        it->second.page_state->release_all(*page_pool_);
    }
    requests_.erase(it);
}

void BatchedExecutor::attach_page_pool(PagePool* pool, uint32_t page_size_tokens) {
    page_pool_ = pool;
    page_size_tokens_ = page_size_tokens > 0 ? page_size_tokens : 64;
}

const RequestKVState* BatchedExecutor::page_state(uint32_t request_id) const {
    auto it = requests_.find(request_id);
    if (it == requests_.end()) return nullptr;
    return it->second.page_state.get();
}

bool BatchedExecutor::advance_page_state(PerRequest& req) {
    if (!page_pool_) return true;
    if (!req.page_state) {
        req.page_state = std::make_unique<RequestKVState>();
        req.page_state->page_size_tokens = page_size_tokens_;
    }
    return req.page_state->extend_one_token(*page_pool_).has_value();
}

size_t BatchedExecutor::known_position(uint32_t request_id) const {
    auto it = requests_.find(request_id);
    return it == requests_.end() ? 0 : it->second.next_position;
}

BatchedExecutor::PerRequest& BatchedExecutor::ensure_request(uint32_t request_id) {
    auto it = requests_.find(request_id);
    if (it != requests_.end()) return it->second;
    PerRequest pr;
    pr.context  = std::make_unique<ExecutionContext>(session_, &graph_);
    pr.executor = std::make_unique<ExecutionExecutor>(graph_,
                                                      &BackendRegistry::Default(),
                                                      pr.context.get());
    // Match chat-repl's startup: zero KV state and invalidate any cached
    // MetalBuffer wrappers. Without this, fuse-mode reads stale GPU
    // buffers cached by the executor singleton during a prior request.
    pr.context->clearStateTensors();
    auto inserted = requests_.emplace(request_id, std::move(pr));
    return inserted.first->second;
}

void BatchedExecutor::prime_token_tensors(ExecutionContext& ctx, uint64_t token) const {
    for (const auto& name : tokenInputNames()) {
        if (graph_.tensors().count(name)) {
            ctx.setTensor(name, {static_cast<float>(token)});
        }
    }
}

BatchedDecodeOutput BatchedExecutor::run_decode(const std::vector<RequestSlot>& slots) {
    // G1: when paged storage AND page pool are both attached, route through
    // the chunked-walk path that uses paged_flash_attention for batched
    // attention. The Scheduler invokes run_decode via the IBatchedExecutor
    // interface; this dispatch keeps the scheduler agnostic to the path.
    if (paged_storage_ && page_pool_ && !slots.empty()) {
        return run_decode_paged_impl(slots);
    }

    BatchedDecodeOutput out;
    out.per_request.reserve(slots.size());
    if (slots.empty()) return out;

    for (const auto& slot : slots) {
        BatchedDecodeOutput::PerRequest per;
        per.request_id = slot.request_id;

        auto& req = ensure_request(slot.request_id);
        ExecutionContext& ctx = *req.context;

        ctx.setToken(slot.input_token);
        ctx.setSequencePosition(slot.sequence_position);
        prime_token_tensors(ctx, slot.input_token);

        auto exec_res = req.executor->run();
        per.success = exec_res.success;
        if (per.success) {
            if (const auto* logits = ctx.getTensor(kLogitsTensorName)) {
                per.logits = *logits;
            }
            req.next_position = slot.sequence_position + 1;
            // B3: extend the request's paged-KV table by one slot. Failure
            // here (pool exhaustion) is reported as a soft failure on the
            // slot — the per-request KV in the contiguous context remains
            // intact, but the scheduler should treat this as a stop signal.
            if (!advance_page_state(req)) {
                per.success = false;
                per.logits.clear();
            }
        }
        out.per_request.push_back(std::move(per));
    }
    return out;
}

// ===========================================================================
// Phase G1: chunked-walk decode using paged_flash_attention.
// ===========================================================================

namespace {
const ExecutionNode* findNode(const ExecutionGraph& g, const std::string& name) {
    for (const auto& n : g.nodes()) if (n.name == name) return &n;
    return nullptr;
}

int parseLayerIndex(const std::string& node_name) {
    // Names like "blk.5.attention" — extract 5.
    auto first_dot = node_name.find('.');
    if (first_dot == std::string::npos) return -1;
    auto second_dot = node_name.find('.', first_dot + 1);
    if (second_dot == std::string::npos) return -1;
    try {
        return std::stoi(node_name.substr(first_dot + 1, second_dot - first_dot - 1));
    } catch (...) { return -1; }
}
} // namespace

void BatchedExecutor::cache_topology() const {
    if (!attention_node_indices_.empty()) return;
    topo_order_ = graph_.topologicalOrder();
    for (size_t i = 0; i < topo_order_.size(); ++i) {
        const auto* n = findNode(graph_, topo_order_[i]);
        if (n && n->op == ExecOpType::Attention) {
            attention_node_indices_.push_back(i);
            attention_layer_index_.push_back(parseLayerIndex(n->name));
        }
    }
}

BatchedDecodeOutput BatchedExecutor::run_decode_paged(const std::vector<RequestSlot>& slots) {
    return run_decode_paged_impl(slots);
}

BatchedDecodeOutput BatchedExecutor::run_decode_paged_impl(
    const std::vector<RequestSlot>& slots) {
    BatchedDecodeOutput out;
    if (slots.empty()) return out;
    if (!page_pool_ || !paged_storage_) {
        // Fall back to the non-paged sequential path.
        return run_decode(slots);
    }

    // H3: when paged storage is attached, dispatch through BatchedWalker
    // (op-by-op batched). Each request's per-op work is gathered into a
    // [N, dim] dispatch instead of N sequential per-request executor calls.
    BatchedWalker walker(*this);
    return walker.step(slots);

    cache_topology();
    const size_t N = slots.size();
    out.per_request.reserve(N);

    auto& metal = MetalExecutor::Instance();

    // Prime each request: setup token / position / token tensor; ensure each
    // request has at least one allocated page in its page table for the new K/V.
    struct PerCall {
        uint32_t request_id;
        PerRequest* req;
        bool ok;
        uint32_t page_id;
        uint32_t slot_in_page;
    };
    std::vector<PerCall> calls;
    calls.reserve(N);

    for (const auto& slot : slots) {
        PerCall c{slot.request_id, &ensure_request(slot.request_id), true, 0, 0};
        ExecutionContext& ctx = *c.req->context;
        ctx.setToken(slot.input_token);
        ctx.setSequencePosition(slot.sequence_position);
        prime_token_tensors(ctx, slot.input_token);
        // Reserve one slot for this token's K/V via the page pool.
        auto loc = c.req->page_state
            ? c.req->page_state->extend_one_token(*page_pool_)
            : std::optional<std::pair<uint32_t, uint32_t>>{};
        if (!c.req->page_state) {
            // First call for this request — create the page state.
            c.req->page_state = std::make_unique<RequestKVState>();
            c.req->page_state->page_size_tokens = page_size_tokens_;
            loc = c.req->page_state->extend_one_token(*page_pool_);
        }
        if (!loc) {
            c.ok = false;
        } else {
            c.page_id = loc->first;
            c.slot_in_page = loc->second;
        }
        calls.push_back(c);
    }

    // Determine attention shape from the first attention node.
    if (attention_node_indices_.empty()) {
        // No attention nodes — model has none? Fall back.
        return run_decode(slots);
    }
    const ExecutionNode* attn0 = findNode(graph_, topo_order_[attention_node_indices_[0]]);
    if (!attn0) return run_decode(slots);
    const size_t num_heads = static_cast<size_t>(attn0->attributes.at("heads"));
    const size_t kv_heads  = static_cast<size_t>(attn0->attributes.at("kv_heads"));
    const size_t head_dim  = static_cast<size_t>(attn0->attributes.at("head_dim"));

    // Validate paged storage matches.
    if (paged_storage_->n_kv_heads() != kv_heads ||
        paged_storage_->head_dim()  != head_dim ||
        paged_storage_->page_size_tokens() != page_size_tokens_) {
        // Mismatch — fall back.
        return run_decode(slots);
    }

    // Rope coefficients.
    const auto& cfg = graph_.modelConfig();
    const size_t rotary_dim = std::min(cfg.rotary_dim, head_dim);
    const float rope_base   = cfg.rope_freq_base > 0 ? cfg.rope_freq_base : 10000.0f;
    const float rope_scale  = cfg.rope_freq_scale > 0 ? cfg.rope_freq_scale : 1.0f;

    // Per-call CPU scratch reused across layers.
    std::vector<std::vector<float>> q_per_call(N);
    std::vector<std::vector<float>> k_per_call(N);
    std::vector<std::vector<float>> v_per_call(N);
    std::vector<float> q_batch_flat(N * num_heads * head_dim, 0.0f);
    std::vector<uint16_t> kv_scratch_f16;  // for one request's K (or V) staged for scatter

    // Allocate output buffer (fp32, batch * heads * head_dim) once.
    void* o_buf = metal.allocateScratchBuffer(N * num_heads * head_dim * sizeof(float));
    void* q_buf = metal.allocateScratchBuffer(q_batch_flat.size() * sizeof(float));

    // Layer-by-layer chunked walk.
    size_t prev_attn_end = 0;  // exclusive end of previous chunk (0 = start of graph)
    for (size_t L_iter = 0; L_iter < attention_node_indices_.size(); ++L_iter) {
        size_t attn_idx = attention_node_indices_[L_iter];
        int layer_idx   = attention_layer_index_[L_iter];
        const ExecutionNode* attn_node = findNode(graph_, topo_order_[attn_idx]);
        if (!attn_node) continue;
        size_t pre_count = (attn_idx > prev_attn_end) ? (attn_idx - prev_attn_end) : 0;

        // Pre-attention chunk per request.
        for (size_t r = 0; r < N; ++r) {
            if (!calls[r].ok) continue;
            PerRequest* req = calls[r].req;
            // Re-prime the token tensors per chunk (the executor's pass setup
            // doesn't carry them across calls).
            ExecutionContext& ctx = *req->context;
            ctx.setToken(slots[r].input_token);
            ctx.setSequencePosition(slots[r].sequence_position);
            prime_token_tensors(ctx, slots[r].input_token);
            auto er = req->executor->run_range(prev_attn_end, pre_count);
            if (!er.success) calls[r].ok = false;
        }

        // Read Q/K/V from each request's context, apply RoPE.
        const std::string& q_name = attn_node->inputs.size() > 0 ? attn_node->inputs[0] : std::string();
        const std::string& k_name = attn_node->inputs.size() > 1 ? attn_node->inputs[1] : std::string();
        const std::string& v_name = attn_node->inputs.size() > 2 ? attn_node->inputs[2] : std::string();

        std::vector<float> rope_cos, rope_sin;
        if (rotary_dim > 0) {
            // Each request may be at a different position; compute coefficients
            // per request inside the loop.
        }

        for (size_t r = 0; r < N; ++r) {
            if (!calls[r].ok) continue;
            ExecutionContext& ctx = *calls[r].req->context;
            const auto* qt = ctx.getTensor(q_name);
            const auto* kt = ctx.getTensor(k_name);
            const auto* vt = ctx.getTensor(v_name);
            if (!qt || !kt || !vt) { calls[r].ok = false; continue; }
            q_per_call[r].assign(qt->begin(), qt->end());
            k_per_call[r].assign(kt->begin(), kt->end());
            v_per_call[r].assign(vt->begin(), vt->end());

            // RoPE: rotate Q (per head) and K (per kv_head).
            if (rotary_dim > 0) {
                std::vector<float> cos, sin;
                computeRotaryCoefficients(slots[r].sequence_position, rotary_dim,
                                          rope_base, rope_scale, cos, sin);
                // Q: num_heads × head_dim; rotate each head's vector.
                for (size_t h = 0; h < num_heads; ++h) {
                    applyRotaryEmbedding(q_per_call[r].data() + h * head_dim,
                                         cos, sin, head_dim, rotary_dim);
                }
                // K: kv_heads × head_dim; rotate each.
                for (size_t h = 0; h < kv_heads; ++h) {
                    applyRotaryEmbedding(k_per_call[r].data() + h * head_dim,
                                         cos, sin, head_dim, rotary_dim);
                }
            }

            // Pack Q into batch-flat buffer.
            std::copy(q_per_call[r].begin(), q_per_call[r].end(),
                      q_batch_flat.begin() + r * num_heads * head_dim);

            // Scatter rotated K + V into paged storage at this layer/slot.
            // Source layout for scatter_kv_paged is [n_kv_heads, tokens=1, head_dim].
            // K stored as [kv_heads, head_dim] which IS [n_kv_heads, 1, head_dim].
            const std::vector<uint32_t> single_page_table = {calls[r].page_id};
            const size_t kv_elems = kv_heads * head_dim;

            // K as fp16.
            kv_scratch_f16.resize(kv_elems);
            castF32toF16(k_per_call[r].data(), kv_scratch_f16.data(), kv_elems);
            void* k_src_buf = metal.allocateScratchBuffer(kv_elems * sizeof(uint16_t));
            metal.uploadToBuffer(k_src_buf, kv_scratch_f16.data(), kv_elems * sizeof(uint16_t));
            metal.scatterKVPaged(paged_storage_->k_buffer(layer_idx),
                                 single_page_table, page_size_tokens_,
                                 kv_heads, head_dim, /*tokens=*/1,
                                 calls[r].slot_in_page, k_src_buf, /*dtype_bytes=*/2);
            metal.releaseScratchBuffer(k_src_buf);

            // V as fp16.
            castF32toF16(v_per_call[r].data(), kv_scratch_f16.data(), kv_elems);
            void* v_src_buf = metal.allocateScratchBuffer(kv_elems * sizeof(uint16_t));
            metal.uploadToBuffer(v_src_buf, kv_scratch_f16.data(), kv_elems * sizeof(uint16_t));
            metal.scatterKVPaged(paged_storage_->v_buffer(layer_idx),
                                 single_page_table, page_size_tokens_,
                                 kv_heads, head_dim, /*tokens=*/1,
                                 calls[r].slot_in_page, v_src_buf, /*dtype_bytes=*/2);
            metal.releaseScratchBuffer(v_src_buf);
        }

        // Build batched paged-flash inputs.
        std::vector<uint32_t> page_tables_flat;
        std::vector<uint32_t> page_table_offsets(N + 1, 0);
        std::vector<uint32_t> seq_lens(N, 0);
        std::vector<uint32_t> q_positions(N, 0);
        for (size_t r = 0; r < N; ++r) {
            page_table_offsets[r] = page_tables_flat.size();
            if (calls[r].ok && calls[r].req->page_state) {
                for (uint32_t pid : calls[r].req->page_state->page_table) {
                    page_tables_flat.push_back(pid);
                }
                seq_lens[r]    = static_cast<uint32_t>(calls[r].req->page_state->total_tokens());
                q_positions[r] = static_cast<uint32_t>(slots[r].sequence_position);
            }
        }
        page_table_offsets[N] = page_tables_flat.size();

        // Upload batched Q.
        metal.uploadToBuffer(q_buf, q_batch_flat.data(),
                             N * num_heads * head_dim * sizeof(float));

        // Dispatch paged-flash for all N.
        bool ok = metal.runPagedFlashAttention(
            q_buf,
            paged_storage_->k_buffer(layer_idx),
            paged_storage_->v_buffer(layer_idx),
            o_buf,
            page_tables_flat, page_table_offsets,
            seq_lens, q_positions,
            N, num_heads, kv_heads, head_dim, page_size_tokens_,
            /*apply_causal=*/true);
        if (!ok) {
            for (auto& c : calls) c.ok = false;
            break;
        }

        // Download batched O and write per-request to attention_mix tensor.
        std::vector<float> o_batch(N * num_heads * head_dim, 0.0f);
        metal.downloadFromBuffer(o_buf, o_batch.data(),
                                 N * num_heads * head_dim * sizeof(float));

        const std::string& o_name = attn_node->outputs.size() > 0 ? attn_node->outputs[0] : std::string();
        for (size_t r = 0; r < N; ++r) {
            if (!calls[r].ok) continue;
            std::vector<float> per_o(o_batch.begin() + r * num_heads * head_dim,
                                     o_batch.begin() + (r + 1) * num_heads * head_dim);
            calls[r].req->context->setTensor(o_name, std::move(per_o));
        }

        prev_attn_end = attn_idx + 1;
    }

    // Post-last-attention chunk per request → run to end of graph.
    for (size_t r = 0; r < N; ++r) {
        BatchedDecodeOutput::PerRequest per;
        per.request_id = slots[r].request_id;
        if (!calls[r].ok) {
            out.per_request.push_back(std::move(per));
            continue;
        }
        ExecutionContext& ctx = *calls[r].req->context;
        ctx.setToken(slots[r].input_token);
        ctx.setSequencePosition(slots[r].sequence_position);
        prime_token_tensors(ctx, slots[r].input_token);
        auto er = calls[r].req->executor->run_range(prev_attn_end, 0);
        per.success = er.success;
        if (per.success) {
            if (const auto* logits = ctx.getTensor("logits")) per.logits = *logits;
            calls[r].req->next_position = slots[r].sequence_position + 1;
        }
        out.per_request.push_back(std::move(per));
    }

    metal.releaseScratchBuffer(o_buf);
    metal.releaseScratchBuffer(q_buf);
    return out;
}

std::vector<float> BatchedExecutor::run_prefill(uint32_t request_id,
                                                const std::vector<uint64_t>& tokens) {
    std::vector<float> last_logits;
    if (tokens.empty()) return last_logits;

    // G1: when paged storage is attached, prefill must populate it
    // incrementally via single-token decode paths so the per-layer paged
    // KV pages reflect the prompt context. Multi-token executor.run() uses
    // the contiguous KV cache and would skip paged storage entirely.
    if (paged_storage_ && page_pool_) {
        size_t cursor = known_position(request_id);
        for (uint64_t tok : tokens) {
            std::vector<RequestSlot> slot{{request_id, tok, cursor}};
            auto out = run_decode_paged_impl(slot);
            if (out.per_request.empty() || !out.per_request.front().success) {
                last_logits.clear();
                return last_logits;
            }
            last_logits = std::move(out.per_request.front().logits);
            ++cursor;
        }
        return last_logits;
    }

    auto& req = ensure_request(request_id);
    ExecutionContext& ctx = *req.context;
    size_t cursor = known_position(request_id);

    // Multi-token batched prefill (matches chat-repl). When seq_len > 1 the
    // executor's fuse path is disabled internally and per-token fallthrough
    // builds the KV cache correctly. Looping single-token decode-with-fuse
    // from a zero KV cache silently produces all-zero logits — the encode
    // path's K/V writes don't take effect until the cache has been seeded
    // by the non-fuse multi-token path. (Confirmed by mlc decode having
    // the same bug; chat-repl avoids it via multi-token prefill.)
    ctx.setTokens(tokens);
    ctx.setSequencePosition(cursor);
    ctx.setActiveSequenceId(0);
    static const std::vector<std::string> kTokenInputs = {"tokens", "token_ids"};
    for (const auto& name : kTokenInputs) {
        if (graph_.tensors().count(name)) {
            std::vector<float> tf;
            tf.reserve(tokens.size());
            for (uint64_t t : tokens) tf.push_back(static_cast<float>(t));
            ctx.setTensor(name, std::move(tf));
        }
    }
    auto exec_res = req.executor->run();
    if (!exec_res.success) {
        return last_logits;
    }
    if (const auto* logits = ctx.getTensor("logits")) {
        last_logits = *logits;
    }
    req.next_position = cursor + tokens.size();
    // Reset seq_len so subsequent run_decode (single-token) sees the right shape.
    ctx.setSeqLen(1);

    // B3 paged-KV: extend the page table by one slot per prefilled token.
    // Failure here releases logits and signals failure.
    if (page_pool_) {
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (!advance_page_state(req)) {
                last_logits.clear();
                return last_logits;
            }
        }
    }
    return last_logits;
}

} // namespace runtime
} // namespace mlc
