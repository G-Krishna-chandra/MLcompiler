#include "runtime/batched_walker.hpp"

#include "runtime/execution_executor.hpp"
#include "runtime/float_convert.hpp"
#include "runtime/metal_runtime.hpp"
#include "runtime/quantization.hpp"
#include "runtime/quant_utils.hpp"
#include "frontends/gguf_loader.hpp"
#include "frontends/ggml_types.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace mlc {
namespace runtime {

namespace {
const ExecutionNode* findNode(const ExecutionGraph& g, const std::string& name) {
    for (const auto& n : g.nodes()) if (n.name == name) return &n;
    return nullptr;
}

const std::string& weightName(const ExecutionNode& n) {
    static const std::string empty;
    auto it = n.annotations.find("weight");
    return it != n.annotations.end() ? it->second : empty;
}
} // namespace

BatchedWalker::BatchedWalker(BatchedExecutor& exec)
    : exec_(exec), session_(exec.session_), graph_(exec.graph_) {}

void BatchedWalker::cache_shape() {
    if (shape_cached_) return;
    const auto& cfg = graph_.modelConfig();
    hidden_size_ = cfg.hidden_size;
    rotary_dim_  = cfg.rotary_dim;
    if (cfg.rope_freq_base > 0)  rope_base_  = cfg.rope_freq_base;
    if (cfg.rope_freq_scale > 0) rope_scale_ = cfg.rope_freq_scale;
    vocab_size_  = cfg.vocab_size;

    // Discover from the graph.
    const auto& tensors = graph_.tensors();
    for (const auto& n : graph_.nodes()) {
        if (n.op == ExecOpType::Attention) {
            num_heads_ = static_cast<size_t>(n.attributes.at("heads"));
            kv_heads_  = static_cast<size_t>(n.attributes.at("kv_heads"));
            head_dim_  = static_cast<size_t>(n.attributes.at("head_dim"));
            n_layers_++;
        }
    }

    // QKV-fused matmul output shape from blk.0.attn_qkv.out_stacked tensor.
    auto qkv_t = tensors.find("blk.0.attn_qkv.out_stacked");
    if (qkv_t != tensors.end() && !qkv_t->second.shape.empty()) {
        qkv_rows_ = static_cast<size_t>(qkv_t->second.shape.back());
    } else {
        qkv_rows_ = num_heads_ * head_dim_ + 2 * kv_heads_ * head_dim_;
    }
    q_rows_ = num_heads_ * head_dim_;
    k_rows_ = kv_heads_ * head_dim_;
    v_rows_ = kv_heads_ * head_dim_;

    // gate+up fused
    auto gu_t = tensors.find("blk.0.ffn_gate_up.out_stacked");
    if (gu_t != tensors.end() && !gu_t->second.shape.empty()) {
        ffn_gate_up_rows_ = static_cast<size_t>(gu_t->second.shape.back());
        ffn_inner_ = ffn_gate_up_rows_ / 2;
    }

    shape_cached_ = true;
}

void BatchedWalker::gather(const std::vector<ExecutionContext*>& ctxs,
                           const std::string& tname,
                           size_t per_req_dim,
                           std::vector<float>& out) const {
    const size_t N = ctxs.size();
    out.assign(N * per_req_dim, 0.0f);
    for (size_t r = 0; r < N; ++r) {
        const auto* t = ctxs[r]->getTensor(tname);
        if (!t) continue;
        size_t copy_n = std::min(per_req_dim, t->size());
        std::copy(t->begin(), t->begin() + copy_n, out.begin() + r * per_req_dim);
    }
}

void BatchedWalker::scatter(const std::vector<ExecutionContext*>& ctxs,
                            const std::string& tname,
                            size_t per_req_dim,
                            const std::vector<float>& in) const {
    const size_t N = ctxs.size();
    for (size_t r = 0; r < N; ++r) {
        std::vector<float> per(per_req_dim);
        std::copy(in.begin() + r * per_req_dim,
                  in.begin() + (r + 1) * per_req_dim,
                  per.begin());
        ctxs[r]->setTensor(tname, std::move(per));
    }
}

void BatchedWalker::op_embedding(const std::vector<ExecutionContext*>& ctxs,
                                 const std::vector<uint64_t>& tokens) {
    // CPU lookup per request: hidden_state_0 = embed_table[token, :]
    const auto& loader = session_.loader();
    const auto& tensor_map = loader.tensors();
    auto it = tensor_map.find("token_embd.weight");
    if (it == tensor_map.end()) {
        throw std::runtime_error("token_embd.weight missing");
    }
    const auto& info = it->second;
    const auto& raw = session_.tensorData(info);
    if (info.shape.size() != 2) throw std::runtime_error("token_embd shape != 2D");
    size_t cols = static_cast<size_t>(info.shape[0]);  // hidden_size
    size_t rows = static_cast<size_t>(info.shape[1]);  // vocab
    (void)rows;

    const size_t N = ctxs.size();
    std::vector<float> dequant_scratch(cols);
    for (size_t r = 0; r < N; ++r) {
        uint64_t tok = tokens[r];
        const uint8_t* row_ptr = raw.data() +
            tok * (info.dtype == frontend::GGML_TYPE_F32 ? cols * sizeof(float)
                                                        : ggmlRowSizeBytes(info.dtype, cols, loader.quantizationVersion()));
        std::vector<float> embed(cols);
        if (info.dtype == frontend::GGML_TYPE_F32) {
            std::memcpy(embed.data(), row_ptr, cols * sizeof(float));
        } else {
            dequantizeRowTo(row_ptr, info.dtype, cols, loader.quantizationVersion(),
                            embed.data());
        }
        ctxs[r]->setTensor("hidden_state_0", std::move(embed));
    }
}

void BatchedWalker::op_norm(const std::vector<ExecutionContext*>& ctxs,
                            const std::string& in_name,
                            const std::string& out_name,
                            const std::string& weight_name,
                            size_t dim) {
    const size_t N = ctxs.size();
    gather(ctxs, in_name, dim, scratch_a_);
    const auto& w = ctxs[0]->getParameter(weight_name);
    auto& metal = MetalExecutor::Instance();
    bool ok = metal.runRmsNormBatched(scratch_a_, w, /*eps=*/1e-5f, N, dim, scratch_b_);
    if (!ok) {
        // CPU fallback per request.
        scratch_b_.assign(N * dim, 0.0f);
        for (size_t r = 0; r < N; ++r) {
            const float* in_row = scratch_a_.data() + r * dim;
            float* out_row = scratch_b_.data() + r * dim;
            double ss = 0.0;
            for (size_t i = 0; i < dim; ++i) ss += static_cast<double>(in_row[i]) * in_row[i];
            float scale = 1.0f / std::sqrt(static_cast<float>(ss / dim) + 1e-5f);
            for (size_t i = 0; i < dim; ++i) out_row[i] = in_row[i] * scale * w[i];
        }
    }
    scatter(ctxs, out_name, dim, scratch_b_);
}

void BatchedWalker::op_q4_matmul(const std::vector<ExecutionContext*>& ctxs,
                                 const std::string& in_name,
                                 const std::string& out_name,
                                 const std::string& weight_name,
                                 size_t in_dim,
                                 size_t out_dim) {
    const size_t N = ctxs.size();
    gather(ctxs, in_name, in_dim, scratch_a_);
    const auto& tmap = session_.loader().tensors();
    auto it = tmap.find(weight_name);
    if (it == tmap.end()) {
        throw std::runtime_error("matmul weight missing: " + weight_name);
    }
    const auto& info = it->second;
    if (info.dtype != frontend::GGML_TYPE_Q4_0) {
        throw std::runtime_error("op_q4_matmul: weight not Q4_0: " + weight_name);
    }
    const auto& raw = session_.tensorData(info);
    size_t rows = static_cast<size_t>(info.shape[1]);
    size_t cols = static_cast<size_t>(info.shape[0]);
    if (rows != out_dim || cols != in_dim) {
        throw std::runtime_error("op_q4_matmul: shape mismatch for " + weight_name);
    }
    size_t row_stride = raw.size() / rows;
    uint32_t qv = session_.loader().quantizationVersion();

    auto& metal = MetalExecutor::Instance();
    bool ok = metal.runMatMulQ4_0Batched(weight_name, raw, scratch_a_,
                                         N, rows, cols, row_stride, qv, scratch_b_);
    if (!ok) throw std::runtime_error("runMatMulQ4_0Batched failed for " + weight_name);
    scatter(ctxs, out_name, out_dim, scratch_b_);
}

void BatchedWalker::op_attention(const std::vector<ExecutionContext*>& ctxs,
                                 const std::vector<RequestSlot>& slots,
                                 const std::vector<BatchedExecutor::PerRequest*>& reqs,
                                 const std::vector<uint32_t>& page_ids,
                                 const std::vector<uint32_t>& slot_ids,
                                 size_t layer_idx) {
    const size_t N = ctxs.size();
    auto& metal = MetalExecutor::Instance();
    auto* storage = exec_.paged_storage_;

    // I4: gather Q, K (per-request) into batched buffers; one GPU rope
    // dispatch for Q and one for K; then scatter K + V into paged storage.
    std::vector<float> q_batch(N * num_heads_ * head_dim_, 0.0f);
    std::vector<float> k_batch(N * kv_heads_  * head_dim_, 0.0f);
    std::vector<float> v_batch(N * kv_heads_  * head_dim_, 0.0f);
    std::vector<float> cos_table(N * (rotary_dim_ ? rotary_dim_ / 2 : 0), 0.0f);
    std::vector<float> sin_table = cos_table;

    for (size_t r = 0; r < N; ++r) {
        const std::string q_name = "blk." + std::to_string(layer_idx) + ".attn_q.out";
        const std::string k_name = "blk." + std::to_string(layer_idx) + ".attn_k.out";
        const std::string v_name = "blk." + std::to_string(layer_idx) + ".attn_v.out";
        const auto* qt = ctxs[r]->getTensor(q_name);
        const auto* kt = ctxs[r]->getTensor(k_name);
        const auto* vt = ctxs[r]->getTensor(v_name);
        if (!qt || !kt || !vt) continue;
        std::copy(qt->begin(), qt->end(), q_batch.begin() + r * num_heads_ * head_dim_);
        std::copy(kt->begin(), kt->end(), k_batch.begin() + r * kv_heads_  * head_dim_);
        std::copy(vt->begin(), vt->end(), v_batch.begin() + r * kv_heads_  * head_dim_);
        if (rotary_dim_ > 0) {
            std::vector<float> cos, sin;
            computeRotaryCoefficients(slots[r].sequence_position, rotary_dim_,
                                      rope_base_, rope_scale_, cos, sin);
            size_t pairs = rotary_dim_ / 2;
            std::copy(cos.begin(), cos.begin() + pairs, cos_table.begin() + r * pairs);
            std::copy(sin.begin(), sin.begin() + pairs, sin_table.begin() + r * pairs);
        }
    }

    // RoPE for Q and K. Per-batch CPU is faster than GPU at the demo
    // batch sizes — each rotation is ~10 µs and the GPU dispatch overhead
    // (upload+dispatch+download) is ~100 µs. For larger batch sizes (N>=16)
    // the GPU rope kernel (runBatchedRope, kept in MetalExecutor) wins.
    if (rotary_dim_ > 0) {
        for (size_t r = 0; r < N; ++r) {
            size_t pairs = rotary_dim_ / 2;
            std::vector<float> cos(cos_table.begin() + r * pairs,
                                   cos_table.begin() + (r + 1) * pairs);
            std::vector<float> sin(sin_table.begin() + r * pairs,
                                   sin_table.begin() + (r + 1) * pairs);
            for (size_t h = 0; h < num_heads_; ++h)
                applyRotaryEmbedding(q_batch.data() + (r * num_heads_ + h) * head_dim_,
                                     cos, sin, head_dim_, rotary_dim_);
            for (size_t h = 0; h < kv_heads_; ++h)
                applyRotaryEmbedding(k_batch.data() + (r * kv_heads_ + h) * head_dim_,
                                     cos, sin, head_dim_, rotary_dim_);
        }
    }

    // Scatter K (rotated) and V into paged storage per request.
    std::vector<uint16_t> kv_scratch_f16(kv_heads_ * head_dim_);
    for (size_t r = 0; r < N; ++r) {
        std::vector<uint32_t> single_pt = {page_ids[r]};
        size_t kv_elems = kv_heads_ * head_dim_;
        castF32toF16(k_batch.data() + r * kv_elems, kv_scratch_f16.data(), kv_elems);
        void* k_src = metal.getOrAllocCachedBuffer("kv_scatter_src",
                                                   kv_elems * sizeof(uint16_t));
        if (!k_src) continue;
        metal.uploadToBuffer(k_src, kv_scratch_f16.data(), kv_elems * sizeof(uint16_t));
        metal.scatterKVPaged(storage->k_buffer(layer_idx), single_pt,
                             storage->page_size_tokens(), kv_heads_, head_dim_,
                             /*tokens=*/1, slot_ids[r], k_src, /*dtype_bytes=*/2);
        castF32toF16(v_batch.data() + r * kv_elems, kv_scratch_f16.data(), kv_elems);
        metal.uploadToBuffer(k_src, kv_scratch_f16.data(), kv_elems * sizeof(uint16_t));
        metal.scatterKVPaged(storage->v_buffer(layer_idx), single_pt,
                             storage->page_size_tokens(), kv_heads_, head_dim_,
                             /*tokens=*/1, slot_ids[r], k_src, /*dtype_bytes=*/2);
    }

    // Build batched paged-flash inputs.
    std::vector<uint32_t> page_tables_flat;
    std::vector<uint32_t> page_table_offsets(N + 1, 0);
    std::vector<uint32_t> seq_lens(N, 0);
    std::vector<uint32_t> q_positions(N, 0);
    for (size_t r = 0; r < N; ++r) {
        page_table_offsets[r] = page_tables_flat.size();
        if (reqs[r] && reqs[r]->page_state) {
            for (uint32_t pid : reqs[r]->page_state->page_table) {
                page_tables_flat.push_back(pid);
            }
            seq_lens[r] = static_cast<uint32_t>(reqs[r]->page_state->total_tokens());
            q_positions[r] = static_cast<uint32_t>(slots[r].sequence_position);
        }
    }
    page_table_offsets[N] = page_tables_flat.size();

    void* q_buf = metal.allocateScratchBuffer(q_batch.size() * sizeof(float));
    metal.uploadToBuffer(q_buf, q_batch.data(), q_batch.size() * sizeof(float));
    void* o_buf = metal.allocateScratchBuffer(N * num_heads_ * head_dim_ * sizeof(float));

    bool ok = metal.runPagedFlashAttention(
        q_buf, storage->k_buffer(layer_idx), storage->v_buffer(layer_idx), o_buf,
        page_tables_flat, page_table_offsets, seq_lens, q_positions,
        N, num_heads_, kv_heads_, head_dim_, storage->page_size_tokens(), true);
    if (!ok) throw std::runtime_error("runPagedFlashAttention failed");

    std::vector<float> o_batch(N * num_heads_ * head_dim_);
    metal.downloadFromBuffer(o_buf, o_batch.data(), o_batch.size() * sizeof(float));

    metal.releaseScratchBuffer(q_buf);
    metal.releaseScratchBuffer(o_buf);

    const std::string mix_name = "blk." + std::to_string(layer_idx) + ".attention_mix";
    scatter(ctxs, mix_name, num_heads_ * head_dim_, o_batch);
}

void BatchedWalker::op_add(const std::vector<ExecutionContext*>& ctxs,
                           const std::string& a_name,
                           const std::string& b_name,
                           const std::string& out_name,
                           size_t dim) {
    const size_t N = ctxs.size();
    gather(ctxs, a_name, dim, scratch_a_);
    gather(ctxs, b_name, dim, scratch_b_);
    auto& metal = MetalExecutor::Instance();
    bool ok = metal.runAdd(scratch_a_, scratch_b_, scratch_c_, /*bias=*/nullptr);
    if (!ok) {
        scratch_c_.assign(N * dim, 0.0f);
        for (size_t i = 0; i < scratch_a_.size(); ++i) scratch_c_[i] = scratch_a_[i] + scratch_b_[i];
    }
    scatter(ctxs, out_name, dim, scratch_c_);
}

void BatchedWalker::op_silu_mul(const std::vector<ExecutionContext*>& ctxs,
                                const std::string& gate_name,
                                const std::string& up_name,
                                const std::string& out_name,
                                size_t dim) {
    const size_t N = ctxs.size();
    gather(ctxs, gate_name, dim, scratch_a_);
    gather(ctxs, up_name, dim, scratch_b_);
    auto& metal = MetalExecutor::Instance();
    bool ok = metal.runFeedForward(scratch_a_, scratch_b_, scratch_c_);
    if (!ok) {
        scratch_c_.assign(N * dim, 0.0f);
        for (size_t i = 0; i < scratch_a_.size(); ++i) {
            float g = scratch_a_[i];
            float silu = g / (1.0f + std::exp(-g));
            scratch_c_[i] = silu * scratch_b_[i];
        }
    }
    scatter(ctxs, out_name, dim, scratch_c_);
}

void BatchedWalker::op_slice(const std::vector<ExecutionContext*>& ctxs,
                             const std::string& src_name,
                             const std::string& dst_name,
                             size_t src_offset,
                             size_t length) {
    const size_t N = ctxs.size();
    for (size_t r = 0; r < N; ++r) {
        const auto* t = ctxs[r]->getTensor(src_name);
        if (!t) continue;
        std::vector<float> out(length, 0.0f);
        if (src_offset + length <= t->size()) {
            std::copy(t->begin() + src_offset, t->begin() + src_offset + length, out.begin());
        }
        ctxs[r]->setTensor(dst_name, std::move(out));
    }
}

void BatchedWalker::op_lm_head(const std::vector<ExecutionContext*>& ctxs,
                               const std::string& in_name,
                               const std::string& out_name) {
    const size_t N = ctxs.size();
    const auto& loader = session_.loader();
    const auto& tmap = loader.tensors();
    const auto& cfg = graph_.modelConfig();
    const std::string& head_name = cfg.head_weight_name;
    auto it = tmap.find(head_name);
    if (it == tmap.end()) throw std::runtime_error("lm_head weight missing");
    const auto& info = it->second;
    const auto& raw = session_.tensorData(info);
    size_t rows = static_cast<size_t>(info.shape[1]);
    size_t cols = static_cast<size_t>(info.shape[0]);
    size_t row_stride = raw.size() / rows;
    uint32_t qv = loader.quantizationVersion();

    auto& metal = MetalExecutor::Instance();

    // I3: batched Q6_K when shape fits and the model uses Q6_K (TinyLlama).
    if (info.dtype == frontend::GGML_TYPE_Q6_K &&
        cols >= 64 && (rows % 4u == 0u)) {
        gather(ctxs, in_name, cols, scratch_a_);
        if (metal.runMatMulQ6KBatched(head_name, raw, scratch_a_, N, rows, cols,
                                      row_stride, scratch_b_)) {
            scatter(ctxs, out_name, rows, scratch_b_);
            return;
        }
    }

    // Fallback: per-request loop (original path).
    for (size_t r = 0; r < N; ++r) {
        const auto* in = ctxs[r]->getTensor(in_name);
        if (!in) continue;
        std::vector<float> logits(rows);
        if (info.dtype == frontend::GGML_TYPE_Q6_K) {
            metal.runMatMulQ6K(head_name, raw, *in, rows, cols, row_stride, logits, nullptr);
        } else if (info.dtype == frontend::GGML_TYPE_Q4_0) {
            metal.runMatMulQ4_0(head_name, raw, *in, rows, cols, row_stride, qv, logits, nullptr);
        } else {
            throw std::runtime_error("lm_head: unsupported dtype");
        }
        ctxs[r]->setTensor(out_name, std::move(logits));
    }
}

BatchedDecodeOutput BatchedWalker::step(const std::vector<RequestSlot>& slots) {
    BatchedDecodeOutput out;
    if (slots.empty()) return out;
    cache_shape();

    // Optional per-op profiling. Set MLC_BATCHED_PROFILE=1 to enable.
    static const bool kProfile = std::getenv("MLC_BATCHED_PROFILE") != nullptr;
    static std::unordered_map<std::string, double> prof_total;
    static std::unordered_map<std::string, size_t> prof_count;
    auto profile = [&](const char* op, std::chrono::steady_clock::time_point t0) {
        if (!kProfile) return;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        prof_total[op] += ms;
        prof_count[op] += 1;
    };
    auto dump_profile = [&]() {
        if (!kProfile) return;
        static int passes = 0;
        if (++passes % 20 != 0) return;
        std::vector<std::pair<std::string, double>> rows(prof_total.begin(), prof_total.end());
        std::sort(rows.begin(), rows.end(),
                  [](const auto& a, const auto& b){ return a.second > b.second; });
        std::fprintf(stderr, "[walker-prof] after %d passes:\n", passes);
        double total = 0.0;
        for (auto& kv : rows) total += kv.second;
        for (auto& kv : rows) {
            std::fprintf(stderr, "  %-22s %8.1f ms total %5zu calls %6.2f ms/call %5.1f%%\n",
                         kv.first.c_str(), kv.second, prof_count[kv.first],
                         kv.second / static_cast<double>(prof_count[kv.first]),
                         100.0 * kv.second / total);
        }
    };

    const size_t N = slots.size();
    std::vector<ExecutionContext*> ctxs;
    std::vector<BatchedExecutor::PerRequest*> reqs;
    std::vector<uint64_t> tokens(N);
    ctxs.reserve(N); reqs.reserve(N);

    // Page id + slot per request (allocated up-front for this token).
    std::vector<uint32_t> page_ids(N, 0);
    std::vector<uint32_t> slot_ids(N, 0);
    bool ok_alloc = true;

    for (size_t r = 0; r < N; ++r) {
        auto& req = exec_.ensure_request(slots[r].request_id);
        ctxs.push_back(req.context.get());
        reqs.push_back(&req);
        tokens[r] = slots[r].input_token;
        ctxs[r]->setToken(slots[r].input_token);
        ctxs[r]->setSequencePosition(slots[r].sequence_position);
        // Allocate one page slot for this token.
        if (!req.page_state) {
            req.page_state = std::make_unique<RequestKVState>();
            req.page_state->page_size_tokens = exec_.page_size_tokens_;
        }
        auto loc = req.page_state->extend_one_token(*exec_.page_pool_);
        if (!loc) { ok_alloc = false; break; }
        page_ids[r] = loc->first;
        slot_ids[r] = loc->second;
    }
    if (!ok_alloc) {
        for (const auto& slot : slots) {
            BatchedDecodeOutput::PerRequest per;
            per.request_id = slot.request_id;
            out.per_request.push_back(std::move(per));
        }
        return out;
    }

    // Embedding.
    auto t_emb = std::chrono::steady_clock::now();
    op_embedding(ctxs, tokens);
    profile("embedding", t_emb);
    // First layer reads 'hidden_state_0'; subsequent reads from prev layer's residual_2.
    std::string current_input = "hidden_state_0";
    const std::string final_norm_in_after_loop = "blk." + std::to_string(n_layers_ - 1) + ".residual_2";

    for (size_t L = 0; L < n_layers_; ++L) {
        std::string prefix = "blk." + std::to_string(L) + ".";
        std::chrono::steady_clock::time_point tp;

        tp = std::chrono::steady_clock::now();
        op_norm(ctxs, current_input, prefix + "attn_norm.out",
                prefix + "attn_norm.weight", hidden_size_);
        profile("norm", tp);

        tp = std::chrono::steady_clock::now();
        op_q4_matmul(ctxs, prefix + "attn_norm.out", prefix + "attn_qkv.out_stacked",
                     prefix + "attn_qkv.weight", hidden_size_, qkv_rows_);
        profile("q4_matmul_qkv", tp);

        tp = std::chrono::steady_clock::now();
        op_slice(ctxs, prefix + "attn_qkv.out_stacked", prefix + "attn_q.out",
                 0, q_rows_);
        op_slice(ctxs, prefix + "attn_qkv.out_stacked", prefix + "attn_k.out",
                 q_rows_, k_rows_);
        op_slice(ctxs, prefix + "attn_qkv.out_stacked", prefix + "attn_v.out",
                 q_rows_ + k_rows_, v_rows_);
        profile("slice", tp);

        tp = std::chrono::steady_clock::now();
        op_attention(ctxs, slots, reqs, page_ids, slot_ids, L);
        profile("attention", tp);

        tp = std::chrono::steady_clock::now();
        op_q4_matmul(ctxs, prefix + "attention_mix", prefix + "attn_output.out",
                     prefix + "attn_output.weight", hidden_size_, hidden_size_);
        profile("q4_matmul_attn_out", tp);

        tp = std::chrono::steady_clock::now();
        op_add(ctxs, current_input, prefix + "attn_output.out",
               prefix + "residual_1", hidden_size_);
        profile("add", tp);

        tp = std::chrono::steady_clock::now();
        op_norm(ctxs, prefix + "residual_1", prefix + "ffn_norm.out",
                prefix + "ffn_norm.weight", hidden_size_);
        profile("norm", tp);

        tp = std::chrono::steady_clock::now();
        op_q4_matmul(ctxs, prefix + "ffn_norm.out", prefix + "ffn_gate_up.out_stacked",
                     prefix + "ffn_gate_up.weight", hidden_size_, ffn_gate_up_rows_);
        profile("q4_matmul_ffn_gu", tp);

        tp = std::chrono::steady_clock::now();
        op_slice(ctxs, prefix + "ffn_gate_up.out_stacked", prefix + "ffn_gate.out",
                 0, ffn_inner_);
        op_slice(ctxs, prefix + "ffn_gate_up.out_stacked", prefix + "ffn_up.out",
                 ffn_inner_, ffn_inner_);
        profile("slice", tp);

        tp = std::chrono::steady_clock::now();
        op_silu_mul(ctxs, prefix + "ffn_gate.out", prefix + "ffn_up.out",
                    prefix + "ffn_mix", ffn_inner_);
        profile("silu_mul", tp);

        tp = std::chrono::steady_clock::now();
        op_q4_matmul(ctxs, prefix + "ffn_mix", prefix + "ffn_down.out",
                     prefix + "ffn_down.weight", ffn_inner_, hidden_size_);
        profile("q4_matmul_ffn_down", tp);

        tp = std::chrono::steady_clock::now();
        op_add(ctxs, prefix + "residual_1", prefix + "ffn_down.out",
               prefix + "residual_2", hidden_size_);
        profile("add", tp);

        current_input = prefix + "residual_2";
    }

    auto t_fn = std::chrono::steady_clock::now();
    op_norm(ctxs, current_input, "final_norm_out", "output_norm.weight", hidden_size_);
    profile("final_norm", t_fn);

    auto t_lm = std::chrono::steady_clock::now();
    op_lm_head(ctxs, "final_norm_out", "logits");
    profile("lm_head", t_lm);

    dump_profile();

    // Collect.
    for (size_t r = 0; r < N; ++r) {
        BatchedDecodeOutput::PerRequest per;
        per.request_id = slots[r].request_id;
        per.success = true;
        if (const auto* lp = ctxs[r]->getTensor("logits")) per.logits = *lp;
        if (per.logits.empty()) per.success = false;
        if (per.success) reqs[r]->next_position = slots[r].sequence_position + 1;
        out.per_request.push_back(std::move(per));
    }
    return out;
}

} // namespace runtime
} // namespace mlc
