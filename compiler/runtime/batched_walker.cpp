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

    for (const auto& n : graph_.nodes()) {
        if (n.op == ExecOpType::Attention) {
            num_heads_ = static_cast<size_t>(n.attributes.at("heads"));
            kv_heads_  = static_cast<size_t>(n.attributes.at("kv_heads"));
            head_dim_  = static_cast<size_t>(n.attributes.at("head_dim"));
            n_layers_++;
        }
    }

    const auto& tensors = graph_.tensors();
    auto qkv_t = tensors.find("blk.0.attn_qkv.out_stacked");
    if (qkv_t != tensors.end() && !qkv_t->second.shape.empty()) {
        qkv_rows_ = static_cast<size_t>(qkv_t->second.shape.back());
    } else {
        qkv_rows_ = num_heads_ * head_dim_ + 2 * kv_heads_ * head_dim_;
    }
    q_rows_ = num_heads_ * head_dim_;
    k_rows_ = kv_heads_ * head_dim_;
    v_rows_ = kv_heads_ * head_dim_;

    auto gu_t = tensors.find("blk.0.ffn_gate_up.out_stacked");
    if (gu_t != tensors.end() && !gu_t->second.shape.empty()) {
        ffn_gate_up_rows_ = static_cast<size_t>(gu_t->second.shape.back());
        ffn_inner_ = ffn_gate_up_rows_ / 2;
    }

    shape_cached_ = true;
}

// Helper: get or alloc a per-walker GPU scratch buffer keyed by name.
// The walker uses these as persistent intermediate state that flows
// between ops on the shared CB without CPU roundtrip.
namespace {
void* scratch(MetalExecutor& m, const char* name, size_t bytes) {
    return m.getOrAllocCachedBuffer(name, bytes);
}
} // namespace

BatchedDecodeOutput BatchedWalker::step(const std::vector<RequestSlot>& slots) {
    BatchedDecodeOutput out;
    if (slots.empty()) return out;
    cache_shape();

    auto& metal = MetalExecutor::Instance();
    auto* storage = exec_.paged_storage_;
    if (!storage) {
        out.per_request.reserve(slots.size());
        for (const auto& s : slots) {
            BatchedDecodeOutput::PerRequest p; p.request_id = s.request_id;
            out.per_request.push_back(std::move(p));
        }
        return out;
    }

    const size_t N = slots.size();

    // Profiling.
    static const bool kProfile = std::getenv("MLC_BATCHED_PROFILE") != nullptr;
    static std::unordered_map<std::string, double> prof_total;
    static std::unordered_map<std::string, size_t> prof_count;
    static int profile_passes = 0;
    auto profile = [&](const char* op, std::chrono::steady_clock::time_point t0) {
        if (!kProfile) return;
        auto t1 = std::chrono::steady_clock::now();
        prof_total[op] += std::chrono::duration<double, std::milli>(t1 - t0).count();
        prof_count[op] += 1;
    };
    auto dump_profile = [&]() {
        if (!kProfile) return;
        if (++profile_passes % 20 != 0) return;
        std::vector<std::pair<std::string, double>> rows(prof_total.begin(), prof_total.end());
        std::sort(rows.begin(), rows.end(),
                  [](const auto& a, const auto& b){ return a.second > b.second; });
        std::fprintf(stderr, "[walker-prof] after %d passes:\n", profile_passes);
        double total = 0.0;
        for (auto& kv : rows) total += kv.second;
        for (auto& kv : rows) {
            std::fprintf(stderr, "  %-22s %8.1f ms total %5zu calls %6.2f ms/call %5.1f%%\n",
                         kv.first.c_str(), kv.second, prof_count[kv.first],
                         kv.second / static_cast<double>(prof_count[kv.first]),
                         100.0 * kv.second / total);
        }
    };

    // Setup per-call: ensure each request, allocate page slot.
    std::vector<BatchedExecutor::PerRequest*> reqs(N, nullptr);
    std::vector<uint32_t> page_ids(N, 0);
    std::vector<uint32_t> slot_ids(N, 0);
    bool ok_alloc = true;
    for (size_t r = 0; r < N; ++r) {
        reqs[r] = &exec_.ensure_request(slots[r].request_id);
        if (!reqs[r]->page_state) {
            reqs[r]->page_state = std::make_unique<RequestKVState>();
            reqs[r]->page_state->page_size_tokens = exec_.page_size_tokens_;
        }
        auto loc = reqs[r]->page_state->extend_one_token(*exec_.page_pool_);
        if (!loc) { ok_alloc = false; break; }
        page_ids[r] = loc->first;
        slot_ids[r] = loc->second;
    }
    if (!ok_alloc) {
        for (const auto& s : slots) {
            BatchedDecodeOutput::PerRequest p; p.request_id = s.request_id;
            out.per_request.push_back(std::move(p));
        }
        return out;
    }

    // Allocate / reuse persistent GPU buffers. Names are stable across passes
    // so the underlying MTLBuffers persist via getOrAllocCachedBuffer.
    const size_t hidden_bytes   = N * hidden_size_     * sizeof(float);
    const size_t qkv_bytes      = N * qkv_rows_        * sizeof(float);
    const size_t q_bytes        = N * q_rows_          * sizeof(float);
    const size_t k_bytes        = N * k_rows_          * sizeof(float);
    const size_t v_bytes        = N * v_rows_          * sizeof(float);
    const size_t gate_up_bytes  = N * ffn_gate_up_rows_ * sizeof(float);
    const size_t ffn_inner_bytes = N * ffn_inner_      * sizeof(float);
    const size_t logits_bytes   = N * vocab_size_      * sizeof(float);
    const size_t kv_f16_bytes   = N * kv_heads_ * head_dim_ * sizeof(uint16_t);

    void* buf_residual = scratch(metal, "w_residual",   hidden_bytes);
    void* buf_norm_o   = scratch(metal, "w_norm_out",   hidden_bytes);
    void* buf_qkv      = scratch(metal, "w_qkv",        qkv_bytes);
    void* buf_q        = scratch(metal, "w_q",          q_bytes);
    void* buf_k        = scratch(metal, "w_k",          k_bytes);
    void* buf_v        = scratch(metal, "w_v",          v_bytes);
    void* buf_attn_mix = scratch(metal, "w_attn_mix",   hidden_bytes);
    void* buf_attn_out = scratch(metal, "w_attn_out",   hidden_bytes);
    void* buf_resid1   = scratch(metal, "w_resid1",     hidden_bytes);
    void* buf_gate_up  = scratch(metal, "w_gate_up",    gate_up_bytes);
    void* buf_gate     = scratch(metal, "w_gate",       ffn_inner_bytes);
    void* buf_up       = scratch(metal, "w_up",         ffn_inner_bytes);
    void* buf_ffn_mix  = scratch(metal, "w_ffn_mix",    ffn_inner_bytes);
    void* buf_ffn_down = scratch(metal, "w_ffn_down",   hidden_bytes);
    void* buf_final_n  = scratch(metal, "w_final_norm", hidden_bytes);
    void* buf_logits   = scratch(metal, "w_logits",     logits_bytes);
    void* buf_kv_f16_k = scratch(metal, "w_kv_f16_k",   kv_f16_bytes);
    void* buf_kv_f16_v = scratch(metal, "w_kv_f16_v",   kv_f16_bytes);

    // ===== CPU prep: embedding lookup + initial residual fill =====
    auto t_emb = std::chrono::steady_clock::now();
    {
        const auto& loader = session_.loader();
        const auto& tmap = loader.tensors();
        auto it = tmap.find("token_embd.weight");
        if (it == tmap.end()) throw std::runtime_error("token_embd.weight missing");
        const auto& info = it->second;
        const auto& raw = session_.tensorData(info);
        size_t cols = static_cast<size_t>(info.shape[0]);
        // Stage all N requests' embeddings into the residual buffer.
        std::vector<float> stage(N * cols, 0.0f);
        std::vector<float> embed(cols);
        for (size_t r = 0; r < N; ++r) {
            uint64_t tok = slots[r].input_token;
            const uint8_t* row_ptr = raw.data() +
                tok * (info.dtype == frontend::GGML_TYPE_F32 ? cols * sizeof(float)
                                                            : ggmlRowSizeBytes(info.dtype, cols, loader.quantizationVersion()));
            if (info.dtype == frontend::GGML_TYPE_F32) {
                std::memcpy(embed.data(), row_ptr, cols * sizeof(float));
            } else {
                dequantizeRowTo(row_ptr, info.dtype, cols, loader.quantizationVersion(),
                                embed.data());
            }
            std::copy(embed.begin(), embed.end(), stage.begin() + r * cols);
        }
        metal.uploadToBuffer(buf_residual, stage.data(), hidden_bytes);
    }
    profile("embedding", t_emb);

    // ===== Open the single shared CB for the entire forward pass =====
    metal.beginForwardPassCB();

    // Per-layer rolling: input/output of the residual stream alternates.
    void* in_buf  = buf_residual;
    void* out_buf = buf_resid1;  // temp; rotates per layer

    for (size_t L = 0; L < n_layers_; ++L) {
        std::string prefix = "blk." + std::to_string(L) + ".";

        // attn_norm: in_buf → buf_norm_o
        {
            auto tp = std::chrono::steady_clock::now();
            const auto& w = reqs[0]->context->getParameter(prefix + "attn_norm.weight");
            metal.encodeRmsNormBatched(in_buf, w, /*eps=*/1e-5f, N, hidden_size_, buf_norm_o);
            profile("norm", tp);
        }

        // attn_qkv matmul: buf_norm_o → buf_qkv
        {
            auto tp = std::chrono::steady_clock::now();
            const std::string wname = prefix + "attn_qkv.weight";
            const auto& tmap = session_.loader().tensors();
            const auto& info = tmap.at(wname);
            const auto& raw = session_.tensorData(info);
            size_t row_stride = raw.size() / qkv_rows_;
            metal.encodeMatMulQ4_0Batched(wname, raw, buf_norm_o, N, qkv_rows_, hidden_size_,
                                          row_stride, session_.loader().quantizationVersion(),
                                          buf_qkv);
            profile("q4_matmul_qkv", tp);
        }

        // We need to slice qkv into Q, K, V. Slicing is a memcpy on GPU
        // buffer's contents — but contents() reads aren't safe until the
        // CB executes. So slice via download → CPU split → upload? That
        // breaks the single-CB. Instead: we pass slice offsets to the
        // attention path directly. For now, do a CB flush + slice + open
        // new CB. This is a temporary intermediate sync; J2 will optimize.
        //
        // Alternative working option: each batched matmul kernel writes
        // outputs in interleaved layout that we can directly slice via
        // pointer arithmetic on the GPU buffer (offset into the same
        // buffer). The qkv_out_stacked layout is [N, qkv_rows] — slicing
        // for request r, q rows is offset r*qkv_rows + 0 .. r*qkv_rows + q_rows.
        //
        // BUT all N requests' Q is interleaved, not contiguous. So we
        // can't pass a single offset pointer for the q_buf to attention.
        //
        // Solution: use a small re-pack kernel (or ditch slice for Q —
        // just have attention read the qkv buffer with strides). For now,
        // let's flush, gather Q/K/V by hand to CPU, re-upload. This costs
        // a CB sync per layer — undermines the point.
        //
        // BETTER: load the qkv buffer into a CPU staging vector once per
        // layer (after CB sync), then proceed. But that means per-layer CB
        // commits.
        //
        // BEST: write a slice/re-pack kernel that takes [N, qkv_rows] and
        // writes into [N, q_rows], [N, k_rows], [N, v_rows] in one
        // dispatch. All on the shared CB. Below: do this CPU-side for now
        // and optimize in J2.

        // Flush to read qkv. Then re-open. (Penalty: CB commit per layer
        // for this slice. To fix in J2 with a re-pack kernel.)
        metal.flushForwardPassCB();
        std::vector<float> qkv_host(N * qkv_rows_);
        metal.downloadFromBuffer(buf_qkv, qkv_host.data(), N * qkv_rows_ * sizeof(float));
        std::vector<float> q_host(N * q_rows_);
        std::vector<float> k_host(N * k_rows_);
        std::vector<float> v_host(N * v_rows_);
        for (size_t r = 0; r < N; ++r) {
            const float* src = qkv_host.data() + r * qkv_rows_;
            std::copy(src, src + q_rows_, q_host.data() + r * q_rows_);
            std::copy(src + q_rows_, src + q_rows_ + k_rows_, k_host.data() + r * k_rows_);
            std::copy(src + q_rows_ + k_rows_, src + qkv_rows_, v_host.data() + r * v_rows_);
        }

        // CPU rope for Q + K (both arrive pre-rope from qkv matmul).
        if (rotary_dim_ > 0) {
            for (size_t r = 0; r < N; ++r) {
                std::vector<float> cos, sin;
                computeRotaryCoefficients(slots[r].sequence_position, rotary_dim_,
                                          rope_base_, rope_scale_, cos, sin);
                for (size_t h = 0; h < num_heads_; ++h)
                    applyRotaryEmbedding(q_host.data() + (r * num_heads_ + h) * head_dim_,
                                         cos, sin, head_dim_, rotary_dim_);
                for (size_t h = 0; h < kv_heads_; ++h)
                    applyRotaryEmbedding(k_host.data() + (r * kv_heads_ + h) * head_dim_,
                                         cos, sin, head_dim_, rotary_dim_);
            }
        }
        // Upload rotated Q to buf_q. K/V to fp16 staging buffers for scatter.
        metal.uploadToBuffer(buf_q, q_host.data(), q_bytes);
        std::vector<uint16_t> k_f16(N * kv_heads_ * head_dim_);
        std::vector<uint16_t> v_f16(N * kv_heads_ * head_dim_);
        castF32toF16(k_host.data(), k_f16.data(), k_f16.size());
        castF32toF16(v_host.data(), v_f16.data(), v_f16.size());
        metal.uploadToBuffer(buf_kv_f16_k, k_f16.data(), kv_f16_bytes);
        metal.uploadToBuffer(buf_kv_f16_v, v_f16.data(), kv_f16_bytes);

        // Re-open CB for the rest of this layer.
        metal.beginForwardPassCB();

        // Scatter K, V into per-layer paged storage.
        {
            auto tp = std::chrono::steady_clock::now();
            std::vector<uint32_t> pid(page_ids.begin(), page_ids.begin() + N);
            std::vector<uint32_t> sid(slot_ids.begin(), slot_ids.begin() + N);
            metal.encodeScatterKVPagedBatched(storage->k_buffer(L), pid, sid,
                                              storage->page_size_tokens(),
                                              kv_heads_, head_dim_, N, buf_kv_f16_k);
            metal.encodeScatterKVPagedBatched(storage->v_buffer(L), pid, sid,
                                              storage->page_size_tokens(),
                                              kv_heads_, head_dim_, N, buf_kv_f16_v);
            profile("kv_scatter", tp);
        }

        // Paged-flash attention.
        {
            auto tp = std::chrono::steady_clock::now();
            std::vector<uint32_t> pt_flat;
            std::vector<uint32_t> pt_off(N + 1, 0);
            std::vector<uint32_t> seq_lens(N), q_pos(N);
            for (size_t r = 0; r < N; ++r) {
                pt_off[r] = pt_flat.size();
                for (uint32_t pid : reqs[r]->page_state->page_table) pt_flat.push_back(pid);
                seq_lens[r] = static_cast<uint32_t>(reqs[r]->page_state->total_tokens());
                q_pos[r] = static_cast<uint32_t>(slots[r].sequence_position);
            }
            pt_off[N] = pt_flat.size();
            metal.encodePagedFlashAttention(buf_q, storage->k_buffer(L), storage->v_buffer(L),
                                            buf_attn_mix, pt_flat, pt_off, seq_lens, q_pos,
                                            N, num_heads_, kv_heads_, head_dim_,
                                            storage->page_size_tokens(), true);
            profile("attention", tp);
        }

        // attn_output projection: buf_attn_mix → buf_attn_out
        {
            auto tp = std::chrono::steady_clock::now();
            const std::string wname = prefix + "attn_output.weight";
            const auto& info = session_.loader().tensors().at(wname);
            const auto& raw = session_.tensorData(info);
            size_t row_stride = raw.size() / hidden_size_;
            metal.encodeMatMulQ4_0Batched(wname, raw, buf_attn_mix, N, hidden_size_, hidden_size_,
                                          row_stride, session_.loader().quantizationVersion(),
                                          buf_attn_out);
            profile("q4_matmul_attn_out", tp);
        }

        // residual_add1: in_buf + buf_attn_out → buf_resid1
        {
            auto tp = std::chrono::steady_clock::now();
            metal.encodeAddBatched(in_buf, buf_attn_out, buf_resid1, N * hidden_size_);
            profile("add", tp);
        }

        // ffn_norm: buf_resid1 → buf_norm_o
        {
            auto tp = std::chrono::steady_clock::now();
            const auto& w = reqs[0]->context->getParameter(prefix + "ffn_norm.weight");
            metal.encodeRmsNormBatched(buf_resid1, w, 1e-5f, N, hidden_size_, buf_norm_o);
            profile("norm", tp);
        }

        // ffn_gate_up matmul: buf_norm_o → buf_gate_up
        {
            auto tp = std::chrono::steady_clock::now();
            const std::string wname = prefix + "ffn_gate_up.weight";
            const auto& info = session_.loader().tensors().at(wname);
            const auto& raw = session_.tensorData(info);
            size_t row_stride = raw.size() / ffn_gate_up_rows_;
            metal.encodeMatMulQ4_0Batched(wname, raw, buf_norm_o, N, ffn_gate_up_rows_, hidden_size_,
                                          row_stride, session_.loader().quantizationVersion(),
                                          buf_gate_up);
            profile("q4_matmul_ffn_gu", tp);
        }

        // Slice gate_up to gate, up — same problem as qkv. Flush and CPU split.
        metal.flushForwardPassCB();
        std::vector<float> gu_host(N * ffn_gate_up_rows_);
        metal.downloadFromBuffer(buf_gate_up, gu_host.data(), gu_host.size() * sizeof(float));
        std::vector<float> gate_host(N * ffn_inner_);
        std::vector<float> up_host(N * ffn_inner_);
        for (size_t r = 0; r < N; ++r) {
            const float* src = gu_host.data() + r * ffn_gate_up_rows_;
            std::copy(src, src + ffn_inner_, gate_host.data() + r * ffn_inner_);
            std::copy(src + ffn_inner_, src + ffn_gate_up_rows_,
                      up_host.data() + r * ffn_inner_);
        }
        metal.uploadToBuffer(buf_gate, gate_host.data(), ffn_inner_bytes);
        metal.uploadToBuffer(buf_up,   up_host.data(),   ffn_inner_bytes);

        metal.beginForwardPassCB();

        // SiLU * mul: buf_gate, buf_up → buf_ffn_mix
        {
            auto tp = std::chrono::steady_clock::now();
            metal.encodeSiluMulBatched(buf_gate, buf_up, buf_ffn_mix, N * ffn_inner_);
            profile("silu_mul", tp);
        }

        // ffn_down matmul: buf_ffn_mix → buf_ffn_down
        {
            auto tp = std::chrono::steady_clock::now();
            const std::string wname = prefix + "ffn_down.weight";
            const auto& info = session_.loader().tensors().at(wname);
            const auto& raw = session_.tensorData(info);
            size_t row_stride = raw.size() / hidden_size_;
            metal.encodeMatMulQ4_0Batched(wname, raw, buf_ffn_mix, N, hidden_size_, ffn_inner_,
                                          row_stride, session_.loader().quantizationVersion(),
                                          buf_ffn_down);
            profile("q4_matmul_ffn_down", tp);
        }

        // residual_add2: buf_resid1 + buf_ffn_down → buf_residual (next layer's input)
        {
            auto tp = std::chrono::steady_clock::now();
            metal.encodeAddBatched(buf_resid1, buf_ffn_down, buf_residual, N * hidden_size_);
            profile("add", tp);
        }

        // Next layer's input is buf_residual.
        in_buf = buf_residual;
    }

    // final_norm: in_buf → buf_final_n
    {
        auto tp = std::chrono::steady_clock::now();
        const auto& w = reqs[0]->context->getParameter("output_norm.weight");
        metal.encodeRmsNormBatched(in_buf, w, 1e-5f, N, hidden_size_, buf_final_n);
        profile("final_norm", tp);
    }

    // lm_head: buf_final_n → buf_logits
    {
        auto tp = std::chrono::steady_clock::now();
        const auto& cfg = graph_.modelConfig();
        const std::string& head_name = cfg.head_weight_name;
        const auto& info = session_.loader().tensors().at(head_name);
        const auto& raw = session_.tensorData(info);
        size_t rows = static_cast<size_t>(info.shape[1]);
        size_t cols = static_cast<size_t>(info.shape[0]);
        size_t row_stride = raw.size() / rows;
        if (info.dtype == frontend::GGML_TYPE_Q6_K && cols >= 64 && rows % 4 == 0) {
            metal.encodeMatMulQ6KBatched(head_name, raw, buf_final_n, N, rows, cols,
                                         row_stride, buf_logits);
        } else if (info.dtype == frontend::GGML_TYPE_Q4_0 && cols >= 64 && rows % 4 == 0) {
            metal.encodeMatMulQ4_0Batched(head_name, raw, buf_final_n, N, rows, cols,
                                          row_stride, session_.loader().quantizationVersion(),
                                          buf_logits);
        } else {
            // Cannot encode; flush and use scalar paths per request.
            metal.flushForwardPassCB();
            std::vector<float> in_host(N * cols);
            metal.downloadFromBuffer(buf_final_n, in_host.data(), in_host.size() * sizeof(float));
            std::vector<float> logits_host(N * rows);
            for (size_t r = 0; r < N; ++r) {
                std::vector<float> in(in_host.begin() + r * cols, in_host.begin() + (r + 1) * cols);
                std::vector<float> lo(rows);
                if (info.dtype == frontend::GGML_TYPE_Q6_K) {
                    metal.runMatMulQ6K(head_name, raw, in, rows, cols, row_stride, lo, nullptr);
                } else {
                    metal.runMatMulQ4_0(head_name, raw, in, rows, cols, row_stride,
                                        session_.loader().quantizationVersion(), lo, nullptr);
                }
                std::copy(lo.begin(), lo.end(), logits_host.begin() + r * rows);
            }
            metal.uploadToBuffer(buf_logits, logits_host.data(), logits_host.size() * sizeof(float));
            // Re-open just so the flush below has a CB to flush (defensive — flush of nil cb is a no-op).
            metal.beginForwardPassCB();
        }
        profile("lm_head", tp);
    }

    // Final commit + wait.
    metal.flushForwardPassCB();

    // Download logits per request.
    std::vector<float> logits_host(N * vocab_size_);
    metal.downloadFromBuffer(buf_logits, logits_host.data(), logits_host.size() * sizeof(float));

    for (size_t r = 0; r < N; ++r) {
        BatchedDecodeOutput::PerRequest per;
        per.request_id = slots[r].request_id;
        per.success = true;
        per.logits.assign(logits_host.begin() + r * vocab_size_,
                          logits_host.begin() + (r + 1) * vocab_size_);
        reqs[r]->next_position = slots[r].sequence_position + 1;
        out.per_request.push_back(std::move(per));
    }

    dump_profile();
    return out;
}

// Stubs for the old per-op functions (kept for header compatibility — no
// longer called by the rewritten step()).
void BatchedWalker::gather(const std::vector<ExecutionContext*>&, const std::string&, size_t, std::vector<float>&) const {}
void BatchedWalker::scatter(const std::vector<ExecutionContext*>&, const std::string&, size_t, const std::vector<float>&) const {}
void BatchedWalker::op_embedding(const std::vector<ExecutionContext*>&, const std::vector<uint64_t>&) {}
void BatchedWalker::op_norm(const std::vector<ExecutionContext*>&, const std::string&, const std::string&, const std::string&, size_t) {}
void BatchedWalker::op_q4_matmul(const std::vector<ExecutionContext*>&, const std::string&, const std::string&, const std::string&, size_t, size_t) {}
void BatchedWalker::op_attention(const std::vector<ExecutionContext*>&, const std::vector<RequestSlot>&, const std::vector<BatchedExecutor::PerRequest*>&, const std::vector<uint32_t>&, const std::vector<uint32_t>&, size_t) {}
void BatchedWalker::op_add(const std::vector<ExecutionContext*>&, const std::string&, const std::string&, const std::string&, size_t) {}
void BatchedWalker::op_silu_mul(const std::vector<ExecutionContext*>&, const std::string&, const std::string&, const std::string&, size_t) {}
void BatchedWalker::op_slice(const std::vector<ExecutionContext*>&, const std::string&, const std::string&, size_t, size_t) {}
void BatchedWalker::op_lm_head(const std::vector<ExecutionContext*>&, const std::string&, const std::string&) {}

void BatchedWalker::prepare_pages_for_pass(const std::vector<BatchedExecutor::PerRequest*>&, std::vector<uint32_t>&, std::vector<uint32_t>&) {}

} // namespace runtime
} // namespace mlc
