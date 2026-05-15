#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "frontends/ggml_types.hpp"

namespace mlc {
namespace runtime {

struct ExecutionNode;

struct MetalBufferHandle {
    void* buffer = nullptr;
    size_t bytes = 0;
    bool host_dirty = false;
    bool device_dirty = false;
};

class MetalExecutor {
public:
    static MetalExecutor& Instance();

    bool isAvailable() const;
    // Metal is required; this throws if unavailable.
    void requireAvailable() const;

    // Centralized dispatch decision: returns true iff the caller should run the
    // Metal kernel for this node. Combines the scheduler's tag, device
    // availability, and the programmatic force-CPU override into a single test
    // so that no caller can accidentally bypass force-CPU by gating on only
    // (node.backend == Metal && isAvailable()).
    bool shouldUseFor(const ExecutionNode& node) const;

    // weight_name is the GGUF tensor name for the weight (e.g.
    // "blk.5.attn_q.weight"). It keys a persistent device-side weight cache
    // so we upload each tensor to Metal exactly once per process lifetime
    // instead of on every matmul call. Pass an empty string to bypass the
    // cache (one-off / synthetic weights).
    bool runMatMul(const std::string& weight_name,
                   const std::vector<float>& weights,
                   const std::vector<float>& input,
                   size_t rows,
                   size_t cols,
                   bool transpose_weight,
                   std::vector<float>& output,
                   const std::vector<float>* bias = nullptr) const;

    bool runMatMulQ4_0(const std::string& weight_name,
                       const std::vector<uint8_t>& weights,
                       const std::vector<float>& input,
                       size_t rows,
                       size_t cols,
                       size_t row_stride,
                       uint32_t quant_version,
                       std::vector<float>& output,
                       const std::vector<float>* bias = nullptr) const;
    bool runMatMulQ4_0Transposed(const std::string& weight_name,
                                 const std::vector<uint8_t>& weights,
                                 const std::vector<float>& input,
                                 size_t rows,
                                 size_t cols,
                                 size_t row_stride,
                                 uint32_t quant_version,
                                 std::vector<float>& output,
                                 const std::vector<float>* bias = nullptr) const;
    bool runMatMulQ4_1(const std::string& weight_name,
                       const std::vector<uint8_t>& weights,
                       const std::vector<float>& input,
                       size_t rows,
                       size_t cols,
                       size_t row_stride,
                       std::vector<float>& output,
                       const std::vector<float>* bias = nullptr) const;
    bool runMatMulQ5_0(const std::string& weight_name,
                       const std::vector<uint8_t>& weights,
                       const std::vector<float>& input,
                       size_t rows,
                       size_t cols,
                       size_t row_stride,
                       std::vector<float>& output,
                       const std::vector<float>* bias = nullptr) const;
    bool runMatMulQ5_1(const std::string& weight_name,
                       const std::vector<uint8_t>& weights,
                       const std::vector<float>& input,
                       size_t rows,
                       size_t cols,
                       size_t row_stride,
                       std::vector<float>& output,
                       const std::vector<float>* bias = nullptr) const;
    bool runMatMulQ2K(const std::string& weight_name,
                      const std::vector<uint8_t>& weights,
                      const std::vector<float>& input,
                      size_t rows,
                      size_t cols,
                      size_t row_stride,
                      std::vector<float>& output,
                      const std::vector<float>* bias = nullptr) const;
    bool runMatMulQ3K(const std::string& weight_name,
                      const std::vector<uint8_t>& weights,
                      const std::vector<float>& input,
                      size_t rows,
                      size_t cols,
                      size_t row_stride,
                      std::vector<float>& output,
                      const std::vector<float>* bias = nullptr) const;
    bool runMatMulQ4K(const std::string& weight_name,
                      const std::vector<uint8_t>& weights,
                      const std::vector<float>& input,
                      size_t rows,
                      size_t cols,
                      size_t row_stride,
                      std::vector<float>& output,
                      const std::vector<float>* bias = nullptr) const;
    bool runMatMulQ5K(const std::string& weight_name,
                      const std::vector<uint8_t>& weights,
                      const std::vector<float>& input,
                      size_t rows,
                      size_t cols,
                      size_t row_stride,
                      std::vector<float>& output,
                      const std::vector<float>* bias = nullptr) const;
    bool runMatMulQ6K(const std::string& weight_name,
                      const std::vector<uint8_t>& weights,
                      const std::vector<float>& input,
                      size_t rows,
                      size_t cols,
                      size_t row_stride,
                      std::vector<float>& output,
                      const std::vector<float>* bias = nullptr) const;
    bool runMatMulQ6KTransposed(const std::string& weight_name,
                                const std::vector<uint8_t>& weights,
                                const std::vector<float>& input,
                                size_t rows,
                                size_t cols,
                                size_t row_stride,
                                std::vector<float>& output,
                                const std::vector<float>* bias = nullptr) const;
    bool runMatMulQ8K(const std::string& weight_name,
                      const std::vector<uint8_t>& weights,
                      const std::vector<float>& input,
                      size_t rows,
                      size_t cols,
                      size_t row_stride,
                      std::vector<float>& output,
                      const std::vector<float>* bias = nullptr) const;
    bool runMatMulQ8_0(const std::string& weight_name,
                       const std::vector<uint8_t>& weights,
                       const std::vector<float>& input,
                       size_t rows,
                       size_t cols,
                       size_t row_stride,
                       std::vector<float>& output,
                       const std::vector<float>* bias = nullptr) const;
    bool runMatMulQ8_1(const std::string& weight_name,
                       const std::vector<uint8_t>& weights,
                       const std::vector<float>& input,
                       size_t rows,
                       size_t cols,
                       size_t row_stride,
                       std::vector<float>& output,
                       const std::vector<float>* bias = nullptr) const;

    // Persistent weight cache reporting. Returns a one-line summary suitable
    // for the chat REPL profiler: "weight cache: H hits, M misses, B MB
    // resident across N tensors". H+M is the total weight-buffer fetch
    // count since process start.
    std::string weightCacheSummary() const;

    // === Deferred-commit fusion window ===
    // Open a window with beginForwardPassCB(), encode supported ops onto its
    // shared MTLCommandBuffer with encodeMatMulQ4_0 / encodeRmsNorm /
    // encodeAdd, then either flushForwardPassCB() (commit + wait + drain
    // pending host memcpys + clear retention) or discardForwardPassCB()
    // (drop without commit on abnormal exit). Always callable on non-Metal
    // builds — returns false for begin/encode and is a no-op for
    // flush/discard.
    bool beginForwardPassCB() const;
    bool hasForwardPassCB() const;
    bool flushForwardPassCB() const;
    // Variant that re-resolves deferred-readback host destinations by tensor
    // name at flush time. Each PendingReadback that was registered with a
    // non-empty tensor_name calls `resolver(name)` to get the current
    // host_dst pointer. Use when the executor's tensor map may have
    // rehashed between encode and flush — invalidating any raw pointers
    // captured at encode time. The fallback (empty tensor_name) still uses
    // the captured pointer.
    bool flushForwardPassCB(const std::function<std::vector<float>*(const std::string&)>& resolver) const;
    void discardForwardPassCB() const;

    // Returns true if there is a pending deferred readback for the given
    // tensor name on the open forward-pass CB. Used by the executor's tap-
    // flush check: a tap on a tensor whose data is deferred must trigger
    // a flush before captureTap reads the host slot. Decoupled from
    // pass_outputs membership so non-fusable ops that defer can also
    // signal "needs flush" without populating pass_outputs.
    bool hasDeferredReadback(const std::string& tensor_name) const;

    // Slice on GPU. Encodes a contiguous-range copy onto the open
    // forward-pass CB so Slice ops don't force a flush + CPU memcpy
    // mid-layer. host_dst is the readback target; tensor_name
    // re-resolves it at flush time. Returns false if no open CB or
    // pipeline unavailable.
    bool encodeSliceFromBuffer(void* input_buffer, size_t input_count,
                               size_t offset_elems, size_t length,
                               void* output_buffer,
                               const std::string& output_tensor_name,
                               std::vector<float>* host_dst,
                               bool needs_host) const;
    bool encodeSliceFromHost(const std::vector<float>& host_input,
                             size_t offset_elems, size_t length,
                             void* output_buffer,
                             const std::string& output_tensor_name,
                             std::vector<float>* host_dst,
                             bool needs_host) const;

    // Encode entry points. Each requires hasForwardPassCB() == true.
    // Output is always a pool-checked-out MTLBuffer (opaque void*; backed
    // by id<MTLBuffer> with ARC strong retention through pass_checked_out_).
    // The host_dst vector is the eventual host destination — its address is
    // captured for the flush drain; it must outlive the window. If
    // needs_host is true, flush memcpys outBuffer contents into *host_dst;
    // otherwise the buffer stays in pool (next consumer reads it directly).
    //
    // Two variants per op, distinguished on input source:
    //   *FromHost — read input from std::vector<float>& (uploaded fresh
    //               via newBufferWithBytes into a transient buffer that
    //               is retained until flush). Used at window boundaries.
    //   *FromBuffer — read input from an existing MTLBuffer (presumed to
    //               carry op-N's output from earlier in the same window).
    //               No fresh upload.
    //
    // For now these reject biased ops (return false) — TinyLlama Q4_0
    // doesn't carry bias on attn/FFN paths, so callers fall back to
    // execute() in the rare bias case.
    bool encodeMatMulQ4_0FromHost(const std::string& weight_name,
                                  const std::vector<uint8_t>& weights,
                                  const std::vector<float>& host_input,
                                  size_t rows, size_t cols, size_t row_stride,
                                  uint32_t quant_version,
                                  void* output_buffer,
                                  std::vector<float>* host_dst,
                                  bool needs_host) const;
    bool encodeMatMulQ4_0FromBuffer(const std::string& weight_name,
                                    const std::vector<uint8_t>& weights,
                                    void* input_buffer,
                                    size_t input_count,
                                    size_t rows, size_t cols, size_t row_stride,
                                    uint32_t quant_version,
                                    void* output_buffer,
                                    std::vector<float>* host_dst,
                                    bool needs_host) const;
    bool encodeRmsNormFromHost(const std::vector<float>& host_input,
                               const std::vector<float>& weight,
                               float epsilon,
                               void* output_buffer,
                               std::vector<float>* host_dst,
                               bool needs_host,
                               const std::string& weight_name = std::string()) const;
    bool encodeRmsNormFromBuffer(void* input_buffer,
                                 size_t input_count,
                                 const std::vector<float>& weight,
                                 float epsilon,
                                 void* output_buffer,
                                 std::vector<float>* host_dst,
                                 bool needs_host,
                                 const std::string& weight_name = std::string()) const;
    bool encodeAddFromHost(const std::vector<float>& host_a,
                           const std::vector<float>& host_b,
                           void* output_buffer,
                           std::vector<float>* host_dst,
                           bool needs_host) const;
    bool encodeAddMixed(const std::vector<float>* host_a, void* buffer_a,
                        const std::vector<float>* host_b, void* buffer_b,
                        size_t element_count,
                        void* output_buffer,
                        std::vector<float>* host_dst,
                        bool needs_host) const;

    // Pool accessors used by the executor. `bytes` is rounded up to the
    // next size class internally. Returned pointer is opaque (id<MTLBuffer>
    // bridged); the executor must record it in its window state so that
    // returnPoolBuffer is called at flush time. ARC strong retention is
    // maintained by Impl's pass_checked_out_ vector.
    void* checkoutPoolBuffer(size_t bytes) const;
    void  trackWindowBuffer(void* buffer) const;  // adds to pass_checked_out_

    // Capability helpers (Metal availability of specific kernels).
    bool hasBiasAddKernel() const;
    bool hasKVWriteKernel() const;
    bool hasDequantQKKernel() const;

    bool runFeedForward(const std::vector<float>& gate,
                        const std::vector<float>& up,
                        std::vector<float>& output,
                        const std::string& output_tensor_name = "") const;

    bool runAdd(const std::vector<float>& a,
                const std::vector<float>& b,
                std::vector<float>& output,
                const std::vector<float>* bias = nullptr) const;

    bool runRmsNorm(const std::vector<float>& input,
                    const std::vector<float>& weight,
                    float epsilon,
                    std::vector<float>& output) const;
    bool runLayerNorm(const std::vector<float>& input,
                      const std::vector<float>& weight,
                      const std::vector<float>* bias,
                      float epsilon,
                      std::vector<float>& output) const;

    bool runSoftmax(const std::vector<float>& input,
                    std::vector<float>& output) const;

    bool hasFeedForwardKernel() const;
    bool hasAddKernel() const;
    bool hasRmsNormKernel() const;
    bool hasLayerNormKernel() const;
    bool hasSoftmaxKernel() const;

    struct CacheDescriptor {
        uint32_t dtype = 0;
        uint32_t quant_version = 0;
        size_t row_stride_bytes = 0;
        MetalBufferHandle* handle = nullptr;
        std::vector<uint8_t>* raw_quant = nullptr;
        std::vector<float>* float_data = nullptr;
    };

    bool runAttention(const std::vector<float>& q,
                      const std::vector<float>& k,
                      const std::vector<float>& v,
                      size_t num_heads,
                      size_t kv_heads,
                      size_t head_dim,
                      size_t context_length,
                      const std::vector<float>& mask,
                      const std::vector<float>* alibi_slopes,
                      size_t position,
                      size_t rotary_dim,
                      float rope_freq_base,
                      float rope_freq_scale,
                      const CacheDescriptor& cache_k,
                      const CacheDescriptor& cache_v,
                      std::vector<float>& output,
                      const std::string& output_tensor_name = "") const;

    // Set after a deferred runAttention call: caller can pull the result
    // buffer (fp32) and element count to insert into the executor's
    // pass_outputs map. Reset to null/0 by clearLastDeferredOutput().
    void clearLastDeferredOutput() const;
    void* lastDeferredOutputBuffer() const;
    size_t lastDeferredOutputElementCount() const;

    // Flash-attention v1 (multi-tile online softmax, no cache, no RoPE).
    // Math correctness validated by mlc test-attention; gated default-off in
    // the production attention path until session-C perf measurement.
    // q:            [num_heads, head_dim]
    // k, v:         [kv_seq, kv_heads, head_dim]   (token-major)
    // output:       [num_heads, head_dim]
    // apply_causal: when true, scores at kv positions > q_position are masked
    //               to -INFINITY. For q_seq=1 decode at position kv_seq-1 the
    //               mask is a no-op.
    // q_position:   the kv position of the active query for causal masking.
    // qk_debug:     optional [num_heads, kv_seq] — pre-softmax scores when non-null
    // sm_debug:     optional [num_heads, kv_seq] — post-softmax weights when non-null
    bool runFlashAttention(const std::vector<float>& q,
                            const std::vector<float>& k,
                            const std::vector<float>& v,
                            size_t num_heads,
                            size_t kv_heads,
                            size_t head_dim,
                            size_t kv_seq,
                            bool apply_causal,
                            size_t q_position,
                            std::vector<float>& output,
                            std::vector<float>* qk_debug = nullptr,
                            std::vector<float>* sm_debug = nullptr) const;

    // Strided variant for the production path. Callers pass K/V from a
    // non-contiguous buffer (e.g., the persistent KV cache laid out as
    // [kv_heads, context_length, head_dim]) plus the per-axis strides
    // measured in floats. Feature stride is assumed to be 1.
    bool runFlashAttentionStrided(const std::vector<float>& q,
                                   const float* k_data, size_t k_size,
                                   const float* v_data, size_t v_size,
                                   size_t num_heads,
                                   size_t kv_heads,
                                   size_t head_dim,
                                   size_t kv_seq,
                                   bool apply_causal,
                                   size_t q_position,
                                   size_t k_stride_token,
                                   size_t k_stride_kv_head,
                                   size_t v_stride_token,
                                   size_t v_stride_kv_head,
                                   std::vector<float>& output) const;

    bool ensureSharedBuffer(std::vector<float>& data, MetalBufferHandle& handle) const;
    void releaseBuffer(MetalBufferHandle& handle) const;
    void markHostModified(MetalBufferHandle& handle,
                          size_t offset_bytes,
                          size_t length_bytes) const;
    bool scatterKVCache(const std::vector<float>& src,
                        MetalBufferHandle& dst,
                        size_t kv_heads,
                        size_t tokens,
                        size_t head_dim,
                        size_t context_length,
                        size_t base_position) const;
    bool dequantQ4Block(const std::vector<uint8_t>& src,
                        size_t cols,
                        std::vector<float>& dst) const;
    bool dequantQ8Block(const std::vector<uint8_t>& src,
                        size_t cols,
                        std::vector<float>& dst) const;
    bool dequantQ4_1Block(const std::vector<uint8_t>& src,
                          size_t cols,
                          std::vector<float>& dst) const;
    bool dequantQ5_0Block(const std::vector<uint8_t>& src,
                          size_t cols,
                          std::vector<float>& dst) const;
    bool dequantQ5_1Block(const std::vector<uint8_t>& src,
                          size_t cols,
                          std::vector<float>& dst) const;
    bool dequantQ8_1Block(const std::vector<uint8_t>& src,
                          size_t cols,
                          std::vector<float>& dst) const;
    bool dequantQKRow(const std::vector<uint8_t>& src,
                      uint32_t dtype,
                      size_t cols,
                      size_t row_stride,
                      std::vector<float>& dst) const;

    // Phase A2 (continuous batching) — gather K or V from a paged buffer
    // into a contiguous [n_kv_heads, num_tokens, head_dim] output.
    //
    // Per-page layout: token-major, [page_size_tokens, n_kv_heads, head_dim]
    // of `dtype_bytes`-wide elements. Page `i`'s offset in the storage
    // buffer is `i * page_size_tokens * n_kv_heads * head_dim * dtype_bytes`.
    // page_table is in token-position order: page_table[t / page_size_tokens]
    // holds the page id for token t.
    //
    // dtype_bytes must be 2 (fp16) or 4 (fp32).
    // Returns false on bad arguments or kernel failure.
    bool gatherKVPages(void* page_storage_buffer,
                       const std::vector<uint32_t>& page_table,
                       size_t page_size_tokens,
                       size_t n_kv_heads,
                       size_t head_dim,
                       size_t num_tokens,
                       void* dst_buffer,
                       size_t dtype_bytes) const;

    // Phase A2 — minimal test helpers for paged-KV unit tests. allocateScratchBuffer
    // returns a fresh shared-storage MTLBuffer of `bytes` bytes (or nullptr).
    // upload/download wrap memcpy to/from contents() for the buffer. Safe to use
    // outside the forward-pass pool; freed by releaseScratchBuffer.
    void* allocateScratchBuffer(size_t bytes) const;
    void  releaseScratchBuffer(void* buffer) const;
    void  uploadToBuffer(void* buffer, const void* src, size_t bytes, size_t dst_offset = 0) const;
    void  downloadFromBuffer(const void* buffer, void* dst, size_t bytes, size_t src_offset = 0) const;

private:
    MetalExecutor();
    ~MetalExecutor();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace runtime
} // namespace mlc
