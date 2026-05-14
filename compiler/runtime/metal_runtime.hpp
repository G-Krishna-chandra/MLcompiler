#pragma once

#include <cstddef>
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
    // Open a window with beginFusionWindow(), encode supported ops onto its
    // shared MTLCommandBuffer with encodeMatMulQ4_0 / encodeRmsNorm /
    // encodeAdd, then either flushFusionWindow() (commit + wait + drain
    // pending host memcpys + clear retention) or discardFusionWindow()
    // (drop without commit on abnormal exit). Always callable on non-Metal
    // builds — returns false for begin/encode and is a no-op for
    // flush/discard.
    bool beginFusionWindow() const;
    bool hasFusionWindow() const;
    bool flushFusionWindow() const;
    void discardFusionWindow() const;

    // Encode entry points. Each requires hasFusionWindow() == true.
    // Output is always a pool-checked-out MTLBuffer (opaque void*; backed
    // by id<MTLBuffer> with ARC strong retention through window_checked_out_).
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
                               bool needs_host) const;
    bool encodeRmsNormFromBuffer(void* input_buffer,
                                 size_t input_count,
                                 const std::vector<float>& weight,
                                 float epsilon,
                                 void* output_buffer,
                                 std::vector<float>* host_dst,
                                 bool needs_host) const;
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
    // maintained by Impl's window_checked_out_ vector.
    void* checkoutPoolBuffer(size_t bytes) const;
    void  trackWindowBuffer(void* buffer) const;  // adds to window_checked_out_

    // Capability helpers (Metal availability of specific kernels).
    bool hasBiasAddKernel() const;
    bool hasKVWriteKernel() const;
    bool hasDequantQKKernel() const;

    bool runFeedForward(const std::vector<float>& gate,
                        const std::vector<float>& up,
                        std::vector<float>& output) const;

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

private:
    MetalExecutor();
    ~MetalExecutor();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace runtime
} // namespace mlc
