#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace mlc {
namespace runtime {

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

    bool runMatMul(const std::vector<float>& weights,
                   const std::vector<float>& input,
                   size_t rows,
                   size_t cols,
                   bool transpose_weight,
                   std::vector<float>& output,
                   const std::vector<float>* bias = nullptr) const;

    bool runMatMulQ4_0(const std::vector<uint8_t>& weights,
                       const std::vector<float>& input,
                       size_t rows,
                       size_t cols,
                       size_t row_stride,
                       uint32_t quant_version,
                       std::vector<float>& output,
                       const std::vector<float>* bias = nullptr) const;
    bool runMatMulQ4_1(const std::vector<uint8_t>& weights,
                       const std::vector<float>& input,
                       size_t rows,
                       size_t cols,
                       size_t row_stride,
                       std::vector<float>& output,
                       const std::vector<float>* bias = nullptr) const;
    bool runMatMulQ5_0(const std::vector<uint8_t>& weights,
                       const std::vector<float>& input,
                       size_t rows,
                       size_t cols,
                       size_t row_stride,
                       std::vector<float>& output,
                       const std::vector<float>* bias = nullptr) const;
    bool runMatMulQ5_1(const std::vector<uint8_t>& weights,
                       const std::vector<float>& input,
                       size_t rows,
                       size_t cols,
                       size_t row_stride,
                       std::vector<float>& output,
                       const std::vector<float>* bias = nullptr) const;
    bool runMatMulQ2K(const std::vector<uint8_t>& weights,
                      const std::vector<float>& input,
                      size_t rows,
                      size_t cols,
                      size_t row_stride,
                      std::vector<float>& output,
                      const std::vector<float>* bias = nullptr) const;
    bool runMatMulQ3K(const std::vector<uint8_t>& weights,
                      const std::vector<float>& input,
                      size_t rows,
                      size_t cols,
                      size_t row_stride,
                      std::vector<float>& output,
                      const std::vector<float>* bias = nullptr) const;
    bool runMatMulQ4K(const std::vector<uint8_t>& weights,
                      const std::vector<float>& input,
                      size_t rows,
                      size_t cols,
                      size_t row_stride,
                      std::vector<float>& output,
                      const std::vector<float>* bias = nullptr) const;
    bool runMatMulQ5K(const std::vector<uint8_t>& weights,
                      const std::vector<float>& input,
                      size_t rows,
                      size_t cols,
                      size_t row_stride,
                      std::vector<float>& output,
                      const std::vector<float>* bias = nullptr) const;
    bool runMatMulQ6K(const std::vector<uint8_t>& weights,
                      const std::vector<float>& input,
                      size_t rows,
                      size_t cols,
                      size_t row_stride,
                      std::vector<float>& output,
                      const std::vector<float>* bias = nullptr) const;
    bool runMatMulQ8K(const std::vector<uint8_t>& weights,
                      const std::vector<float>& input,
                      size_t rows,
                      size_t cols,
                      size_t row_stride,
                      std::vector<float>& output,
                      const std::vector<float>* bias = nullptr) const;
    bool runMatMulQ8_0(const std::vector<uint8_t>& weights,
                       const std::vector<float>& input,
                       size_t rows,
                       size_t cols,
                       size_t row_stride,
                       std::vector<float>& output,
                       const std::vector<float>* bias = nullptr) const;
    bool runMatMulQ8_1(const std::vector<uint8_t>& weights,
                       const std::vector<float>& input,
                       size_t rows,
                       size_t cols,
                       size_t row_stride,
                       std::vector<float>& output,
                       const std::vector<float>* bias = nullptr) const;

    bool runFeedForward(const std::vector<float>& gate,
                        const std::vector<float>& up,
                        std::vector<float>& output) const;

    bool runAdd(const std::vector<float>& a,
                const std::vector<float>& b,
                std::vector<float>& output) const;

    bool runRmsNorm(const std::vector<float>& input,
                    const std::vector<float>& weight,
                    float epsilon,
                    std::vector<float>& output) const;

    bool runSoftmax(const std::vector<float>& input,
                    std::vector<float>& output) const;

    bool hasFeedForwardKernel() const;
    bool hasAddKernel() const;
    bool hasRmsNormKernel() const;
    bool hasSoftmaxKernel() const;

    bool runAttention(const std::vector<float>& q,
                      const std::vector<float>& k,
                      const std::vector<float>& v,
                      size_t num_heads,
                      size_t kv_heads,
                      size_t head_dim,
                      size_t context_length,
                      const std::vector<float>& mask,
                      size_t position,
                      size_t rotary_dim,
                      float rope_freq_base,
                      float rope_freq_scale,
                      std::vector<float>& kv_cache_k,
                      std::vector<float>& kv_cache_v,
                      MetalBufferHandle* cache_k_handle,
                      MetalBufferHandle* cache_v_handle,
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

private:
    MetalExecutor();
    ~MetalExecutor();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace runtime
} // namespace mlc
