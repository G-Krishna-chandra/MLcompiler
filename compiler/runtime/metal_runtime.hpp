#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "frontends/ggml_types.hpp"

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
    // Metal is required; this throws if unavailable.
    void requireAvailable() const;

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
    bool runMatMulQ4_0Transposed(const std::vector<uint8_t>& weights,
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
    bool runMatMulQ6KTransposed(const std::vector<uint8_t>& weights,
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
