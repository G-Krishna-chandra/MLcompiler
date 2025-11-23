#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlc {
namespace runtime {

size_t q4_0RowSize(size_t cols, uint32_t quant_version);
void dequantizeRowQ4_0(const uint8_t* src, size_t cols, uint32_t quant_version, float* dst);
float dotProductRowQ4_0(const uint8_t* src,
                        size_t cols,
                        uint32_t quant_version,
                        const float* vec);
void quantizeRowQ4_0(const float* src,
                     size_t cols,
                     uint32_t quant_version,
                     std::vector<uint8_t>& dst);

size_t q4_1RowSize(size_t cols);
void dequantizeRowQ4_1(const uint8_t* src, size_t cols, float* dst);
float dotProductRowQ4_1(const uint8_t* src,
                        size_t cols,
                        const float* vec);
void quantizeRowQ4_1(const float* src,
                     size_t cols,
                     std::vector<uint8_t>& dst);

size_t q5_0RowSize(size_t cols);
void dequantizeRowQ5_0(const uint8_t* src, size_t cols, float* dst);
float dotProductRowQ5_0(const uint8_t* src,
                        size_t cols,
                        const float* vec);
void quantizeRowQ5_0(const float* src,
                     size_t cols,
                     std::vector<uint8_t>& dst);

size_t q5_1RowSize(size_t cols);
void dequantizeRowQ5_1(const uint8_t* src, size_t cols, float* dst);
float dotProductRowQ5_1(const uint8_t* src,
                        size_t cols,
                        const float* vec);
void quantizeRowQ5_1(const float* src,
                     size_t cols,
                     std::vector<uint8_t>& dst);

size_t q8_0RowSize(size_t cols);
void dequantizeRowQ8_0(const uint8_t* src, size_t cols, float* dst);
float dotProductRowQ8_0(const uint8_t* src,
                        size_t cols,
                        const float* vec);
void quantizeRowQ8_0(const float* src,
                     size_t cols,
                     std::vector<uint8_t>& dst);

size_t q8_1RowSize(size_t cols);
void dequantizeRowQ8_1(const uint8_t* src, size_t cols, float* dst);
float dotProductRowQ8_1(const uint8_t* src,
                        size_t cols,
                        const float* vec);
void quantizeRowQ8_1(const float* src,
                     size_t cols,
                     std::vector<uint8_t>& dst);

void applyRotaryEmbedding(float* data,
                          const std::vector<float>& cos,
                          const std::vector<float>& sin,
                          size_t head_dim,
                          size_t rotary_dim);

inline void applyRotaryEmbedding(std::vector<float>& data,
                                 const std::vector<float>& cos,
                                 const std::vector<float>& sin,
                                 size_t head_dim,
                                 size_t rotary_dim) {
    if (data.empty()) return;
    applyRotaryEmbedding(data.data(), cos, sin, head_dim, rotary_dim);
}

void computeRotaryCoefficients(size_t position,
                               size_t rotary_dim,
                               float freq_base,
                               float freq_scale,
                               std::vector<float>& cos,
                               std::vector<float>& sin);

size_t q2_kRowSize(size_t cols);
void dequantizeRowQ2_K(const uint8_t* src, size_t cols, float* dst);
float dotProductRowQ2_K(const uint8_t* src,
                        size_t cols,
                        const float* vec);
void quantizeRowQ2_K(const float* src,
                     size_t cols,
                     std::vector<uint8_t>& dst);

size_t q3_kRowSize(size_t cols);
void dequantizeRowQ3_K(const uint8_t* src, size_t cols, float* dst);
float dotProductRowQ3_K(const uint8_t* src,
                        size_t cols,
                        const float* vec);
void quantizeRowQ3_K(const float* src,
                     size_t cols,
                     std::vector<uint8_t>& dst);

size_t q4_kRowSize(size_t cols);
void dequantizeRowQ4_K(const uint8_t* src, size_t cols, float* dst);
float dotProductRowQ4_K(const uint8_t* src,
                        size_t cols,
                        const float* vec);
void quantizeRowQ4_K(const float* src,
                     size_t cols,
                     std::vector<uint8_t>& dst);

size_t q5_kRowSize(size_t cols);
void dequantizeRowQ5_K(const uint8_t* src, size_t cols, float* dst);
float dotProductRowQ5_K(const uint8_t* src,
                        size_t cols,
                        const float* vec);
void quantizeRowQ5_K(const float* src,
                     size_t cols,
                     std::vector<uint8_t>& dst);

size_t q6_kRowSize(size_t cols);
void dequantizeRowQ6_K(const uint8_t* src, size_t cols, float* dst);
float dotProductRowQ6_K(const uint8_t* src,
                        size_t cols,
                        const float* vec);
void quantizeRowQ6_K(const float* src,
                     size_t cols,
                     std::vector<uint8_t>& dst);

size_t q8_kRowSize(size_t cols);
void dequantizeRowQ8_K(const uint8_t* src, size_t cols, float* dst);
float dotProductRowQ8_K(const uint8_t* src,
                        size_t cols,
                        const float* vec);
void quantizeRowQ8_K(const float* src,
                     size_t cols,
                     std::vector<uint8_t>& dst);

} // namespace runtime
} // namespace mlc
