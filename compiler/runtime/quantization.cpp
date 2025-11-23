#include "runtime/quantization.hpp"

#include "runtime/float_convert.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace mlc {
namespace runtime {

namespace {
constexpr float kEpsilon = 1e-8f;

template <typename T>
inline T clampValue(T v, T lo, T hi) {
    return std::min(std::max(v, lo), hi);
}

inline size_t ceilDiv(size_t value, size_t block) {
    return (value + block - 1) / block;
}

constexpr size_t QK4_0 = 32;
constexpr size_t QK4_1 = 32;
constexpr size_t QK5_0 = 32;
constexpr size_t QK5_1 = 32;
constexpr size_t QK8_0 = 32;
constexpr size_t QK8_1 = 32;
constexpr size_t QK_K = 256;
constexpr size_t K_SCALE_SIZE = 12;

inline void packQ2Values(const std::array<uint8_t, QK_K>& values, uint8_t* dst) {
    uint8_t* out = dst;
    for (size_t base = 0; base < QK_K; base += 128) {
        for (size_t l = 0; l < 32; ++l) {
            uint8_t v0 = values[base + l];
            uint8_t v1 = values[base + l + 32];
            uint8_t v2 = values[base + l + 64];
            uint8_t v3 = values[base + l + 96];
            out[l] = static_cast<uint8_t>(v0 | (v1 << 2) | (v2 << 4) | (v3 << 6));
        }
        out += 32;
    }
}

inline void encodeScaleMinK4(const std::array<uint8_t, QK_K / 32>& scales,
                             const std::array<uint8_t, QK_K / 32>& mins,
                             uint8_t* dst) {
    std::memset(dst, 0, K_SCALE_SIZE);
    for (size_t j = 0; j < QK_K / 32; ++j) {
        uint8_t ls = scales[j];
        uint8_t lm = mins[j];
        if (j < 4) {
            dst[j] = ls;
            dst[j + 4] = lm;
        } else {
            dst[j + 4] = static_cast<uint8_t>((ls & 0xF) | ((lm & 0xF) << 4));
            dst[j - 4] |= static_cast<uint8_t>((ls >> 4) << 6);
            dst[j - 0] |= static_cast<uint8_t>((lm >> 4) << 6);
        }
    }
}

#pragma pack(push, 1)
struct BlockQ4_0_V1 {
    uint16_t d;
    uint8_t qs[QK4_0 / 2];
};

struct BlockQ4_0_V2 {
    uint16_t d[2];
    uint8_t qs[2][QK4_0 / 2];
};

struct BlockQ4_1 {
    uint16_t d;
    uint16_t m;
    uint8_t qs[QK4_1 / 2];
};

struct BlockQ5_0 {
    uint16_t d;
    uint8_t qh[4];
    uint8_t qs[QK5_0 / 2];
};

struct BlockQ5_1 {
    uint16_t d;
    uint16_t m;
    uint8_t qh[4];
    uint8_t qs[QK5_1 / 2];
};

struct BlockQ8_0 {
    uint16_t d;
    int8_t qs[QK8_0];
};
#pragma pack(pop)

#pragma pack(push, 1)
struct BlockQ8_1 {
    uint16_t d;
    uint16_t s;
    int8_t qs[QK8_1];
};

struct BlockQ2_K {
    uint8_t scales[QK_K / 16];
    uint8_t qs[QK_K / 4];
    uint16_t d;
    uint16_t dmin;
};

struct BlockQ3_K {
    uint8_t hmask[QK_K / 8];
    uint8_t qs[QK_K / 4];
    uint8_t scales[12];
    uint16_t d;
};

struct BlockQ4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K / 2];
};

struct BlockQ5_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K / 8];
    uint8_t qs[QK_K / 2];
};

struct BlockQ6_K {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t scales[QK_K / 16];
    uint16_t d;
};

struct BlockQ8_K {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K / 16];
};
#pragma pack(pop)

constexpr size_t BLOCK_Q4_0_V1 = sizeof(BlockQ4_0_V1);
constexpr size_t BLOCK_Q4_0_V2 = sizeof(BlockQ4_0_V2);
constexpr size_t BLOCK_Q4_1 = sizeof(BlockQ4_1);
constexpr size_t BLOCK_Q5_0 = sizeof(BlockQ5_0);
constexpr size_t BLOCK_Q5_1 = sizeof(BlockQ5_1);
constexpr size_t BLOCK_Q8_0 = sizeof(BlockQ8_0);
constexpr size_t BLOCK_Q8_1 = sizeof(BlockQ8_1);
constexpr size_t BLOCK_Q2_K = sizeof(BlockQ2_K);
constexpr size_t BLOCK_Q3_K = sizeof(BlockQ3_K);
constexpr size_t BLOCK_Q4_K = sizeof(BlockQ4_K);
constexpr size_t BLOCK_Q5_K = sizeof(BlockQ5_K);
constexpr size_t BLOCK_Q6_K = sizeof(BlockQ6_K);
constexpr size_t BLOCK_Q8_K = sizeof(BlockQ8_K);
static_assert(BLOCK_Q4_0_V1 == 18, "Unexpected block_q4_0_v1 size");
static_assert(BLOCK_Q4_0_V2 == 36, "Unexpected block_q4_0_v2 size");
static_assert(BLOCK_Q4_1 == 20, "Unexpected block_q4_1 size");
static_assert(BLOCK_Q5_0 == 22, "Unexpected block_q5_0 size");
static_assert(BLOCK_Q5_1 == 24, "Unexpected block_q5_1 size");
static_assert(BLOCK_Q8_0 == 34, "Unexpected block_q8_0 size");
static_assert(BLOCK_Q8_1 == 36, "Unexpected block_q8_1 size");
static_assert(BLOCK_Q2_K == 84, "Unexpected block_q2_k size");
static_assert(BLOCK_Q3_K == 110, "Unexpected block_q3_k size");
static_assert(BLOCK_Q4_K == 144, "Unexpected block_q4_k size");
static_assert(BLOCK_Q5_K == 176, "Unexpected block_q5_k size");
static_assert(BLOCK_Q6_K == 210, "Unexpected block_q6_k size");
static_assert(BLOCK_Q8_K == 292, "Unexpected block_q8_k size");

inline size_t numBlocks(size_t cols) {
    return (cols + QK4_0 - 1) / QK4_0;
}

inline size_t numBlocks(size_t cols, size_t block_elems) {
    return (cols + block_elems - 1) / block_elems;
}

void decodeV1(const uint8_t* src, size_t cols, float* dst) {
    const BlockQ4_0_V1* blocks = reinterpret_cast<const BlockQ4_0_V1*>(src);
    size_t out = 0;
    size_t nb = numBlocks(cols);
    for (size_t b = 0; b < nb && out < cols; ++b) {
        float d = fp16ToFloat(blocks[b].d);
        for (size_t j = 0; j < QK4_0 / 2 && out < cols; ++j) {
            uint8_t byte = blocks[b].qs[j];
            int8_t lo = static_cast<int8_t>(byte & 0x0F) - 8;
            int8_t hi = static_cast<int8_t>((byte >> 4) & 0x0F) - 8;
            dst[out++] = d * static_cast<float>(lo);
            if (out < cols) {
                dst[out++] = d * static_cast<float>(hi);
            }
        }
    }
}

inline void getScaleMinK4(int index, const uint8_t* data, uint8_t* scale, uint8_t* min) {
    if (index < 4) {
        *scale = data[index] & 63;
        *min = data[index + 4] & 63;
    } else {
        *scale = static_cast<uint8_t>((data[index + 4] & 0xF) | ((data[index - 4] >> 6) << 4));
        *min = static_cast<uint8_t>((data[index + 4] >> 4) | ((data[index] >> 6) << 4));
    }
}

} // namespace

size_t q4_0RowSize(size_t cols, uint32_t quant_version) {
    // Newer quantization versions still store the same per-block payload for Q4_0.
    (void)quant_version;
    size_t block_bytes = BLOCK_Q4_0_V1;
    return numBlocks(cols) * block_bytes;
}

void dequantizeRowQ4_0(const uint8_t* src, size_t cols, uint32_t quant_version, float* dst) {
    (void)quant_version;
    decodeV1(src, cols, dst);
}

float dotProductRowQ4_0(const uint8_t* src,
                        size_t cols,
                        uint32_t quant_version,
                        const float* vec) {
    (void)quant_version;
    const BlockQ4_0_V1* blocks = reinterpret_cast<const BlockQ4_0_V1*>(src);
    size_t nb = numBlocks(cols);
    size_t idx = 0;
    float acc = 0.0f;
    for (size_t b = 0; b < nb && idx < cols; ++b) {
        float d = fp16ToFloat(blocks[b].d);
        for (size_t j = 0; j < QK4_0 / 2 && idx < cols; ++j) {
            uint8_t byte = blocks[b].qs[j];
            int8_t lo = static_cast<int8_t>(byte & 0x0F) - 8;
            acc += d * static_cast<float>(lo) * vec[idx++];
            if (idx < cols) {
                int8_t hi = static_cast<int8_t>((byte >> 4) & 0x0F) - 8;
                acc += d * static_cast<float>(hi) * vec[idx++];
            }
        }
    }
    return acc;
}

void quantizeRowQ4_0(const float* src,
                     size_t cols,
                     uint32_t quant_version,
                     std::vector<uint8_t>& dst) {
    dst.resize(q4_0RowSize(cols, quant_version));
    uint8_t* out = dst.data();
    size_t blocks = (cols + QK4_0 - 1) / QK4_0;
    size_t idx = 0;
    size_t row_stride = q4_0RowSize(QK4_0, quant_version);
    for (size_t b = 0; b < blocks; ++b) {
        const float* block = src + idx;
        size_t block_size = std::min<size_t>(QK4_0, cols - idx);
        float max_abs = 0.0f;
        for (size_t i = 0; i < block_size; ++i) {
            max_abs = std::max(max_abs, std::fabs(block[i]));
        }
        float scale = max_abs / 7.0f;
        if (scale == 0.0f) scale = 1e-8f;
        uint16_t fp = floatToFp16(scale);
        uint8_t* block_out = out + b * row_stride;
        reinterpret_cast<uint16_t*>(block_out)[0] = fp;
        uint8_t* qs = block_out + 2;
        for (size_t i = 0; i < block_size; i += 2) {
            int8_t q0 = static_cast<int8_t>(std::round(block[i] / scale));
            q0 = std::max<int8_t>(-8, std::min<int8_t>(7, q0));
            int8_t q1 = 0;
            if (i + 1 < block_size) {
                q1 = static_cast<int8_t>(std::round(block[i + 1] / scale));
                q1 = std::max<int8_t>(-8, std::min<int8_t>(7, q1));
            }
            qs[i / 2] = (static_cast<uint8_t>(q1 & 0xF) << 4) | (static_cast<uint8_t>(q0) & 0xF);
        }
        idx += block_size;
    }
}

size_t q4_1RowSize(size_t cols) {
    size_t blocks = (cols + QK4_1 - 1) / QK4_1;
    return blocks * BLOCK_Q4_1;
}

void dequantizeRowQ4_1(const uint8_t* src, size_t cols, float* dst) {
    const BlockQ4_1* blocks = reinterpret_cast<const BlockQ4_1*>(src);
    size_t out = 0;
    size_t nb = (cols + QK4_1 - 1) / QK4_1;
    for (size_t b = 0; b < nb && out < cols; ++b) {
        float d = fp16ToFloat(blocks[b].d);
        float m = fp16ToFloat(blocks[b].m);
        for (size_t j = 0; j < QK4_1 / 2 && out < cols; ++j) {
            uint8_t byte = blocks[b].qs[j];
            int x0 = (byte & 0x0F);
            int x1 = (byte >> 4);
            dst[out++] = x0 * d + m;
            if (out < cols) {
                dst[out++] = x1 * d + m;
            }
        }
    }
}

float dotProductRowQ4_1(const uint8_t* src,
                        size_t cols,
                        const float* vec) {
    const BlockQ4_1* blocks = reinterpret_cast<const BlockQ4_1*>(src);
    size_t nb = (cols + QK4_1 - 1) / QK4_1;
    size_t idx = 0;
    float acc = 0.0f;
    for (size_t b = 0; b < nb && idx < cols; ++b) {
        float d = fp16ToFloat(blocks[b].d);
        float m = fp16ToFloat(blocks[b].m);
        for (size_t j = 0; j < QK4_1 / 2 && idx < cols; ++j) {
            uint8_t byte = blocks[b].qs[j];
            int x0 = byte & 0x0F;
            acc += (x0 * d + m) * vec[idx++];
            if (idx < cols) {
                int x1 = byte >> 4;
                acc += (x1 * d + m) * vec[idx++];
            }
        }
    }
    return acc;
}

void quantizeRowQ4_1(const float* src,
                     size_t cols,
                     std::vector<uint8_t>& dst) {
    size_t blocks = ceilDiv(cols, QK4_1);
    dst.resize(blocks * BLOCK_Q4_1);
    BlockQ4_1* out = reinterpret_cast<BlockQ4_1*>(dst.data());
    size_t idx = 0;
    for (size_t b = 0; b < blocks; ++b) {
        size_t block_size = std::min(QK4_1, cols - idx);
        if (block_size == 0) {
            out[b].d = floatToFp16(0.0f);
            out[b].m = floatToFp16(0.0f);
            std::memset(out[b].qs, 0, sizeof(out[b].qs));
            continue;
        }
        float minv = std::numeric_limits<float>::max();
        float maxv = -std::numeric_limits<float>::max();
        for (size_t i = 0; i < block_size; ++i) {
            float v = src[idx + i];
            minv = std::min(minv, v);
            maxv = std::max(maxv, v);
        }
        float d = (maxv - minv) / 15.0f;
        if (d < kEpsilon) d = kEpsilon;
        out[b].d = floatToFp16(d);
        out[b].m = floatToFp16(minv);
        std::memset(out[b].qs, 0, sizeof(out[b].qs));
        for (size_t i = 0; i < block_size; i += 2) {
            float v0 = (src[idx + i] - minv) / d;
            uint8_t q0 = static_cast<uint8_t>(std::round(v0));
            q0 = clampValue<uint8_t>(q0, 0, 15);
            uint8_t q1 = 0;
            if (i + 1 < block_size) {
                float v1 = (src[idx + i + 1] - minv) / d;
                q1 = static_cast<uint8_t>(std::round(v1));
                q1 = clampValue<uint8_t>(q1, 0, 15);
            }
            out[b].qs[i / 2] = static_cast<uint8_t>((q1 << 4) | q0);
        }
        idx += block_size;
    }
}

size_t q5_0RowSize(size_t cols) {
    size_t blocks = (cols + QK5_0 - 1) / QK5_0;
    return blocks * BLOCK_Q5_0;
}

void dequantizeRowQ5_0(const uint8_t* src, size_t cols, float* dst) {
    const BlockQ5_0* blocks = reinterpret_cast<const BlockQ5_0*>(src);
    size_t out = 0;
    size_t nb = (cols + QK5_0 - 1) / QK5_0;
    for (size_t b = 0; b < nb && out < cols; ++b) {
        float d = fp16ToFloat(blocks[b].d);
        uint32_t qh;
        std::memcpy(&qh, blocks[b].qh, sizeof(qh));
        for (size_t j = 0; j < QK5_0 / 2 && out < cols; ++j) {
            uint8_t byte = blocks[b].qs[j];
            uint8_t xh0 = ((qh >> (j + 0)) << 4) & 0x10;
            uint8_t xh1 = ((qh >> (j + 12))     ) & 0x10;
            int x0 = (byte & 0x0F) | xh0;
            int x1 = (byte >> 4) | xh1;
            dst[out++] = (x0 - 16) * d;
            if (out < cols) {
                dst[out++] = (x1 - 16) * d;
            }
        }
    }
}

float dotProductRowQ5_0(const uint8_t* src,
                        size_t cols,
                        const float* vec) {
    const BlockQ5_0* blocks = reinterpret_cast<const BlockQ5_0*>(src);
    size_t idx = 0;
    size_t nb = (cols + QK5_0 - 1) / QK5_0;
    float acc = 0.0f;
    for (size_t b = 0; b < nb && idx < cols; ++b) {
        float d = fp16ToFloat(blocks[b].d);
        uint32_t qh;
        std::memcpy(&qh, blocks[b].qh, sizeof(qh));
        for (size_t j = 0; j < QK5_0 / 2 && idx < cols; ++j) {
            uint8_t byte = blocks[b].qs[j];
            uint8_t xh0 = ((qh >> (j + 0)) << 4) & 0x10;
            uint8_t xh1 = ((qh >> (j + 12))     ) & 0x10;
            int x0 = ((byte & 0x0F) | xh0) - 16;
            acc += static_cast<float>(x0) * d * vec[idx++];
            if (idx < cols) {
                int x1 = ((byte >> 4) | xh1) - 16;
                acc += static_cast<float>(x1) * d * vec[idx++];
            }
        }
    }
    return acc;
}

void quantizeRowQ5_0(const float* src,
                     size_t cols,
                     std::vector<uint8_t>& dst) {
    size_t blocks = ceilDiv(cols, QK5_0);
    dst.resize(blocks * BLOCK_Q5_0);
    BlockQ5_0* out = reinterpret_cast<BlockQ5_0*>(dst.data());
    size_t idx = 0;
    for (size_t b = 0; b < blocks; ++b) {
        size_t block_size = std::min(QK5_0, cols - idx);
        float max_abs = 0.0f;
        for (size_t i = 0; i < block_size; ++i) {
            max_abs = std::max(max_abs, std::fabs(src[idx + i]));
        }
        float d = max_abs / 16.0f;
        if (d < kEpsilon) d = kEpsilon;
        out[b].d = floatToFp16(d);
        std::memset(out[b].qs, 0, sizeof(out[b].qs));
        uint32_t qh = 0;
        for (size_t j = 0; j < QK5_0 / 2; ++j) {
            size_t i0 = j;
            size_t i1 = j + QK5_0 / 2;
            float v0 = (i0 < block_size) ? src[idx + i0] / d : 0.0f;
            float v1 = (i1 < block_size) ? src[idx + i1] / d : 0.0f;
            int qi0 = static_cast<int>(std::round(v0)) + 16;
            int qi1 = static_cast<int>(std::round(v1)) + 16;
            qi0 = clampValue(qi0, 0, 31);
            qi1 = clampValue(qi1, 0, 31);
            uint8_t nib0 = static_cast<uint8_t>(qi0 & 0x0F);
            uint8_t nib1 = static_cast<uint8_t>(qi1 & 0x0F);
            out[b].qs[j] = static_cast<uint8_t>(nib0 | (nib1 << 4));
            if (qi0 & 0x10) qh |= (1u << (j + 0));
            if (qi1 & 0x10) qh |= (1u << (j + QK5_0 / 2));
        }
        std::memcpy(out[b].qh, &qh, sizeof(qh));
        idx += block_size;
    }
}

size_t q5_1RowSize(size_t cols) {
    size_t blocks = (cols + QK5_1 - 1) / QK5_1;
    return blocks * BLOCK_Q5_1;
}

void dequantizeRowQ5_1(const uint8_t* src, size_t cols, float* dst) {
    const BlockQ5_1* blocks = reinterpret_cast<const BlockQ5_1*>(src);
    size_t out = 0;
    size_t nb = (cols + QK5_1 - 1) / QK5_1;
    for (size_t b = 0; b < nb && out < cols; ++b) {
        float d = fp16ToFloat(blocks[b].d);
        float m = fp16ToFloat(blocks[b].m);
        uint32_t qh;
        std::memcpy(&qh, blocks[b].qh, sizeof(qh));
        for (size_t j = 0; j < QK5_1 / 2 && out < cols; ++j) {
            uint8_t byte = blocks[b].qs[j];
            uint8_t xh0 = ((qh >> (j + 0)) << 4) & 0x10;
            uint8_t xh1 = ((qh >> (j + 12))     ) & 0x10;
            int x0 = ((byte & 0x0F) | xh0);
            int x1 = ((byte >> 4) | xh1);
            dst[out++] = x0 * d + m;
            if (out < cols) {
                dst[out++] = x1 * d + m;
            }
        }
    }
}

float dotProductRowQ5_1(const uint8_t* src,
                        size_t cols,
                        const float* vec) {
    const BlockQ5_1* blocks = reinterpret_cast<const BlockQ5_1*>(src);
    size_t idx = 0;
    size_t nb = (cols + QK5_1 - 1) / QK5_1;
    float acc = 0.0f;
    for (size_t b = 0; b < nb && idx < cols; ++b) {
        float d = fp16ToFloat(blocks[b].d);
        float m = fp16ToFloat(blocks[b].m);
        uint32_t qh;
        std::memcpy(&qh, blocks[b].qh, sizeof(qh));
        for (size_t j = 0; j < QK5_1 / 2 && idx < cols; ++j) {
            uint8_t byte = blocks[b].qs[j];
            uint8_t xh0 = ((qh >> (j + 0)) << 4) & 0x10;
            uint8_t xh1 = ((qh >> (j + 12))     ) & 0x10;
            int x0 = ((byte & 0x0F) | xh0);
            acc += (x0 * d + m) * vec[idx++];
            if (idx < cols) {
                int x1 = ((byte >> 4) | xh1);
                acc += (x1 * d + m) * vec[idx++];
            }
        }
    }
    return acc;
}

void quantizeRowQ5_1(const float* src,
                     size_t cols,
                     std::vector<uint8_t>& dst) {
    size_t blocks = ceilDiv(cols, QK5_1);
    dst.resize(blocks * BLOCK_Q5_1);
    BlockQ5_1* out = reinterpret_cast<BlockQ5_1*>(dst.data());
    size_t idx = 0;
    for (size_t b = 0; b < blocks; ++b) {
        size_t block_size = std::min(QK5_1, cols - idx);
        if (block_size == 0) {
            out[b].d = floatToFp16(0.0f);
            out[b].m = floatToFp16(0.0f);
            std::memset(out[b].qs, 0, sizeof(out[b].qs));
            std::memset(out[b].qh, 0, sizeof(out[b].qh));
            continue;
        }
        float minv = std::numeric_limits<float>::max();
        float maxv = -std::numeric_limits<float>::max();
        for (size_t i = 0; i < block_size; ++i) {
            float v = src[idx + i];
            minv = std::min(minv, v);
            maxv = std::max(maxv, v);
        }
        float d = (maxv - minv) / 31.0f;
        if (d < kEpsilon) d = kEpsilon;
        out[b].d = floatToFp16(d);
        out[b].m = floatToFp16(minv);
        std::memset(out[b].qs, 0, sizeof(out[b].qs));
        uint32_t qh = 0;
        for (size_t j = 0; j < QK5_1 / 2; ++j) {
            size_t i0 = j;
            size_t i1 = j + QK5_1 / 2;
            float v0 = (i0 < block_size) ? (src[idx + i0] - minv) / d : 0.0f;
            float v1 = (i1 < block_size) ? (src[idx + i1] - minv) / d : 0.0f;
            int qi0 = clampValue(static_cast<int>(std::round(v0)), 0, 31);
            int qi1 = clampValue(static_cast<int>(std::round(v1)), 0, 31);
            uint8_t nib0 = static_cast<uint8_t>(qi0 & 0x0F);
            uint8_t nib1 = static_cast<uint8_t>(qi1 & 0x0F);
            out[b].qs[j] = static_cast<uint8_t>(nib0 | (nib1 << 4));
            if (qi0 & 0x10) qh |= (1u << (j + 0));
            if (qi1 & 0x10) qh |= (1u << (j + QK5_1 / 2));
        }
        std::memcpy(out[b].qh, &qh, sizeof(qh));
        idx += block_size;
    }
}

size_t q8_0RowSize(size_t cols) {
    size_t blocks = (cols + QK8_0 - 1) / QK8_0;
    return blocks * BLOCK_Q8_0;
}

void dequantizeRowQ8_0(const uint8_t* src, size_t cols, float* dst) {
    const BlockQ8_0* blocks = reinterpret_cast<const BlockQ8_0*>(src);
    size_t out = 0;
    size_t nb = (cols + QK8_0 - 1) / QK8_0;
    for (size_t b = 0; b < nb && out < cols; ++b) {
        float d = fp16ToFloat(blocks[b].d);
        for (size_t j = 0; j < QK8_0 && out < cols; ++j) {
            dst[out++] = static_cast<float>(blocks[b].qs[j]) * d;
        }
    }
}

float dotProductRowQ8_0(const uint8_t* src,
                        size_t cols,
                        const float* vec) {
    const BlockQ8_0* blocks = reinterpret_cast<const BlockQ8_0*>(src);
    size_t idx = 0;
    size_t nb = (cols + QK8_0 - 1) / QK8_0;
    float acc = 0.0f;
    for (size_t b = 0; b < nb && idx < cols; ++b) {
        float d = fp16ToFloat(blocks[b].d);
        for (size_t j = 0; j < QK8_0 && idx < cols; ++j) {
            acc += static_cast<float>(blocks[b].qs[j]) * d * vec[idx++];
        }
    }
    return acc;
}

void quantizeRowQ8_0(const float* src,
                     size_t cols,
                     std::vector<uint8_t>& dst) {
    size_t blocks = ceilDiv(cols, QK8_0);
    dst.resize(blocks * BLOCK_Q8_0);
    BlockQ8_0* out = reinterpret_cast<BlockQ8_0*>(dst.data());
    size_t idx = 0;
    for (size_t b = 0; b < blocks; ++b) {
        size_t block_size = std::min(QK8_0, cols - idx);
        float max_abs = 0.0f;
        for (size_t i = 0; i < block_size; ++i) {
            max_abs = std::max(max_abs, std::fabs(src[idx + i]));
        }
        float d = max_abs / 127.0f;
        if (d < kEpsilon) d = kEpsilon;
        out[b].d = floatToFp16(d);
        for (size_t i = 0; i < QK8_0; ++i) {
            float v = (i < block_size) ? src[idx + i] / d : 0.0f;
            int q = clampValue(static_cast<int>(std::round(v)), -128, 127);
            out[b].qs[i] = static_cast<int8_t>(q);
        }
        idx += block_size;
    }
}

size_t q8_1RowSize(size_t cols) {
    size_t blocks = (cols + QK8_1 - 1) / QK8_1;
    return blocks * BLOCK_Q8_1;
}

void dequantizeRowQ8_1(const uint8_t* src, size_t cols, float* dst) {
    const BlockQ8_1* blocks = reinterpret_cast<const BlockQ8_1*>(src);
    size_t out = 0;
    size_t nb = (cols + QK8_1 - 1) / QK8_1;
    for (size_t b = 0; b < nb && out < cols; ++b) {
        float d = fp16ToFloat(blocks[b].d);
        float sum = fp16ToFloat(blocks[b].s);
        float bias = (QK8_1 > 0) ? (sum / static_cast<float>(QK8_1)) : 0.0f;
        for (size_t j = 0; j < QK8_1 && out < cols; ++j) {
            dst[out++] = d * static_cast<float>(blocks[b].qs[j]) + bias;
        }
    }
}

float dotProductRowQ8_1(const uint8_t* src,
                        size_t cols,
                        const float* vec) {
    const BlockQ8_1* blocks = reinterpret_cast<const BlockQ8_1*>(src);
    size_t nb = (cols + QK8_1 - 1) / QK8_1;
    size_t idx = 0;
    float acc = 0.0f;
    for (size_t b = 0; b < nb && idx < cols; ++b) {
        float d = fp16ToFloat(blocks[b].d);
        float sum = fp16ToFloat(blocks[b].s);
        float bias = sum / static_cast<float>(QK8_1);
        for (size_t j = 0; j < QK8_1 && idx < cols; ++j) {
            float val = d * static_cast<float>(blocks[b].qs[j]) + bias;
            acc += val * vec[idx++];
        }
    }
    return acc;
}

void quantizeRowQ8_1(const float* src,
                     size_t cols,
                     std::vector<uint8_t>& dst) {
    size_t blocks = ceilDiv(cols, QK8_1);
    dst.resize(blocks * BLOCK_Q8_1);
    BlockQ8_1* out = reinterpret_cast<BlockQ8_1*>(dst.data());
    size_t idx = 0;
    for (size_t b = 0; b < blocks; ++b) {
        size_t block_size = std::min(QK8_1, cols - idx);
        float sum = 0.0f;
        for (size_t i = 0; i < block_size; ++i) {
            sum += src[idx + i];
        }
        float bias = (block_size > 0) ? (sum / static_cast<float>(block_size)) : 0.0f;
        float max_abs = 0.0f;
        for (size_t i = 0; i < block_size; ++i) {
            max_abs = std::max(max_abs, std::fabs(src[idx + i] - bias));
        }
        float d = max_abs / 127.0f;
        if (d < kEpsilon) d = kEpsilon;
        out[b].d = floatToFp16(d);
        out[b].s = floatToFp16(bias * static_cast<float>(QK8_1));
        for (size_t i = 0; i < QK8_1; ++i) {
            float v = (i < block_size) ? (src[idx + i] - bias) / d : 0.0f;
            int q = clampValue(static_cast<int>(std::round(v)), -128, 127);
            out[b].qs[i] = static_cast<int8_t>(q);
        }
        idx += block_size;
    }
}

void applyRotaryEmbedding(float* data,
                          const std::vector<float>& cos,
                          const std::vector<float>& sin,
                          size_t head_dim,
                          size_t rotary_dim) {
    if (!data || head_dim == 0 || rotary_dim == 0) {
        return;
    }
    if (rotary_dim > head_dim) {
        rotary_dim = head_dim;
    }
    size_t pairs = rotary_dim / 2;
    for (size_t i = 0; i < pairs; ++i) {
        float c = (i < cos.size()) ? cos[i] : 1.0f;
        float s = (i < sin.size()) ? sin[i] : 0.0f;
        float x0 = data[2 * i];
        float x1 = data[2 * i + 1];
        data[2 * i] = x0 * c - x1 * s;
        data[2 * i + 1] = x0 * s + x1 * c;
    }
}

void computeRotaryCoefficients(size_t position,
                               size_t rotary_dim,
                               float freq_base,
                               float freq_scale,
                               std::vector<float>& cos,
                               std::vector<float>& sin) {
    if (rotary_dim < 2) {
        cos.clear();
        sin.clear();
        return;
    }
    if (freq_base <= 0.0f) {
        freq_base = 10000.0f;
    }
    if (freq_scale == 0.0f) {
        freq_scale = 1.0f;
    }
    size_t pairs = rotary_dim / 2;
    cos.resize(pairs);
    sin.resize(pairs);
    float pos = static_cast<float>(position);
    for (size_t i = 0; i < pairs; ++i) {
        float exponent = static_cast<float>(2 * i) / static_cast<float>(rotary_dim);
        float inv_freq = std::pow(freq_base, -exponent);
        float angle = pos * inv_freq * freq_scale;
        cos[i] = std::cos(angle);
        sin[i] = std::sin(angle);
    }
}

size_t q2_kRowSize(size_t cols) {
    size_t blocks = numBlocks(cols, QK_K);
    return blocks * BLOCK_Q2_K;
}

void dequantizeRowQ2_K(const uint8_t* src, size_t cols, float* dst) {
    const BlockQ2_K* blocks = reinterpret_cast<const BlockQ2_K*>(src);
    size_t nb = numBlocks(cols, QK_K);
    size_t out = 0;
    for (size_t b = 0; b < nb && out < cols; ++b) {
        float d = fp16ToFloat(blocks[b].d);
        float min = fp16ToFloat(blocks[b].dmin);
        const uint8_t* q_ptr = blocks[b].qs;
        int is = 0;
        for (size_t n = 0; n < QK_K && out < cols; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4 && out < cols; ++j) {
                uint8_t sc = blocks[b].scales[is++];
                float dl = d * static_cast<float>(sc & 0xF);
                float ml = min * static_cast<float>(sc >> 4);
                for (int l = 0; l < 16 && out < cols; ++l) {
                    int8_t val = static_cast<int8_t>((q_ptr[l] >> shift) & 0x3);
                    dst[out++] = dl * static_cast<float>(val) - ml;
                }

                sc = blocks[b].scales[is++];
                dl = d * static_cast<float>(sc & 0xF);
                ml = min * static_cast<float>(sc >> 4);
                for (int l = 0; l < 16 && out < cols; ++l) {
                    int8_t val = static_cast<int8_t>((q_ptr[l + 16] >> shift) & 0x3);
                    dst[out++] = dl * static_cast<float>(val) - ml;
                }
                shift += 2;
            }
            q_ptr += 32;
        }
    }
}

float dotProductRowQ2_K(const uint8_t* src,
                        size_t cols,
                        const float* vec) {
    const BlockQ2_K* blocks = reinterpret_cast<const BlockQ2_K*>(src);
    size_t nb = numBlocks(cols, QK_K);
    float acc = 0.0f;
    for (size_t b = 0; b < nb; ++b) {
        float d = fp16ToFloat(blocks[b].d);
        float min = fp16ToFloat(blocks[b].dmin);
        const uint8_t* q_ptr = blocks[b].qs;
        int is = 0;
        size_t block_start = b * QK_K;
        size_t block_end = std::min(cols, block_start + QK_K);
        size_t out = block_start;
        for (size_t n = 0; n < QK_K && out < block_end; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4 && out < block_end; ++j) {
                uint8_t sc = blocks[b].scales[is++];
                float dl = d * static_cast<float>(sc & 0xF);
                float ml = min * static_cast<float>(sc >> 4);
                for (int l = 0; l < 16 && out < block_end; ++l) {
                    int8_t val = static_cast<int8_t>((q_ptr[l] >> shift) & 0x3);
                    float f = dl * static_cast<float>(val) - ml;
                    acc += f * vec[out++];
                }

                sc = blocks[b].scales[is++];
                dl = d * static_cast<float>(sc & 0xF);
                ml = min * static_cast<float>(sc >> 4);
                for (int l = 0; l < 16 && out < block_end; ++l) {
                    int8_t val = static_cast<int8_t>((q_ptr[l + 16] >> shift) & 0x3);
                    float f = dl * static_cast<float>(val) - ml;
                    acc += f * vec[out++];
                }
                shift += 2;
            }
            q_ptr += 32;
        }
    }
    return acc;
}

void quantizeRowQ2_K(const float* src,
                     size_t cols,
                     std::vector<uint8_t>& dst) {
    size_t blocks = ceilDiv(cols, QK_K);
    dst.resize(blocks * BLOCK_Q2_K);
    BlockQ2_K* out = reinterpret_cast<BlockQ2_K*>(dst.data());
    size_t offset = 0;
    constexpr size_t kChunks = QK_K / 16;
    for (size_t b = 0; b < blocks; ++b) {
        size_t block_elems = std::min(QK_K, cols - offset);
        std::array<float, QK_K> block_data{};
        if (block_elems > 0) {
            std::memcpy(block_data.data(), src + offset, block_elems * sizeof(float));
        }
        std::array<float, kChunks> chunk_scales{};
        std::array<float, kChunks> chunk_offsets{};
        float max_scale = 0.0f;
        float max_offset = 0.0f;
        for (size_t j = 0; j < kChunks; ++j) {
            size_t start = j * 16;
            size_t end = start + 16;
            if (start >= block_elems) {
                chunk_scales[j] = 0.0f;
                chunk_offsets[j] = 0.0f;
                continue;
            }
            float minv = block_data[start];
            float maxv = block_data[start];
            for (size_t i = start; i < end && i < block_elems; ++i) {
                minv = std::min(minv, block_data[i]);
                maxv = std::max(maxv, block_data[i]);
            }
            float offset_val = (minv < 0.0f) ? -minv : 0.0f;
            float shifted_max = maxv + offset_val;
            float scale = shifted_max / 3.0f;
            if (scale < kEpsilon) scale = 0.0f;
            chunk_scales[j] = scale;
            chunk_offsets[j] = offset_val;
            max_scale = std::max(max_scale, scale);
            max_offset = std::max(max_offset, offset_val);
        }
        float scale_base = (max_scale > 0.0f) ? (max_scale / 15.0f) : 0.0f;
        float offset_base = (max_offset > 0.0f) ? (max_offset / 15.0f) : 0.0f;
        out[b].d = floatToFp16(scale_base);
        out[b].dmin = floatToFp16(offset_base);
        std::array<uint8_t, QK_K> q_values{};
        for (size_t j = 0; j < kChunks; ++j) {
            uint8_t scale_code = 0;
            uint8_t offset_code = 0;
            if (scale_base > 0.0f && chunk_scales[j] > 0.0f) {
                int code = static_cast<int>(std::round(chunk_scales[j] / scale_base));
                code = clampValue(code, 0, 15);
                scale_code = static_cast<uint8_t>(code);
            }
            if (offset_base > 0.0f && chunk_offsets[j] > 0.0f) {
                int code = static_cast<int>(std::round(chunk_offsets[j] / offset_base));
                code = clampValue(code, 0, 15);
                offset_code = static_cast<uint8_t>(code);
            }
            out[b].scales[j] = static_cast<uint8_t>(scale_code | (offset_code << 4));
            float actual_scale = scale_base * static_cast<float>(scale_code);
            float actual_offset = offset_base * static_cast<float>(offset_code);
            for (size_t i = 0; i < 16; ++i) {
                size_t data_idx = j * 16 + i;
                if (data_idx >= QK_K) break;
                float value = (data_idx < block_elems) ? block_data[data_idx] : 0.0f;
                float shifted = value + actual_offset;
                float qf = (actual_scale > 0.0f) ? std::round(shifted / actual_scale) : 0.0f;
                int q = clampValue(static_cast<int>(qf), 0, 3);
                q_values[data_idx] = static_cast<uint8_t>(q);
            }
        }
        packQ2Values(q_values, out[b].qs);
        offset += block_elems;
    }
}

size_t q3_kRowSize(size_t cols) {
    size_t blocks = numBlocks(cols, QK_K);
    return blocks * BLOCK_Q3_K;
}

void dequantizeRowQ3_K(const uint8_t* src, size_t cols, float* dst) {
    const BlockQ3_K* blocks = reinterpret_cast<const BlockQ3_K*>(src);
    size_t nb = numBlocks(cols, QK_K);
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    size_t out = 0;
    for (size_t b = 0; b < nb && out < cols; ++b) {
        float d_all = fp16ToFloat(blocks[b].d);
        const uint8_t* q = blocks[b].qs;
        const uint8_t* hm = blocks[b].hmask;
        uint32_t aux[4];
        std::memcpy(aux, blocks[b].scales, sizeof(aux));
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t* scales = reinterpret_cast<const int8_t*>(aux);

        int is = 0;
        uint8_t m = 1;
        for (size_t n = 0; n < QK_K && out < cols; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4 && out < cols; ++j) {
                float dl = d_all * static_cast<float>(scales[is++] - 32);
                for (int l = 0; l < 16 && out < cols; ++l) {
                    int8_t val = static_cast<int8_t>((q[l + 0] >> shift) & 0x3);
                    int8_t delta = (hm[l + 0] & m) ? 0 : 4;
                    dst[out++] = dl * static_cast<float>(val - delta);
                }

                dl = d_all * static_cast<float>(scales[is++] - 32);
                for (int l = 0; l < 16 && out < cols; ++l) {
                    int8_t val = static_cast<int8_t>((q[l + 16] >> shift) & 0x3);
                    int8_t delta = (hm[l + 16] & m) ? 0 : 4;
                    dst[out++] = dl * static_cast<float>(val - delta);
                }
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

float dotProductRowQ3_K(const uint8_t* src,
                        size_t cols,
                        const float* vec) {
    const BlockQ3_K* blocks = reinterpret_cast<const BlockQ3_K*>(src);
    size_t nb = numBlocks(cols, QK_K);
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    float acc = 0.0f;
    for (size_t b = 0; b < nb; ++b) {
        float d_all = fp16ToFloat(blocks[b].d);
        const uint8_t* q = blocks[b].qs;
        const uint8_t* hm = blocks[b].hmask;
        uint32_t aux[4];
        std::memcpy(aux, blocks[b].scales, sizeof(aux));
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t* scales = reinterpret_cast<const int8_t*>(aux);

        int is = 0;
        uint8_t m = 1;
        size_t base = b * QK_K;
        size_t end = std::min(cols, base + QK_K);
        size_t out = base;
        for (size_t n = 0; n < QK_K && out < end; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4 && out < end; ++j) {
                float dl = d_all * static_cast<float>(scales[is++] - 32);
                for (int l = 0; l < 16 && out < end; ++l) {
                    int8_t val = static_cast<int8_t>((q[l + 0] >> shift) & 0x3);
                    int8_t delta = (hm[l + 0] & m) ? 0 : 4;
                    float vf = dl * static_cast<float>(val - delta);
                    acc += vf * vec[out++];
                }
                dl = d_all * static_cast<float>(scales[is++] - 32);
                for (int l = 0; l < 16 && out < end; ++l) {
                    int8_t val = static_cast<int8_t>((q[l + 16] >> shift) & 0x3);
                    int8_t delta = (hm[l + 16] & m) ? 0 : 4;
                    float vf = dl * static_cast<float>(val - delta);
                    acc += vf * vec[out++];
                }
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
    return acc;
}

void quantizeRowQ3_K(const float* src,
                     size_t cols,
                     std::vector<uint8_t>& dst) {
    size_t blocks = ceilDiv(cols, QK_K);
    dst.resize(blocks * BLOCK_Q3_K);
    BlockQ3_K* out = reinterpret_cast<BlockQ3_K*>(dst.data());
    size_t offset = 0;
    constexpr size_t kChunks = QK_K / 16;
    for (size_t b = 0; b < blocks; ++b) {
        size_t block_elems = std::min(QK_K, cols - offset);
        std::array<float, QK_K> block_data{};
        if (block_elems > 0) {
            std::memcpy(block_data.data(), src + offset, block_elems * sizeof(float));
        }
        std::array<float, kChunks> chunk_scales{};
        float max_scale = 0.0f;
        for (size_t j = 0; j < kChunks; ++j) {
            size_t start = j * 16;
            if (start >= block_elems) {
                chunk_scales[j] = 0.0f;
                continue;
            }
            float max_abs = 0.0f;
            for (size_t i = start; i < start + 16 && i < block_elems; ++i) {
                max_abs = std::max(max_abs, std::fabs(block_data[i]));
            }
            float scale = max_abs / 4.0f;
            if (scale < kEpsilon) scale = 0.0f;
            chunk_scales[j] = scale;
            max_scale = std::max(max_scale, scale);
        }
        float base = (max_scale > 0.0f) ? (max_scale / 31.0f) : 0.0f;
        out[b].d = floatToFp16(base);
        std::array<uint8_t, kChunks> scale_codes{};
        std::array<uint8_t, QK_K> raw_values{};
        for (size_t j = 0; j < kChunks; ++j) {
            uint8_t code = 32;
            if (base > 0.0f && chunk_scales[j] > 0.0f) {
                int v = static_cast<int>(std::round(chunk_scales[j] / base));
                v = clampValue(v, 0, 31);
                code = static_cast<uint8_t>(32 + v);
            }
            scale_codes[j] = code;
            float actual_scale = base * static_cast<float>(static_cast<int>(code) - 32);
            for (size_t i = 0; i < 16; ++i) {
                size_t data_idx = j * 16 + i;
                if (data_idx >= QK_K) break;
                float value = (data_idx < block_elems) ? block_data[data_idx] : 0.0f;
                float qf = (actual_scale > 0.0f) ? std::round(value / actual_scale) : 0.0f;
                int q = clampValue(static_cast<int>(qf), -4, 3);
                raw_values[data_idx] = static_cast<uint8_t>(q + 4);
            }
        }
        std::memset(out[b].scales, 0, sizeof(out[b].scales));
        for (size_t j = 0; j < kChunks; ++j) {
            uint8_t l = scale_codes[j];
            if (j < 8) {
                out[b].scales[j] = static_cast<uint8_t>(l & 0xF);
            } else {
                out[b].scales[j - 8] |= static_cast<uint8_t>((l & 0xF) << 4);
            }
            l >>= 4;
            out[b].scales[j % 4 + 8] |= static_cast<uint8_t>(l << (2 * (j / 4)));
        }
        std::memset(out[b].hmask, 0, QK_K / 8);
        int m = 0;
        uint8_t hm = 1;
        for (size_t i = 0; i < QK_K; ++i) {
            if (raw_values[i] > 3) {
                out[b].hmask[m] |= hm;
                raw_values[i] -= 4;
            }
            if (++m == static_cast<int>(QK_K / 8)) {
                m = 0;
                hm = static_cast<uint8_t>(hm << 1);
            }
        }
        packQ2Values(raw_values, out[b].qs);
        offset += block_elems;
    }
}

size_t q4_kRowSize(size_t cols) {
    size_t blocks = numBlocks(cols, QK_K);
    return blocks * BLOCK_Q4_K;
}

void dequantizeRowQ4_K(const uint8_t* src, size_t cols, float* dst) {
    const BlockQ4_K* blocks = reinterpret_cast<const BlockQ4_K*>(src);
    size_t nb = numBlocks(cols, QK_K);
    size_t out = 0;
    for (size_t b = 0; b < nb && out < cols; ++b) {
        float d = fp16ToFloat(blocks[b].d);
        float min = fp16ToFloat(blocks[b].dmin);
        const uint8_t* q = blocks[b].qs;
        int is = 0;
        for (size_t j = 0; j < QK_K && out < cols; j += 64) {
            uint8_t sc, m;
            getScaleMinK4(is + 0, blocks[b].scales, &sc, &m);
            float d1 = d * static_cast<float>(sc);
            float m1 = min * static_cast<float>(m);
            getScaleMinK4(is + 1, blocks[b].scales, &sc, &m);
            float d2 = d * static_cast<float>(sc);
            float m2 = min * static_cast<float>(m);
            for (int l = 0; l < 32 && out < cols; ++l) {
                dst[out++] = d1 * static_cast<float>(q[l] & 0xF) - m1;
            }
            for (int l = 0; l < 32 && out < cols; ++l) {
                dst[out++] = d2 * static_cast<float>(q[l] >> 4) - m2;
            }
            q += 32;
            is += 2;
        }
    }
}

float dotProductRowQ4_K(const uint8_t* src,
                        size_t cols,
                        const float* vec) {
    const BlockQ4_K* blocks = reinterpret_cast<const BlockQ4_K*>(src);
    size_t nb = numBlocks(cols, QK_K);
    float acc = 0.0f;
    for (size_t b = 0; b < nb; ++b) {
        float d = fp16ToFloat(blocks[b].d);
        float min = fp16ToFloat(blocks[b].dmin);
        const uint8_t* scales = blocks[b].scales;
        const uint8_t* qs = blocks[b].qs;
        size_t base = b * QK_K;
        size_t end = std::min(cols, base + QK_K);
        size_t out = base;
        for (size_t n = 0; n < QK_K && out < end; n += 32) {
            uint8_t sc = scales[n / 32];
            uint8_t sc2 = scales[n / 32 + 8];
            float dl = d * static_cast<float>(sc & 0xF);
            float ml = min * static_cast<float>(sc >> 4);
            for (int l = 0; l < 16 && out < end; ++l) {
                uint8_t val = (qs[n / 2 + l] & 0x0F);
                float vf = dl * static_cast<float>(val) - ml;
                acc += vf * vec[out++];
            }
            dl = d * static_cast<float>(sc2 & 0xF);
            ml = min * static_cast<float>(sc2 >> 4);
            for (int l = 0; l < 16 && out < end; ++l) {
                uint8_t val = ((qs[n / 2 + l] >> 4) & 0x0F);
                float vf = dl * static_cast<float>(val) - ml;
                acc += vf * vec[out++];
            }
        }
    }
    return acc;
}

void quantizeRowQ4_K(const float* src,
                     size_t cols,
                     std::vector<uint8_t>& dst) {
    size_t blocks = ceilDiv(cols, QK_K);
    dst.resize(blocks * BLOCK_Q4_K);
    BlockQ4_K* out = reinterpret_cast<BlockQ4_K*>(dst.data());
    size_t offset = 0;
    constexpr size_t kChunks = QK_K / 32;
    for (size_t b = 0; b < blocks; ++b) {
        size_t block_elems = std::min(QK_K, cols - offset);
        std::array<float, QK_K> block_data{};
        if (block_elems > 0) {
            std::memcpy(block_data.data(), src + offset, block_elems * sizeof(float));
        }
        std::array<float, kChunks> chunk_scales{};
        std::array<float, kChunks> chunk_offsets{};
        float max_scale = 0.0f;
        float max_offset = 0.0f;
        for (size_t j = 0; j < kChunks; ++j) {
            size_t start = j * 32;
            if (start >= block_elems) {
                chunk_scales[j] = 0.0f;
                chunk_offsets[j] = 0.0f;
                continue;
            }
            float minv = block_data[start];
            float maxv = block_data[start];
            for (size_t i = start; i < start + 32 && i < block_elems; ++i) {
                minv = std::min(minv, block_data[i]);
                maxv = std::max(maxv, block_data[i]);
            }
            float offset_val = (minv < 0.0f) ? -minv : 0.0f;
            float scale = (maxv + offset_val) / 15.0f;
            if (scale < kEpsilon) scale = 0.0f;
            chunk_scales[j] = scale;
            chunk_offsets[j] = offset_val;
            max_scale = std::max(max_scale, scale);
            max_offset = std::max(max_offset, offset_val);
        }
        float scale_base = (max_scale > 0.0f) ? (max_scale / 63.0f) : 0.0f;
        float offset_base = (max_offset > 0.0f) ? (max_offset / 63.0f) : 0.0f;
        out[b].d = floatToFp16(scale_base);
        out[b].dmin = floatToFp16(offset_base);
        std::array<uint8_t, kChunks> scale_codes{};
        std::array<uint8_t, kChunks> offset_codes{};
        std::array<uint8_t, QK_K> quantized{};
        for (size_t j = 0; j < kChunks; ++j) {
            uint8_t sc_code = 0;
            uint8_t off_code = 0;
            if (scale_base > 0.0f && chunk_scales[j] > 0.0f) {
                int code = static_cast<int>(std::round(chunk_scales[j] / scale_base));
                code = clampValue(code, 0, 63);
                sc_code = static_cast<uint8_t>(code);
            }
            if (offset_base > 0.0f && chunk_offsets[j] > 0.0f) {
                int code = static_cast<int>(std::round(chunk_offsets[j] / offset_base));
                code = clampValue(code, 0, 63);
                off_code = static_cast<uint8_t>(code);
            }
            scale_codes[j] = sc_code;
            offset_codes[j] = off_code;
            float actual_scale = scale_base * static_cast<float>(sc_code);
            float actual_offset = offset_base * static_cast<float>(off_code);
            for (size_t i = 0; i < 32; ++i) {
                size_t data_idx = j * 32 + i;
                if (data_idx >= QK_K) break;
                float value = (data_idx < block_elems) ? block_data[data_idx] : 0.0f;
                float shifted = value + actual_offset;
                float qf = (actual_scale > 0.0f) ? std::round(shifted / actual_scale) : 0.0f;
                int q = clampValue(static_cast<int>(qf), 0, 15);
                quantized[data_idx] = static_cast<uint8_t>(q);
            }
        }
        encodeScaleMinK4(scale_codes, offset_codes, out[b].scales);
        uint8_t* q = out[b].qs;
        for (size_t j = 0; j < QK_K; j += 64) {
            for (size_t l = 0; l < 32; ++l) {
                q[l] = static_cast<uint8_t>(quantized[j + l] | (quantized[j + l + 32] << 4));
            }
            q += 32;
        }
        offset += block_elems;
    }
}

size_t q5_kRowSize(size_t cols) {
    size_t blocks = numBlocks(cols, QK_K);
    return blocks * BLOCK_Q5_K;
}

void dequantizeRowQ5_K(const uint8_t* src, size_t cols, float* dst) {
    const BlockQ5_K* blocks = reinterpret_cast<const BlockQ5_K*>(src);
    size_t nb = numBlocks(cols, QK_K);
    size_t out = 0;
    for (size_t b = 0; b < nb && out < cols; ++b) {
        float d = fp16ToFloat(blocks[b].d);
        float min = fp16ToFloat(blocks[b].dmin);
        const uint8_t* ql = blocks[b].qs;
        const uint8_t* qh = blocks[b].qh;
        int is = 0;
        uint8_t u1 = 1;
        uint8_t u2 = 2;
        for (size_t j = 0; j < QK_K && out < cols; j += 64) {
            uint8_t sc, m;
            getScaleMinK4(is + 0, blocks[b].scales, &sc, &m);
            float d1 = d * static_cast<float>(sc);
            float m1 = min * static_cast<float>(m);
            getScaleMinK4(is + 1, blocks[b].scales, &sc, &m);
            float d2 = d * static_cast<float>(sc);
            float m2 = min * static_cast<float>(m);
            for (int l = 0; l < 32 && out < cols; ++l) {
                uint8_t hi = (qh[l] & u1) ? 16 : 0;
                dst[out++] = d1 * static_cast<float>((ql[l] & 0xF) + hi) - m1;
            }
            for (int l = 0; l < 32 && out < cols; ++l) {
                uint8_t hi = (qh[l] & u2) ? 16 : 0;
                dst[out++] = d2 * static_cast<float>((ql[l] >> 4) + hi) - m2;
            }
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

float dotProductRowQ5_K(const uint8_t* src,
                        size_t cols,
                        const float* vec) {
    const BlockQ5_K* blocks = reinterpret_cast<const BlockQ5_K*>(src);
    size_t nb = numBlocks(cols, QK_K);
    float acc = 0.0f;
    for (size_t b = 0; b < nb; ++b) {
        float d = fp16ToFloat(blocks[b].d);
        float min = fp16ToFloat(blocks[b].dmin);
        const uint8_t* scales = blocks[b].scales;
        const uint8_t* qs = blocks[b].qs;
        const uint8_t* qh = blocks[b].qh;
        size_t base = b * QK_K;
        size_t end = std::min(cols, base + QK_K);
        size_t out = base;
        for (size_t n = 0; n < QK_K && out < end; n += 32) {
            uint8_t sc = scales[n / 32];
            uint8_t sc2 = scales[n / 32 + 8];
            float dl = d * static_cast<float>(sc & 0xF);
            float ml = min * static_cast<float>(sc >> 4);
            for (int l = 0; l < 16 && out < end; ++l) {
                uint8_t vh = ((qh[n / 8 + l / 8] >> (l % 8)) & 1) << 4;
                uint8_t val = (qs[n / 2 + l] & 0x0F) | vh;
                float vf = dl * static_cast<float>(val - 16) - ml;
                acc += vf * vec[out++];
            }
            dl = d * static_cast<float>(sc2 & 0xF);
            ml = min * static_cast<float>(sc2 >> 4);
            for (int l = 0; l < 16 && out < end; ++l) {
                uint8_t vh = ((qh[n / 8 + l / 8 + 4] >> (l % 8)) & 1) << 4;
                uint8_t val = ((qs[n / 2 + l] >> 4) & 0x0F) | vh;
                float vf = dl * static_cast<float>(val - 16) - ml;
                acc += vf * vec[out++];
            }
        }
    }
    return acc;
}

void quantizeRowQ5_K(const float* src,
                     size_t cols,
                     std::vector<uint8_t>& dst) {
    size_t blocks = ceilDiv(cols, QK_K);
    dst.resize(blocks * BLOCK_Q5_K);
    BlockQ5_K* out = reinterpret_cast<BlockQ5_K*>(dst.data());
    size_t offset = 0;
    constexpr size_t kChunks = QK_K / 32;
    for (size_t b = 0; b < blocks; ++b) {
        size_t block_elems = std::min(QK_K, cols - offset);
        std::array<float, QK_K> block_data{};
        if (block_elems > 0) {
            std::memcpy(block_data.data(), src + offset, block_elems * sizeof(float));
        }
        std::array<float, kChunks> chunk_scales{};
        std::array<float, kChunks> chunk_offsets{};
        float max_scale = 0.0f;
        float max_offset = 0.0f;
        for (size_t j = 0; j < kChunks; ++j) {
            size_t start = j * 32;
            if (start >= block_elems) {
                chunk_scales[j] = 0.0f;
                chunk_offsets[j] = 0.0f;
                continue;
            }
            float minv = block_data[start];
            float maxv = block_data[start];
            for (size_t i = start; i < start + 32 && i < block_elems; ++i) {
                minv = std::min(minv, block_data[i]);
                maxv = std::max(maxv, block_data[i]);
            }
            float offset_val = (minv < 0.0f) ? -minv : 0.0f;
            float scale = (maxv + offset_val) / 31.0f;
            if (scale < kEpsilon) scale = 0.0f;
            chunk_scales[j] = scale;
            chunk_offsets[j] = offset_val;
            max_scale = std::max(max_scale, scale);
            max_offset = std::max(max_offset, offset_val);
        }
        float scale_base = (max_scale > 0.0f) ? (max_scale / 63.0f) : 0.0f;
        float offset_base = (max_offset > 0.0f) ? (max_offset / 63.0f) : 0.0f;
        out[b].d = floatToFp16(scale_base);
        out[b].dmin = floatToFp16(offset_base);
        std::array<uint8_t, kChunks> scale_codes{};
        std::array<uint8_t, kChunks> offset_codes{};
        std::array<uint8_t, QK_K> quantized{};
        for (size_t j = 0; j < kChunks; ++j) {
            uint8_t sc_code = 0;
            uint8_t off_code = 0;
            if (scale_base > 0.0f && chunk_scales[j] > 0.0f) {
                int code = static_cast<int>(std::round(chunk_scales[j] / scale_base));
                code = clampValue(code, 0, 63);
                sc_code = static_cast<uint8_t>(code);
            }
            if (offset_base > 0.0f && chunk_offsets[j] > 0.0f) {
                int code = static_cast<int>(std::round(chunk_offsets[j] / offset_base));
                code = clampValue(code, 0, 63);
                off_code = static_cast<uint8_t>(code);
            }
            scale_codes[j] = sc_code;
            offset_codes[j] = off_code;
            float actual_scale = scale_base * static_cast<float>(sc_code);
            float actual_offset = offset_base * static_cast<float>(off_code);
            for (size_t i = 0; i < 32; ++i) {
                size_t data_idx = j * 32 + i;
                if (data_idx >= QK_K) break;
                float value = (data_idx < block_elems) ? block_data[data_idx] : 0.0f;
                float shifted = value + actual_offset;
                float qf = (actual_scale > 0.0f) ? std::round(shifted / actual_scale) : 0.0f;
                int q = clampValue(static_cast<int>(qf), 0, 31);
                quantized[data_idx] = static_cast<uint8_t>(q);
            }
        }
        encodeScaleMinK4(scale_codes, offset_codes, out[b].scales);
        std::memset(out[b].qh, 0, QK_K / 8);
        uint8_t* ql = out[b].qs;
        uint8_t m1 = 1;
        uint8_t m2 = 2;
        for (size_t n = 0; n < QK_K; n += 64) {
            for (size_t j = 0; j < 32; ++j) {
                int l1 = quantized[n + j];
                int l2 = quantized[n + j + 32];
                if (l1 > 15) {
                    l1 -= 16;
                    out[b].qh[j] |= m1;
                }
                if (l2 > 15) {
                    l2 -= 16;
                    out[b].qh[j] |= m2;
                }
                ql[j] = static_cast<uint8_t>(l1 | (l2 << 4));
            }
            m1 = static_cast<uint8_t>(m1 << 2);
            m2 = static_cast<uint8_t>(m2 << 2);
            ql += 32;
        }
        offset += block_elems;
    }
}

size_t q6_kRowSize(size_t cols) {
    size_t blocks = numBlocks(cols, QK_K);
    return blocks * BLOCK_Q6_K;
}

void dequantizeRowQ6_K(const uint8_t* src, size_t cols, float* dst) {
    const BlockQ6_K* blocks = reinterpret_cast<const BlockQ6_K*>(src);
    size_t nb = numBlocks(cols, QK_K);
    for (size_t b = 0; b < nb; ++b) {
        float block_buf[QK_K] = {0.0f};
        float d = fp16ToFloat(blocks[b].d);
        const uint8_t* ql = blocks[b].ql;
        const uint8_t* qh = blocks[b].qh;
        const int8_t* sc = blocks[b].scales;
        float* y = block_buf;
        for (size_t n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int8_t q1 = static_cast<int8_t>((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = static_cast<int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = static_cast<int8_t>((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = static_cast<int8_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = d * static_cast<float>(sc[is + 0]) * static_cast<float>(q1);
                y[l + 32] = d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2);
                y[l + 64] = d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3);
                y[l + 96] = d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4);
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
        size_t block_elems = std::min(QK_K, cols - b * QK_K);
        std::memcpy(dst + b * QK_K, block_buf, block_elems * sizeof(float));
    }
}

size_t q8_kRowSize(size_t cols) {
    size_t blocks = numBlocks(cols, QK_K);
    return blocks * BLOCK_Q8_K;
}

void dequantizeRowQ8_K(const uint8_t* src, size_t cols, float* dst) {
    const BlockQ8_K* blocks = reinterpret_cast<const BlockQ8_K*>(src);
    size_t nb = numBlocks(cols, QK_K);
    size_t out = 0;
    for (size_t b = 0; b < nb && out < cols; ++b) {
        for (size_t j = 0; j < QK_K && out < cols; ++j) {
            dst[out++] = blocks[b].d * static_cast<float>(blocks[b].qs[j]);
        }
    }
}

void quantizeRowQ8_K(const float* src,
                     size_t cols,
                     std::vector<uint8_t>& dst) {
    size_t blocks = ceilDiv(cols, QK_K);
    dst.resize(blocks * BLOCK_Q8_K);
    BlockQ8_K* out = reinterpret_cast<BlockQ8_K*>(dst.data());
    size_t offset = 0;
    for (size_t b = 0; b < blocks; ++b) {
        size_t block_elems = std::min(QK_K, cols - offset);
        float max_abs = 0.0f;
        for (size_t i = 0; i < block_elems; ++i) {
            max_abs = std::max(max_abs, std::fabs(src[offset + i]));
        }
        float d = max_abs / 127.0f;
        if (d < kEpsilon) d = kEpsilon;
        out[b].d = d;
        std::fill(std::begin(out[b].bsums), std::end(out[b].bsums), 0);
        for (size_t i = 0; i < QK_K; ++i) {
            float v = (i < block_elems) ? src[offset + i] / d : 0.0f;
            int q = clampValue(static_cast<int>(std::round(v)), -128, 127);
            out[b].qs[i] = static_cast<int8_t>(q);
        }
        for (size_t g = 0; g < QK_K / 16; ++g) {
            int32_t sum = 0;
            for (size_t i = 0; i < 16; ++i) {
                sum += out[b].qs[g * 16 + i];
            }
            out[b].bsums[g] = static_cast<int16_t>(sum);
        }
        offset += block_elems;
    }
}

float dotProductRowQ8_K(const uint8_t* src,
                        size_t cols,
                        const float* vec) {
    const BlockQ8_K* blocks = reinterpret_cast<const BlockQ8_K*>(src);
    size_t nb = numBlocks(cols, QK_K);
    float acc = 0.0f;
    for (size_t b = 0; b < nb; ++b) {
        float d = blocks[b].d;
        const int8_t* qs = blocks[b].qs;
        size_t base = b * QK_K;
        size_t end = std::min(cols, base + QK_K);
        size_t out = base;
        for (size_t i = 0; i < QK_K && out < end; ++i) {
            float vf = d * static_cast<float>(qs[i]);
            acc += vf * vec[out++];
        }
    }
    return acc;
}

float dotProductRowQ6_K(const uint8_t* src,
                        size_t cols,
                        const float* vec) {
    const BlockQ6_K* blocks = reinterpret_cast<const BlockQ6_K*>(src);
    size_t nb = numBlocks(cols, QK_K);
    float acc = 0.0f;
    for (size_t b = 0; b < nb; ++b) {
        float d = fp16ToFloat(blocks[b].d);
        const uint8_t* ql = blocks[b].ql;
        const uint8_t* qh = blocks[b].qh;
        const int8_t* scales = blocks[b].scales;
        size_t base = b * QK_K;
        size_t end = std::min(cols, base + QK_K);
        size_t out = base;
        for (size_t i = 0; i < QK_K / 16 && out < end; ++i) {
            int8_t sc = scales[i];
            float dl = d * static_cast<float>(sc);
            for (size_t j = 0; j < 16 && out < end; ++j) {
                size_t idx = i * 16 + j;
                int bit = (idx & 1) ? (ql[idx / 2] >> 4) : (ql[idx / 2] & 0xF);
                int high = (qh[idx / 4] >> (2 * (idx % 4))) & 0x3;
                int val = bit | (high << 4);
                if (val > 31) val -= 64;
                float vf = dl * static_cast<float>(val);
                acc += vf * vec[out++];
            }
        }
    }
    return acc;
}

void quantizeRowQ6_K(const float* src,
                     size_t cols,
                     std::vector<uint8_t>& dst) {
    size_t blocks = ceilDiv(cols, QK_K);
    dst.resize(blocks * BLOCK_Q6_K);
    BlockQ6_K* out = reinterpret_cast<BlockQ6_K*>(dst.data());
    size_t offset = 0;
    constexpr size_t kChunks = QK_K / 16;
    for (size_t b = 0; b < blocks; ++b) {
        size_t block_elems = std::min(QK_K, cols - offset);
        std::array<float, QK_K> block_data{};
        if (block_elems > 0) {
            std::memcpy(block_data.data(), src + offset, block_elems * sizeof(float));
        }
        std::array<float, kChunks> chunk_scales{};
        float max_scale = 0.0f;
        for (size_t j = 0; j < kChunks; ++j) {
            size_t start = j * 16;
            if (start >= block_elems) {
                chunk_scales[j] = 0.0f;
                continue;
            }
            float max_abs = 0.0f;
            for (size_t i = start; i < start + 16 && i < block_elems; ++i) {
                max_abs = std::max(max_abs, std::fabs(block_data[i]));
            }
            float scale = max_abs / 32.0f;
            if (scale < kEpsilon) scale = 0.0f;
            chunk_scales[j] = scale;
            max_scale = std::max(max_scale, scale);
        }
        float base = (max_scale > 0.0f) ? (max_scale / 127.0f) : 0.0f;
        out[b].d = floatToFp16(base);
        std::array<int8_t, kChunks> scale_codes{};
        std::array<uint8_t, QK_K> q_encoded{};
        for (size_t j = 0; j < kChunks; ++j) {
            int8_t sc = 0;
            if (base > 0.0f && chunk_scales[j] > 0.0f) {
                int code = static_cast<int>(std::round(chunk_scales[j] / base));
                code = clampValue(code, -127, 127);
                sc = static_cast<int8_t>(code);
            }
            scale_codes[j] = sc;
            float actual_scale = base * static_cast<float>(sc);
            for (size_t i = 0; i < 16; ++i) {
                size_t data_idx = j * 16 + i;
                if (data_idx >= QK_K) break;
                float value = (data_idx < block_elems) ? block_data[data_idx] : 0.0f;
                float qf = (actual_scale != 0.0f) ? std::round(value / actual_scale) : 0.0f;
                int q = clampValue(static_cast<int>(qf), -32, 31);
                q_encoded[data_idx] = static_cast<uint8_t>(q + 32);
            }
        }
        std::memcpy(out[b].scales, scale_codes.data(), kChunks * sizeof(int8_t));
        std::memset(out[b].ql, 0, sizeof(out[b].ql));
        std::memset(out[b].qh, 0, sizeof(out[b].qh));
        uint8_t* ql = out[b].ql;
        uint8_t* qh = out[b].qh;
        for (size_t n = 0; n < QK_K; n += 128) {
            for (size_t l = 0; l < 32; ++l) {
                uint8_t q1 = q_encoded[n + l];
                uint8_t q2 = q_encoded[n + l + 32];
                uint8_t q3 = q_encoded[n + l + 64];
                uint8_t q4 = q_encoded[n + l + 96];
                ql[l + 0] = static_cast<uint8_t>((q1 & 0xF) | ((q3 & 0xF) << 4));
                ql[l + 32] = static_cast<uint8_t>((q2 & 0xF) | ((q4 & 0xF) << 4));
                qh[l] = static_cast<uint8_t>(((q1 >> 4) & 0x3) |
                                             (((q2 >> 4) & 0x3) << 2) |
                                             (((q3 >> 4) & 0x3) << 4) |
                                             (((q4 >> 4) & 0x3) << 6));
            }
            ql += 64;
            qh += 32;
        }
        offset += block_elems;
    }
}

} // namespace runtime
} // namespace mlc
