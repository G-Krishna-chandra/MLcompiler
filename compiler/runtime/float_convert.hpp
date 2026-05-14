#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace mlc {
namespace runtime {

inline float fp16ToFloat(uint16_t h) {
    uint16_t h_exp = (h & 0x7C00u);
    uint16_t h_sig = (h & 0x03FFu);
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t mantissa;
    uint32_t exp;

    if (h_exp == 0) {
        if (h_sig == 0) {
            mantissa = 0;
            exp = 0;
        } else {
            exp = 0;
            while ((h_sig & 0x0400u) == 0) {
                h_sig <<= 1;
                exp++;
            }
            h_sig &= 0x03FFu;
            mantissa = static_cast<uint32_t>(h_sig) << 13;
            // Subnormal values use exponent = 1 - bias, then adjust for shift count.
            exp = 127 - 14 - exp;
        }
    } else if (h_exp == 0x7C00u) {
        exp = 255;
        mantissa = static_cast<uint32_t>(h_sig) << 13;
    } else {
        exp = ((h_exp >> 10) - 15 + 127);
        mantissa = static_cast<uint32_t>(h_sig) << 13;
    }

    uint32_t word = sign | (exp << 23) | mantissa;
    float result;
    std::memcpy(&result, &word, sizeof(result));
    return result;
}

inline float bf16ToFloat(uint16_t b16) {
    uint32_t word = static_cast<uint32_t>(b16) << 16;
    float result;
    std::memcpy(&result, &word, sizeof(result));
    return result;
}

inline uint16_t floatToFp16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFFu;

    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa = (mantissa | 0x800000u) >> (1 - exp);
        return static_cast<uint16_t>(sign | ((mantissa + 0x1000u) >> 13));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) |
                                 ((mantissa + 0x1000u) >> 13));
}

// Bulk fp16 → fp32 cast (NEON 8-elem fast path, scalar tail). Used by KV cache
// dequant and the fp16-attention input/output staging.
inline void castF16toF32(const uint16_t* src, float* dst, size_t n) {
    size_t i = 0;
#if defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
    for (; i + 8 <= n; i += 8) {
        float16x8_t h = vld1q_f16(reinterpret_cast<const __fp16*>(src + i));
        vst1q_f32(dst + i,     vcvt_f32_f16(vget_low_f16(h)));
        vst1q_f32(dst + i + 4, vcvt_f32_f16(vget_high_f16(h)));
    }
#endif
    for (; i < n; ++i) dst[i] = fp16ToFloat(src[i]);
}

// Bulk fp32 → fp16 cast (NEON 8-elem fast path, scalar tail).
inline void castF32toF16(const float* src, uint16_t* dst, size_t n) {
    size_t i = 0;
#if defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
    for (; i + 8 <= n; i += 8) {
        float32x4_t lo = vld1q_f32(src + i);
        float32x4_t hi = vld1q_f32(src + i + 4);
        float16x8_t h  = vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi));
        vst1q_f16(reinterpret_cast<__fp16*>(dst + i), h);
    }
#endif
    for (; i < n; ++i) dst[i] = floatToFp16(src[i]);
}

} // namespace runtime
} // namespace mlc
