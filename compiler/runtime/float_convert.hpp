#pragma once

#include <cstdint>
#include <cstring>

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
            exp = 127 - 15 - exp;
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

} // namespace runtime
} // namespace mlc
