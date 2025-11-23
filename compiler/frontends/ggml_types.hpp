#pragma once

#include <cstdint>

namespace mlc {
namespace frontend {

constexpr uint32_t GGML_TYPE_F32     = 0;
constexpr uint32_t GGML_TYPE_F16     = 1;
constexpr uint32_t GGML_TYPE_Q4_0    = 2;
constexpr uint32_t GGML_TYPE_Q4_1    = 3;
constexpr uint32_t GGML_TYPE_Q5_0    = 6;
constexpr uint32_t GGML_TYPE_Q5_1    = 7;
constexpr uint32_t GGML_TYPE_Q8_0    = 8;
constexpr uint32_t GGML_TYPE_Q8_1    = 9;
constexpr uint32_t GGML_TYPE_Q2_K    = 10;
constexpr uint32_t GGML_TYPE_Q3_K    = 11;
constexpr uint32_t GGML_TYPE_Q4_K    = 12;
constexpr uint32_t GGML_TYPE_Q5_K    = 13;
constexpr uint32_t GGML_TYPE_Q6_K    = 14;
constexpr uint32_t GGML_TYPE_Q8_K    = 15;
constexpr uint32_t GGML_TYPE_I8      = 24;
constexpr uint32_t GGML_TYPE_I16     = 25;
constexpr uint32_t GGML_TYPE_I32     = 26;
constexpr uint32_t GGML_TYPE_I64     = 27;
constexpr uint32_t GGML_TYPE_F64     = 28;
constexpr uint32_t GGML_TYPE_BF16    = 30;

} // namespace frontend
} // namespace mlc

