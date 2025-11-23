#include "runtime/quant_utils.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "frontends/ggml_types.hpp"
#include "runtime/quantization.hpp"

namespace mlc {
namespace runtime {

size_t ggmlRowSizeBytes(uint32_t dtype, size_t cols, uint32_t quant_version) {
    switch (dtype) {
        case frontend::GGML_TYPE_F32:
            return cols * sizeof(float);
        case frontend::GGML_TYPE_F16:
        case frontend::GGML_TYPE_BF16:
            return cols * sizeof(uint16_t);
        case frontend::GGML_TYPE_I8:
            return cols * sizeof(int8_t);
        case frontend::GGML_TYPE_Q4_0:
            return q4_0RowSize(cols, quant_version);
        case frontend::GGML_TYPE_Q4_1:
            return q4_1RowSize(cols);
        case frontend::GGML_TYPE_Q5_0:
            return q5_0RowSize(cols);
        case frontend::GGML_TYPE_Q5_1:
            return q5_1RowSize(cols);
        case frontend::GGML_TYPE_Q8_0:
            return q8_0RowSize(cols);
        case frontend::GGML_TYPE_Q8_1:
            return q8_1RowSize(cols);
        case frontend::GGML_TYPE_Q2_K:
            return q2_kRowSize(cols);
        case frontend::GGML_TYPE_Q3_K:
            return q3_kRowSize(cols);
        case frontend::GGML_TYPE_Q4_K:
            return q4_kRowSize(cols);
        case frontend::GGML_TYPE_Q5_K:
            return q5_kRowSize(cols);
        case frontend::GGML_TYPE_Q6_K:
            return q6_kRowSize(cols);
        case frontend::GGML_TYPE_Q8_K:
            return q8_kRowSize(cols);
        default:
            throw std::runtime_error("Unsupported GGML dtype for row size");
    }
}

void dequantizeRowTo(const uint8_t* src,
                     uint32_t dtype,
                     size_t cols,
                     uint32_t quant_version,
                     float* dst) {
    switch (dtype) {
        case frontend::GGML_TYPE_F32: {
            const float* ptr = reinterpret_cast<const float*>(src);
            std::copy(ptr, ptr + cols, dst);
            break;
        }
        case frontend::GGML_TYPE_I8: {
            const int8_t* ptr = reinterpret_cast<const int8_t*>(src);
            for (size_t i = 0; i < cols; ++i) dst[i] = static_cast<float>(ptr[i]);
            break;
        }
        case frontend::GGML_TYPE_Q4_0:
            dequantizeRowQ4_0(src, cols, quant_version, dst);
            break;
        case frontend::GGML_TYPE_Q4_1:
            dequantizeRowQ4_1(src, cols, dst);
            break;
        case frontend::GGML_TYPE_Q5_0:
            dequantizeRowQ5_0(src, cols, dst);
            break;
        case frontend::GGML_TYPE_Q5_1:
            dequantizeRowQ5_1(src, cols, dst);
            break;
        case frontend::GGML_TYPE_Q8_0:
            dequantizeRowQ8_0(src, cols, dst);
            break;
        case frontend::GGML_TYPE_Q8_1:
            dequantizeRowQ8_1(src, cols, dst);
            break;
        case frontend::GGML_TYPE_Q2_K:
            dequantizeRowQ2_K(src, cols, dst);
            break;
        case frontend::GGML_TYPE_Q3_K:
            dequantizeRowQ3_K(src, cols, dst);
            break;
        case frontend::GGML_TYPE_Q4_K:
            dequantizeRowQ4_K(src, cols, dst);
            break;
        case frontend::GGML_TYPE_Q5_K:
            dequantizeRowQ5_K(src, cols, dst);
            break;
        case frontend::GGML_TYPE_Q6_K:
            dequantizeRowQ6_K(src, cols, dst);
            break;
        case frontend::GGML_TYPE_Q8_K:
            dequantizeRowQ8_K(src, cols, dst);
            break;
        default:
            throw std::runtime_error("Unsupported GGML dtype for dequantization");
    }
}

void quantizeRowFrom(const float* src,
                     uint32_t dtype,
                     size_t cols,
                     uint32_t quant_version,
                     uint8_t* dst) {
    switch (dtype) {
        case frontend::GGML_TYPE_F32: {
            float* ptr = reinterpret_cast<float*>(dst);
            std::copy(src, src + cols, ptr);
            break;
        }
        case frontend::GGML_TYPE_I8: {
            int8_t* ptr = reinterpret_cast<int8_t*>(dst);
            for (size_t i = 0; i < cols; ++i) ptr[i] = static_cast<int8_t>(std::round(src[i]));
            break;
        }
        case frontend::GGML_TYPE_Q4_0: {
            std::vector<uint8_t> tmp;
            quantizeRowQ4_0(src, cols, quant_version, tmp);
            std::copy(tmp.begin(), tmp.end(), dst);
            break;
        }
        case frontend::GGML_TYPE_Q4_1: {
            std::vector<uint8_t> tmp;
            quantizeRowQ4_1(src, cols, tmp);
            std::copy(tmp.begin(), tmp.end(), dst);
            break;
        }
        case frontend::GGML_TYPE_Q5_0: {
            std::vector<uint8_t> tmp;
            quantizeRowQ5_0(src, cols, tmp);
            std::copy(tmp.begin(), tmp.end(), dst);
            break;
        }
        case frontend::GGML_TYPE_Q5_1: {
            std::vector<uint8_t> tmp;
            quantizeRowQ5_1(src, cols, tmp);
            std::copy(tmp.begin(), tmp.end(), dst);
            break;
        }
        case frontend::GGML_TYPE_Q8_0: {
            std::vector<uint8_t> tmp;
            quantizeRowQ8_0(src, cols, tmp);
            std::copy(tmp.begin(), tmp.end(), dst);
            break;
        }
        case frontend::GGML_TYPE_Q8_1: {
            std::vector<uint8_t> tmp;
            quantizeRowQ8_1(src, cols, tmp);
            std::copy(tmp.begin(), tmp.end(), dst);
            break;
        }
        case frontend::GGML_TYPE_Q2_K: {
            std::vector<uint8_t> tmp;
            quantizeRowQ2_K(src, cols, tmp);
            std::copy(tmp.begin(), tmp.end(), dst);
            break;
        }
        case frontend::GGML_TYPE_Q3_K: {
            std::vector<uint8_t> tmp;
            quantizeRowQ3_K(src, cols, tmp);
            std::copy(tmp.begin(), tmp.end(), dst);
            break;
        }
        case frontend::GGML_TYPE_Q4_K: {
            std::vector<uint8_t> tmp;
            quantizeRowQ4_K(src, cols, tmp);
            std::copy(tmp.begin(), tmp.end(), dst);
            break;
        }
        case frontend::GGML_TYPE_Q5_K: {
            std::vector<uint8_t> tmp;
            quantizeRowQ5_K(src, cols, tmp);
            std::copy(tmp.begin(), tmp.end(), dst);
            break;
        }
        case frontend::GGML_TYPE_Q6_K: {
            std::vector<uint8_t> tmp;
            quantizeRowQ6_K(src, cols, tmp);
            std::copy(tmp.begin(), tmp.end(), dst);
            break;
        }
        case frontend::GGML_TYPE_Q8_K: {
            std::vector<uint8_t> tmp;
            quantizeRowQ8_K(src, cols, tmp);
            std::copy(tmp.begin(), tmp.end(), dst);
            break;
        }
        default:
            throw std::runtime_error("Unsupported GGML dtype for quantization");
    }
}

} // namespace runtime
} // namespace mlc
