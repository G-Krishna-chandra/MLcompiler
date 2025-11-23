#include "runtime/session.hpp"
#include "frontends/ggml_types.hpp"

#include "runtime/float_convert.hpp"
#include "runtime/quantization.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

namespace mlc {
namespace runtime {

namespace {

void ensureRowStride(const std::string& tensor_name,
                     size_t actual,
                     size_t expected) {
    if (actual != expected) {
        throw std::runtime_error("Unexpected tensor data size for '" + tensor_name + "'");
    }
}

void decodeTensorRow(const frontend::GGUFTensorInfo& tensor,
                     const std::vector<uint8_t>& data,
                     size_t row_stride_bytes,
                     uint32_t quant_version,
                     uint64_t cols_u64,
                     uint64_t row_index,
                     std::vector<float>& buffer) {
    size_t cols = static_cast<size_t>(cols_u64);
    if (cols == 0) {
        throw std::runtime_error("Tensor '" + tensor.name + "' has zero columns");
    }
    if (buffer.size() != cols) {
        buffer.assign(cols, 0.0f);
    }
    size_t offset = static_cast<size_t>(row_index) * row_stride_bytes;
    if (offset + row_stride_bytes > data.size()) {
        throw std::runtime_error("Tensor '" + tensor.name + "' row out of range");
    }
    const uint8_t* row_ptr = data.data() + offset;

    switch (tensor.dtype) {
        case frontend::GGML_TYPE_F32: {
            size_t expected = cols * sizeof(float);
            ensureRowStride(tensor.name, row_stride_bytes, expected);
            const float* row_f32 = reinterpret_cast<const float*>(row_ptr);
            std::copy(row_f32, row_f32 + cols, buffer.begin());
            break;
        }
        case frontend::GGML_TYPE_F16: {
            size_t expected = cols * sizeof(uint16_t);
            ensureRowStride(tensor.name, row_stride_bytes, expected);
            const uint16_t* row_half = reinterpret_cast<const uint16_t*>(row_ptr);
            for (size_t i = 0; i < cols; ++i) {
                buffer[i] = fp16ToFloat(row_half[i]);
            }
            break;
        }
        case frontend::GGML_TYPE_BF16: {
            size_t expected = cols * sizeof(uint16_t);
            ensureRowStride(tensor.name, row_stride_bytes, expected);
            const uint16_t* row_bf16 = reinterpret_cast<const uint16_t*>(row_ptr);
            for (size_t i = 0; i < cols; ++i) {
                buffer[i] = bf16ToFloat(row_bf16[i]);
            }
            break;
        }
        case frontend::GGML_TYPE_I8: {
            size_t expected = cols * sizeof(int8_t);
            ensureRowStride(tensor.name, row_stride_bytes, expected);
            const int8_t* row_i8 = reinterpret_cast<const int8_t*>(row_ptr);
            for (size_t i = 0; i < cols; ++i) {
                buffer[i] = static_cast<float>(row_i8[i]);
            }
            break;
        }
        case frontend::GGML_TYPE_Q4_0: {
            size_t expected = q4_0RowSize(cols, quant_version);
            ensureRowStride(tensor.name, row_stride_bytes, expected);
            dequantizeRowQ4_0(row_ptr, cols, quant_version, buffer.data());
            break;
        }
        case frontend::GGML_TYPE_Q4_1: {
            size_t expected = q4_1RowSize(cols);
            ensureRowStride(tensor.name, row_stride_bytes, expected);
            dequantizeRowQ4_1(row_ptr, cols, buffer.data());
            break;
        }
        case frontend::GGML_TYPE_Q5_0: {
            size_t expected = q5_0RowSize(cols);
            ensureRowStride(tensor.name, row_stride_bytes, expected);
            dequantizeRowQ5_0(row_ptr, cols, buffer.data());
            break;
        }
        case frontend::GGML_TYPE_Q5_1: {
            size_t expected = q5_1RowSize(cols);
            ensureRowStride(tensor.name, row_stride_bytes, expected);
            dequantizeRowQ5_1(row_ptr, cols, buffer.data());
            break;
        }
        case frontend::GGML_TYPE_Q8_0: {
            size_t expected = q8_0RowSize(cols);
            ensureRowStride(tensor.name, row_stride_bytes, expected);
            dequantizeRowQ8_0(row_ptr, cols, buffer.data());
            break;
        }
        case frontend::GGML_TYPE_Q8_1: {
            size_t expected = q8_1RowSize(cols);
            ensureRowStride(tensor.name, row_stride_bytes, expected);
            dequantizeRowQ8_1(row_ptr, cols, buffer.data());
            break;
        }
        case frontend::GGML_TYPE_Q2_K: {
            size_t expected = q2_kRowSize(cols);
            ensureRowStride(tensor.name, row_stride_bytes, expected);
            dequantizeRowQ2_K(row_ptr, cols, buffer.data());
            break;
        }
        case frontend::GGML_TYPE_Q3_K: {
            size_t expected = q3_kRowSize(cols);
            ensureRowStride(tensor.name, row_stride_bytes, expected);
            dequantizeRowQ3_K(row_ptr, cols, buffer.data());
            break;
        }
        case frontend::GGML_TYPE_Q4_K: {
            size_t expected = q4_kRowSize(cols);
            ensureRowStride(tensor.name, row_stride_bytes, expected);
            dequantizeRowQ4_K(row_ptr, cols, buffer.data());
            break;
        }
        case frontend::GGML_TYPE_Q5_K: {
            size_t expected = q5_kRowSize(cols);
            ensureRowStride(tensor.name, row_stride_bytes, expected);
            dequantizeRowQ5_K(row_ptr, cols, buffer.data());
            break;
        }
        case frontend::GGML_TYPE_Q6_K: {
            size_t expected = q6_kRowSize(cols);
            ensureRowStride(tensor.name, row_stride_bytes, expected);
            dequantizeRowQ6_K(row_ptr, cols, buffer.data());
            break;
        }
        case frontend::GGML_TYPE_Q8_K: {
            size_t expected = q8_kRowSize(cols);
            ensureRowStride(tensor.name, row_stride_bytes, expected);
            dequantizeRowQ8_K(row_ptr, cols, buffer.data());
            break;
        }
        default:
            throw std::runtime_error("Tensor '" + tensor.name + "' uses unsupported dtype for embeddings/linear ops");
    }
}

} // namespace


Session::Session(const std::string& gguf_path) : loader_(gguf_path) {
    if (!loader_.load()) {
        throw std::runtime_error("Failed to load GGUF file: " + gguf_path);
    }
}

std::vector<float> Session::runLinear(const std::string& tensor_name,
                                      const std::vector<float>& input) const {
    const auto& tensors = loader_.tensors();
    auto it = tensors.find(tensor_name);
    if (it == tensors.end()) {
        throw std::runtime_error("Tensor '" + tensor_name + "' not found");
    }

    const auto& tensor = it->second;
    if (tensor.shape.size() != 2) {
        throw std::runtime_error("Tensor '" + tensor_name + "' is not 2D");
    }

    uint64_t rows = tensor.shape[0];
    uint64_t cols = tensor.shape[1];
    bool matches_cols = input.size() == cols;
    bool matches_rows = input.size() == rows;
    if (!matches_cols && !matches_rows) {
        throw std::runtime_error("Input vector size mismatch for tensor '" + tensor_name + "'");
    }

    auto data = loader_.loadTensorData(tensor);
    if (rows == 0 || cols == 0) {
        throw std::runtime_error("Tensor '" + tensor_name + "' has zero dimensions");
    }
    if (data.size() % rows != 0) {
        throw std::runtime_error("Unexpected tensor data size for '" + tensor_name + "'");
    }
    size_t row_stride_bytes = data.size() / rows;
    uint32_t quant_version = loader_.quantizationVersion();
    size_t cols_size = static_cast<size_t>(cols);
    std::vector<float> row_buffer(cols_size, 0.0f);

    if (matches_cols) {
        std::vector<float> output(rows, 0.0f);
#if defined(__APPLE__)
        if (cols >= 64 && tensor.dtype == frontend::GGML_TYPE_F32) {
            const float* weights = reinterpret_cast<const float*>(data.data());
            cblas_sgemv(CblasRowMajor,
                        CblasNoTrans,
                        static_cast<int>(rows),
                        static_cast<int>(cols),
                        1.0f,
                        weights,
                        static_cast<int>(cols),
                        input.data(),
                        1,
                        0.0f,
                        output.data(),
                        1);
            return output;
        }
#endif
        if (tensor.dtype == frontend::GGML_TYPE_Q4_0) {
            for (uint64_t r = 0; r < rows; ++r) {
                const uint8_t* row_ptr = data.data() + r * row_stride_bytes;
                output[r] = dotProductRowQ4_0(row_ptr,
                                              cols_size,
                                              quant_version,
                                              input.data());
            }
            return output;
        } else if (tensor.dtype == frontend::GGML_TYPE_Q2_K) {
            for (uint64_t r = 0; r < rows; ++r) {
                const uint8_t* row_ptr = data.data() + r * row_stride_bytes;
                output[r] = dotProductRowQ2_K(row_ptr,
                                              cols_size,
                                              input.data());
            }
            return output;
        } else if (tensor.dtype == frontend::GGML_TYPE_Q3_K) {
            for (uint64_t r = 0; r < rows; ++r) {
                const uint8_t* row_ptr = data.data() + r * row_stride_bytes;
                output[r] = dotProductRowQ3_K(row_ptr,
                                              cols_size,
                                              input.data());
            }
            return output;
        } else if (tensor.dtype == frontend::GGML_TYPE_Q4_K) {
            for (uint64_t r = 0; r < rows; ++r) {
                const uint8_t* row_ptr = data.data() + r * row_stride_bytes;
                output[r] = dotProductRowQ4_K(row_ptr,
                                              cols_size,
                                              input.data());
            }
            return output;
        } else if (tensor.dtype == frontend::GGML_TYPE_Q5_K) {
            for (uint64_t r = 0; r < rows; ++r) {
                const uint8_t* row_ptr = data.data() + r * row_stride_bytes;
                output[r] = dotProductRowQ5_K(row_ptr,
                                              cols_size,
                                              input.data());
            }
            return output;
        } else if (tensor.dtype == frontend::GGML_TYPE_Q6_K) {
            for (uint64_t r = 0; r < rows; ++r) {
                const uint8_t* row_ptr = data.data() + r * row_stride_bytes;
                output[r] = dotProductRowQ6_K(row_ptr,
                                              cols_size,
                                              input.data());
            }
            return output;
        } else if (tensor.dtype == frontend::GGML_TYPE_Q8_K) {
            for (uint64_t r = 0; r < rows; ++r) {
                const uint8_t* row_ptr = data.data() + r * row_stride_bytes;
                output[r] = dotProductRowQ8_K(row_ptr,
                                              cols_size,
                                              input.data());
            }
            return output;
        } else if (tensor.dtype == frontend::GGML_TYPE_Q4_1) {
            for (uint64_t r = 0; r < rows; ++r) {
                const uint8_t* row_ptr = data.data() + r * row_stride_bytes;
                output[r] = dotProductRowQ4_1(row_ptr,
                                              cols_size,
                                              input.data());
            }
            return output;
        } else if (tensor.dtype == frontend::GGML_TYPE_Q5_0) {
            for (uint64_t r = 0; r < rows; ++r) {
                const uint8_t* row_ptr = data.data() + r * row_stride_bytes;
                output[r] = dotProductRowQ5_0(row_ptr,
                                              cols_size,
                                              input.data());
            }
            return output;
        } else if (tensor.dtype == frontend::GGML_TYPE_Q5_1) {
            for (uint64_t r = 0; r < rows; ++r) {
                const uint8_t* row_ptr = data.data() + r * row_stride_bytes;
                output[r] = dotProductRowQ5_1(row_ptr,
                                              cols_size,
                                              input.data());
            }
            return output;
        } else if (tensor.dtype == frontend::GGML_TYPE_Q8_0) {
            for (uint64_t r = 0; r < rows; ++r) {
                const uint8_t* row_ptr = data.data() + r * row_stride_bytes;
                output[r] = dotProductRowQ8_0(row_ptr,
                                              cols_size,
                                              input.data());
            }
            return output;
        } else if (tensor.dtype == frontend::GGML_TYPE_Q8_1) {
            for (uint64_t r = 0; r < rows; ++r) {
                const uint8_t* row_ptr = data.data() + r * row_stride_bytes;
                output[r] = dotProductRowQ8_1(row_ptr,
                                              cols_size,
                                              input.data());
            }
            return output;
        }
        for (uint64_t r = 0; r < rows; ++r) {
            decodeTensorRow(tensor, data, row_stride_bytes, quant_version, cols, r, row_buffer);
            float acc = 0.0f;
            for (uint64_t c = 0; c < cols; ++c) {
                acc += row_buffer[c] * static_cast<float>(input[c]);
            }
            output[r] = acc;
        }
        return output;
    } else {
        std::vector<float> output(cols, 0.0f);
#if defined(__APPLE__)
        if (rows >= 64 && tensor.dtype == frontend::GGML_TYPE_F32) {
            const float* weights = reinterpret_cast<const float*>(data.data());
            cblas_sgemv(CblasRowMajor,
                        CblasTrans,
                        static_cast<int>(rows),
                        static_cast<int>(cols),
                        1.0f,
                        weights,
                        static_cast<int>(cols),
                        input.data(),
                        1,
                        0.0f,
                        output.data(),
                        1);
            return output;
        }
#endif
        for (uint64_t r = 0; r < rows; ++r) {
            decodeTensorRow(tensor, data, row_stride_bytes, quant_version, cols, r, row_buffer);
            float multiplier = static_cast<float>(input[r]);
            for (uint64_t c = 0; c < cols; ++c) {
                output[c] += row_buffer[c] * multiplier;
            }
        }
        return output;
    }
}

std::vector<float> Session::getEmbedding(const std::string& tensor_name,
                                         uint64_t token_id) const {
    const auto& tensors = loader_.tensors();
    auto it = tensors.find(tensor_name);
    if (it == tensors.end()) {
        throw std::runtime_error("Tensor '" + tensor_name + "' not found");
    }

    const auto& tensor = it->second;
    if (tensor.shape.size() != 2) {
        throw std::runtime_error("Tensor '" + tensor_name + "' is not 2D");
    }

    uint64_t rows = tensor.shape[0];
    uint64_t cols = tensor.shape[1];

    auto data = loader_.loadTensorData(tensor);
    if (rows == 0 || cols == 0) {
        throw std::runtime_error("Tensor '" + tensor_name + "' has zero dimensions");
    }
    if (data.size() % rows != 0) {
        fprintf(stderr, "[Session] tensor '%s': data_size=%zu rows=%llu\n",
                tensor_name.c_str(),
                data.size(),
                (unsigned long long)rows);
        throw std::runtime_error("Unexpected tensor data size for '" + tensor_name + "'");
    }
    size_t row_stride_bytes = data.size() / rows;

    if (std::getenv("MLC_VERBOSE")) {
        fprintf(stderr, "[Session] tensor='%s' dtype=%u rows=%llu cols=%llu data_size=%zu row_stride=%zu quant_v=%u\n",
                tensor_name.c_str(),
                tensor.dtype,
                (unsigned long long)rows,
                (unsigned long long)cols,
                data.size(),
                row_stride_bytes,
                loader_.quantizationVersion());
    }

    bool tokens_as_rows = true;
    std::string lower_name = tensor_name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    bool looks_like_embedding =
        (lower_name.find("tok") != std::string::npos && lower_name.find("emb") != std::string::npos) ||
        lower_name.find("token_embd") != std::string::npos;
    if (looks_like_embedding && rows < cols) {
        tokens_as_rows = false;
    }
    uint64_t vocab_dim = tokens_as_rows ? rows : cols;
    uint64_t embed_dim = tokens_as_rows ? cols : rows;
    if (token_id >= vocab_dim) {
        throw std::runtime_error("Token id out of range for tensor '" + tensor_name + "'");
    }

    uint32_t quant_version = loader_.quantizationVersion();
    if (tokens_as_rows) {
        std::vector<float> output(static_cast<size_t>(embed_dim), 0.0f);
        decodeTensorRow(tensor, data, row_stride_bytes, quant_version, cols, token_id, output);
        return output;
    }

    std::vector<float> output(static_cast<size_t>(embed_dim), 0.0f);
    std::vector<float> temp_row(static_cast<size_t>(cols), 0.0f);
    size_t token_index = static_cast<size_t>(token_id);
    for (uint64_t r = 0; r < rows; ++r) {
        decodeTensorRow(tensor, data, row_stride_bytes, quant_version, cols, r, temp_row);
        output[r] = temp_row[token_index];
    }
    return output;
}

} // namespace runtime
} // namespace mlc
