#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "frontends/ggml_types.hpp"

namespace mlc {
namespace runtime {

struct TensorStorage {
    uint32_t dtype = frontend::GGML_TYPE_F32;
    uint32_t quant_version = 0;
    size_t row_stride_bytes = 0;
    std::vector<uint8_t> raw_data;
    std::vector<float> float_data;

    TensorStorage() = default;

    static TensorStorage FromFloatVector(std::vector<float>&& values) {
        TensorStorage storage;
        storage.dtype = frontend::GGML_TYPE_F32;
        storage.row_stride_bytes = values.size() * sizeof(float);
        storage.float_data = std::move(values);
        return storage;
    }

    const std::vector<float>* tryFloatData() const {
        if (dtype != frontend::GGML_TYPE_F32) {
            return nullptr;
        }
        return &float_data;
    }

    std::vector<float>* tryMutableFloatData() {
        if (dtype != frontend::GGML_TYPE_F32) {
            return nullptr;
        }
        return &float_data;
    }

    void assignRawData(uint32_t type,
                       size_t row_stride,
                       uint32_t quant_ver,
                       std::vector<uint8_t>&& bytes) {
        dtype = type;
        row_stride_bytes = row_stride;
        quant_version = quant_ver;
        raw_data = std::move(bytes);
        float_data.clear();
    }

    size_t elementCount() const {
        if (dtype == frontend::GGML_TYPE_F32) {
            return float_data.size();
        }
        if (row_stride_bytes == 0) return 0;
        return raw_data.size() / row_stride_bytes;
    }
};

} // namespace runtime
} // namespace mlc
