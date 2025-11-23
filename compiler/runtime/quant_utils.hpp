#pragma once

#include <cstddef>
#include <cstdint>

namespace mlc {
namespace runtime {

size_t ggmlRowSizeBytes(uint32_t dtype, size_t cols, uint32_t quant_version);
void dequantizeRowTo(const uint8_t* src,
                     uint32_t dtype,
                     size_t cols,
                     uint32_t quant_version,
                     float* dst);
void quantizeRowFrom(const float* src,
                     uint32_t dtype,
                     size_t cols,
                     uint32_t quant_version,
                     uint8_t* dst);

} // namespace runtime
} // namespace mlc
