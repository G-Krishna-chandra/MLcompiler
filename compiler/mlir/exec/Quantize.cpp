#include "compiler/mlir/exec/Quantize.h"
#include "compiler/runtime/quant_utils.hpp"

#include <stdexcept>

namespace mlir::mlc::exec {

using ::mlc::frontend::GGUFLoader;
using ::mlc::frontend::GGUFTensorInfo;

std::vector<float> dequantizeToF32(const GGUFLoader &loader,
                                   const std::string &tensor_name) {
  const auto &tensors = loader.tensors();
  auto it = tensors.find(tensor_name);
  if (it == tensors.end())
    throw std::runtime_error("dequantizeToF32: missing tensor " + tensor_name);
  const GGUFTensorInfo &info = it->second;

  // Total element count = product of dims.
  size_t n_elements = 1;
  for (uint64_t d : info.shape)
    n_elements *= static_cast<size_t>(d);

  std::vector<uint8_t> raw = loader.loadTensorData(info);

  // GGUF stores tensor dimensions in REVERSE numpy order: shape[0] is the
  // INNERMOST (fastest-varying) dim, i.e. elements-per-row of the row-major
  // byte layout. shape.back() is the OUTERMOST = number of rows. The runtime
  // uses the same convention (see batched_walker.cpp:191 for token_embd).
  size_t cols = static_cast<size_t>(info.shape[0]);
  size_t rows = n_elements / cols;
  size_t row_bytes = raw.size() / rows;
  if (rows * row_bytes != raw.size())
    throw std::runtime_error("dequantizeToF32: byte count not divisible by row count for " + tensor_name);

  std::vector<float> out(n_elements);
  uint32_t qv = loader.quantizationVersion();
  for (size_t r = 0; r < rows; ++r) {
    ::mlc::runtime::dequantizeRowTo(raw.data() + r * row_bytes, info.dtype,
                                    cols, qv, out.data() + r * cols);
  }
  return out;
}

} // namespace mlir::mlc::exec
