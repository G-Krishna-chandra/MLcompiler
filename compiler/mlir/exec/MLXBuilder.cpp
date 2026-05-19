#include "compiler/mlir/exec/MLXBuilder.h"
#include "compiler/mlir/exec/Quantize.h"

#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

#include <cstring>
#include <stdexcept>

namespace mx = mlx::core;

namespace mlir::mlc::exec {

namespace {

// Copy `data` into a fresh malloc'd buffer the mx::array can own. MLX's
// raw-pointer constructor takes a deleter we install to free() at array
// teardown.
mx::array makeMLXFromCopy(const float *src, size_t n, mx::Shape shape) {
  // mx::array takes a raw pointer + deleter; we malloc + memcpy + free on
  // destruction.
  float *buf = static_cast<float *>(std::malloc(n * sizeof(float)));
  if (!buf)
    throw std::bad_alloc();
  std::memcpy(buf, src, n * sizeof(float));
  return mx::array(static_cast<void *>(buf), std::move(shape), mx::float32,
                   [](void *p) { std::free(p); });
}

} // namespace

mx::array fp32ToMLX(const std::vector<float> &data,
                    const std::vector<int> &shape) {
  size_t expected = 1;
  for (int d : shape)
    expected *= static_cast<size_t>(d);
  if (data.size() != expected)
    throw std::runtime_error("fp32ToMLX: data size doesn't match shape");
  mx::Shape s(shape.begin(), shape.end());
  return makeMLXFromCopy(data.data(), data.size(), std::move(s));
}

mx::array gufWeightToMLX(const ::mlc::frontend::GGUFLoader &loader,
                         const std::string &tensor_name) {
  auto deq = dequantizeToF32(loader, tensor_name);
  const auto &info = loader.tensors().at(tensor_name);
  std::vector<int> shape;
  shape.reserve(info.shape.size());
  for (uint64_t d : info.shape)
    shape.push_back(static_cast<int>(d));
  return fp32ToMLX(deq, shape);
}

mx::array matmul(const mx::array &x, const mx::array &w, bool transpose_b) {
  return mx::matmul(x, transpose_b ? mx::transpose(w) : w);
}

std::vector<float> mlxToF32(const mx::array &a) {
  // Force evaluation; eval is a free function on a single array via the
  // public C++ API.
  mx::array tmp = a;
  if (tmp.dtype() != mx::float32)
    tmp = mx::astype(tmp, mx::float32);
  mx::eval(tmp);
  const float *src = tmp.data<float>();
  return std::vector<float>(src, src + tmp.size());
}

} // namespace mlir::mlc::exec
