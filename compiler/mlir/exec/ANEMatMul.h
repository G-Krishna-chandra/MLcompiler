#pragma once

#include <mlx/array.h>

#include <memory>
#include <string>

namespace mlir::mlc::exec {

// A single CoreML model wrapping one fp16 matmul `y = x @ w` where the
// weight is baked into the compiled model. Constructed once per weight
// at executor startup (slow — ~1s per compile). Per-prediction dispatch
// is fast (~50-100 µs on Apple silicon).
//
// The MLX-side bridge owns its own MLMultiArray scratch buffers and
// copies in/out on each predict. fp16 only.
class ANEMatMul {
public:
  // Wrap an already-built .mlpackage at `path` (e.g. from
  // `compiler/coreml/gen_matmul_model.py`). `M` is the fixed batch /
  // seq size of the model's input; for single-token decode use M=1.
  // Throws if the package can't compile or the MLModel can't load.
  ANEMatMul(const std::string &mlpackage_path, int M, int K, int N);
  ~ANEMatMul();

  ANEMatMul(const ANEMatMul &) = delete;
  ANEMatMul &operator=(const ANEMatMul &) = delete;
  ANEMatMul(ANEMatMul &&) noexcept;
  ANEMatMul &operator=(ANEMatMul &&) noexcept;

  // x must be shape [M, K] fp16 (will be evaluated). Returns [M, N] fp16.
  mlx::core::array predict(const mlx::core::array &x);

  int M() const;
  int K() const;
  int N() const;

private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

// Run the offline gen_matmul_model.py script via the Python 3.12 venv
// to produce a .mlpackage. Path returned points to the generated
// directory. Throws on any subprocess error.
std::string buildANEMatMulPackage(const std::string &out_dir,
                                  const std::string &cache_key,
                                  int M, int K, int N,
                                  const mlx::core::array &weight_fp16);

} // namespace mlir::mlc::exec
