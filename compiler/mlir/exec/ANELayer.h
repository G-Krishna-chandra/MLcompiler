#pragma once

#include <mlx/array.h>

#include <memory>
#include <string>
#include <utility>

namespace mlir::mlc::exec {

// One CoreML model that packs two independent matmuls into a single
// predict() call:
//
//   out_qkv  = x_qkv  @ W_QKV   (shape [M, K_qkv] @ [K_qkv, N_qkv])
//   out_attn = x_attn @ W_O     (shape [M, K_attn] @ [K_attn, N_attn])
//
// The point is U1's hypothesis: if the boundary cost is paid per
// predict() rather than per op, doubling the work inside one predict()
// should be much cheaper than two predicts. ANELayer is the experimental
// vehicle; an A/B against `2 * ANEMatMul::predict()` tells us whether
// to keep going.
class ANELayer {
public:
  ANELayer(const std::string &mlpackage_path, int M,
           int K_qkv, int N_qkv, int K_attn, int N_attn);
  ~ANELayer();
  ANELayer(const ANELayer &) = delete;
  ANELayer &operator=(const ANELayer &) = delete;
  ANELayer(ANELayer &&) noexcept;
  ANELayer &operator=(ANELayer &&) noexcept;

  // Run both subgraphs in one CoreML predict(). Returns (out_qkv, out_attn).
  // Inputs evaluated to fp16 internally; outputs come back in CoreML's
  // native output dtype (fp32 in our config), as with ANEMatMul.
  std::pair<mlx::core::array, mlx::core::array>
  predict(const mlx::core::array &x_qkv, const mlx::core::array &x_attn);

  int M() const;
  int K_qkv() const;
  int N_qkv() const;
  int K_attn() const;
  int N_attn() const;

private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

// Bake a packed-layer .mlpackage via the Python venv + gen_layer_model.py.
// Returns the path to the saved package. Caches by `cache_key` so reruns
// don't regenerate.
std::string buildANELayerPackage(const std::string &out_dir,
                                 const std::string &cache_key,
                                 int M, int K_qkv, int N_qkv,
                                 int K_attn, int N_attn,
                                 const mlx::core::array &w_qkv_fp16,
                                 const mlx::core::array &w_o_fp16);

} // namespace mlir::mlc::exec
