// mlc-kernel-bench: per-shape kernel selection profiler.
//
// For every distinct matmul shape in TinyLlama Q4_0 at M=1, 4, 8:
//   - Benchmarks mx::quantized_matmul (MLX native)
//   - Benchmarks the custom Q4_0 Metal kernel (q4_0_matmul)
//   - Reports median µs and declares winner
//
// This is the W1 data collection step for the per-shape kernel selection moat.
// Run with: mlc-kernel-bench <gguf-path>

#include "compiler/frontends/gguf_loader.hpp"
#include "compiler/mlir/exec/MLXBuilder.h"
#include "compiler/mlir/exec/MLXQuantize.h"
#include "compiler/mlir/exec/Q4MatMul.h"

#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace mx = mlx::core;
using ::mlc::frontend::GGUFLoader;

namespace {

// ── Timing helpers ───────────────────────────────────────────────────────────

// Evaluate arr and return wall-clock µs (GPU sync included).
double timedEval(mx::array &arr) {
  auto t0 = std::chrono::steady_clock::now();
  mx::eval(arr);
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

// Run `fn` nwarmup+ntimed times; return median µs of the timed portion.
template <typename Fn>
double bench(Fn fn, int nwarmup = 20, int ntimed = 100) {
  // Warmup.
  for (int i = 0; i < nwarmup; ++i) {
    auto arr = fn();
    mx::eval(arr);
  }
  // Timed.
  std::vector<double> samples(ntimed);
  for (int i = 0; i < ntimed; ++i) {
    auto arr = fn();
    samples[i] = timedEval(arr);
  }
  std::sort(samples.begin(), samples.end());
  return samples[ntimed / 2];  // median
}

// ── Random activation builder ────────────────────────────────────────────────

mx::array randomAct(int M, int K) {
  // fp32 activations: small Gaussian-ish values.
  std::vector<float> buf(static_cast<size_t>(M) * K);
  for (auto &v : buf) v = 0.01f * (static_cast<float>(std::rand()) / RAND_MAX - 0.5f);
  mx::Shape shape{M, K};
  float *p = static_cast<float *>(std::malloc(buf.size() * sizeof(float)));
  std::memcpy(p, buf.data(), buf.size() * sizeof(float));
  return mx::array(static_cast<void *>(p), std::move(shape), mx::float32,
                   [](void *q) { std::free(q); });
}

// ── Per-shape benchmark ──────────────────────────────────────────────────────

struct ShapeResult {
  int M, K, N;
  std::string label;
  double mlx_us;    // median µs for mx::quantized_matmul
  double custom_us; // median µs for custom Q4_0 kernel
};

// Benchmark one (M, K, N) shape using the provided MLXQuantWeights and raw bytes.
ShapeResult benchShape(const std::string &label, int M, int K, int N,
                       const mlir::mlc::exec::MLXQuantWeights &mlx_pkg,
                       const mx::array &raw_bytes) {
  auto x = randomAct(M, K);
  mx::eval(x);

  // MLX path: mx::quantized_matmul.
  double mlx_us = bench([&] {
    return mx::quantized_matmul(x, mlx_pkg.w_q, mlx_pkg.scales, mlx_pkg.biases,
                                /*transpose=*/true, 32, 4, "affine");
  });

  // Custom path: q4_0_matmul.
  double custom_us = bench([&] {
    return mlir::mlc::exec::q4_0_matmul(x, raw_bytes, K, N);
  });

  return {M, K, N, label, mlx_us, custom_us};
}

// ── Weight loading helpers ───────────────────────────────────────────────────

// Build the concat'd MLXQuantWeights for a list of tensor names (along out-dim).
mlir::mlc::exec::MLXQuantWeights loadConcat(const GGUFLoader &loader,
                                             const std::vector<std::string> &names) {
  if (names.size() == 1) return mlir::mlc::exec::ggufQ4_0ToMLXQuantized(loader, names[0]);
  std::vector<mx::array> wqs, scs, bis;
  int in_dim = 0, out_total = 0;
  for (const auto &nm : names) {
    auto pkg = mlir::mlc::exec::ggufQ4_0ToMLXQuantized(loader, nm);
    wqs.push_back(pkg.w_q);
    scs.push_back(pkg.scales);
    bis.push_back(pkg.biases);
    in_dim  = pkg.in_dim;
    out_total += pkg.out_dim;
  }
  auto w_q_cat    = mx::concatenate(wqs, 0);
  auto scales_cat = mx::concatenate(scs, 0);
  auto biases_cat = mx::concatenate(bis, 0);
  mx::eval({w_q_cat, scales_cat, biases_cat});
  return {std::move(w_q_cat), std::move(scales_cat), std::move(biases_cat), in_dim, out_total};
}

// Build the concat'd raw Q4_0 bytes for the custom kernel.
mx::array loadBytesConcat(const GGUFLoader &loader,
                          const std::vector<std::string> &names) {
  if (names.size() == 1) return mlir::mlc::exec::gufWeightToBytesMLX(loader, names[0]);
  std::vector<mx::array> arrays;
  for (const auto &nm : names)
    arrays.push_back(mlir::mlc::exec::gufWeightToBytesMLX(loader, nm));
  auto cat = mx::concatenate(arrays, 0);
  mx::eval(cat);
  return cat;
}

// ── Report formatting ────────────────────────────────────────────────────────

void printTable(const std::vector<ShapeResult> &results) {
  std::printf("\n%-30s  %5s  %8s  %9s  %9s  %8s\n",
              "shape (label)",
              "M", "mlx µs", "custom µs", "winner", "speedup");
  std::printf("%s\n", std::string(80, '-').c_str());
  for (const auto &r : results) {
    bool custom_wins = r.custom_us < r.mlx_us;
    double speedup = custom_wins ? r.mlx_us / r.custom_us : r.custom_us / r.mlx_us;
    std::printf("%-30s  %5d  %8.1f  %9.1f  %9s  %7.2fx\n",
                r.label.c_str(), r.M, r.mlx_us, r.custom_us,
                custom_wins ? "custom" : "mlx", speedup);
  }
  std::printf("%s\n", std::string(80, '-').c_str());
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: mlc-kernel-bench <gguf-path>\n");
    return 1;
  }

  const std::string model_path = argv[1];
  GGUFLoader loader(model_path);
  if (!loader.load()) {
    std::fprintf(stderr, "failed to load %s\n", model_path.c_str());
    return 1;
  }

  // ── Load weights for each distinct shape ──────────────────────────────────
  // TinyLlama 1.1B Q4_0 shapes (all Q4_0 matmul ops in the compiler path):
  //
  //  QKV fused   : K=2048, N=2560  (Q=2048, K_kv=256, V_kv=256, concat along out)
  //  attn_output : K=2048, N=2048
  //  gate+up fused: K=2048, N=11264 (gate=5632, up=5632)
  //  ffn_down    : K=5632, N=2048

  struct ShapeDef {
    std::string label;
    std::vector<std::string> weight_names;
  };

  const std::vector<ShapeDef> shape_defs = {
    {"qkv_fused (K=2048, N=2560)",
     {"blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight"}},
    {"attn_out (K=2048, N=2048)",
     {"blk.0.attn_output.weight"}},
    {"gate_up_fused (K=2048, N=11264)",
     {"blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"}},
    {"ffn_down (K=5632, N=2048)",
     {"blk.0.ffn_down.weight"}},
  };

  std::printf("Loading weights…\n");
  struct ShapeWeights {
    std::string label;
    int K, N;
    mlir::mlc::exec::MLXQuantWeights mlx_pkg;
    mx::array raw_bytes;
  };
  std::vector<ShapeWeights> shapes;
  for (const auto &sd : shape_defs) {
    auto mlx_pkg   = loadConcat(loader, sd.weight_names);
    auto raw_bytes = loadBytesConcat(loader, sd.weight_names);
    mx::eval(raw_bytes);
    shapes.push_back({sd.label, mlx_pkg.in_dim, mlx_pkg.out_dim,
                      std::move(mlx_pkg), std::move(raw_bytes)});
    std::printf("  loaded %s  K=%d N=%d\n", sd.label.c_str(),
                shapes.back().K, shapes.back().N);
  }

  // ── Benchmark at M=1, 4, 8 ───────────────────────────────────────────────
  const std::vector<int> M_vals = {1, 4, 8};
  std::vector<ShapeResult> results;

  std::printf("\nBenchmarking (20 warmup + 100 timed iterations each)…\n");
  for (int M : M_vals) {
    for (const auto &sw : shapes) {
      auto r = benchShape(sw.label, M, sw.K, sw.N, sw.mlx_pkg, sw.raw_bytes);
      results.push_back(r);
      std::printf("  M=%d %s: mlx=%.1f µs  custom=%.1f µs\n",
                  M, sw.label.c_str(), r.mlx_us, r.custom_us);
      std::fflush(stdout);
    }
  }

  // ── Print table ───────────────────────────────────────────────────────────
  printTable(results);

  // ── Assess stop conditions ───────────────────────────────────────────────
  int mlx_wins = 0, custom_wins = 0;
  for (const auto &r : results) {
    if (r.custom_us < r.mlx_us) ++custom_wins;
    else ++mlx_wins;
  }
  int total = static_cast<int>(results.size());
  std::printf("\nSummary: mlx wins %d/%d shapes, custom wins %d/%d shapes\n",
              mlx_wins, total, custom_wins, total);
  if (mlx_wins == total)
    std::printf("STOP: MLX wins all shapes — no kernel selection advantage.\n");
  else if (custom_wins == total)
    std::printf("NOTE: Custom wins all shapes — could use custom-only, "
                "but check M=4,8 separately.\n");
  else
    std::printf("MIXED: Proceed to W2 (KernelSelect pass).\n");

  // ── Write markdown table to logs/ ────────────────────────────────────────
  const std::string outpath = "logs/kernel-selection-profile.md";
  FILE *f = std::fopen(outpath.c_str(), "w");
  if (f) {
    std::fprintf(f, "# Kernel Selection Profile\n\n");
    std::fprintf(f, "Model: TinyLlama 1.1B Q4_0, M3 Pro\n");
    std::fprintf(f, "Method: 20 warmup + 100 timed GPU-synced iterations, median µs\n\n");
    std::fprintf(f, "| shape | M | mlx µs | custom µs | winner | speedup |\n");
    std::fprintf(f, "|-------|---|--------|-----------|--------|---------|\n");
    for (const auto &r : results) {
      bool cw = r.custom_us < r.mlx_us;
      double sp = cw ? r.mlx_us / r.custom_us : r.custom_us / r.mlx_us;
      std::fprintf(f, "| %s | %d | %.1f | %.1f | %s | %.2fx |\n",
                   r.label.c_str(), r.M, r.mlx_us, r.custom_us,
                   cw ? "custom" : "mlx", sp);
    }
    std::fprintf(f, "\n**MLX wins:** %d/%d  **Custom wins:** %d/%d\n",
                 mlx_wins, total, custom_wins, total);
    std::fclose(f);
    std::printf("Results written to %s\n", outpath.c_str());
  }

  return 0;
}
