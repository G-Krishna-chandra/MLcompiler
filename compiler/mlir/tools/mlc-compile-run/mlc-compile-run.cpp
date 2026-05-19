// mlc-compile-run: end-to-end inference driver running through the Phase-2
// compiler path (GGUF → emitted MLIR → optional fusion → MLX walker →
// greedy decode).
//
// Usage:
//   mlc-compile-run <gguf> --prompt "..." [--max-tokens N] [--no-fuse]
//
// The runtime path lives in `mlc` and is unchanged; this binary is the
// compiler-path counterpart.

#include "compiler/frontends/gguf_loader.hpp"
#include "compiler/mlir/emit/GGUFToMLIR.h"
#include "compiler/mlir/exec/MLIRExecutor.h"
#include "compiler/mlir/passes/FuseNormMatMul.h"
#include "compiler/mlir/passes/FuseQKVMatMul.h"
#include "compiler/runtime/tokenizer.hpp"

#include "mlir/IR/MLIRContext.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
  std::string model_path;
  std::string prompt = "The capital of France is";
  int max_tokens = 16;
  bool fuse = true;
};

Args parseArgs(int argc, char **argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    if (s == "--prompt" && i + 1 < argc)      a.prompt = argv[++i];
    else if (s == "--max-tokens" && i + 1 < argc) a.max_tokens = std::atoi(argv[++i]);
    else if (s == "--no-fuse")                a.fuse = false;
    else if (s.size() && s[0] != '-')         a.model_path = s;
    else {
      std::fprintf(stderr, "Unknown arg: %s\n", argv[i]);
      std::exit(2);
    }
  }
  if (a.model_path.empty()) {
    std::fprintf(stderr,
                 "usage: mlc-compile-run <gguf> --prompt \"...\" "
                 "[--max-tokens N] [--no-fuse]\n");
    std::exit(2);
  }
  return a;
}

int argmaxRow(const float *row, int n) {
  int best = 0;
  float best_v = row[0];
  for (int i = 1; i < n; ++i) {
    if (row[i] > best_v) { best_v = row[i]; best = i; }
  }
  return best;
}

} // namespace

int main(int argc, char **argv) {
  Args args = parseArgs(argc, argv);

  mlc::frontend::GGUFLoader loader(args.model_path);
  if (!loader.load()) {
    std::fprintf(stderr, "failed to load %s\n", args.model_path.c_str());
    return 1;
  }

  mlc::runtime::Tokenizer tok(loader);
  if (!tok.valid()) {
    std::fprintf(stderr, "tokenizer init failed (no vocab in GGUF?)\n");
    return 1;
  }

  mlir::MLIRContext ctx;
  auto module = mlir::mlc::emitMLIR(ctx, loader);
  unsigned fused = 0;
  unsigned qkv_fused = 0;
  if (args.fuse) {
    fused = mlir::mlc::fuseNormMatMul(*module);
    qkv_fused = mlir::mlc::fuseQKVMatMul(*module);
  }
  std::cout << "[compile] fusion " << (args.fuse ? "ON" : "OFF")
            << ", " << fused << " norm+matmul fused, "
            << qkv_fused << " QKV triples merged\n";

  mlir::mlc::exec::MLIRExecutor exec(*module, loader);

  // Tokenize. TinyLlama uses BOS=1 and the chat-repl convention adds it by
  // default. Use add_bos=true to match the runtime.
  auto enc = tok.encode(args.prompt, mlc::runtime::TokenizerConfig{
                                          /*add_bos=*/true, /*add_eos=*/false});
  std::vector<int32_t> tokens;
  tokens.reserve(enc.size());
  for (uint64_t id : enc)
    tokens.push_back(static_cast<int32_t>(id));

  std::cout << "[prompt] " << args.prompt << "\n";
  std::cout << "[encoded] " << tokens.size() << " tokens\n";
  std::cout << "[generation] ";
  std::cout.flush();

  auto t_start = std::chrono::steady_clock::now();
  // Prefill: feed the whole prompt in a single run() to seed the KV cache.
  // Decode: feed one new token per call, advancing the absolute position.
  exec.reset();
  std::vector<int32_t> prefill_positions(tokens.size());
  std::iota(prefill_positions.begin(), prefill_positions.end(), 0);
  auto out = exec.run(tokens, prefill_positions);

  int abs_pos = static_cast<int>(tokens.size()) - 1;
  int new_tokens = 0;
  for (int step = 0; step < args.max_tokens; ++step) {
    int seq = out.shape[0];
    int vocab = out.shape[1];
    const float *last_row = out.data.data() + static_cast<size_t>(seq - 1) * vocab;
    int next = argmaxRow(last_row, vocab);

    if (tok.isEogToken(static_cast<uint64_t>(next)))
      break;

    std::cout << tok.tokenString(static_cast<uint64_t>(next)) << std::flush;
    tokens.push_back(next);
    ++new_tokens;
    ++abs_pos;
    if (step + 1 >= args.max_tokens) break;

    // One-token decode call. KV cache carries forward.
    out = exec.run({next}, {abs_pos});
  }
  auto t_end = std::chrono::steady_clock::now();
  std::cout << "\n";

  double secs =
      std::chrono::duration<double>(t_end - t_start).count();
  double toks_per_s = new_tokens > 0 ? (new_tokens / secs) : 0.0;
  std::cout << "[stats] generated " << new_tokens << " tokens in "
            << secs << "s (" << toks_per_s << " tok/s, "
            << "Q4_0 weights + custom Metal kernel, KV cache, "
            << "single-token decode)\n";

  return 0;
}
