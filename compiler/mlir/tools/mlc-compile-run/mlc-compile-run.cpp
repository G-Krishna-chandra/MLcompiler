// mlc-compile-run: end-to-end inference driver running through the Phase-2
// compiler path (GGUF → emitted MLIR → optional fusion → MLX walker →
// greedy decode).
//
// Single-stream:
//   mlc-compile-run <gguf> --prompt "..." [--max-tokens N] [--no-fuse]
//
// Batched:
//   mlc-compile-run <gguf> --prompt "A" --prompt "B" ... --max-tokens N
//   (each --prompt becomes one concurrent request; batch size = #prompts)
//
// The runtime path lives in `mlc` and is unchanged; this binary is the
// compiler-path counterpart.

#include "compiler/frontends/gguf_loader.hpp"
#include "compiler/mlir/emit/GGUFToMLIR.h"
#include "compiler/mlir/exec/MLIRExecutor.h"
#include "compiler/mlir/passes/FuseNormMatMul.h"
#include "compiler/mlir/passes/FuseQKVMatMul.h"
#include "compiler/mlir/passes/ScheduleDevices.h"
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
  std::vector<std::string> prompts;   // one per concurrent request
  int max_tokens = 16;
  bool fuse = true;
  bool ane = false;
};

Args parseArgs(int argc, char **argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    if (s == "--prompt" && i + 1 < argc)          a.prompts.push_back(argv[++i]);
    else if (s == "--max-tokens" && i + 1 < argc) a.max_tokens = std::atoi(argv[++i]);
    else if (s == "--no-fuse")                    a.fuse = false;
    else if (s == "--ane")                        a.ane = true;
    else if (s.size() && s[0] != '-')             a.model_path = s;
    else {
      std::fprintf(stderr, "Unknown arg: %s\n", argv[i]);
      std::exit(2);
    }
  }
  if (a.model_path.empty()) {
    std::fprintf(stderr,
                 "usage: mlc-compile-run <gguf> [--prompt \"...\" ...] "
                 "[--max-tokens N] [--no-fuse]\n");
    std::exit(2);
  }
  // Default to one prompt if none supplied.
  if (a.prompts.empty())
    a.prompts.push_back("The capital of France is");
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
  int N = static_cast<int>(args.prompts.size());

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
  unsigned fused = 0, qkv_fused = 0;
  if (args.fuse) {
    fused     = mlir::mlc::fuseNormMatMul(*module);
    qkv_fused = mlir::mlc::fuseQKVMatMul(*module);
  }
  if (args.ane) mlir::mlc::scheduleDevices(*module);
  std::cout << "[compile] fusion " << (args.fuse ? "ON" : "OFF")
            << ", " << fused << " norm+matmul fused, "
            << qkv_fused << " QKV triples merged"
            << ", ANE " << (args.ane ? "ON" : "OFF") << "\n";

  mlir::mlc::exec::MLIRExecutor exec(*module, loader, args.ane);

  const char *q4_mode = std::getenv("MLC_Q4_CUSTOM_KERNEL");
  bool use_custom = q4_mode && std::string(q4_mode) == "1";

  // ── Single-stream path ───────────────────────────────────────────────────
  if (N == 1) {
    auto enc = tok.encode(args.prompts[0],
                          mlc::runtime::TokenizerConfig{true, false});
    std::vector<int32_t> tokens;
    for (uint64_t id : enc) tokens.push_back(static_cast<int32_t>(id));

    std::cout << "[prompt] " << args.prompts[0] << "\n";
    std::cout << "[encoded] " << tokens.size() << " tokens: [";
    for (size_t i = 0; i < tokens.size(); ++i)
      std::cout << (i ? "," : "") << tokens[i];
    std::cout << "]\n[generation] ";
    std::cout.flush();

    exec.reset();
    std::vector<int32_t> prefill_pos(tokens.size());
    std::iota(prefill_pos.begin(), prefill_pos.end(), 0);
    auto out = exec.run(tokens, prefill_pos);

    auto t_start = std::chrono::steady_clock::now();
    int abs_pos = static_cast<int>(tokens.size()) - 1;
    int new_tokens = 0;
    for (int step = 0; step < args.max_tokens; ++step) {
      int seq = out.shape[0], vocab = out.shape[1];
      const float *last = out.data.data() + static_cast<size_t>(seq - 1) * vocab;
      int next = argmaxRow(last, vocab);
      if (tok.isEogToken(static_cast<uint64_t>(next))) break;
      std::cout << tok.tokenString(static_cast<uint64_t>(next)) << std::flush;
      tokens.push_back(next);
      ++new_tokens; ++abs_pos;
      if (step + 1 >= args.max_tokens) break;
      out = exec.run({next}, {abs_pos});
    }
    auto t_end = std::chrono::steady_clock::now();
    std::cout << "\n";
    double secs = std::chrono::duration<double>(t_end - t_start).count();
    double tps  = new_tokens > 0 ? new_tokens / secs : 0.0;
    std::cout << "[stats] generated " << new_tokens << " tokens in " << secs
              << "s (" << tps << " tok/s, "
              << (use_custom ? "custom Q4_0 Metal kernel [MLC_Q4_CUSTOM_KERNEL=1]"
                             : "mx::quantized_matmul (group_size=32 bits=4 affine)")
              << ", KV cache, batch=1)\n";
    exec.printProfile();
    return 0;
  }

  // ── Batched path (N ≥ 2) ─────────────────────────────────────────────────
  std::cout << "[batch] " << N << " concurrent requests\n";

  // Tokenize all prompts; require equal length for V1.
  std::vector<std::vector<int32_t>> all_tokens(N);
  for (int i = 0; i < N; ++i) {
    auto enc = tok.encode(args.prompts[i],
                          mlc::runtime::TokenizerConfig{true, false});
    for (uint64_t id : enc) all_tokens[i].push_back(static_cast<int32_t>(id));
    std::cout << "[prompt " << i << "] " << args.prompts[i]
              << " (" << all_tokens[i].size() << " tokens)\n";
  }
  for (int i = 1; i < N; ++i) {
    if (all_tokens[i].size() != all_tokens[0].size()) {
      std::fprintf(stderr,
                   "[batch] V1: all prompts must have the same token length.\n"
                   "  prompt 0: %zu tokens, prompt %d: %zu tokens\n",
                   all_tokens[0].size(), i, all_tokens[i].size());
      return 1;
    }
  }

  // Prefill: ONE shared forward pass, KV cache replicated N times.
  exec.reset();
  auto prefill_outs = exec.prefillBatch(all_tokens);
  int prompt_len = static_cast<int>(all_tokens[0].size());

  // Per-request generated token sequences and next-token state.
  std::vector<std::vector<int32_t>> gen_tokens(N);
  std::vector<int32_t> next_toks(N);
  for (int i = 0; i < N; ++i) {
    const auto &out = prefill_outs[i];
    int vocab = out.shape[1];
    const float *last = out.data.data() + static_cast<size_t>(out.shape[0] - 1) * vocab;
    next_toks[i] = argmaxRow(last, vocab);
  }

  auto t_start = std::chrono::steady_clock::now();
  int step = 0;
  for (; step < args.max_tokens; ++step) {
    // positions: all requests at prompt_len + step (common in V1)
    int abs_pos = prompt_len + step;
    std::vector<int32_t> pos_vec(N, abs_pos);

    bool all_done = true;
    for (int i = 0; i < N; ++i) {
      if (!tok.isEogToken(static_cast<uint64_t>(next_toks[i])))
        all_done = false;
    }
    if (all_done) break;

    // Record current next tokens before the decode step.
    for (int i = 0; i < N; ++i) {
      if (!tok.isEogToken(static_cast<uint64_t>(next_toks[i])))
        gen_tokens[i].push_back(next_toks[i]);
    }
    if (step + 1 >= args.max_tokens) break;

    auto batch_outs = exec.runBatch(next_toks, pos_vec);
    for (int i = 0; i < N; ++i) {
      if (tok.isEogToken(static_cast<uint64_t>(next_toks[i]))) continue;
      const auto &out = batch_outs[i];
      int vocab = out.shape[1];
      next_toks[i] = argmaxRow(out.data.data(), vocab);
    }
  }
  auto t_end = std::chrono::steady_clock::now();

  // Report per-request generations.
  for (int i = 0; i < N; ++i) {
    std::cout << "[req " << i << "] ";
    for (int32_t tid : gen_tokens[i])
      std::cout << tok.tokenString(static_cast<uint64_t>(tid));
    std::cout << "\n";
  }

  int total_tokens = 0;
  for (auto &g : gen_tokens) total_tokens += static_cast<int>(g.size());
  double secs = std::chrono::duration<double>(t_end - t_start).count();
  double agg_tps = secs > 0 ? total_tokens / secs : 0.0;
  double per_req_tps = agg_tps / N;
  std::cout << "[stats] batch=" << N
            << " total_tokens=" << total_tokens
            << " wall=" << secs << "s"
            << " aggregate=" << agg_tps << " tok/s"
            << " per_req=" << per_req_tps << " tok/s"
            << " (mx::quantized_matmul, KV cache)\n";

  exec.printProfile();
  return 0;
}
