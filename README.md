# MLcompiler

An MLIR-based LLM inference compiler for Apple Silicon, built from scratch in C++17.

## What this is

A two-phase project. Built by one person as a deep dive into LLM inference on Apple Silicon.

- **Phase 1** — a from-scratch inference runtime: GGUF parser, tokenizer, custom Metal kernels (Q4_0/Q6_K mat-vec, RMSNorm, paged flash attention), paged KV cache, continuous batching scheduler, and a three-way parity harness against llama.cpp. The goal was to understand the full stack end-to-end before reaching for any library.
- **Phase 2** — an MLIR compiler that emits a model graph from GGUF, runs optimization passes (QKV batching, FFN gate+up batching), and lowers to MLX for execution. Achieves **185 tok/s aggregate at batch=32** on TinyLlama 1.1B Q4_0 on M3 Pro — 28% faster than Ollama's 144 tok/s single-stream ceiling.

Both phases ship as binaries; the Phase-1 runtime remains untouched in Phase-2 sessions so prior parity guarantees hold.

## Results

**Phase 1 — Single-stream throughput trajectory** (TinyLlama 1.1B Q4_0, M3 Pro, `mlc chat-repl`):

| Milestone                       | tok/s | What changed                               |
|---------------------------------|-------|--------------------------------------------|
| First working inference         | 1.9   | Naive CPU + Metal embeddings only          |
| CB batching + fp16 KV           | 21.0  | Dispatch overhead reduction, fp16 KV cache |
| RMSNorm kernel rewrite          | 30.0  | `simd_sum` parallel reduction              |
| Q4_0 matmul rewrite             | 40.0  | Simdgroup mat-vec, blocked nibble decode   |
| Final sweep                     | 54    | Multi-row kernel, persistent buffers       |

**Phase 2 — Compiler path batched scaling** (same model, M3 Pro, `mlc-compile-run`, 32 tok/req):

| Batch | Aggregate tok/s | Per-request tok/s | vs Ollama (144) |
|-------|-----------------|-------------------|-----------------|
| 1     | 78              | 78                | 0.54×           |
| 4     | 129             | 32.3              | 0.90×           |
| 8     | 141             | 17.6              | 0.98×           |
| 16    | 160             | 10.0              | **1.11×**       |
| 32    | 185             | 5.8               | **1.28×**       |
| 64    | 187             | 2.9               | 1.30× (plateau) |

The GPU is saturated at batch=32. No OOM up to batch=64 (6.3 GB of 18 GB unified memory).

## Architecture

A short summary. See [docs/architecture.md](docs/architecture.md) for the deep dive.

**Phase 1 — Runtime (`compiler/runtime/`, `compiler/main.cpp`)**
- Custom Metal kernels: Q4_0/Q6_K mat-vec with simdgroup `simd_sum` reductions; RMSNorm with simdgroup variance reduction; paged flash attention with online softmax and runtime-scaled threadgroup tile sizing.
- Paged KV cache: 64-token pages, fp16 storage, GPU scatter/gather (`scatter_kv_paged_batched_f16`), per-request page tables.
- Continuous batching: iteration-level scheduler with FIFO admission; batched per-op walker dispatching each kernel once across all N requests; single command buffer per forward pass with persistent GPU-resident intermediate buffers.
- Three-way parity harness: handwritten reference / executor-CPU / executor-Metal compared per-layer against llama.cpp dumps; cosine ≥ 0.9999 enforced.

**Phase 2 — Compiler (`compiler/mlir/`)**
- MLIR dialect (`compiler/mlir/dialect/`): 7 ops covering a transformer forward pass — `mlc.embedding`, `mlc.norm`, `mlc.matmul`, `mlc.feedforward`, `mlc.attention`, `mlc.lm_head`, plus fused variants `mlc.fused_norm_matmul` and `mlc.fused_norm_qkv_matmul`.
- IR emission from GGUF (`compiler/mlir/emit/GGUFToMLIR.cpp`): one MLIR function per model, weight tensors as func args carrying `mlc.name` attributes.
- Fusion passes (`compiler/mlir/passes/`): `FuseNormMatMul` (norm + matmul) and `FuseQKVMatMul` (three QKV matmuls sharing a norm) — net 1–3% at batch≥2.
- Execution (`compiler/mlir/exec/MLIRExecutor.cpp`): an IR walker that lowers each op to MLX — `mx::quantized_matmul`, `mx::fast::scaled_dot_product_attention`, `mx::fast::rms_norm`. GGUF Q4_0 weights are converted bit-exact to MLX's native packed format at load time (`MLXQuantize.cpp`).
- Batched execution: `prefillBatch` runs one shared prefill and replicates the KV cache N times; `runBatch` does an [N, hidden] forward pass with per-request attention.
- Optional fused multi-op Metal kernels (`FusedKernels.cpp`, gated by `MLC_FUSED_KERNELS=1`): correct, but did not improve throughput at this scale — kept as infrastructure.

## What we learned

The most useful part of this project, in roughly the order things were learned:

1. **MLX's lazy evaluation already fuses adjacent ops.** Our MLIR norm+matmul fusion pass added 0% improvement. The cross-op fusions that actually moved a number (QKV projection batching, FFN gate+up batching) added 1–3% at batch≥2 and regressed −2% at batch=1. The compiler's value at this scale is in **batched execution orchestration**, not in adjacent-op fusion that MLX discovers for free.

2. **ANE via CoreML doesn't work for inference.** Per-matmul CoreML overhead is ~250 µs regardless of how much you pack into a subgraph. Measured, falsified, documented (see U1 in [docs/lessons.md](docs/lessons.md)). The ANE hardware is fast on paper; the framework overhead kills it for sub-millisecond ops.

3. **Metal memory coherence breaks at scale.** A single-encoder forward pass with `memoryBarrierWithScope:Buffers` works correctly at 10 dispatches across 4 buffers, but produces garbage outputs at 260 dispatches across 20 reused buffers. The minimal reproducer is in `tools/metal_hazard_test.mm`. Not documented elsewhere that we've found.

4. **85% of decode-step *measured* time is dispatch overhead — but only when you measure wrong.** Forced-eval profiling (`mx::eval()` after every op) reports 110 dispatches × ~100 µs = 11 ms of "dispatch overhead." That 100 µs is the **CPU-GPU sync cost from each forced eval**, not the GPU command processor cost. In lazy-eval production, MLX submits all ops in one command buffer and the GPU pipelines them — per-command overhead is closer to 1–5 µs. We confirmed this by writing fused multi-op kernels that reduced dispatches from 110 to 22 and saw **−1.8% throughput**: the fused kernels also did more compute per dispatch and weren't as tuned as Apple's separate kernels.

5. **Custom kernels match but don't beat MLX at decode shapes.** Our hand-written Q4_0 mat-vec kernel was within 0.5% of `mx::quantized_matmul` end-to-end (78.2 vs 78.5 tok/s at batch=1). Per-shape micro-benchmarks show the custom kernel wins at M=1 by 1.1–1.5× on individual ops, but the win vanishes in production because dispatch overhead dominates these small ops anyway. Use Apple's path as the default.

6. **Batching is the real lever on Apple Silicon.** Single-stream is bottlenecked at ~78 tok/s on TinyLlama (MLX's ceiling for this matmul shape). Batched execution reaches 185 tok/s at batch=32 by widening matmuls from [1, 2048] × [K, 2048] to [32, 2048] × [K, 2048], where MLX's tiled `quantized_matmul` is highly efficient. The compiler advantage is in orchestrating the batch — converting weights once at load time, replicating KV cache via deep copy, and dispatching per-op kernels once across all requests.

## Build and run

### Prerequisites

- macOS, Apple Silicon (M-series). Metal is the primary backend.
- CMake 3.15+, Xcode 14+ command-line tools.
- MLX installed via Homebrew: `brew install mlx` (provides `libmlx.dylib`).
- Optional: llama.cpp as a sibling directory at `../llama.cpp` (built with `LLAMA_BUILD_SHARED=ON`) for the BPE tokenizer and `mlc compare --vs-llamacpp` reference dumps.

### Build

```bash
git clone https://github.com/G-Krishna-chandra/MLcompiler.git
cd MLcompiler
git submodule update --init --recursive
mkdir -p build && cd build
cmake ..
make -j8
```

Produces `build/bin/mlc` (Phase-1 runtime CLI), `build/bin/mlc-compile-run` (Phase-2 driver), `build/bin/mlc-emit` / `mlc-opt` (MLIR tools), `build/bin/mlc-kernel-bench`, and `build/bin/mlc_tests` / `mlir_exec_tests` (gtest harnesses).

### Get a model

```bash
mkdir -p models
# TinyLlama 1.1B Q4_0 (~600 MB):
curl -L -o models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
    https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf
```

### Run

```bash
# --- Phase 1: from-scratch runtime ---

# Interactive chat (BPE tokenizer needs ../llama.cpp built)
./build/bin/mlc chat-repl models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf

# Continuous-batching scaling benchmark
./build/bin/mlc serve models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
    --benchmark --benchmark-batches 1,2,4,8 --max-tokens 32

# Per-layer parity vs CPU reference
./build/bin/mlc compare --metal-vs-cpu \
    models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
    --prompt "The capital of France is"

# --- Phase 2: compiler path ---

# Single-stream greedy decode
./build/bin/mlc-compile-run models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
    --prompt "The capital of France is" --max-tokens 64

# Batched: pass --prompt N times (V1 requires equal-length prompts)
./build/bin/mlc-compile-run models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
    --prompt "The capital of France is" \
    --prompt "The capital of France is" \
    --prompt "The capital of France is" \
    --prompt "The capital of France is" \
    --prompt "The capital of France is" \
    --prompt "The capital of France is" \
    --prompt "The capital of France is" \
    --prompt "The capital of France is" \
    --max-tokens 32

# Walk the scaling curve (1, 2, 4, 8, 16, 32)
PROMPT="The capital of France is"
for N in 1 2 4 8 16 32; do
  ARGS=""
  for i in $(seq 1 $N); do ARGS="$ARGS --prompt \"$PROMPT\""; done
  eval ./build/bin/mlc-compile-run \
      models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf $ARGS --max-tokens 32 \
      2>/dev/null | grep stats
done

# Per-op profile (forces mx::eval per category)
MLC_COMPILER_PROFILE=1 ./build/bin/mlc-compile-run \
    models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
    --prompt "The capital of France is" --max-tokens 64

# Per-shape custom-kernel vs MLX micro-benchmark
./build/bin/mlc-kernel-bench models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf

# --- MLIR tooling ---
./build/bin/mlc-emit models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf > model.mlir
./build/bin/mlc-opt --mlc-fuse-norm-matmul --mlc-fuse-qkv-matmul < model.mlir
```

### Environment variables

| Variable                  | Effect                                                                   |
|---------------------------|--------------------------------------------------------------------------|
| `MLC_FUSE_LAYER=1`        | Phase-1: enable layer-window fusion in the runtime walker                |
| `MLC_PROFILE_NODES=1`     | Phase-1: per-op timing trace                                              |
| `MLC_COMPILER_PROFILE=1`  | Phase-2: forced-eval per-op-category timing table                         |
| `MLC_FUSED_KERNELS=1`     | Phase-2: use fused multi-op Metal kernels (correct, no speedup at scale) |
| `MLC_Q4_CUSTOM_KERNEL=1`  | Phase-2: fall back to the hand-written Q4_0 kernel instead of MLX        |
| `MLC_FORCE_CPU=1`         | Phase-1: pin everything to CPU (debugging)                                |

## Project structure

```
compiler/
  frontends/       GGUF parser, ggml type tables
  runtime/         Phase-1 runtime: walker, scheduler, paged KV, Metal kernels (.mm)
  mlir/
    dialect/       MLIR dialect (7 ops + 2 fused variants)
    emit/          GGUF → MLIR emission
    passes/        FuseNormMatMul, FuseQKVMatMul, ScheduleDevices
    exec/          MLIRExecutor walker, MLXQuantize, FusedKernels, Q4MatMul
    tools/         mlc-opt, mlc-emit, mlc-compile-run, mlc-kernel-bench
  main.cpp         Phase-1 CLI dispatch

tools/             Phase-1 diagnostics: metal_hazard_test, paged_flash_bench,
                   llamacpp_dump_activations, mlc_dump_kv_cache
tests/             gtest unit + integration tests
docs/              architecture and lessons-learned writeups
third_party/
  httplib.h        vendored cpp-httplib (HTTP server for mlc api)
  llama.cpp/       optional submodule for tokenizer + reference dumps
```

## Docs

- [docs/architecture.md](docs/architecture.md) — deep technical writeup of both phases: forward-pass pipeline, memory management, kernel design choices, the Metal coherence finding.
- [docs/lessons.md](docs/lessons.md) — engineering log of what worked and what didn't, arc-by-arc.
- [docs/compiler-path-v5.md](docs/compiler-path-v5.md) — compiler optimization history from emission through fusion + execution.
- [ROADMAP.md](ROADMAP.md) — the perf sequence that drove the Phase-1 numbers.

## License

See LICENSE.
