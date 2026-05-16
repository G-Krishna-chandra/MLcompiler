# MLcompiler

A from-scratch C++17 LLM inference runtime for Apple Silicon. Not a
wrapper around llama.cpp. Every component — GGUF parser, tokenizer,
Metal compute kernels, paged KV cache, continuous-batching scheduler,
parity harness — is implemented from scratch to understand and
demonstrate the full inference stack.

## What it does

- **Single-stream inference** on TinyLlama 1.1B Q4_0: 50–54 tok/s on
  M3 Pro (`mlc chat-repl`, fuse-mode).
- **Continuous batching**: 97 tok/s aggregate at batch=8 via paged-KV
  attention (`mlc serve --paged --benchmark`).
- **Multi-model support**: Llama-architecture GGUF (TinyLlama 1.1B,
  Llama 3.2 3B verified end-to-end).
- **Three-way parity validation** at every layer boundary against
  llama.cpp.

## Architecture highlights

- **Custom Metal kernels**: Q4_0 / Q6_K mat-vec with simdgroup
  parallelism (`simd_sum` reductions, 32 threads/threadgroup,
  unroll-by-4 row blocks); RMSNorm with simdgroup variance reduction;
  paged flash-attention with online softmax and runtime-scaled
  threadgroup tile sizing (handles head_dim 64 → 128 within the
  M3 Pro's 32 KB threadgroup memory budget).
- **Paged KV cache**: fixed-size 64-token pages, GPU scatter/gather
  (`scatter_kv_paged_batched_f16`), per-layer fp16 page storage,
  request-level page tables.
- **Continuous batching**: iteration-level scheduler with FIFO
  admission and per-request streaming; op-by-op batched walker that
  dispatches every per-op kernel once across all N requests'
  inputs; single command buffer per forward pass with persistent
  GPU-resident intermediate buffers.
- **Parity harness**: three-way comparison (handwritten CPU
  reference / executor CPU / executor Metal) at every layer
  boundary; cosine similarity asserted ≥ 0.9999 vs llama.cpp;
  CSV output for offline analysis.

## Performance trajectory

Single-stream tok/s on TinyLlama 1.1B Q4_0 (M3 Pro):

```
1.9   → baseline (CPU-only naive)
3.7   → Metal embedding + Q4_0 matmul
10.7  → fp16 KV cache + MPS attention
19.1  → fp16 attention path
26.5  → batched prefill + Q4_0 mat-vec rewrite
37.4  → fuse-layer pass (deferred CB)
54.0  → single CB per forward pass + persistent buffers
```

Continuous batching scaling (TinyLlama, paged KV):

```
1.10× → G arc (per-layer paged attention, per-request walker)
3.07× → I arc (batched per-op kernels)
8.31× → J arc (single CB + persistent buffers + tiled paged-flash)
```

Aggregate tok/s today (`mlc serve --paged --benchmark`):

```
batch  1  →  45  tok/s
batch  2  →  72  tok/s
batch  4  →  86  tok/s
batch  8  →  97  tok/s
batch 16  → 101  tok/s   (M3 Pro compute ceiling)
```

## Honest limitations

- **Single-stream is ~2× slower than llama.cpp** on TinyLlama
  (~54 tok/s vs ~100–150 tok/s on the same chip). The gap is
  per-kernel maturity, not architecture: llama.cpp's Q4_0
  mat-vec has years of micro-optimization.
- **3B-class models lose batched scaling** to ollama by ~8× at
  batch=4 (6.0 tok/s vs 48 tok/s) — per-layer command-buffer
  sync overhead dominates the larger model's per-pass time.
  Documented Metal-coherence finding in
  [docs/architecture.md](docs/architecture.md) and
  [logs/single-encoder-diagnosis.md](logs/single-encoder-diagnosis.md).
- Only Q4_0 and Q6_K quantization formats are production-ready;
  Q4_1 / Q5_0 / Q5_1 Metal kernels have a known split-half
  indexing bug (tests `MatMulQ4_1MatchesCPU…`, `MatMulQ5…` red).
- Only Llama-architecture models (Llama 1/2/3, TinyLlama). No
  Mistral/Mixtral/Gemma kernel support yet.

## Build and run

### Prerequisites

- macOS on Apple Silicon (M-series). The Metal backend is the
  primary target.
- CMake 3.15+, Apple Clang from Xcode 14+.
- (Optional but recommended) `llama.cpp` checked out as a sibling
  directory `../llama.cpp` and built with
  `LLAMA_BUILD_SHARED=ON` — provides the BPE tokenizer for chat
  and the reference dumps for `mlc compare --vs-llamacpp`.

### Build

```bash
git clone https://github.com/G-Krishna-chandra/MLcompiler
cd MLcompiler
mkdir build && cd build
cmake ..
make -j8
```

Produces `build/bin/mlc` (CLI) and `build/bin/mlc_tests` (gtest harness).

### Get a model

```bash
mkdir -p models
# TinyLlama 1.1B Q4_0 (~600 MB):
curl -L -o models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
    https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf

# Llama 3.2 3B Q4_0 (~1.9 GB):
curl -L -o models/Llama-3.2-3B.Q4_0.gguf \
    https://huggingface.co/QuantFactory/Llama-3.2-3B-GGUF/resolve/main/Llama-3.2-3B.Q4_0.gguf
```

### Run

```bash
# Single-stream chat (fuse-mode for best perf)
MLC_FUSE_LAYER=1 ./build/bin/mlc chat-repl models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf

# Batched scaling sweep
./build/bin/mlc serve models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
    --benchmark --benchmark-batches 1,2,4,8 --max-tokens 32

# Parity vs CPU reference (catches kernel regressions)
./build/bin/mlc compare --metal-vs-cpu models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
    --prompt "The capital of France is"

# HTTP API for the wedge demo
./build/bin/mlc api models/Llama-3.2-3B.Q4_0.gguf --port 8080 \
    --static-dir tools/demo
# then open http://localhost:8080 in a browser
```

## Project structure

```
compiler/
  frontends/       GGUF parser, tensor metadata, type system
  ir/              graph-based intermediate representation
  passes/          IR transformations (layout, fusion, tiling, arch hints)
  pipeline/        IR pipeline that drives passes
  codegen/         CPU + Metal code generation
  runtime/         the runtime: session, executor, walker, scheduler,
                   paged KV, parity harness, Metal kernels (in .mm)
  main.cpp         CLI dispatch (chat, chat-repl, serve, api, compare,
                   decode, inspect, plan, ...)
tools/
  demo/            web demo: side-by-side race UI vs ollama
  metal_hazard_test.mm   Metal sync-primitive reproducer
  paged_flash_bench.cpp  standalone paged-flash perf bench
  metal_diag.mm          Metal capability check
  llamacpp_dump_activations.cpp  reference dumps for --vs-llamacpp parity
  mlc_dump_kv_cache.cpp  per-step Q/K/V offline-analysis dumper
tests/             gtest unit + integration tests
docs/              architecture and lessons-learned writeups
third_party/
  httplib.h        vendored cpp-httplib (HTTP server for mlc api)
  llama.cpp/       optional submodule for tokenizer + reference dumps
```

## Documentation

- [docs/architecture.md](docs/architecture.md) — deeper technical
  writeup: forward-pass pipeline, memory management, kernel design
  choices, the Metal coherence finding.
- [docs/lessons.md](docs/lessons.md) — what worked, what didn't,
  and what each arc taught about Apple-Silicon LLM inference.
- [ROADMAP.md](ROADMAP.md) — the perf sequence that led to today's state.

## License

See LICENSE.
