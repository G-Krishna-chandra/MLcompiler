# ML Compiler

A modern ML compiler project written in C++17. It ingests GGUF models, lifts
them into a structured IR, runs layout/tiling/fusion passes, schedules an
execution graph across CPU and Metal backends, and runs Llama-class quantized
LLMs token-by-token on Apple Silicon.

## Status

Metal backend is end-to-end functional on Q4_0 quantized models. Numerical
parity against the CPU backend is verified by the in-tree parity harness
(`mlc compare --metal-vs-cpu`): on TinyLlama 1.1B every layer-boundary tensor
matches between CPU and Metal at cosine ≥ 0.999999 (max abs error in the
1e-5 range — float-rounding noise) and the final logits agree on top-1,
top-5, and top-10 token rankings.

The runtime does not yet match llama.cpp byte-for-byte at the model-output
level. A small-magnitude numerical drift between mlc CPU and llama.cpp causes
the greedy first generated token to differ on some prompts. That gap is
orthogonal to the CPU↔Metal parity above and is the next milestone; the
specific Metal kernel families that still need the same split-half fix
that landed for Q4_0 are listed under "Known issues" below.

## Project Structure

```
project-root/
  CMakeLists.txt          # Main CMake configuration
  /compiler               # Compiler source code
    /frontends           # Frontend parsers and analyzers
    /ir                  # Intermediate representation
    /passes              # Compiler passes
    /codegen             # Code generation backends
      /cpu               # CPU code generation
      /metal             # Metal (GPU) code generation
    /runtime             # Runtime system
  /third_party           # Third-party dependencies
  /tests                 # Test suite
  /python                # Python bindings (future)
```

## Building

### Prerequisites

- CMake 3.15 or higher
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- Make or Ninja build system

### Build Instructions

```bash
mkdir build
cd build
cmake ..
make -j
```

### Build Options

- `BUILD_TESTS`: Enable/disable tests (default: ON)
  ```bash
  cmake -DBUILD_TESTS=OFF ..
  ```

### Running Tests

After building, run tests with:

```bash
cd build
ctest
```

Or run the test executable directly:

```bash
./bin/mlc_tests
```

## Using the CLI

Place `.gguf` models anywhere inside the repository (e.g. `models/`). The `mlc`
CLI can introspect, simulate, and execute them.

### Inspect the Model / Plan

```bash
# Inspect metadata and tensor previews
./build/bin/mlc inspect models/tinyllama.gguf --dump-tensors

# Build the execution plan and simulate scheduling
./build/bin/mlc run --simulate models/tinyllama.gguf 42
```

Simulation prints the execution order and verifies that tensor dependencies are
met before running any kernels.

### Execute the Graph

Use `--execute` to run the compiled execution graph for a single token. By
default Metal is used for embedding, attention, FFN, RMSNorm, softmax, RoPE,
fused bias-add, and the Q4_0 / Q6_K quantized matmuls that TinyLlama needs;
the executor falls back to the CPU backend automatically when Metal is
unavailable or for ops that don't yet have a Metal kernel. Setting
`MLC_FORCE_CPU=1` pins every kernel to CPU for parity comparisons.

```bash
./build/bin/mlc run --execute models/tinyllama.gguf 42
```

The CLI prints embedding previews, dry-run logits, and finally the logits
produced by the real execution path. Combine `--simulate` and `--execute` to
see both traces and results in one invocation.

### Useful Flags

| Flag | Description |
| ---- | ----------- |
| `--preview=N` | Limit float preview length (default 16) |
| `--no-logits` | Skip dry-run logits computation |
| `--simulate` | Dump execution plan traces |
| `--simulate-limit=N` | Stop simulation after `N` nodes |
| `--execute` | Run the actual execution graph and print logits |
| `--position=N` | Set the KV-cache position / decode step (default 0) |
| `--verbose` | Dump the full execution graph when building plans |

### Compare CPU vs Metal (parity harness)

`mlc compare` runs the same prompt through two execution paths in-process and
reports per-tensor numerical divergence at every layer-boundary tensor —
embedding output, per-block attn_output / residual_1 / ffn_down / residual_2,
the final RMSNorm output, and the final logits.

```bash
./build/bin/mlc compare --metal-vs-cpu models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  --prompt "The capital of France is" \
  --csv-out logs/compare.csv
```

Each row reports max abs diff, mean abs diff, RMS, and cosine similarity;
the logits row also reports top-1 / top-5 / top-10 overlap. Useful for
catching regressions when adding or modifying Metal kernels.

A second mode, `mlc compare --vs-llamacpp ... --reference-dir DIR`, consumes
pre-dumped llama.cpp tensor values (one `<sanitized_name>.f32.bin` per
boundary tensor) and reports mlc-CPU vs llama.cpp divergence. The reference
dump format is documented inline in `mlc compare --help`.

## Architecture Overview

1. **GGUF Loader** (`compiler/frontends/gguf_loader.*`): Parses GGUF v1–v3 files
   plus KV metadata and tensor directories.
2. **IR Builder** (`compiler/ir/ir_builder.*`): Converts GGUF tensors + metadata
   into a high-level operator graph (embedding → per-layer attention/FFN →
   logits).
3. **Passes** (`compiler/passes/*`): Apply layout normalization, matmul fusion,
   and tiling hints to the IR.
4. **Kernel Scheduler** (`compiler/runtime/kernel_scheduler.*`): Lowers the IR
   into an execution graph with backend hints.
5. **Executor** (`compiler/runtime/execution_*`): Creates an
   `ExecutionContext`, loads tensors, and dispatches CPU or Metal kernels.
   Metal kernels cover embedding, attention (with GQA + sliding-window),
   FFN, RMSNorm / LayerNorm, softmax, RoPE, fused bias-add, KV-cache
   scatter, and the Q4_0 / Q6_K quantized matmuls. The CPU backend is the
   reference path and the automatic fallback. Numerical parity between
   the two backends is asserted by the parity harness on every layer
   boundary (see "Compare CPU vs Metal" above).

## Apple Silicon Optimizations

- `Session::runLinear` switches to Accelerate BLAS (`cblas_sgemv`) for large F32
  weights and uses direct dot-product helpers for every GGUF quant format
  (Q4/Q5/Q6/Q8, K-series included), minimizing CPU dequant work.
- RMSNorm, softmax, and residual adds execute via vDSP on CPU, and Metal kernels
  now cover embeddings, attention, FFN, add, norm, softmax, rotary position
  updates, and quantized matmuls; LM-head matmuls fuse their bias adds on
  Metal so logits stay on the GPU. Q4_0 and Q6_K matmul kernels are
  parity-tested against the CPU reference; Q4_1 / Q5_0 / Q5_1 are known
  broken (see Known issues). Remaining ops (logit post-processing) fall
  back to CPU until their GPU path lands.
- KV caches live in shared Metal buffers so attention reads/writes the entire
  prefix without re-uploading per head; multi-token batches reuse the same
  resident cache slices while staying coherent with the CPU mirrors.
- Incoming KV tokens are written directly on the GPU via scatter kernels, so
  cache updates stay in GPU memory even when decoding multiple tokens per step.
- The runtime exposes reusable KV-buffer APIs (`ensureSharedBuffer`,
  `scatterKVCache`) so other passes can batch cache updates without duplicating
  attention-specific logic.
- CLI exposes `--position` so you can step through tokens while reusing the
  KV-cache, mirroring how real decoders drive Apple’s unified memory subsystem.
- When nodes target the Metal backend, the executor runs them on the Apple GPU
  using MPS + custom compute shaders, falling back to Accelerate automatically
  when Metal isn’t available.

## Known issues

- **mlc CPU output diverges from llama.cpp on greedy decode.** Small per-layer
  numerical drift (RMSNorm epsilon / RoPE / Q4_0 dequant rounding) accumulates
  across the 22 transformer blocks of TinyLlama and flips the top-1 logit on
  some prompts (e.g. greedy completion of "The capital of France is" picks "a"
  instead of " Paris"). The drift is bounded; mlc CPU and llama.cpp logits
  remain highly correlated. Under investigation.
- **Q4_1 / Q5_0 / Q5_1 Metal kernels have a known split-half indexing bug.**
  Same pattern as the Q4_0 bug fixed in commit d9720c7: the kernel walks
  `col_index++` against an assumed interleaved layout, but Q-quant storage is
  split-half (lo nibbles fill positions 0..15, hi nibbles fill 16..31). Test
  cases `MetalRuntimeTest.MatMulQ4_1MatchesCPUWhenAvailable` and
  `MetalRuntimeTest.MatMulQ5MatchesCPUWhenAvailable` document the failure and
  are currently red. TinyLlama Q4_0 does not exercise these kernels.
- **Internal BPE fallback in the tokenizer is incomplete.** Production paths
  go through the optional llama.cpp tokenizer (linked when
  `MLC_ENABLE_LLAMA_TOKENIZER=1`); the fallback BPE merge logic in
  `compiler/runtime/tokenizer.cpp` does not always merge correctly. Flagged
  by `TokenizerTest.EncodesAndDecodesText`.

## Future Integration

The project is structured to easily integrate:
- **MLIR**: Uncomment MLIR find_package in main CMakeLists.txt
- **LLVM**: Uncomment LLVM find_package in main CMakeLists.txt

## License



