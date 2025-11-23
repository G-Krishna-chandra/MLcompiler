# ML Compiler

A modern ML compiler project written in C++17. It ingests GGUF models, lifts
them into a structured IR, runs layout/tiling/fusion passes, schedules an
execution graph across CPU/Metal backends, and can already execute simple
graphs end-to-end.

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

Use `--execute` to run the compiled execution graph for a single token. The CPU
backend currently powers execution (Metal routes through CPU until we add real
GPU kernels).

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
   `ExecutionContext`, loads tensors, and dispatches CPU/Metal kernels.
   Embedding + matmul/linear kernels already run end-to-end on Apple GPUs with
   CPU fallbacks; attention/FFN/softmax/add/norm kernels also have GPU coverage.

## Apple Silicon Optimizations

- `Session::runLinear` switches to Accelerate BLAS (`cblas_sgemv`) for large F32
  weights and uses direct dot-product helpers for every GGUF quant format
  (Q4/Q5/Q6/Q8, K-series included), minimizing CPU dequant work.
- RMSNorm, softmax, and residual adds execute via vDSP on CPU, and Metal kernels
  now cover embeddings, attention, FFN, add, norm, softmax, rotary position
  updates, and all quantized matmuls; LM-head matmuls fuse their bias adds on
  Metal so logits stay on the GPU. Remaining ops (logit post-processing) fall
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

## Future Integration

The project is structured to easily integrate:
- **MLIR**: Uncomment MLIR find_package in main CMakeLists.txt
- **LLVM**: Uncomment LLVM find_package in main CMakeLists.txt

## License

[Add your license here]

