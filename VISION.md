# MLcompiler — Project Vision

## What this project is

An MLIR-based compiler for LLM inference on Apple Silicon. The compiler takes a model graph, runs optimization passes (op fusion, tiling, batch-aware scheduling), and lowers to MLX for execution. Written in C++17.

The project has two phases:
- Phase 1 (complete): from-scratch runtime proving we understand the full inference stack. 1.9 → 54 tok/s single-stream, 97 tok/s batched aggregate. Paged KV, continuous batching, custom Metal kernels, three-way parity harness.
- Phase 2 (current): MLIR compiler that optimizes the model graph before lowering to MLX. 1.02 → 71 tok/s in five sessions. This is the active development direction.

## The moat

The moat is the MLIR compilation pipeline. Specifically: cross-op fusion passes that reduce dispatch count and improve data locality in ways that calling MLX directly cannot achieve.

Nobody else has this on Apple Silicon:
- vllm-mlx: Python serving framework calling MLX ops directly. No compiler, no cross-op fusion.
- Ollama: wrapper around MLX/llama.cpp. No compiler.
- llama.cpp: hand-written Metal shaders. No compiler, no MLX integration.
- mlx-lm: direct MLX inference. No compiler.

Our compiler sees the full graph and can fuse adjacent ops (QKV projection batching, FFN gate+up batching, norm+matmul), eliminate intermediate buffers, and lower to optimal MLX dispatch sequences. This is a structural advantage that compounds with model complexity.

## What we build vs what we import

### WE IMPORT (do not rebuild):
- **mx::quantized_matmul** for all matmul ops. This is Apple's optimized fused dequant+matmul kernel. Do not write custom Q4_0/Q6_K Metal kernels for the compiler path. Convert GGUF weights to MLX's native quantized format at model load time.
- **mx::fast::scaled_dot_product_attention** for attention. Use MLX's SDPA, not a custom attention kernel, for the compiler path.
- **mx::fast::rms_norm** for normalization.
- **Scheduler design from vllm-mlx**. Their continuous batching scheduler is adapted from vLLM's production scheduler (preemption, prefix caching, priority ordering). Adopt their design rather than extending our FIFO scheduler.
- **MLX's lazy evaluation** for adjacent-op fusion. MLX already fuses norm+matmul etc. within its graph. Don't duplicate this in MLIR passes.

### WE BUILD (our unique contribution):
- **MLIR dialect and IR emission.** Model graph represented as MLIR that optimization passes can transform.
- **Cross-op fusion passes.** Transforms MLX can't discover:
  - QKV projection batching: three matmuls sharing one input → one batched matmul + split. Saves 2 kernel launches per layer.
  - FFN gate+up batching: two matmuls sharing one input → one batched matmul + split. Saves 1 kernel launch per layer.
  - Multi-output kernels where mx::slice overhead exceeds launch savings.
  - Future: attention+output projection fusion, full FFN fusion.
- **Batch-aware tiling passes.** When batch_size > 1, tile across both batch and output dimensions jointly for optimal GPU occupancy.
- **Device scheduling pass.** Annotate ops with target device. Currently GPU-only (ANE via CoreML was falsified — per-op dispatch overhead doesn't amortize). Future: direct ANE via private APIs if the overhead problem can be bypassed.
- **The C++ execution engine.** Lower latency than Python for agentic workloads where per-token overhead matters.

## What we do NOT build:
- Custom matmul Metal kernels for the compiler path. Use mx::quantized_matmul.
- Custom attention Metal kernels for the compiler path. Use mx::fast::scaled_dot_product_attention.
- A from-scratch scheduler. Adopt vllm-mlx's design.
- CoreML/ANE integration. Falsified (U1). Per-matmul CoreML overhead is ~250 µs regardless of subgraph packing. Revisit only via direct _ANEClient APIs, not CoreML.

## Current state (as of last session)

Compiler path: 71 tok/s on TinyLlama 1.1B Q4_0, M3 Pro.
- MLIR dialect: 7 ops, IR emission from GGUF, round-trip through mlc-opt
- Fusion passes: QKV batching (+13%), FFN gate+up batching, multi-output Q4_0 kernel
- Execution: custom Q4_0 Metal kernel via mx::fast::metal_kernel, in-place KV cache, MLX SDPA
- ANE: investigated and falsified via CoreML (T2, U1). Code retained, off by default.

## Immediate next steps (priority order)

1. **Switch matmul to mx::quantized_matmul.** Convert GGUF Q4_0 weights to MLX native quantized format at load time. Replace custom Q4_0 kernel with mx::quantized_matmul calls. This should close the gap to vllm-mlx on per-op matmul speed (71 → potentially 100+ tok/s).

2. **Continuous batching through the compiler path.** Add batched execution to the MLIR pipeline. The MLIR fusion passes (QKV batching, FFN gate+up) should reduce dispatch count per forward pass, improving batched throughput over vanilla MLX.

3. **Adopt vllm-mlx scheduler design.** Replace FIFO scheduler with preemption-capable, prefix-caching scheduler based on vllm-mlx's approach.

4. **Benchmark against vllm-mlx directly.** Same model, same hardware, same workload. Demonstrate that MLIR fusion passes produce measurably better throughput than calling MLX directly.

5. **Direct ANE investigation (future).** Bypass CoreML via _ANEClient/_ANECompiler private APIs. Only if the dispatch overhead problem can be eliminated at the API level.

## Validation rules

- Compiler path output must match runtime path at cosine >= 0.999 per layer
- Greedy decode must produce coherent output
- Batched path per-request output must match single-stream at cosine >= 0.999
- Phase 1 runtime (DecodeRunner, chat-repl, parity harness) must remain untouched and passing

## Key empirical findings (do not re-litigate)

1. **ANE via CoreML doesn't work for inference.** Per-matmul CoreML overhead is ~250 µs regardless of subgraph packing. The S4 "5x ANE win" was an apples-to-oranges benchmark (fresh weights vs resident weights). Falsified in U1.
2. **Metal memory coherence breaks at scale.** Single-encoder forward pass with barriers works at small scale (10 dispatches) but produces garbage at walker scale (260 dispatches, 20 reused buffers). Diagnosed in N1, documented in tools/metal_hazard_test.mm.
3. **MLX lazy evaluation already fuses adjacent ops.** MLIR norm+matmul fusion added zero wall-clock improvement because MLX was already doing it internally. Only cross-op fusions (shared-input batching) move the clock.
4. **Custom Q4_0 kernel < MLX quantized_matmul.** Our hand-written kernel is slower than Apple's optimized path. Use theirs.
