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

Compiler path: 77 tok/s single-stream / 143 tok/s batch=8, TinyLlama Q4_0, M3 Pro.
- MLIR dialect: 7 ops, IR emission from GGUF, round-trip through mlc-opt
- Fusion passes: QKV batching, FFN gate+up batching (marginal advantage at batch≥2)
- Execution: mx::quantized_matmul (group_size=32, bits=4, affine), MLX SDPA, KV cache
- Batching: prefillBatch + runBatch, N concurrent requests; 143 tok/s at batch=8
- Driver: `mlc-compile-run --prompt A --prompt B ... --max-tokens N`
- Profiling: `MLC_COMPILER_PROFILE=1` per-op timing table
- `MLC_Q4_CUSTOM_KERNEL=1` falls back to original custom kernel (debugging)

## Immediate next steps (priority order)

1. ~~**Switch matmul to mx::quantized_matmul.**~~ DONE (U6). Speed is 66 tok/s
   (comparable to the old custom kernel). Generation matches Python/MLX reference.
   Both quantized paths produce "Paris, the city of love" (repeating) at temp=0 on
   the standard prompt — this is TinyLlama's failure mode, not a regression. The
   "Paris." output from U5 was from the fp32 dequant path (1 tok/s), not Q4_0.

2. ~~**Continuous batching through the compiler path.**~~ DONE (U7). batch=8
   reaches 143 tok/s aggregate, matching Ollama's 144 tok/s at batch=1.
   Phase-1 runtime batch=8: 97 tok/s; compiler path: 1.47× faster.
   **STOP CONDITION triggered:** fusion advantage < 2% at batch≥4. The QKV
   and FFN gate+up fusion passes provide at most 3.2% (batch=2) and regress
   -2% at batch=1. MLX's lazy evaluation already fuses these ops; MLIR passes
   duplicate work MLX does for free. The moat, as currently implemented, is
   not measurable at this scale.

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
3. **MLX lazy evaluation already fuses adjacent ops.** MLIR norm+matmul fusion added zero wall-clock improvement because MLX was already doing it internally. QKV batching and FFN gate+up batching fusion passes gave at most 3.2% advantage (batch=2) and regressed by -2% at batch=1. MLX handles the same cross-op fusion in its graph automatically. Only cross-compile-boundary information (device assignment, scheduler-driven batch formation) can produce structural advantages that MLX cannot match alone.
4. **Custom Q4_0 kernel ≈ mx::quantized_matmul in decode mode.** Both give ~65–66
   tok/s on TinyLlama batch=1 decode. mx::quantized_matmul is the default because it
   is Apple's optimized path and matches the Python/MLX reference. The custom kernel
   is retained behind `MLC_Q4_CUSTOM_KERNEL=1`.
5. **Batching at 143 tok/s matches Ollama's batch=1 ceiling.** The compiler path's
   continuous batching (V1–V3) reaches 143 tok/s aggregate at batch=8, matching Ollama
   (144 tok/s) which has no batch serving mode. Phase-1 runtime batch=8 was 97 tok/s;
   compiler path is 1.47× faster. The thin-matmul inefficiency at batch=1 (MLX
   vs custom Metal) is solved by batching — at batch=8 the mats are [8,2048]×[K,2048],
   fully utilizing MLX's tiled quantized_matmul.
6. **The 2× gap to Ollama batch=1 is thin-matmul.** At seq=1 batch=1, matmuls are
   [1,2048]×[K,2048] — too thin for MLX's tiled kernel. Llama.cpp's custom simdgroup
   reduction handles this shape ~2× better. Not fixable without custom Metal shaders
   (which we explicitly don't write for the compiler path).
