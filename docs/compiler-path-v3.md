# Compiler path v3 — bench + analysis

End of the S1–S6 session: TinyLlama-1.1B Q4_0 now runs through a custom
Q4_0 Metal kernel + in-place KV cache + ANE-aware scheduling annotations,
**~3.6x the Phase-1 runtime** (64 tok/s vs 17.7 tok/s) on the same prompt
and hardware. We're ~44% of Ollama's measured throughput on the same
model. Memory footprint dropped to ~660 MB of Q4_0 weights resident on
device (was 1.3 GB of fp16 in v2).

## Side-by-side, same prompt

Prompt: `"The capital of France is"`, greedy decode, 64 generated tokens,
TinyLlama-1.1B Q4_0, batch=1. M-series Mac.

| Path                              | tok/s     | Memory | Notes |
|-----------------------------------|----------:|-------:|-------|
| Phase-1 runtime (`mlc serve`)     |    17.7   | ~600 MB| Hand Q4_0 Metal kernels, KV cache |
| Compiler v1 (Q1–Q5)               |     1.02  | ~5 GB  | CPU dequant every forward |
| Compiler v2 (R1–R6)               |    38.32  | 1.2 GB | fp16 resident, MLX matmul |
| Compiler v3 fuse ON               | ~52       | ~660 MB| All fusion passes, Q4_0 kernel, in-place KV |
| **Compiler v3 fuse OFF**          | **~64**   | ~660 MB| Q4_0 kernel + in-place KV; fusion passes off |
| Ollama (TinyLlama)                |   143.75  | n/a    | Production reference; uses MLX or llama.cpp |

Three back-to-back v3 runs at 64 tokens:
* fuse ON: 42.5, 59.3, 56.0 tok/s
* fuse OFF: 64.8, 64.1, 63.7 tok/s

## How we got from v2 to v3

| Commit | What changed                                  | tok/s | Δ vs prev |
|--------|-----------------------------------------------|-------|-----------|
| v2 R6  | fp16 resident, MLX matmul, in-place KV unimpl |  38.3 | — |
| S1     | Q4_0 Metal kernel (probe only, not wired)     | n/a   | — |
| S2     | wire Q4_0 kernel into compiler path           |  42.8 | +12% |
| S3     | in-place KV cache (slice_update)              |  69.2 | +62% (at 32 tok) |
| —      | longer run, fusion ON                          |  ~52  | (variance shows below) |
| —      | longer run, fusion OFF                         |  ~64  | +20% over fusion ON |

The two wins that mattered: **resident Q4_0 + in-place KV**. S2 cut weight
memory in half and trimmed dispatch bandwidth; S3 broke the O(seq)
per-token concat cost. Combined effect on 64-token decode: 38 → 64 tok/s.

## Why fusion ON is now slower

This is the surprise of v3. Reasoning by inspection of the walker:

* In v2, the QKV-fusion pass (R4) collapsed 3 matmuls (~2048×2048 fp16
  each) into one batched matmul over 2560×2048. The batched call was
  about the same speed as 3 separate big calls, so fusion was roughly
  neutral (+13% from R5's FFN gate+up batching only).
* In v3, those 3 matmuls go through the custom Q4_0 kernel, which is
  small and cheap per launch. Fusing them means MLX has to slice the
  combined result into Q/K/V — three `mx::slice` calls that allocate
  intermediate arrays. The slice overhead exceeds the per-launch saving.

Same story for FFN gate+up: separate gate + up matmuls each return their
own array directly; the fused path allocates a 2× width result then
slices in half.

The IR passes still buy us something the runtime path can't (whole-
program structure), but the *runtime* shape of that win flipped sign at
Q4_0 + custom-kernel. Fixing it is a v4 problem: either keep the slice
fully in the kernel (multi-output kernel), or run the batched matmul +
split inline without an `mx::slice` round-trip.

## What still costs time (v3 → v4)

1. **Embedding + LM head stay fp16.** Embedding is 32000 × 2048 fp16
   resident = 130 MB; LM head is also 32000 × 2048 fp16 = 130 MB. Both
   matmul-style; both could move to the Q4_0 kernel (or Q6_K kernel for
   the lm_head, which is Q6_K in the GGUF). Largest remaining bandwidth
   target.
2. **Attention is dense + recomputes RoPE in fp32.** The fp16→fp32 round
   trip around RoPE is a workaround for the seq_len=1 NaN we hit in R2.
   A fast::rope-replacement that handles fp16 correctly removes one
   eval+copy per attention. Plus flash attention would help.
3. **`mx::slice_update` allocates a new full-size array each call.** Per
   layer, per token, we copy 2048 × 64 = 128k elements that haven't
   changed. Native in-place slice (or a kernel that does the update in
   the existing buffer) eliminates that.
4. **MLX slice overhead.** Every fusion / split that goes through
   `mx::slice` pays an allocation. A multi-output Q4_0 kernel that
   writes Q, K, V into separate output buffers in one launch closes
   the fusion ON vs OFF gap.
5. **Ollama uses paged attention + warm kernels.** We're ~44% there
   without batched serving, paged KV, or a true flash kernel. The next
   ~2x is in attention.

## The ANE story (S4)

The investigation that's the real Phase-2 lever for this session.

Direct measurement, fp16 matmul, M=1, M-series Mac:

| Shape (K × N) | CoreML/ANE | MLX/GPU | ANE win |
|---------------|------------|---------|---------|
| 2048 × 256    |    49 µs   | 178 µs  | 3.6x    |
| 2048 × 2048   |    55 µs   | 286 µs  | 5.2x    |
| 2048 × 5632   |   367 µs   | 344 µs  | 0.94x   |

ANE wins by 3-5x on attention-projection shapes (Q/K/V/O for TinyLlama
all fit). Crossover with GPU is around N≈4096. The S5 scheduling pass
annotates 44 TinyLlama ops as ANE-targets (22 QKV-merge + 22
attn_output) and 113 as GPU.

**The actual ANE backend isn't wired yet.** Compiler v3 still runs
everything on MLX-GPU. Next session: CoreML model preparation at build
time (via the Python venv per `compiler/coreml/README.md`), and a
lowering pass that routes `target_device = ane` ops to
`MLPrediction:fromFeatures:` through the compiled `.mlmodelc`. If the
end-to-end ANE matmul cost in-flight (including data marshaling) keeps
the 5x win on attn projections, the compiler path crosses ~80 tok/s and
makes a real run at parity with Ollama.

## Validation status

| Check                                                  | Status |
|--------------------------------------------------------|--------|
| S1: Q4_0 kernel vs CPU triple-loop                     | PASS (cosine 1.000) |
| S2: Q4_0 path same output as fp16 path                 | PASS |
| S3: in-place KV produces same generation, flat tok/s   | PASS (tested 32 vs 64 tok) |
| S4: CoreML matmul lands on ANE                         | DONE (per-call latency reported) |
| S5: scheduling pass annotations match heuristic        | PASS (44 ANE / 113 GPU on TinyLlama) |
| Compiler v3 output == compiler v2 output               | PASS ("Paris, the city of love…") |

## Files added this session

```
compiler/mlir/exec/
  Q4MatMul.{h,cpp}            custom Q4_0 Metal kernel via mx::fast::metal_kernel
  MLIRExecutor.{h,cpp}        (mod) Q4_0 cache + dispatch; pre-alloc KV; slice_update
compiler/mlir/passes/
  ScheduleDevices.{h,cpp}     target_device annotation pass
compiler/coreml/
  gen_matmul_model.py         coremltools script → .mlpackage
  coreml_matmul_probe.mm      Objective-C++ ANE timer
  mlx_matmul_probe.cpp        MLX-GPU companion timer
  README.md                   reproduction + integration notes
tests/mlir/
  test_q4_kernel.cpp
  test_mlx_metal_kernel_smoke.cpp
  test_schedule_devices.cpp
```

The Phase-1 runtime (`compiler/runtime/`, the hand-written op IR, the
existing passes, `compiler/main.cpp`) was not modified this session.
