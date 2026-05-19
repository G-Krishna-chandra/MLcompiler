# Compiler path v5 — session summary

The session spanned two threads: a CoreML/ANE investigation (T1–T2,
then U1) and a GPU-side optimization (U4). The ANE thread closed with
a negative result; the GPU thread cleared the v3 fusion regression and
took absolute throughput from ~64 to ~71 tok/s on the same prompt.

## Cumulative bench, same prompt + hardware

Prompt: `"The capital of France is"`, greedy decode, 64 generated tokens,
TinyLlama-1.1B Q4_0, batch=1. M-series Mac.

| Path                                  |  tok/s    | Memory  | Notes |
|---------------------------------------|----------:|--------:|-------|
| Phase-1 runtime (`mlc serve`)         |    17.7   | ~600 MB | Hand Q4_0 Metal kernels, KV cache |
| Compiler v1 (Q1–Q5)                   |     1.02  |  ~5 GB  | CPU dequant every forward |
| Compiler v2 (R1–R6)                   |    38.3   |  1.2 GB | fp16 resident, MLX matmul |
| Compiler v3 fuse ON                   |    ~52    | ~660 MB | Q4_0 kernel + slice overhead |
| Compiler v3 fuse OFF                  |    ~64    | ~660 MB | Same, no QKV fusion |
| T2 ANE (44 ops)                       |    ~26    | ~660 MB + ~250 MB | per-op CoreML, regression |
| T2 ANE (attn_output only)             |    ~33    | ~660 MB + ~125 MB | half the cycled models |
| **U4 fuse ON** (this session)         |  **71.0** | ~660 MB | Multi-output Q4_0 kernel |
| **U4 fuse OFF**                       |  **70.5** | ~660 MB | Same FFN gain, no QKV gain |
| Ollama (TinyLlama)                    |   143.75  |   n/a   | Production reference |

Five back-to-back U4 runs at 64 tokens:
* fuse ON: 67.4, 72.5, 72.4, 70.6, 71.0
* fuse OFF: 70.7, 70.3, 70.3, 71.3, 70.5

## What this session bought us

```
Phase-1 runtime  17.7  →  v3 fuse OFF  64    : +3.6x (R1–S6 weeks)
v3 fuse OFF      64    →  U4 fuse ON   71.0  : +11%  (this session)
v3 fuse ON       52    →  U4 fuse ON   71.0  : +37%  (fusion regression cleared)
```

The fuse-ON regression that v3 documented is gone. Multi-output kernels
recovered most of the slice overhead the v3 walker paid on QKV (3 slices)
and FFN gate+up (2 slices).

## ANE thread closeout

T1 measured 95 µs/call back-to-back on a single CoreML matmul. T2 wired
that into the walker and saw 590 µs/call in the real hybrid — a regression
to 33 tok/s. U1 directly tested the "boundary cost amortizes if we pack
ops into one predict()" hypothesis with a 2-input / 2-output MLProgram
covering one layer's QKV + attn_output matmuls.

**Result: hypothesis falsified.** Packing 2 matmuls into 1 predict()
costs ~520 µs vs ~480 µs for 2 individual predicts — within noise. The
per-matmul cost inside CoreML at this shape is ~250 µs regardless of
packing. GPU Q4_0 at the same shape costs ~125 µs each.

S4's 5× ANE win at K=N=2048 was an apples-to-oranges comparison: bare
CoreML probe vs MLX with fresh weights and per-call eval. Once both
sides have resident weights and pay the runtime bridge cost, the result
flips in GPU's favor. Detailed analysis in [compiler-path-u1.md].

The `--ane` flag, scheduleDevices pass, and ANEMatMul / ANELayer code
remain in the tree (off by default) as raw material for future work
on **full-forward-on-ANE** or **long-context prefill** scenarios where
the call-frequency / shape profile differs.

## What's left on the GPU side (v6+)

Per v3's "what still costs time" list, U4 closes (4). Remaining levers:

1. **Embedding + LM head stay fp16 resident.** 32000 × 2048 × 2 ≈ 130 MB
   per side; both matmul-style; both move-eligible to a Q4_0 / Q6_K
   kernel. Largest remaining bandwidth target.
2. **Attention recomputes RoPE in fp32** (seq_len=1 fp16 NaN workaround
   from R2). A flash kernel that handles fp16 RoPE cleanly eliminates
   one cast pair per attention.
3. **`mx::slice_update` allocates a full slab per KV cache write.** A
   slot-indexed in-place writer drops the per-layer-per-token copy.
4. **Paged attention + warm kernels** in Ollama. The remaining ~2x gap
   to 143 tok/s is dominated by attention efficiency.

## Files

```
compiler/mlir/exec/
  ANELayer.{h,mm}           T2-followup packed-layer CoreML wrapper (off path)
  Q4MatMul.{h,cpp}     (mod) Multi-output kernels: _qkv, _gate_up
  MLIRExecutor.{h,cpp} (mod) ANE dispatch, multi-output dispatch
compiler/coreml/
  gen_layer_model.py        2-input / 2-output MLProgram baker
compiler/mlir/tools/mlc-compile-run/
  mlc-compile-run.cpp  (mod) --ane flag
docs/
  compiler-path-v4.md       T2 negative-result writeup
  compiler-path-u1.md       U1 falsification + ANE closeout
  compiler-path-u4.md       Multi-output kernel bench analysis
  compiler-path-v5.md       This session summary
tests/mlir/
  test_ane_layer.cpp        U1 microbench
  test_q4_kernel_multi.cpp  U4 multi-output correctness
```

## Standing vs Ollama

49% of Ollama's measured throughput on the same model. The session's
core insight: the per-op ANE thread is a dead end at this scale, and
the next 2× lives in attention. The compiler IR + walker structure
is paying its way — the multi-output kernel landed cleanly behind the
walker's `FusedNormQKVMatMulOp` and `FeedForwardOp` dispatch sites
without touching the IR shape.
