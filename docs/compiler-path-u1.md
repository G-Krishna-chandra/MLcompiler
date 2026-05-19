# U1 — subgraph packing falsifies the boundary-amortization hypothesis

The premise from T2's negative result: per-op ANE routing in the walker
ran ~33 tok/s vs ~64 tok/s GPU-only, root-caused as boundary cost (each
`mx::eval` between ANE and GPU breaks MLX command-buffer batching). The
proposed fix was to **pack many matmuls into one CoreML model** so the
boundary cost is paid once per predict() rather than once per matmul.

U1 tests this hypothesis directly on TinyLlama layer-0 attention
projections (`Q‖K‖V` concat and `attn_output`), at the simplest possible
scale: two matmuls.

## What was built

- `compiler/coreml/gen_layer_model.py` — coremltools script that bakes
  a fp16 MLProgram with 2 inputs / 2 outputs / 2 baked weights.
- `compiler/mlir/exec/ANELayer.{h,mm}` — mirror of T1's `ANEMatMul`
  with `predict(x_qkv, x_attn) → (out_qkv, out_attn)`. Pre-allocated
  `MLMultiArray` for each input; per-call memcpy in/out.
- `tests/mlir/test_ane_layer.cpp` — bake + correctness + 4-way bench.

## Bench (per-iter wall-clock, M=1 fp16, K=2048, layer-0 weights)

`N_qkv = 2560` (Q 2048 + K 256 + V 256), `N_attn = 2048`. Per-iter
includes input memcpy + predict + output memcpy. 200 iters after warmup,
averaged. Three independent runs each:

| Configuration             |  run 1  |  run 2  |  run 3  |
|---------------------------|--------:|--------:|--------:|
| (a') ANELayer (isolated)  |    478  |    564  |   —     |
| (a)  ANELayer + siblings  |    520  |    548  |    482  |
| (b)  2× ANEMatMul         |    445  |    511  |    492  |
| (c)  2× Q4_0 Metal kernel |    242  |    257  |    269  |

Units: µs/iter.

The packed-isolated (a') ≈ packed-with-siblings (a) ≈ 2× individual (b).
The model-cycling effect we hypothesized in T2 doesn't really hurt this
microbench. All three ANE configurations are **~1.9× slower** than the
GPU Q4_0 baseline at the same shape.

## What this means

The premise of U1 was: boundary cost is paid per predict() call.
Implication: 1 predict() with 2 ops would beat 2 predict()s with 1 op.

**Result: the cost is per-op inside the model, not per call.** The 2-op
packed model is essentially the same wall-clock as 2× 1-op models. If
anything, the packed version trends very slightly slower — CoreML's
2-input / 2-output graph has a small extra overhead vs two simpler
1-input / 1-output graphs.

This explains the T2 regression in a way the S4/T1 microbenches
couldn't predict: when CoreML evaluates a model, the per-matmul cost
is ~250 µs at K=2048, N=2048, regardless of the surrounding context.
T1's 110 µs/call was measured in a tight loop on a single matmul model;
that's already the fastest case. Real walker contexts and packed
contexts both land at ~250 µs/matmul.

Single matmuls at this shape on GPU through the Q4_0 kernel are
~130 µs each (entry (c) is 2× ~130). The crossover is bigger than S4
suggested — the apples-to-apples comparison at the *call* level (with
MLX eval included, not just the kernel itself) puts GPU clearly ahead.

## Why the S4 number was misleading

S4 measured `coreml_matmul_probe.mm` at 55 µs (2048×2048, fp16) vs MLX
at 286 µs. That MLX number was a fresh `mx::matmul` + `mx::eval` per
call, with no resident weight — almost worst-case for MLX. The
production v3 path keeps the Q4_0 bytes resident and uses a fused
dequant+matmul kernel, so MLX-with-Q4_0 lands ~130 µs/call — and the
CoreML side gains ~50 µs of unavoidable bridge overhead (memcpy +
MLX→CoreML buffer wrap) once it's in the walker.

What once looked like a 5× ANE win at the matmul level becomes a
roughly 2× ANE *loss* at the call level once you account for the
runtime path on both sides.

## Stop condition hit

The U-series plan said:

> If one-layer subgraph is slower: the overhead is in CoreML model
> compilation/execution at this graph size, not switching. Report and
> stop.

The packed ANE result is slower than the GPU Q4_0 baseline by ~2×, and
neither faster nor slower than 2× individual ANE. The hypothesis is
falsified; the stop condition is hit.

## What's left for ANE

Two structurally different uses remain plausible — both are session-
sized investigations, not quick wins:

1. **Full-forward on ANE (Option C).** Pack the entire decode step
   (embed → 22 × layer → norm → lm_head) into one CoreML model. One
   predict() per token; one boundary crossing per token. Requires
   CoreML to support: RoPE (or fp32 workaround), causal-mask softmax,
   KV cache as model state, and fp16 GQA. Unclear if all of that is
   reachable inside a single MIL program.
2. **Long-context prefill** where the input is M=512+ instead of M=1.
   ANE matmul throughput scales much better with M than GPU (within
   limits). The boundary cost is paid once for many tokens of work.

Neither belongs in the current TinyLlama-decode workload. The natural
next step is **GPU-only optimization**: the multi-output Q4_0 kernel
that lets fusion ON beat fusion OFF (see compiler-path-v3.md "What
still costs time"). That's where the remaining headroom is.
