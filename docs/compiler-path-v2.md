# Compiler path v2 — bench + analysis

End of the R1–R6 session: compiler path now **outperforms the Phase-1
runtime by 2.4x** on TinyLlama-1.1B (`mlc-compile-run` 38.3 tok/s vs
`mlc serve` 15.8 tok/s on the same prompt, M-series Mac, Q4_0). 38x faster
than the v1 baseline (1.02 tok/s). Output unchanged across all variants.

## Side-by-side, same prompt

Prompt: `"The capital of France is"`, greedy decode, 32 generated tokens,
TinyLlama-1.1B Q4_0. Both binaries `--batch-size 1`.

| Path                                | tok/s | vs runtime | Notes |
|-------------------------------------|-------|-----------|-------|
| Compiler v1 (Q1–Q5)                 |  1.02 |  0.07x    | CPU dequant every forward, dense attention, no KV cache |
| Compiler v2 — no fuse               | 38.25 |  2.42x    | All v2 changes except `fuseNormMatMul` + `fuseQKVMatMul` (FFN gate+up batching is executor-internal) |
| Compiler v2 — full                  | 38.32 |  2.43x    | All passes on |
| Phase-1 runtime (`mlc serve`)       | 15.80 |  1.00x    | Hand-written Q4_0 Metal kernels, paged KV, fused dispatch |

All compiler-v2 runs produce identical output (sanity check):

```
Paris, the city of love, the city of love, the city of love, the city of
the world.
The city of love, the city of
```

Greedy decoding loops on TinyLlama-1.1B — that's a model-quality artifact
(seen identically on the runtime side too), not a compiler-path issue.

## How we got from 1.02 to 38.32

| Commit | What changed                            | tok/s | Δ vs prev |
|--------|-----------------------------------------|-------|-----------|
| Q4     | Baseline v1                              |  1.02 | —         |
| R1     | Resident fp16 weights                    |  7.80 | +7.6x     |
| R2     | KV cache + MLX SDPA                      | 31.67 | +4.1x     |
| R3     | (probe only — MLX affine quant skipped)  |   —   | —         |
| R4     | norm+QKV matmul fusion (IR pass)         | 31.15 | ~0%       |
| R5     | FFN gate+up batched matmul (executor)    | 35.36 | +13%      |
| R6     | Final 32-token bench                     | 38.32 | (longer run, JIT warm) |

R2 was the dominant win — eliminating per-forward dequantization (R1) and
moving from "rebuild the whole graph every token" to "feed one token + a
KV cache" (R2). Past that, MLX's lazy-graph fusion picks up most of the
no-brainer cross-op savings; the explicit IR-level passes (R3, R4) end up
roughly flat. R5 was the surprise win — FFN gate/up shares an input and
amortizes one big matmul over what was two of three matmuls per layer.

## Why fusion ON ≈ fusion OFF in v2

Fusion OFF still has:
* R1 resident fp16 weights
* R2 KV cache + MLX SDPA
* R5 FFN gate+up batching (executor-internal, doesn't go through the IR
  pass machinery)

Fusion OFF is missing:
* R4's norm+QKV merge — the three QKV matmuls each recompute the
  norm

In numbers, that's worth essentially zero on TinyLlama. The norm is cheap
relative to the matmul, and MLX's lazy graph captures the
"three-ops-share-an-input" pattern via expression DAG fusion at
`mx::eval()` time anyway. The IR-level pass machinery still earns its
keep — it gives us a hook for transforms MLX *can't* do alone (device
split, weight residency policy, kernel selection) — but the wins on
existing MLX-only deployments are slim.

## What still costs time

Profile by inspection of the walker (no instrumentation built in this
session):

1. **Per-layer attention** — even with `scaled_dot_product_attention`,
   the GQA broadcast + transpose + concat-onto-cache dance allocates
   intermediates each step. The KV cache concat itself is the most
   suspect: it copies the whole prefix every token. Switching to a
   pre-allocated ring/page buffer that updates in place would cut this.
2. **Embedding lookup as fp16 `mx::take` on a 32000-row table** —
   single-token decode does one row gather; cheap, but the row is read
   non-contiguously because the table is stored `[vocab, hidden]`.
3. **LM head matmul** — projects 2048 → 32000 every step; 130 MB of fp16
   weight bandwidth per token. Quantizing this one weight is the easiest
   bandwidth win.

## R3: why we didn't integrate MLX quantized_matmul

Probe test (`test_mlx_quant_roundtrip.cpp`): single-matmul cosine between
fp32 reference and `mx::quantized_matmul` (affine, 4-bit) hits ≈ 0.99 —
exactly the threshold the session plan flagged as "skip". Group_size 32
and 64 both behave the same. The issue is double-quantization: GGUF Q4_0
→ fp32 → MLX-affine → quantized matmul. Each step loses precision; 22
layers compound. The proper fix is a custom Metal kernel that reads
GGUF Q4_0 bytes directly. That's the next external-kernel work, not an
MLX one-liner.

## Validation status

| Check                                       | Status |
|---------------------------------------------|--------|
| Q1: MLX matmul vs CPU triple-loop           | PASS (cosine 1.000) |
| Q2: full forward produces finite logits     | PASS (logits ∈ [-13.9, 18.0]) |
| Q3: norm+matmul fusion preserves logits     | PASS (cosine 1.000) |
| Q4: top-1 = "Paris" after prompt            | PASS |
| R1–R5: same continuation as v1, ~38x faster | PASS |
| R3 probe: MLX affine 4-bit cosine           | 0.990 (borderline; not integrated) |

## What to optimize next, priority order

1. **Custom Metal Q4_0 quantized_matmul kernel** — reads GGUF bytes
   directly, no double-quantization. Closes the precision gap from R3
   and saves ~1.3 GB → ~660 MB of weight memory. Expected throughput:
   close to runtime parity at lower memory; might unlock 60+ tok/s.
2. **In-place KV cache (ring or paged)** — eliminate the per-token
   concat copy. Memory-allocation churn shows up at long sequences.
3. **LM head quantization specifically** — biggest single weight
   (130 MB fp16); quantizing just this one to Q4 saves bandwidth on
   every token.
4. **ANE MIL lowering** — R1–R6 are GPU-only. The whole heterogeneous
   thesis is that sequential / latency-bound steps belong on ANE. The
   dialect's `target_device` attribute is already the placeholder.
5. **Proper scheduling pass** — per-op device assignment that the
   `target_device` attribute points at.

## Files added this session

```
compiler/mlir/exec/  (modified)
  MLIRExecutor.{h,cpp}   resident weights, KV cache, MLX SDPA,
                         QKV+FFN concat caches
compiler/mlir/passes/
  FuseQKVMatMul.{h,cpp}  the QKV fusion pass
compiler/mlir/dialect/MLCOps.td  (modified)
  + mlc.fused_norm_qkv_matmul op
tests/mlir/
  test_mlx_quant_roundtrip.cpp   R3 probe
```

The Phase-1 runtime (`compiler/runtime/`, the hand-written op IR, the
existing passes, `compiler/main.cpp`) was not modified this session.
