# Compiler path v1 — bench + analysis

End of the Q1–Q5 session (commits `8e11d1b` through `6000074` on `main`):
the new MLIR-driven compiler path runs TinyLlama-1.1B end-to-end on Apple
silicon via MLX, alongside the existing hand-written runtime. Output is
correct; throughput is, as expected, slower than the runtime by ~15x.

The Phase-1 runtime (`compiler/runtime/`, the hand-written op IR, the
existing passes, `compiler/main.cpp`) was not modified.

## Side-by-side, same prompt

Prompt: `"The capital of France is"`, greedy decode, 16 generated tokens,
TinyLlama-1.1B Q4_0. Both binaries called with batch size 1.

| Path                              | tok/s | Wall (16 tok) | Notes |
|-----------------------------------|-------|---------------|-------|
| `mlc serve` (Phase-1 runtime)     | 14.8  |  1.08 s       | Hand-written Q4_0 Metal kernels, KV cache, fused dispatch |
| `mlc-compile-run` (fusion ON)     |  1.02 | 15.67 s       | MLX fp32 matmul, CPU Q4_0→f32 dequant per forward, dense attention, no KV cache |
| `mlc-compile-run` (`--no-fuse`)   |  1.12 | 14.26 s       | Same, no norm+matmul fusion |

Both compiler-path runs produce the same coherent generation:

```
Paris.

2. The capital of Spain is Madrid.

3
```

The runtime's first generated token is also " Paris", so the compiler
path is on the same top-1 trajectory as the reference implementation.

## Where the compiler path spends its time

Per generated token, the compiler path does a full forward pass over the
entire current sequence. No KV cache; each step recomputes attention from
position 0. By inspection of the walker, the cost breaks down as:

1. **CPU Q4_0 → fp32 dequantization, every weight, every step.** TinyLlama
   has ~660 MB of Q4_0 weights expanding to ~5.3 GB of fp32. The walker
   re-dequantizes everything on every `run()`. This is the dominant cost.
2. **MLX matmul on fp32** — 88 matmuls per layer set. MLX's Metal path is
   not the bottleneck; the feed-in is.
3. **MLX graph realization** — one `mx::eval()` per forward; the whole
   computation graph (22 layers, all activations) lives until eval.
4. **Dense attention** — O(seq²) softmax matmul. For peak seq = 22 in
   this run, negligible vs the weight matmuls.

Empirically: doubling the prompt length roughly doubles per-token wall
time, consistent with dequant + matmul being linear in seq.

## What the fusion pass actually did

The norm+matmul fusion (Q3) collapses each TinyLlama attn-norm into the 3
matmuls that consume it (q/k/v), eliminating 22 IR ops. Cosine vs
unfused: 1.000.

It did **not** speed up generation. Fusion-ON 1.02 tok/s vs fusion-OFF
1.12 tok/s — fusion is marginally slower in our v1. Two reasons:

* The fused op **recomputes RMSNorm** inside each of the three matmuls
  that pulled it in. Unfused, the norm is computed once and reused by
  q/k/v. Fusion trades intermediate-buffer savings for redundant norm
  work. RMSNorm is cheap relative to matmul, but not free.
* MLX's lazy graph already fuses adjacent ops at `mx::eval()` time, so
  the no-intermediate-buffer benefit was largely captured before our
  pass ran.

This is the expected "the compiler's win was already MLX's win" effect at
this stage. A proper norm+QKV fusion (one op doing norm + 3 matmuls,
sharing the norm) would be a real save — a clean next pass.

## What to optimize next, priority order

1. **Resident dequantized weights** — load each weight tensor into an
   `mx::array` once at `MLIRExecutor` construction, keep it for the life
   of the run. Removes per-forward dequant. Trade-off: 5.3 GB peak memory
   for TinyLlama; lazy-load + LRU is the obvious follow-up. Expected
   speedup: **3–5x**.
2. **Fused Q4_0 dequant + matmul Metal kernel** — once (1) lands, the
   matmul itself becomes the bottleneck. An MLX-side "weight unpack on
   dispatch" kernel matches the runtime path. Expected speedup: **2–3x**.
3. **KV cache through the compiler** — even with (1) and (2), dense
   attention re-projects Q/K/V for every prefix token on every step. A
   persistent KV cache makes per-step attention O(1) in seq. Big win on
   longer generations. Expected speedup at 100+ tokens: **5–10x**.
4. **norm + QKV fusion (proper)** — one fused op `RMSNorm(x) → q/k/v
   matmul` that shares the norm. Together with (1) and (2), this is the
   shape of the "real" compiler win over MLX-alone.
5. **ANE MIL lowering** — Q1–Q5 only target GPU/MLX. The whole point of
   the heterogeneous-compiler thesis is that the sequential,
   latency-bound steps of the agent workload should run on ANE. The
   dialect's `target_device` attribute is already the placeholder;
   lowering for `#mlc.device<ane>` is the next visible piece of the moat.

## Validation status

| Check                                       | Status |
|---------------------------------------------|--------|
| Q1: MLX matmul vs CPU triple-loop, cosine ≥ .999 | PASS (1.000) |
| Q2: full forward produces finite logits      | PASS (logits in [-13.9, 18.0]) |
| Q3: fusion preserves logits, cosine ≥ .999   | PASS (1.000) |
| Q4: top-1 token is "Paris"                   | PASS (" Paris.\n\n2. The capital of Spain…") |
| Q5: tok/s table (this doc)                   | DONE |

## Lessons from this session

* GGUF stores tensor dimensions **innermost-first** at the metadata
  level, even though the byte layout is row-major C. Earlier commits
  treated `info.shape[0]` as numpy-rows; square matrices hid the
  mismatch. Q1's cosine = 1 came from comparing two equally-wrong
  interpretations of the same buffer. Non-square tensors (attn_k/v,
  ffn_*, embedding, lm_head) exposed it only once `mlc-compile-run`
  actually generated a token. **Cosine-on-self doesn't catch
  transposed-buffer bugs; end-to-end token decode does.**
* RoPE convention for GGUF Llamas is `traditional=true`
  (consecutive-pair rotation), per the runtime's
  `applyRotaryToBuffer` in `attention_cpu.cpp`. MLX's `traditional=false`
  is the rotate-half convention used by HF LLaMA's Python code, which is
  mathematically equivalent under a different storage permutation but
  not what GGUF gives us.
* MLX's lazy-graph fusion already captures the wins of most simple
  adjacent-op fusions. Future compiler passes need to do something MLX
  can't do alone — cross-op fusion that requires whole-program info,
  ANE/GPU device split, weight residency across calls. The Q3 pass
  proved the IR-rewrite mechanic works; meaningful wins from this
  mechanic will come from the passes ranked 4 and 5 above.

## Files added this session

```
compiler/mlir/exec/
  MLIRExecutor.{h,cpp}    walker / interpreter over emitted IR
  MLXBuilder.{h,cpp}      fp32/GGUF ↔ mx::array conversions
  Quantize.{h,cpp}        GGUF → fp32 dequant (wraps runtime helpers)
compiler/mlir/passes/
  FuseNormMatMul.{h,cpp}  norm+matmul fusion pass
compiler/mlir/tools/mlc-compile-run/
  mlc-compile-run.cpp     end-to-end CLI driver
tests/mlir/
  test_mlx_matmul_lowering.cpp
  test_compiler_path_forward.cpp
  test_norm_matmul_fusion.cpp
```
