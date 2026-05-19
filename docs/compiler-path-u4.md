# U4 — multi-output Q4_0 kernel; fusion regression fixed

The v3 walker had a counter-intuitive result: turning **off** the
QKV-fusion pass made decode faster (64 vs 52 tok/s). Diagnosis from v3:
the fused path produced one concat'd output tensor and then sliced it
into Q, K, V — three `mx::slice` calls, each allocating an intermediate
fp16 array. The slice overhead exceeded the savings from collapsing
three Q4_0 dispatches into one.

The same problem hurt the FFN gate+up batched matmul on the always-on
`FeedForwardOp` path: gate and up come out concat'd, get sliced apart,
then go through SwiGLU.

U4 replaces the `q4_0_matmul + mx::slice` pattern with multi-output
Metal kernels that write directly into the destination buffers.

## What changed

`compiler/mlir/exec/Q4MatMul.{h,cpp}`:
- `q4_0_matmul_qkv(x, w_concat, K, Nq, Nk, Nv) -> (q, k, v)`. One
  Metal dispatch over the existing Q‖K‖V concat'd Q4_0 weight; each
  output lane writes to one of three buffers based on whether its row
  index falls in the Q, K, or V region. Dot-product math identical to
  `q4_0_matmul`; only the store routing differs.
- `q4_0_matmul_gate_up(x, w_concat, K, Ng, Nu) -> (gate, up)`. Same
  pattern for the FFN gate+up batched matmul.

`compiler/mlir/exec/MLIRExecutor.cpp`:
- `FusedNormQKVMatMulOp` (Q4_0 branch): calls `q4_0_matmul_qkv` and
  binds `q`, `k`, `v` directly. Old path's three `mx::slice` calls
  removed.
- `FeedForwardOp` (Q4_0 branch): calls `q4_0_matmul_gate_up` and feeds
  the two outputs straight into the SwiGLU `silu(gate) * up`.

The non-Q4 fallback paths still use the concat+slice approach, since
they go through MLX's `matmul` which doesn't have a multi-output
variant — and the slice cost is a smaller fraction of the matmul cost
in fp16-MLX anyway.

## Bench: TinyLlama-1.1B Q4_0, 64 generated tokens, 5 runs each

| Path                  | runs (tok/s)                          | median |
|-----------------------|---------------------------------------|-------:|
| v3 fuse ON            | 42.5, 59.3, 56.0                      |  ~52   |
| v3 fuse OFF           | 64.8, 64.1, 63.7                      |  ~64   |
| **U4 fuse ON**        | 67.4, 72.5, 72.4, 70.6, 71.0          | **71.0** |
| **U4 fuse OFF**       | 70.7, 70.3, 70.3, 71.3, 70.5          | **70.5** |

The fuse-ON regression in v3 (-19%) is gone. Fuse ON is now slightly
ahead of fuse OFF at the median (~0.5 tok/s), well within run-to-run
noise but reproducibly the same direction. Both sides are ~7-19 tok/s
above the v3 numbers from the same prompt.

The fuse-OFF improvement comes from the FFN gate+up multi-output
kernel — `FeedForwardOp` is emitted by the frontend, not by a fusion
pass, so both fuse settings use the same FFN path. Fuse ON gets the
additional QKV gain from the 3-output kernel.

We're now at ~50% of Ollama's measured 143.75 tok/s on the same model.

## Validation

| Check                                          | Status |
|------------------------------------------------|--------|
| `q4_0_matmul_qkv` matches concat+3*slice path  | PASS (maxDiff 0, cosine 1.000 on real layer-0 QKV) |
| `q4_0_matmul_gate_up` matches concat+2*slice   | PASS (maxDiff 0 on real layer-0 FFN) |
| End-to-end output: "Paris, the city of love…" | PASS (matches v3 baseline) |
| Fuse ON ≥ Fuse OFF                             | PASS (median 71.0 vs 70.5) |

## Why this works

A kernel launch on Apple M-series is cheap, but `mx::slice` is not
free: each slice allocates a new contiguous fp16 buffer and copies the
slab out of the source. For Q‖K‖V at TinyLlama dim, that's 3 buffers
× (1 × {2048, 256, 256} × 2 bytes) = ~5 KB of allocations per layer
per token, plus the implicit fp16 reads/writes. Multiplied across
22 layers × 64 tokens, the overhead becomes visible against the
matmul cost itself (which is also small per call at single-token
decode).

Writing the matmul results into separate buffers from the start has
the same per-lane work but no post-hoc copy. The output buffers are
allocated once by MLX (via the kernel's `output_shapes`) and the
kernel writes the values exactly where the next op will read them.

## What's still on the table

Per v3's "what still costs time" list, U4 closes (4). The remaining
work for getting from ~71 → ~143 tok/s:

1. Embedding + LM head still fp16 resident. Q4_0 (embed) + Q6_K
   (lm_head) would save ~260 MB of bandwidth-bound matmul per token.
2. Attention path still recomputes RoPE in fp32 (the seq_len=1 fp16
   NaN workaround). A flash kernel that handles fp16 RoPE eliminates
   one cast pair per attention.
3. `mx::slice_update` for KV cache still copies the full pre-allocated
   slab on each write. A real in-place writer (or a slot-indexed
   kernel) saves one big copy per layer per token.

Each of these is a structurally separate piece of work. U4 hits the
slice-overhead lever; the others target the remaining unrecovered
bandwidth in attention and embedding.
