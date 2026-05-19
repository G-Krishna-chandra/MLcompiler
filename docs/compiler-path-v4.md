# Compiler path v4 — T2 hybrid GPU+ANE wiring

T1 built a per-weight CoreML/ANE matmul executor with ~95 µs/call
back-to-back on a real TinyLlama attn_q weight — under the 200 µs cap
the S4 plan flagged as the threshold for "ANE could win". T2 wires that
executor into the runtime walker for every op the S5 scheduler marked
`target_device = ane`.

**Output: correct.** "Paris, the city of love…" continuation matches
the GPU path token-for-token.

**Throughput: regression.** ANE on, fuse on, 64 generated tokens,
TinyLlama-1.1B Q4_0, three back-to-back runs:

| Config                                  |   tok/s (3 runs)    | Mean   |
|-----------------------------------------|---------------------|--------|
| GPU only, fuse ON  (v3 baseline)        |  69.4, 64.9, 59.3   |  64.5  |
| GPU only, fuse OFF (v3 baseline)        |  51.3, 60.4, 58.5   |  56.7  |
| ANE on, all 44 ops (22 QKV + 22 attnO)  |  24.8, 26.2, 26.2   |  25.7  |
| ANE on, attn_output only (22 ops)       |  29.2, 35.3, 34.5   |  33.0  |

Numbers on the M-series host, prompt `"The capital of France is"`,
greedy decode, all models warm (`.mlpackage` cached on disk under
`/tmp/mlc-ane-cache/`).

## Why the microbench didn't predict the real-workload regression

The T1 microbench had ANE at ~95 µs/call. Naive math says 44 calls × 95
µs = 4.2 ms/token of ANE work — a small chunk of the ~15 ms/token
baseline budget. We expected a flat or net-positive result.

The actual cost in the hybrid is ~590 µs/call (= (1/35 − 1/65) sec /
22 calls), about **6× the back-to-back microbench**. Two effects stack:

1. **MLX command-buffer coalescing breaks at the ANE boundary.** Each
   `predict()` does `mx::eval(x16)` to materialize the input bytes, then
   memcpys into a CoreML `MLMultiArray`. The eval is a GPU sync. With
   pure-GPU decode, MLX queues a dozen+ ops into one Metal command
   buffer and submits asynchronously; the ANE round-trip forces a fence
   every N ops, exposing per-launch overhead that was hidden before.
2. **44 models is too many to keep "warm" on ANE.** The microbench hits
   one model 200× in a row, so model state stays resident. In the
   hybrid, we cycle through 44 distinct `.mlpackage`s once per token,
   and CoreML appears to pay a setup cost per switch. Halving to 22
   models recovers ~7 tok/s (26 → 33), supporting the "switching cost"
   hypothesis — but not enough to break above baseline.

The S4 microbenchmarks are still correct *for what they measured*. They
don't model batched-decode pipeline interleaving, which is the dominant
effect.

## What this means for Phase-2

The S5 scheduling pass currently routes 44 ops to ANE on TinyLlama based
on a per-op latency model. That model is right per-op and wrong
end-to-end. A better cost model would account for:

* **Boundary cost.** Each GPU→ANE→GPU transition costs ~400 µs of
  command-buffer flush + CoreML setup. Group consecutive ANE ops or
  amortize the boundary by batching many weights into one ANE model.
* **Model count.** Past ~10 cycled models the per-switch cost dominates
  the per-call win. Cap ANE-routed ops at a small set, or batch them.

The cleanest fix is structural: bake **one** ANE model per shape that
covers all 22 layers in one predict call (the input becomes [22, K]
instead of [1, K], output [22, N]). That's a v5 problem — needs
multi-layer weight packing and a way to index per-layer outputs from
a single ANE result tensor.

## What changed in T2

```
compiler/mlir/exec/MLIRExecutor.{h,cpp}
  + `use_ane` ctor flag (off by default)
  + Third constructor pass: for every op annotated Device::ANE by S5,
    dequant Q4_0 → fp16, call buildANEMatMulPackage to bake a
    .mlpackage (cached on disk so reruns skip ~1s per op), and cache
    an ANEMatMul keyed on the mlir::Operation*.
  + run() dispatch: if the op has an ANE entry AND input M matches
    the baked M (=1), call ANEMatMul::predict() and adapt the output
    dtype. Prefill (seq > 1) falls through to the Q4_0 kernel — the
    bake is M=1-only today.
  + Env knob MLC_ANE_QKV=0 disables the FusedNormQKVMatMulOp branch
    for ANE so we can A/B the attn_output-only configuration.

compiler/mlir/tools/mlc-compile-run/mlc-compile-run.cpp
  + --ane flag; runs ScheduleDevices and passes use_ane=true to the
    executor. Default off (matches v3 baseline behavior).
```

## Validation

| Check                                            | Status |
|--------------------------------------------------|--------|
| --ane output ≡ no-flag output                    | PASS (Paris continuation matches verbatim) |
| --ane bake cached on disk (rerun is fast)        | PASS (44 .mlpackage in /tmp/mlc-ane-cache) |
| Prefill falls back to Q4_0 (M=1 baked, seq>1)    | PASS (no shape-mismatch crash at prompt length 6) |
| Decode throughput vs baseline                    | **REGRESSION** (35 tok/s best ANE config vs 64 tok/s baseline) |

The wiring works. The result says per-op ANE routing at TinyLlama
scale doesn't pay off; the cost model in S5 needs to know about
boundary cost and model-switch overhead to make better calls.
