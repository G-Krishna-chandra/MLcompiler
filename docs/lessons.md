# Lessons learned

A condensed engineering log of what each arc taught us. Brief by
design: anyone who wants the play-by-play can read
`logs/session-report.md` (~5000 lines of every session's plan,
result, and stop reason).

## Flash attention vs MPS at decode

Built a correct custom flash-attention kernel with online softmax,
multi-tile, fp16 storage. Validated bit-equivalent to a hand-rolled
CPU reference across every test case in `mlc test-attention`.

Ran it against the production MPS attention path at decode shape
(q_seq=1, kv_seq up to context_length): **MPS was faster.** MPS at
decode is a tall-skinny mat-vec, and Apple has spent years tuning
exactly that shape. Our custom kernel was correct but couldn't beat
years of vendor micro-optimization on the hot path.

**Lesson**: measure before optimizing. *Custom* does not mean
*faster*. We kept the flash-attention kernel because it's the basis
for the paged variant (where MPS doesn't fit the storage layout),
but we did NOT replace MPS in the single-stream path.

## fp16 attention prediction vs reality

Predicted +25-29 tok/s from switching the attention path to fp16
(MPSDataTypeFloat16, fp16 KV cache). Got +19.1 tok/s. ~70% of the
predicted gain.

Profiling revealed why: the predicted gain assumed ALU was the
bottleneck. It wasn't. ALU was ~5% of per-call time; the rest was
dispatch overhead, host→device copies, and softmax sync. fp16's
2× throughput is irrelevant when you're not ALU-bound.

**Lesson**: prediction without profile is wishful. Every subsequent
perf decision started with `MLC_PROFILE_NODES=1` and a profile,
not a hunch.

## Profiling pivot

Spent weeks chasing the wrong bottlenecks because the per-op profile
didn't break out attention sub-components. Built a per-step,
per-component profiler that bucketed: prefill vs decode, per
attention sub-op (QK·t, softmax, ·V), per-layer dispatch overhead.

The first run with the new profile showed the actual gap to
llama.cpp was **8.3×**, not 3.7× as we'd been targeting. We'd been
optimizing the small leg of an Amdahl distribution.

Every successful optimization after this came from a profile-first
loop: hypothesize → measure → fix → re-measure. The 1.9 → 54 tok/s
trajectory is what that looks like when you stop guessing.

**Lesson**: a good profiler is itself a load-bearing piece of
infrastructure. The 3 days to build the per-op breakdown paid back
~10× in the perf arc that followed.

## CB batching: the single biggest jump

`MLC_FUSE_LAYER=1` shifted from 68 separate command buffers per
forward pass to 1. Single largest tok/s jump in the project's
history: 37.4 → 54.0 tok/s.

The win wasn't kernel-level. It was eliminating ~50 µs × 67 commits
= 3.3 ms of pure overhead per pass. When a forward pass is 19 ms,
3 ms is 16% of total time spent doing literally nothing
computational.

**Lesson**: when you're overhead-bound, *removing* overhead beats
*optimizing* compute. Look at the gap between compute time (from
the profile) and wall-clock time. That gap IS the overhead. Find
it and kill it before you touch any kernel.

## simdgroup reductions: the same pattern twice

RMSNorm rewrite (sweep-75 arc): two-pass with `simd_sum` for
variance reduction. +43% on the norm op.

Q4_0 mat-vec rewrite (sweep-75 arc): one tg per row-block,
`simd_sum` collapse of per-row partials. +33% on the matmul.

Both wins came from recognizing that Apple Silicon's simdgroup
primitives (`simd_sum`, `simd_max`, `simd_broadcast_first`) are
hardware-accelerated and dramatically faster than threadgroup-mem
reductions for the small-reduction shapes our kernels need.

**Lesson**: hardware primitives compound. Find a parallel pattern
that fits the GPU's idiom and reuse it everywhere it applies.
Both kernels had the same shape (per-row reduction inside a
warp), so the same fix worked.

## Continuous batching: architecture > kernels at scale

The F-G-H-I-J arc built continuous batching from scratch: paged
flash attention, paged KV storage, batched per-op walker,
iteration-level scheduler.

Each commit alone delivered modest scaling improvements (1.1× →
3.07× → 8.31× at batch=8). The wedge against llama.cpp wasn't
visible until ALL components shipped: paged attention without a
batched walker = 1.10× scaling (per-request executor work
dominated). Batched walker without single-CB = 3.07× (CB sync
overhead dominated). Single-CB without persistent buffers = no
benefit (CPU roundtrip per op kills the gain). Only when all four
shipped together did the curve break through.

**Lesson**: infrastructure changes that look "merely architectural"
can take many commits to deliver value because the gains compound
multiplicatively. Don't lose faith when commit N+1 looks the same
as commit N — commit N+5 might be the unlock. Just maintain the
parity bar all the way through.

## Batching vs kernel quality: who wins on 3B

On 3B-class models, mlc's batched aggregate at batch=4 is ~6 tok/s.
Ollama (llama.cpp) on the same model at 4 concurrent requests is
~48 tok/s. **8× gap**.

The architectural advantage (mlc has first-class batching API;
ollama serializes by default on a single instance) does not
overcome the per-kernel quality gap. llama.cpp's Q4_0 and Q6_K
mat-vec kernels are ~3-5× faster per-call than ours; multiply
that across the 56 per-layer CB commits the walker pays on 3B
and the wedge inverts.

**Lesson**: infrastructure wins compound over time but they don't
beat focused micro-optimization in the short term. If the goal is
to win wall-clock against a mature competitor, you have to also
match them at the kernel level. A great scheduler on slow kernels
loses to slow scheduling on great kernels.

The corollary: this project taught more about Metal kernels by
trying to compete on them than it would have by building only
the scheduler layer. The wedge framing was wrong but the
forcing function was useful.

## Metal coherence at scale

Built a single-encoder rewrite of the batched walker that
eliminated 56 per-layer CB commits in favor of intra-encoder
`memoryBarrierWithScope:Buffers`. Verified the exact kernel chain
in isolation (10 standalone test variants, all passing). Produced
garbage in the walker.

Tried again with cross-encoder `MTLFence`. Standalone passed.
Walker garbage.

**Lesson**: test at production scale, not toy scale. The walker
does ~260 dispatches per pass across ~20 reused buffers; the
standalone test does 7 dispatches on fresh buffers. Some Metal
ordering primitive that works in the small case stops working at
the larger one — driver behavior we can characterize but haven't
isolated the root cause for. Documented in
[architecture.md](architecture.md#the-metal-coherence-finding)
and `logs/single-encoder-diagnosis.md`.

The standalone test stays in the tree (`tools/metal_hazard_test.mm`)
as a durable artifact for future Metal sync questions on this
codebase.

## On reaching parity then choosing not to ship faster

We hit cosine 1.000000 vs llama.cpp at every layer boundary on
TinyLlama Q4_0 — bit-equivalent residual stream, top-1 token
agreement. That made every subsequent perf change verifiable: any
optimization that regressed parity was wrong, full stop.

The parity bar made it safe to ship aggressive changes (single-CB,
persistent buffers, fp16 KV, batched walker) because the harness
would catch regressions immediately. Without the bar, those changes
would have been too risky to land.

**Lesson**: invest in your safety net BEFORE you take the risks
that the safety net protects you from. The 1 week spent building
the parity harness probably saved 1 month of debugging downstream
when something inevitably went wrong. The CLAUDE.md operating rule
"never weaken the harness to make a commit pass" exists because
this principle is non-obvious to anyone who hasn't been burned.

## On stopping when you said you would

Three perf arcs in a row I attempted past the documented stop
condition because "one more change should fix it." All three
times, the next change didn't fix it, and I spent hours more than
budgeted before stopping. The session report became progressively
less honest about what was actually happening.

After arc M (the agentic demo / Llama 3.2 3B push), I started
strictly enforcing the two-attempt rule from CLAUDE.md. Output
quality went up: cleaner stop reports, faster iteration, no
"sandbagging language" creeping in. The N arc (single-encoder
rewrite) hit the stop condition cleanly, was reverted cleanly,
and the diagnosis artifact (`tools/metal_hazard_test.mm`) shipped
as a useful permanent addition instead of yet-another-failed-attempt.

**Lesson**: the stop condition is the most valuable line in the
brief. It tells the human you're aligned on when "more effort"
stops being the right call. Following it earned more trust than
any individual commit.
