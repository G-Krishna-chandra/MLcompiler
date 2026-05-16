# Architecture

This is the technical writeup. The README has the elevator pitch and
the build/run instructions; this doc covers how the pieces fit together
under the hood.

## Forward-pass pipeline

```
   GGUF file
       │
       ▼
   GGUFLoader       parses headers, tensor directory, KV metadata.
       │            Builds a flat tensor_map keyed by name. Returns
       │            tensor offsets but doesn't read tensor bytes
       │            (mmap'd on demand by Session::tensorData).
       ▼
   Session          owns the loader, the raw-tensor cache, and a
       │            cached fused-weight builder (attn_qkv from q+k+v,
       │            ffn_gate_up from gate+up). Fused weights are
       │            synthetic concatenations stored in synthetic_data_
       │            and exposed through the same tensor_map.
       ▼
   IRBuilder        translates GGUF tensors into a graph-based IR
       │            (ir::Graph). One node per logical op (Embedding,
       │            Norm, MatMul, Attention, Add, ...). Reads model
       │            metadata (num_layers, hidden_size, head_count,
       │            kv_head_count, rope_freq_base, rotary_dim,
       │            head_weight_name) from GGUF KV.
       ▼
   IR passes        layout normalization, matmul fusion hints,
       │            architecture family hints, tiling hints. Each
       │            pass is a separate translation unit in
       │            compiler/passes/.
       ▼
   KernelScheduler  Schedule() converts ir::Graph into an
       │            ExecutionGraph: per-node backend choice (CPU /
       │            Metal), kernel id, tensor allocations.
       ▼
   ExecutionPlanBuilder  BuildFromLoader() finalizes the
       │            ExecutionGraph: sets head_weight_name (handles
       │            tied embeddings — Llama 3.x, Gemma), infers
       │            vocab_size from the head tensor's non-hidden
       │            dimension, calls setModelConfig.
       ▼
   ExecutionGraph + ModelConfig  — the runtime artifact.
```

The graph is built once per Session and reused across all decode
steps and all requests.

### Two execution paths from ExecutionGraph

There are two ways the runtime drives an ExecutionGraph through a
forward pass:

**1. Single-stream (ExecutionExecutor).** Walks the graph
node-by-node. `MLC_FUSE_LAYER=1` enables the fuse-mode optimization:
fusable ops (Norm, Q4_0 matmul, Add, Slice) encode onto a shared
`MTLCommandBuffer` opened once at the start of the forward pass;
metal-eligible non-fusable ops (MPS-attention, FeedForward) defer
their dispatches onto the same CB and chain their outputs via a
`pass_outputs` map for zero-CPU-roundtrip handoff. One CB commits
+ waits at the end of the pass.

Used by: `mlc chat-repl`, `mlc decode`, `mlc compare`, the parity
harness. This is the canonical numerically-validated path.

**2. Continuous batching (BatchedExecutor + BatchedWalker).**
`mlc serve --paged` and `mlc api` route through this. The
BatchedWalker has the model topology hard-coded for Llama-arch
(RMSNorm → fused QKV matmul → split + RoPE + cast → scatter K/V →
paged-flash → attn_out matmul → residual → ffn_norm → fused
gate_up matmul → split + silu·up → ffn_down → residual). It
dispatches every per-op kernel once across all N requests' inputs,
all on a single CB per forward pass.

## Memory management

Three distinct allocation patterns coexist:

**Persistent name-keyed buffers** for the BatchedWalker's
intermediate activations: `MetalExecutor::getOrAllocCachedBuffer
(name, bytes)` returns the same `MTLBuffer` across calls keyed by
name. Reallocates if the requested size exceeds the cached size.
These buffers persist for the process lifetime; their values are
overwritten each forward pass. Used for `w_residual`, `w_qkv`,
`w_q`, `w_k`, `w_v`, `w_attn_mix`, `w_gate_up`, `w_gate`, `w_up`,
`w_ffn_mix`, `w_ffn_down`, `w_final_norm`, `w_logits`, plus the
fp16 staging buffers `w_kv_f16_k`/`w_kv_f16_v` and the RoPE
`w_rope_cos`/`w_rope_sin` tables.

**Pool buffers** for single-stream fuse-mode: a pool of
intermediate-sized `MTLBuffer`s that `ExecutionExecutor` checks
out for each fusable op's output. Returned to the pool at
`flushForwardPassCB()` time. Pool sizing is driven by the
pre-analysis pass that runs at the start of each `executor.run()`.

**Paged KV storage** for continuous batching: `PagedKVStorage`
owns one bulk per-layer `MTLBuffer` of size
`capacity_pages * page_size_tokens * n_kv_heads * head_dim *
sizeof(fp16)`. `PagePool` is a logical free list of page IDs;
`RequestKVState` holds a per-request page table. New tokens grab
the next slot via `extend_one_token`; pages return to the pool on
request completion. Default `page_size_tokens = 64`.

GPU↔CPU traffic in the BatchedWalker is intentionally minimized:
the only per-pass uploads are the embedding row(s) and the RoPE
cos/sin tables; the only per-pass download is the final logits.

## Metal kernel design

### Q4_0 mat-vec (q4_0_matmul_v3_batched)

Q4_0 stores each 32-element block as 16 packed bytes (4-bit
nibbles, low half in positions 0..15, high half in positions
16..31) plus one fp16 scale. The kernel:

- Dispatch: `(batch * row_blocks)` threadgroups, 32 threads/tg
  (one simdgroup).
- Each tg processes 4 output rows in parallel (unrolled).
- Each thread strides through cols by 32 (one simdgroup width),
  dequantizes the block and accumulates the partial dot product.
- After the inner loop: `simd_sum(p[i])` collapses per-row
  partial sums to thread 0, which writes the result.

The single thread/row choice keeps register pressure low and
exploits Apple Silicon's strong simdgroup primitives. Per-tg
work is dominated by Q4_0 dequant; the matmul fits comfortably
within one simdgroup's compute budget at TinyLlama's
hidden_size=2048.

### Q6_K mat-vec (q6_k_matmul_v3_batched)

Q6_K is more complex than Q4_0: 256-element superblocks with
6-bit quants split as a 4-bit `ql` field, a 2-bit `qh` field,
and 16 8-bit scales. The kernel walks each superblock unpacking
the three streams per element, applies the appropriate scale,
and accumulates exactly like Q4_0. Same 32-thread / 4-row
threadgroup layout.

This kernel handles the lm_head matmul. At 3B-model scale
(3072 × 128256 rows), the per-pass cost is ~6 ms on M3 Pro
— a significant fraction of total per-pass time and a known
gap vs llama.cpp's matrix-vector kernels.

### RMSNorm (rms_norm_kernel_v2_batched)

Two-pass: variance reduction via simdgroup primitives, then
scale + multiply by weight. 256 threads/tg, one tg per request
in the batch. The simdgroup reduction pattern (broadcast partial
sums via `simd_sum`, then complete with a small threadgroup-mem
reduce) gave a 43% speedup over a naive scan implementation
during the perf arc.

### Paged flash attention (paged_flash_attention)

Online-softmax flash attention reading K/V from paged storage.
Dispatch: 1D grid of `(batch * num_heads)` threadgroups,
`head_dim` threads each. Each threadgroup runs the standard
online-softmax recurrence over kv-position tiles of size
`tile_size`:

1. Load Q for (request, head) into threadgroup memory.
2. For each tile: cooperative paged load of K and V; QK·t with
   causal mask; online softmax update; weighted V accumulation
   into the running output.
3. Final normalize by total softmax denominator; write output.

**Runtime tile sizing.** The threadgroup memory budget is
`D + 2*T*D + T + D` floats for (shared_q, tile_K, tile_V,
tile_scores, partial_buf). M3 Pro caps this at 32 KB = 8192
floats, so `T ≤ (8192 - 2D) / (2D + 1)`. For `head_dim=64`
(TinyLlama), T=32 fits; for `head_dim=128` (Llama 3.2 3B), the
math caps T at 30 and we round down to 24 for clean load
alignment. The tile size is passed as a kernel param at encode
time so one compiled kernel handles both regimes.

**GQA support.** The kernel maps each query head to a key/value
head via `kv_h = head_idx * kv_heads / num_heads`. Tested with
TinyLlama (32/4 = 8:1) and Llama 3.2 3B (24/8 = 3:1) — any
divisible ratio works.

### Strided slice + cast (strided_copy_f32, cast_f32_to_f16)

The BatchedWalker needs to split a fused QKV matmul output into
Q/K/V buffers and cast K/V to fp16 for the paged scatter — all
without a host roundtrip. Two small kernels:

- `strided_copy_f32`: each thread copies one `(request, dim)`
  element from a strided source. Used to extract Q at offset 0,
  K at offset q_rows, V at offset q_rows+k_rows of the QKV
  output, and gate/up similarly for the FFN.
- `cast_f32_to_f16`: bulk fp32→fp16 cast. One thread per element.

These were originally bundled into a "compound" encoder with
intra-encoder `memoryBarrierWithScope:Buffers` but Apple's Metal
driver did not consistently track the resulting hazards at the
walker's 260-dispatches-per-pass scale (see *Metal coherence at
scale* below).

## Continuous batching

### Per decode step

Three components coordinate per step:

**Scheduler** (`runtime/scheduler.cpp`). Maintains an FIFO admission
queue and an "active" list of in-flight requests. Each iteration:
1. Promote any waiting prompts onto the active list (up to
   max_batch).
2. Prefill any newly-active requests (one token per `walker.step`
   call — yes, prefill goes through the same batched walker as
   decode for paged storage population).
3. Call `walker.step(active_slots)` to produce one new token per
   active request.
4. Append generated tokens; fire per-request streaming callback;
   on EOS or max-tokens, mark complete and release pages.

**BatchedWalker** (`runtime/batched_walker.cpp`). One forward pass
across all N active requests. Single MTLCommandBuffer holds the
entire pass. Per layer:
- RMSNorm (batched: `[N, hidden]`).
- Q4_0 fused QKV matmul: input `[N, hidden]` → output
  `[N, qkv_rows]`.
- (Per-layer flush + GPU strided split + GPU rope + GPU fp16
  cast for K/V — *the workaround for the coherence finding
  below*).
- Paged scatter K, V into the per-layer page storage at the
  request's current slot.
- Paged flash attention: per-request Q reads → page-table-driven
  K/V reads → per-request output.
- attn_output matmul + residual_add1.
- RMSNorm + Q4_0 fused gate_up matmul.
- Per-layer flush + GPU strided split → silu·up → ffn_down +
  residual_add2.

Then final_norm + lm_head + CB commit + logits download.

**PagedKVStorage** (`runtime/paged_kv.cpp`). Per-layer bulk fp16
buffer. Layout per page: `[page_size_tokens, n_kv_heads,
head_dim]`. Allocator pre-sizes for `(max_batch + margin) *
pages_per_request` pages. Each forward pass scatters newly
computed K/V at the request's current page+slot; paged-flash
reads via per-request page tables.

### Single-CB design pivot

Before the J arc, the batched walker created ~660 MTLCommandBuffer
objects per forward pass (one per op per request). The CB
commit/wait overhead on M3 Pro is ~50 µs per CB; at 660 CBs ×
50 µs = 33 ms of pure overhead per pass — a ~30% tax on
single-stream and a ceiling on aggregate throughput.

The J1 commit collapsed this to *one* CB per forward pass with
persistent name-keyed GPU buffers carrying state between ops. The
two changes are coupled: single CB only helps if state stays
GPU-resident between ops (otherwise the commit+wait+download+upload
pattern is forced anyway); persistent buffers only help if there's
a single CB carrying the dispatches that read/write them.

Result: TinyLlama batch=8 aggregate jumped from 53.8 → 94.0 tok/s
in one commit. Batch=1 stayed the same (no batching benefit
without multiple requests).

## The Metal coherence finding

After the J arc closed the single-CB design, an M1 follow-up
attempted to replace the walker's per-layer CB flushes (forced by
slice ops that needed CPU roundtrip) with GPU kernels chained on
the shared CB. The basic chain — strided_copy → batched_rope →
cast_f32_to_f16 — verified bit-identical to CPU expected when run
with a forced flush at each layer.

But running the same chain *without* the per-layer flushes
produced garbage: the downstream scatter kernel read stale data
from the cached fp16 buffers, even with
`MTLResourceHazardTrackingModeTracked` set on every buffer and
`memoryBarrierWithScope:MTLBarrierScopeBuffers` between dispatches
inside one encoder.

A standalone reproducer ([tools/metal_hazard_test.mm](
../tools/metal_hazard_test.mm)) tested ten variants of the same
chain on the same buffer setup: single-encoder with and without
barriers, multi-encoder with and without MTLFence, multi-encoder
with explicit `useResource`. **All ten passed.** Cross-encoder
hazard tracking and intra-encoder memory barriers both work as
documented for our kernels at small scale.

So the coherence issue is *scale*-dependent. The walker does ~260
dispatches per forward pass across ~12 kernels and ~20 cached
buffers reused across 22 layers and across multiple forward passes.
Something in that scale-up breaks ordering in a way the standalone
doesn't reproduce. The current best guesses (no smaller reproducer
yet):

- Per-CB resource use limit exhausted at ~260 dispatches.
- WAR hazards on per-layer cached buffers across same-CB fence
  boundaries (standalone tested RAW, not WAR).
- Pipeline state caching across kernel switches at high counts.
- Threadgroup memory state persistence across dispatches when
  paged_flash's tg-memory request (~25 KB on 3B) differs from
  the next dispatch's (<1 KB).

**The current production workaround**: each compound encoder is
followed by a CB commit+wait boundary. Each flush is ~1.5 ms on
M3 Pro (vs ~3-5 ms for the original per-layer
flush+download+CPU-split+upload pattern), so M1 still ships a
~30-60% improvement over pre-M1, but the per-layer CB-sync floor
remains the binding cost on 3B-class models.

Resolving this — either by isolating the actual repro and
restructuring, or by demonstrating it's a Metal driver bug —
would unlock ~84 ms/pass of overhead on 3B (estimated 2× tok/s
improvement). Recorded for future work in
[logs/single-encoder-diagnosis.md](
../logs/single-encoder-diagnosis.md).

## Parity harness

The harness (`runtime/parity_harness.cpp`, driven by
`mlc compare`) runs the same prompt through two execution paths
in-process and diffs every layer-boundary tensor.

Two modes:

- `--metal-vs-cpu`: runs the model twice, once with
  `MLC_FORCE_CPU=1` (every kernel CPU-only), once Metal. Diffs at
  every layer boundary (embedding, per-block attn_output /
  residual_1 / ffn_down / residual_2, final_norm, logits).
  Cosine target ≥ 0.999.

- `--vs-llamacpp`: runs mlc once with `MLC_FORCE_CPU=1`, compares
  against pre-dumped llama.cpp tensor values at the same
  boundaries. Reference dumps produced by
  `tools/llamacpp_dump_activations`. Cosine target ≥ 0.9999.

### Why three-way comparison catches bugs that two-way doesn't

A two-way comparison (Metal vs CPU) catches kernel divergence
but doesn't catch *both sides agreeing on a wrong answer*. The
canonical example from this project's history:

Early Metal Q4_0 matmul had a split-half nibble indexing bug.
The CPU reference (also written from scratch in this repo) had
the SAME bug. Both produced wrong values that AGREED. Two-way
parity passed. mlc generated "Pari?" instead of "Paris".

The fix was to add llama.cpp as a third reference. mlc-CPU vs
llama.cpp diverged cleanly — Q4_0 dequant was wrong on our side,
right on theirs. Fixed the dequant; both mlc-CPU and mlc-Metal
agreed with llama.cpp; greedy decode produced "Paris".

This is the rule the parity harness enforces today: every kernel
must agree with both an independent CPU reference AND with
llama.cpp. The CSV output makes any divergence point obvious
(`logs/compare.csv`).

## Selected per-op profile (TinyLlama batch=1, post-J1)

`MLC_PROFILE_NODES=1` exposes per-op timing. Decode-pass averages
for TinyLlama 1.1B Q4_0 at batch=1 on M3 Pro (chat-repl, fuse-mode):

```
MatMul        7.7 ms (44%)  4 calls/pass  (QKV per layer)
Linear        7.1 ms (38%)  4 calls/pass  (attn_out + ffn_gate_up + ffn_down + lm_head)
Attention     4.0 ms (16%)  1 call/pass   (MPS-batched attention)
Norm          3.2 ms (12%)  4 calls/pass  (attn_norm + ffn_norm)
Add           2.9 ms (11%)  2 calls/pass
FeedForward   0.7 ms ( 3%)
Embedding     0.3 ms ( 1%)  CPU lookup
Slice         0.1 ms (<1%)  (Q4_0 row slicing in fuse-mode)
```

Per-pass total ~19.9 ms → 50 tok/s. The matmul (QKV + attn_out
+ FFN + lm_head) collectively accounts for 82% of per-pass time.
Closing the kernel gap to llama.cpp on Q4_0 mat-vec is the
single largest single-stream lever remaining.
