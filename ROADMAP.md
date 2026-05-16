# Roadmap

## North star

Build the LLM inference runtime that Apple Silicon needs and doesn't have.

Two milestones, in order:

1. **Parity with llama.cpp on single-stream inference.** Target: TinyLlama 1.1B Q4_0 on M3 Pro Metal at parity with llama.cpp's ~70-100 tok/s. This is the credibility floor. Anything slower than this, nobody takes the project seriously regardless of architectural merit.

2. **Surpass llama.cpp on concurrent / batched workloads via continuous batching.** Once parity is reached, the wedge work begins: paged KV cache, dynamic batching, multi-request scheduling. This is where the project differentiates — local LLM serving for agentic workflows on Apple Silicon, which nothing else does well.

Both are achievable. Llama.cpp took years to reach its current state; we are not constrained by that timeline because the diagnostic and validation infrastructure to ship correct kernels in single sessions already exists in this repo (parity harness, dispatch trace, micro-tests). Match the floor, then push past it on the workload nobody else targets.

## Current state

- **Throughput:** 12.1 tok/s on TinyLlama Q4_0 Metal (M3 Pro), 6.4× over the post-correctness baseline.
- **Parity:** end-to-end on TinyLlama Q4_0 against llama.cpp; residual stream bit-equal through tested blocks; final logits top-1/5/10 identical.
- **Harness:** `mlc compare --metal-vs-cpu` and `mlc compare --vs-llamacpp` exist and gate every correctness claim.
- **Latest commit on origin/main:** ece5829 (per-window command-buffer fusion for MatMul/Norm/Add ops).

## Perf sequence to parity

Execute in order. Each item builds on the previous. After every commit, run the validation cascade (below). Commit if green, push.

### 1. GPU-resident intermediates (next up)

Direct continuation of ece5829. Today's fusion windows degenerate to single-op for dependent pairs (attn_norm → attn_q, ffn_norm → ffn_gate, ffn_down → residual_2) because op N+1 reads its input from a host vector that op N hasn't populated yet. With GPU-resident intermediates, op N's output stays in an MTLBuffer; op N+1 reads it directly via `setBuffer`. No host roundtrip, no flush between dependent fusable ops.

**Design (already worked through, do not re-derive):**

- **Buffer pool.** `MetalIntermediateBufferPool` inside `MetalExecutor::Impl`. Size classes by power-of-2 byte count. For TinyLlama Q4_0 the relevant classes are 1 KB (attn_k/v outputs), 8 KB (most hidden states), 32 KB (FFN gate/up), 128 KB (logits) — 4 classes total, max footprint ~4 MB. Pool is process-scoped, single-threaded, no eviction needed at this scale. API: `checkout(bytes) → id<MTLBuffer>`, `returnBuffer(buf)`, `returnAll(vec)`. Per-window checkout tracking in a `window_checked_out_` vector cleared on flush.

- **Encode API.** Two named variants per fusable op (6 entry points total): `encodeMatMulQ4_0FromHost` and `encodeMatMulQ4_0FromBuffer`, same for RmsNorm and Add. Output is always a pool-checked-out `MTLBuffer`. Whether that buffer drains back to a host vec at flush is decoupled from encode and decided by per-tensor `needs_host_output` flag. Both variants share an internal inner helper. Reject Option B (TensorRef union with kind enum) — explicit signatures keep the type system on our side. Reject Option C (always-buffer internally) — Apple's `newBufferWithBytesNoCopy` requires page-aligned memory which arbitrary `std::vector::data()` doesn't satisfy, so copy-at-boundary is unavoidable.

- **Executor pre-analysis.** Before the main dispatch loop, compute `tensor_needs_host_output: map<string, bool>`. A tensor needs host materialization if any consumer is a CPU op, a non-fusable Metal op (Attention, FeedForward, Embedding, Softmax, F32 matmul), a registered tap target, or an exported graph output. **Critical correctness rule:** also mark needs_host=true for any tensor whose consumer comes after a non-fusable op in topological order, even if that consumer is itself fusable. Otherwise the buffer gets returned to the pool at a mid-block flush and a later fusable consumer would read stale data. Implement as a precomputed "is there a non-fusable op between positions i and j" prefix table; O(nodes × consumers).

- **Per-node dispatch logic.** When fusable: check if any input is in `window_outputs` (still GPU-resident); dispatch to `FromBuffer` variant if yes, `FromHost` otherwise. Checkout output buffer from pool. Allocate stable host destination via `context->allocateTensor(name, count, false)`. Record `{name, buffer, host_dst, needs_host}` in `window_outputs`. At flush: for each record where `needs_host`, memcpy buffer contents to host_dst; return buffer to pool unconditionally; clear `window_outputs`.

- **Multi-input ops.** Add has two inputs. Handle (gpu_a, host_b), (host_a, gpu_b), (gpu_a, gpu_b) inline via nullability on both input slots in a single Add entry point. Add is the only multi-input fusable op; this stays small.

- **Pool eviction.** Never. Buffers live forever, max ~4 MB on TinyLlama. Revisit if larger models push footprint past 50 MB.

**Expected impact:** dependent-pair windows collapse from 1-op back into multi-op. Should fuse most of a block into one commit instead of the current ~5 commits/block. Predicted 12.1 → 18-22 tok/s.

**Estimated LOC:** ~520, distributed across MetalIntermediateBufferPool (~80), encode entry-point variants (~200), header + pool accessors (~40), executor pre-analysis + window state + flush logic (~120), and operator_backend encode() extensions (~80).

### 2. Fused QKV + fused gate/up at loader level

Q, K, V projections share the same input. Concatenate the three weight matrices at GGUF load time into one `[input_dim, q_out + k_out + v_out]` tensor. Replace three matmuls with one bigger matmul + output slice. Same for FFN gate + up (both consume the same post-norm hidden state, produce stacked outputs).

**Why this is worth doing:** fewer dispatches AND larger matmuls (better GPU utilization — MPS or the custom Q4_0 kernel runs more efficiently on larger weight matrices). Architecturally clean: doesn't touch the executor or scheduler, only the loader and the IR rewrite passes.

**Files likely touched:** `compiler/frontends/gguf_loader.cpp` (concatenation at load), a new pass in `compiler/passes/` that rewrites the IR (recognize Q+K+V pattern, emit one fused MatMul node with a slice consumer), and minor scheduler updates. Reuse existing `MatMulFusionPass` machinery if applicable.

**Expected impact:** 1.3–1.5×. Predicted ~25–30 tok/s on top of item 1.

### 3. Batched prefill

Currently the executor processes one token per `executor.run()` call during prompt prefill. For a 29-token templated prompt, that's 29 full forward passes before the first generated token. llama.cpp does prompt eval as a single multi-token forward pass. Match that.

**What this requires:** thread `seq_len > 1` through the IR (already partially supported — the attention path handles multi-token K/V scatter), update RoPE to apply per-position rotations across a batch dimension, update the executor's per-step loop to consume the whole prompt at once.

**Architectural payoff beyond the perf win:** batched prefill is structurally the same shape as "multiple requests' prefills happening in one forward pass" — i.e., half the infrastructure needed for continuous batching gets built here. Treat this as a wedge-prep item, not just a perf item.

**Expected impact:** prefill ms/tok drops dramatically (3–5×), generation ms/tok unchanged. Big win for first-token latency. Predicted: prefill at ~30 ms/tok or better, generation steady-state unchanged from item 2.

### 4. Re-profile, iterate

After items 1–3, run `MLC_PROFILE_NODES=1` and identify the new dominant cost. Likely candidates:
- F32 activations throughout (llama.cpp uses fp16 for some intermediates; same tradeoff exists for us)
- Persistent encoder reuse / command buffer pooling
- Custom fused-attention kernel (replace MPS with a single Metal shader that does Q·K, mask, softmax, att·V in shared threadgroup memory — flash-attention-style)
- Eliminating host roundtrips at backend boundaries (e.g., logit post-processing on CPU when it could stay on GPU)

Pick the largest item, propose a fix, implement, validate, commit. Repeat until tok/s reaches the parity target.

### 5. Continuous batching (wedge work begins)

After parity. Out of scope for this roadmap document; will get its own design doc when item 4 completes.

## Validation cascade

Run for every perf commit, in order. Stop and report if any step fails. Do not commit if any step fails.

**A. Build clean.**
```
cmake --build build -j 8
```

**B. Quant kernel parity (unit-level).**
```
./build/bin/mlc test-matmul-q4 models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf
```
All 5 weights must show flag=OK on the 3-way table (H↔C, H↔M, C↔M all cosine ≥ 0.9999).

**C. Backend parity (graph-level).**
```
./build/bin/mlc compare --metal-vs-cpu models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  --prompt "The capital of France is"
```
Every layer-boundary tensor must show cosine = 1.000000. Final logits row: top-1 1/1, top-5 5/5, top-10 10/10. Any divergence stops the cascade.

**D. Reference parity (vs llama.cpp).**
```
./build/bin/mlc compare --vs-llamacpp models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  --prompt "The capital of France is" \
  --reference-dir logs/llamacpp_ref_capital_of_france
```
Residual stream tensors (residual_1, ffn_down.out, residual_2) must be cosine = 1.000000 through all tested blocks against the pre-dumped llama.cpp activations.

**E. End-to-end greedy decode.**
```
./build/bin/mlc chat models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  --raw-prompt --prompt-text "The capital of France is" \
  --temperature 0 --topk 1 --topp 1.0 --max-new 30
```
Output must start with `Paris.` and continue coherently. Token IDs should match the previous commit's output for this prompt (greedy is deterministic; any change in token IDs is a regression signal worth investigating even if the output still "looks right").

**F. Perf measurement.**
```
printf 'Hello!\nexit\n' | MLC_PROFILE_NODES=1 ./build/bin/mlc chat-repl \
  models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  --temperature 0.7 --topp 0.9 --topk 40 --max-new 50
```
Capture `[timing]` and `[profile]` output. Compare against the previous commit's numbers. **Important:** do NOT use `mlc compare` for perf measurement — taps registered there collapse fusion windows by design (see flush-on-tap comment in execution_executor.cpp). Use `chat-repl` for the perf truth.

If the new commit's tok/s is lower than the previous commit's, that's a perf regression. Stop, investigate, do not commit until resolved or until the regression is consciously accepted with a written justification.

## Working rules

- Every commit goes through the full validation cascade. No skipping steps because they passed last time.
- Parity claims require harness verification. "It looks right" is not parity.
- Commit messages describe what changed, what the measured speedup was, and what the next bottleneck is. Future-you (or future contributors) read these to understand the project's trajectory.
- When in doubt about a design decision, prefer the choice that's architecturally compatible with continuous batching (item 5). If a perf optimization makes batching harder later, flag it in the commit body or stop and write a note.
- The parity harness is load-bearing infrastructure. Do not weaken its assertions to make a commit pass. If the harness catches a regression, the regression is real.

## Project facts worth remembering

- **Working model:** `models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf` is the primary validation target. Llama-2 7B Q2_K is also on disk for stress testing but uses Q2_K kernels not yet exercised end-to-end.
- **llama.cpp reference dumps:** `logs/llamacpp_ref_capital_of_france/` contains pre-dumped activations for the standard prompt. The tool that produces them is `tools/llamacpp_dump_activations.cpp`; regenerate via `./build/bin/llamacpp_dump_activations --model <gguf> --prompt "..." --out-dir <dir> --cache-type-k f16 --cache-type-v f16` if reference is needed for a new prompt. The `--cache-type-*` flags default to `f16`, matching both llama.cpp's `llama_context_default_params()` defaults and our default-on fp16 KV cache; pass `f32` for a fp32-cache reference if needed.
- **Env vars that matter:** `MLC_FORCE_CPU=1` (pin everything to CPU), `MLC_FUSE_LAYER=1` (enable fusion path), `MLC_PROFILE_NODES=1` (per-op timing), `MLC_PARITY_DUMP=DIR` (dump both sides' tap tensors), `MLC_HARNESS_STRICT=1` (turn dispatch-leak warnings into errors), `MLC_FP16_KVCACHE=0` (opt out of fp16 KV cache; default ON), `MLC_FP16_ATTN=0` (opt out of fp16 MPS attention; default ON).
- **Hardware:** M3 Pro, 18 GB unified memory. Optimizations should assume this baseline.
- **Backend dispatch:** routes through `MetalExecutor::shouldUseFor(node)` which folds in `node.backend == Metal`, `isAvailable()`, and `KernelDescriptorRegistry::forceCpu()`. Any new dispatch site that doesn't use this helper is a bug.
