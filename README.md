# MLcompiler

A from-scratch C++17 ML compiler and runtime that runs GGUF LLMs on Apple
Silicon. Lifts GGUF tensors into a structured IR, schedules an execution
graph across CPU + Metal backends, and runs Llama-class quantized LLMs
token-by-token with a paged KV cache and continuous batching.

## What this is for

The single-stream perf race on Apple Silicon is owned by `llama.cpp`:
~70-100 tok/s on TinyLlama 1.1B Q4_0 on an M3 Pro, single request,
greedy decode. That ceiling is real and hard-won.

llama.cpp does not implement **continuous batching** — multiple
in-flight requests at different sequence positions sharing the GPU,
each streaming tokens independently, with new requests admitted as
old ones finish. On Apple Silicon there is no production runtime
that does this today. That is the wedge.

This repo demonstrates a working continuous-batching runtime for
Llama-class models on Apple Silicon, end-to-end:

- Paged KV cache (per-request page tables backed by per-layer Metal
  buffers).
- Custom paged flash-attention kernel in MSL — single dispatch
  serves all in-flight requests.
- Op-by-op batched walker with one Metal command buffer per forward
  pass and persistent GPU-resident intermediate state.
- Iteration-level scheduler with FIFO admission, EOS handling, and
  per-request streaming.
- `mlc serve` CLI for end-to-end multi-request serving.

## Performance

TinyLlama 1.1B Q4_0, M3 Pro:

### Single-stream (one request)

| Path | tok/s | ms/pass |
|---|---|---|
| `mlc chat-repl` (fuse-mode, single CB + persistent buffers) | ~50 | ~20 |
| llama.cpp (reference) | 70-100 | 10-14 |

mlc single-stream is bounded by the Q4_0 mat-vec kernel and the MPS
attention path; closing the gap to llama.cpp is a kernel-rewrite arc,
not a control-plane arc. Numerical parity vs CPU is held to cosine
1.000000 across every layer boundary on every commit (`mlc compare`).

### Continuous batching (multiple concurrent requests)

`mlc serve --paged --benchmark` on the same model + chip:

| batch | aggregate tok/s | per-request tok/s | scaling vs N=1 |
|------:|----------------:|------------------:|---------------:|
| 1     |            45.1 |              45.1 |          1.00× |
| 2     |            74.9 |              37.8 |          1.66× |
| 4     |            87.7 |              22.4 |          1.94× |
| 8     |            96.4 |              12.6 |          2.14× |

Aggregate throughput at batch≥4 matches or exceeds llama.cpp's
single-stream ceiling, while serving multiple concurrent requests.
That gap — *aggregate* throughput at batch — is the wedge.

llama.cpp at batch=4 is still ~70-100 tok/s total, because it serves
one request at a time. mlc at batch=4 is ~88 tok/s total *across
four concurrent requests*. The architectural gain is not about
beating llama.cpp at the kernel level; it's about being able to
serve N requests in parallel at all.

### Reproduce

```bash
# single-stream
MLC_FUSE_LAYER=1 ./build/bin/mlc chat-repl models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf

# batched scaling sweep
./build/bin/mlc serve models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
    --benchmark --benchmark-batches 1,2,4,8 --max-tokens 32
```

## Building

### Prerequisites

- macOS on Apple Silicon (M-series). The Metal backend is the
  primary target; CPU-only build works but is slow.
- CMake 3.15+ and a C++17 compiler (Apple clang from Xcode is fine).
- Optional: `llama.cpp` checkout next to this repo for the parity
  harness's `--vs-llamacpp` reference dump path and the bundled
  llama.cpp tokenizer.

### Build

```bash
mkdir build && cd build
cmake ..
make -j8
```

Produces `build/bin/mlc` (the CLI) and `build/bin/mlc_tests` (the
gtest harness).

### Tests

```bash
cd build && ctest --output-on-failure
# or directly:
./bin/mlc_tests
```

The test suite covers IR lowering, GGUF loading, kernel parity vs
CPU reference, paged KV scatter/gather, the batched walker, the
scheduler, and end-to-end greedy-decode match across batch sizes.

## CLI tour

### Single-stream chat (`mlc chat-repl`)

Interactive REPL. Fuse-mode (`MLC_FUSE_LAYER=1`) keeps one Metal
command buffer open across the entire forward pass and chains
GPU-resident output buffers between ops. KV cache is reused across
turns via prefix matching — turn N+1 only prefills the new suffix,
not the whole conversation.

```bash
MLC_FUSE_LAYER=1 ./build/bin/mlc chat-repl models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
    --max-new 64 --temperature 0.8

You: Hello
[timing] prefill=798 ms (24 prompt tokens, 30 kv-reused, 33.3 ms/tok)
        generated=8 tok in 152.8 ms (19.1 ms/tok, 52.4 tok/s)
```

`MLC_PROFILE_NODES=1` adds a per-op breakdown each turn.
`MLC_KV_REUSE_DEBUG=1` logs the LCP detection per turn.

### Continuous-batch serve (`mlc serve`)

Multi-prompt streaming with iteration-level scheduling. Add
`--paged` to route attention through the paged-flash kernel.

```bash
./build/bin/mlc serve models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --paged \
    --prompt "Capital of France?" \
    --prompt "Capital of Japan?" \
    --prompt "Capital of Italy?" \
    --prompt "Capital of Germany?" \
    --max-tokens 32 --batch-size 4
```

Each request streams tokens with a `[req N]` prefix; the trailer
prints aggregate tok/s across all in-flight requests.

`--benchmark` turns it into a standardized scaling sweep with a
fixed prompt:

```bash
./build/bin/mlc serve models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
    --benchmark --benchmark-batches 1,2,4,8 --max-tokens 32
```

### Parity harness (`mlc compare`)

Runs the same prompt through two execution paths in-process and
diffs every layer-boundary tensor (embedding, per-block
attn_output / residual_1 / ffn_down / residual_2, final_norm,
logits). Used as the load-bearing correctness gate on every perf
commit.

```bash
# CPU vs Metal — catches kernel regressions
./build/bin/mlc compare --metal-vs-cpu models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
    --prompt "The capital of France is"

# mlc CPU vs llama.cpp — catches drift from upstream reference
./build/bin/mlc compare --vs-llamacpp models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
    --prompt "The capital of France is" \
    --reference-dir logs/llamacpp_dump/
```

Cosine, max abs diff, mean abs diff, RMS reported per tensor;
logits row also shows top-k overlap. Exits non-zero on any
divergence past the configured thresholds.

### Diagnostics

| Command | Purpose |
|---|---|
| `mlc inspect <gguf>` | Metadata + tensor listing |
| `mlc plan <gguf>` | Build and print the execution graph |
| `mlc run --execute <gguf> <token>` | Single-token forward dispatch |
| `mlc decode <gguf> <ids>` | Token-by-token decode loop |
| `mlc capabilities` | Runtime + Metal capability dump |

## Architecture

```
  GGUF file                      compiler/frontends/gguf_loader
       ↓
  IR (operator graph)            compiler/ir/, compiler/passes/
       ↓
  Execution graph                compiler/runtime/execution_plan_builder
       ↓
  ┌─────────────────┬────────────────────────────┐
  │ Single-stream   │ Continuous batching         │
  │ ExecutionExec   │ BatchedExecutor + Walker    │
  │ (chat-repl,     │ (serve, scheduler)          │
  │  decode, compare)                              │
  └────────┬────────┴────────┬───────────────────┘
           ↓                 ↓
  Metal kernels (mm Q4_0/Q6_K, RMSNorm, attention,
                 paged-flash, scatter, rope, silu*mul, add)
  CPU fallback (Accelerate, vDSP)
```

### Single-stream path

`ExecutionExecutor::run()` walks the execution graph node-by-node.
With `MLC_FUSE_LAYER=1`, fusable ops (Norm, Q4_0 matmul, Add,
Slice) encode onto a shared `MTLCommandBuffer` opened once at the
start of the forward pass; metal-eligible non-fusable ops
(Attention via MPS, FeedForward) defer their dispatches onto the
same CB and chain their outputs via a `pass_outputs` map for
zero-CPU-roundtrip handoff. The CB commits + waits once at the end
of the pass.

This path is the parity-harness ground truth — every kernel is
asserted byte-equal to a hand-written CPU reference.

### Batched / paged path

`BatchedExecutor` + `BatchedWalker` (compiler/runtime/) implement
op-by-op batched dispatch across N requests:

- **Paged KV cache**: `PagePool` (logical free list of page IDs),
  `RequestKVState` (per-request page table), `PagedKVStorage`
  (per-layer Metal-backed bulk page memory). New tokens grab the
  next slot; pages can be released and reused as requests finish.
- **Batched kernels**: `q4_0_matmul_v3_batched`,
  `q6_k_matmul_v3_batched`, `rms_norm_kernel_v2_batched`,
  element-wise `add` / `silu_mul` — all take a batch dimension in
  the grid and process all N requests in one dispatch.
- **`paged_flash_attention`**: 1D grid `batch * num_heads`, online
  softmax, fp16 K/V → fp32 accumulator. One dispatch per layer
  serves all in-flight requests.
- **Single CB per pass**: BatchedWalker opens one
  `MTLCommandBuffer`, encodes ~660 dispatches across 22 layers,
  commits + waits once. Persistent named buffers (`w_residual`,
  `w_qkv`, `w_attn_mix`, etc.) survive across passes and carry
  state op-to-op without CPU roundtrip.
- **Scheduler**: `compiler/runtime/kernel_scheduler` runs requests
  to completion via iteration-level scheduling — every step picks
  the live batch, calls the walker, streams tokens, releases
  pages on EOS.

## Ground rules for changes

The parity harness is load-bearing infrastructure. Every perf
commit goes through:

1. `mlc compare --metal-vs-cpu` cosine ≥ 0.999 at every block.
2. `mlc compare --vs-llamacpp` cosine ≥ 0.9999 at every block.
3. Greedy-decode coherence on `chat-repl`.
4. `mlc serve --paged --benchmark` no batched-path regression.

The harness's assertions are not loosened. If a change disagrees
with the harness, the change is wrong.

See `CLAUDE.md` for the full operating rules and `ROADMAP.md` for
the perf sequence.

## License

See LICENSE.
