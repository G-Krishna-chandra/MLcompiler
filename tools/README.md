# tools

Out-of-tree binaries built alongside `mlc`. None are required for normal use of
the runtime — they exist to support parity investigations and one-off
diagnostics. Each has its own `add_executable` line in the top-level
`CMakeLists.txt`.

## `mlc_dump_kv_cache`

Dump per-step Q/K/V tap tensors and the final K/V cache contents to disk for
offline analysis against expected values. Runs prefill of a prompt on the CPU
backend, writes every registered tap (one F32 file per step) and the
end-of-run K/V cache for block 0. Used in conjunction with
`mlc compare --metal-vs-cpu` when localizing kernel bugs to the cache write
path.

Usage:

    mlc_dump_kv_cache --model PATH --prompt "..." --out-dir DIR \
                     [--tap blk.0.attn_k.out] [--tap blk.0.attn_v.out] ...
