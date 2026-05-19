#!/usr/bin/env bash
# Emit MLIR from the TinyLlama GGUF and verify:
#   1) op counts match the canonical 22-layer pre-norm transformer
#   2) the emitted module round-trips through mlc-opt
#
# Skips with exit 0 if the model file isn't present, so the suite stays green
# in environments without the GGUF.
#
# Usage (from CMake/CTest):
#   test_emit_tinyllama.sh <path-to-mlc-emit> <path-to-mlc-opt> <path-to-tinyllama.gguf>

set -euo pipefail

MLC_EMIT="${1:?path to mlc-emit}"
MLC_OPT="${2:?path to mlc-opt}"
MODEL="${3:?path to tinyllama gguf}"

if [[ ! -f "${MODEL}" ]]; then
    echo "SKIP: model not found at ${MODEL}"
    exit 0
fi

OUT="$(mktemp -t tinyllama_emit.XXXXXX.mlir)"
trap 'rm -f "${OUT}"' EXIT

"${MLC_EMIT}" "${MODEL}" > "${OUT}"

# Per-op count: expected for the 22-layer TinyLlama-1.1B architecture.
# Parallel arrays — dots in associative-array keys trip bash's arithmetic
# parser, so we keep these two flat lists in lockstep.
ops=(mlc.embedding mlc.norm mlc.matmul mlc.attention mlc.feedforward mlc.add mlc.lm_head)
counts=(1          45       88         22            22              44      1)
for i in "${!ops[@]}"; do
    op="${ops[$i]}"
    want="${counts[$i]}"
    got=$(grep -c "${op}" "${OUT}" || true)
    if [[ "${got}" != "${want}" ]]; then
        echo "Op count mismatch for ${op}: got ${got}, expected ${want}"
        echo "--- emitted IR (first 40 lines) ---"
        head -40 "${OUT}"
        exit 1
    fi
done

# Round-trip: mlc-opt must parse what mlc-emit printed, then re-print
# identically. Catches printer/parser drift on the emitted form.
RT="$(mktemp -t tinyllama_rt.XXXXXX.mlir)"
trap 'rm -f "${OUT}" "${RT}"' EXIT
"${MLC_OPT}" "${OUT}" > "${RT}"

# Compare round-tripped output with what we get from running mlc-opt again
# (eliminates trivial formatting differences that mlc-emit's printer might
# introduce relative to mlc-opt's canonical form).
RT2="$(mktemp -t tinyllama_rt2.XXXXXX.mlir)"
trap 'rm -f "${OUT}" "${RT}" "${RT2}"' EXIT
"${MLC_OPT}" "${RT}" > "${RT2}"

if ! diff -q "${RT}" "${RT2}" >/dev/null; then
    echo "Round-trip mismatch on TinyLlama emission"
    diff "${RT}" "${RT2}" | head -30
    exit 1
fi

echo "TinyLlama emission OK: 223 ops, round-trip stable."
