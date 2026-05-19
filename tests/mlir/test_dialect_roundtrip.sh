#!/usr/bin/env bash
# Round-trip test: parse the fixture, print it, then re-parse what we printed.
# A round-trip means the dialect's printer is the inverse of its parser.
#
# Usage (from CMake/CTest):
#   test_dialect_roundtrip.sh <path-to-mlc-opt> <path-to-fixture.mlir>
set -euo pipefail

MLC_OPT="${1:?path to mlc-opt}"
FIXTURE="${2:?path to .mlir fixture}"

# First pass: parse + verify + print.
FIRST=$("${MLC_OPT}" "${FIXTURE}")

# Second pass: re-parse what we just printed. If the printer drifts from the
# parser, mlir-opt will reject this on the second pass.
SECOND=$(echo "${FIRST}" | "${MLC_OPT}")

if [[ "${FIRST}" != "${SECOND}" ]]; then
    echo "Round-trip mismatch between first and second pass."
    diff <(echo "${FIRST}") <(echo "${SECOND}") || true
    exit 1
fi

# Sanity: the printed IR must contain each dialect op we expect.
for op in mlc.norm mlc.matmul mlc.attention mlc.feedforward mlc.add mlc.embedding mlc.lm_head; do
    if ! echo "${FIRST}" | grep -q "${op}"; then
        echo "Round-trip output is missing ${op}"
        echo "${FIRST}"
        exit 1
    fi
done

echo "Round-trip OK."
