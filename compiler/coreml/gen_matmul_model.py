#!/usr/bin/env python3
"""Generate a CoreML model that does fp16 matmul with a baked weight.

Usage:
  python3 gen_matmul_model.py <out_path.mlpackage> <M> <K> <N> [weight.bin]

If weight.bin is given, it must contain K*N fp16 values in row-major
[K, N] layout. Otherwise a deterministic seeded random weight is used
(for the S4-style timing probes that don't care about correctness).
"""
import os
import sys
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types as mil_types


def load_weight(path, K, N):
    """Read a raw fp16 binary blob of shape [K, N], row-major."""
    arr = np.fromfile(path, dtype=np.float16)
    if arr.size != K * N:
        raise SystemExit(
            f"weight {path}: expected {K*N} fp16 elements, got {arr.size}")
    return arr.reshape(K, N)


def build_matmul_program(M, K, N, weight):
    """Build a single-matmul MIL program: x @ w (both fp16)."""
    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(M, K), dtype=mil_types.fp16),
        ]
    )
    def prog(x):
        return mb.matmul(x=x, y=weight, transpose_x=False, transpose_y=False)
    return prog


def main():
    if len(sys.argv) < 5:
        sys.stderr.write(__doc__)
        sys.exit(2)
    out_path = sys.argv[1]
    M, K, N = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    weight_path = sys.argv[5] if len(sys.argv) >= 6 else None
    if weight_path:
        w = load_weight(weight_path, K, N)
        print(f"loaded weight: shape={w.shape} dtype={w.dtype} "
              f"w[0,0..3]={w[0, :3]} w[1,0..3]={w[1, :3]}")
    else:
        rng = np.random.default_rng(0xC0DE)
        w = rng.standard_normal((K, N)).astype(np.float16)

    prog = build_matmul_program(M, K, N, w)
    # MLProgram-format CoreML model. precision=FP16 keeps the model
    # eligible for ANE; FP32 would force CPU/GPU fallback for many ops.
    model = ct.convert(
        prog,
        source="milinternal",
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        # `compute_units=ALL` is the default; saved into the model so
        # runtime loaders pick it up.
        compute_units=ct.ComputeUnit.ALL,
    )
    # `save()` accepts a .mlpackage directory path on macOS 13+.
    model.save(out_path)
    print(f"wrote {out_path} (M={M} K={K} N={N})")


if __name__ == "__main__":
    main()
