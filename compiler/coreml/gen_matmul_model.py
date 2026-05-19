#!/usr/bin/env python3
"""Generate a minimal CoreML model that does fp16 matmul.

Used by the S4 investigation: tells us whether a CoreML-compiled matmul
can actually land on the Neural Engine, and what the latency looks like
vs MLX-on-GPU for the same shape.

Usage:
  python3 gen_matmul_model.py <out_path.mlpackage> <M> <K> <N>

Example:
  python3 gen_matmul_model.py /tmp/matmul.mlpackage 1 2048 2048
"""
import os
import sys
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types as mil_types


def build_matmul_program(M, K, N):
    """Build a single-matmul MIL program: x @ w (both fp16)."""
    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(M, K), dtype=mil_types.fp16),
        ]
    )
    def prog(x):
        # Bake the weight as a constant — that's how production models load
        # frozen Q/K/V/etc. into a CoreML graph.
        w = np.random.randn(K, N).astype(np.float16)
        return mb.matmul(x=x, y=w, transpose_x=False, transpose_y=False)
    return prog


def main():
    if len(sys.argv) != 5:
        sys.stderr.write(__doc__)
        sys.exit(2)
    out_path = sys.argv[1]
    M, K, N = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

    prog = build_matmul_program(M, K, N)
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
