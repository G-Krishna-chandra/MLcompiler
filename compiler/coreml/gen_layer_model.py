#!/usr/bin/env python3
"""Generate a CoreML model that packs two matmuls into one .mlpackage.

The model has two independent subgraphs sharing a single predict() call:
  out_qkv  = x_qkv  @ W_QKV  (Q ‖ K ‖ V concat; shape [M, K_qkv] @ [K_qkv, N_qkv])
  out_attn = x_attn @ W_O    (attn_output projection; [M, K_attn] @ [K_attn, N_attn])

U1 hypothesis: cycling many CoreML models per token costs ~590 µs/call
in the real walker (vs ~95 µs/call back-to-back on a single model). If
packing reduces the per-token call count, the boundary-flush overhead
amortizes.

Usage:
  gen_layer_model.py <out.mlpackage> <M> <K_qkv> <N_qkv> <K_attn> <N_attn> \\
      <w_qkv.bin> <w_o.bin>

w_qkv.bin: K_qkv * N_qkv fp16 values, [K_qkv, N_qkv] row-major.
w_o.bin:   K_attn * N_attn fp16 values, [K_attn, N_attn] row-major.
"""
import os
import sys
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types as mil_types


def load_weight(path, K, N):
    arr = np.fromfile(path, dtype=np.float16)
    if arr.size != K * N:
        raise SystemExit(
            f"weight {path}: expected {K*N} fp16 elements, got {arr.size}")
    return arr.reshape(K, N)


def build_program(M, K_qkv, N_qkv, K_attn, N_attn, w_qkv, w_o):
    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(M, K_qkv), dtype=mil_types.fp16),
            mb.TensorSpec(shape=(M, K_attn), dtype=mil_types.fp16),
        ]
    )
    def prog(x_qkv, x_attn):
        out_qkv = mb.matmul(x=x_qkv, y=w_qkv,
                            transpose_x=False, transpose_y=False,
                            name="out_qkv")
        out_attn = mb.matmul(x=x_attn, y=w_o,
                             transpose_x=False, transpose_y=False,
                             name="out_attn")
        return out_qkv, out_attn
    return prog


def main():
    if len(sys.argv) < 9:
        sys.stderr.write(__doc__)
        sys.exit(2)
    out_path = sys.argv[1]
    M = int(sys.argv[2])
    K_qkv, N_qkv = int(sys.argv[3]), int(sys.argv[4])
    K_attn, N_attn = int(sys.argv[5]), int(sys.argv[6])
    w_qkv_path = sys.argv[7]
    w_o_path = sys.argv[8]

    w_qkv = load_weight(w_qkv_path, K_qkv, N_qkv)
    w_o = load_weight(w_o_path, K_attn, N_attn)
    print(f"loaded W_QKV {w_qkv.shape} {w_qkv.dtype}, "
          f"W_O {w_o.shape} {w_o.dtype}")

    prog = build_program(M, K_qkv, N_qkv, K_attn, N_attn, w_qkv, w_o)
    model = ct.convert(
        prog,
        source="milinternal",
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
    )
    model.save(out_path)
    print(f"wrote {out_path} "
          f"(M={M} K_qkv={K_qkv} N_qkv={N_qkv} K_attn={K_attn} N_attn={N_attn})")


if __name__ == "__main__":
    main()
