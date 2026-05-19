# ANE/CoreML investigation (S4)

The Phase-2 moat is heterogeneous compilation: GPU for batched / throughput
work, ANE for sequential / latency-bound work. S4 measures whether that
second half is real — can we actually dispatch a single-token matmul to ANE
through CoreML, and does it beat MLX-on-GPU at the same shape?

**Answer: yes, by 3-5x on attention-projection shapes.**

## How to reproduce

```bash
# 1. Generate the .mlpackage with coremltools (needs Python ≤ 3.13).
python3.12 -m venv /tmp/coreml-venv
/tmp/coreml-venv/bin/pip install coremltools
/tmp/coreml-venv/bin/python compiler/coreml/gen_matmul_model.py \
    /tmp/matmul_2048.mlpackage 1 2048 2048

# 2. Time CoreML/ANE.
build/bin/coreml_matmul_probe /tmp/matmul_2048.mlpackage 1 2048 2048 200

# 3. Time MLX-on-GPU at the same shape.
build/bin/mlx_matmul_probe 1 2048 2048 200
```

## Measured single-matmul latency

M-series Mac, fp16 matmul, M=1 (decode-style single-batch), 200 iters
averaged after warm-up:

| K     | N     | CoreML/ANE | MLX/GPU | ANE win |
|-------|-------|-----------:|--------:|--------:|
| 2048  | 256   |   49 µs    | 178 µs  | 3.6x    |
| 2048  | 2048  |   55 µs    | 286 µs  | 5.2x    |
| 2048  | 5632  |  367 µs    | 344 µs  | 0.94x   |

ANE dominates at attention-projection shapes (Q, K, V, O for TinyLlama all
fall in the K=2048, N≤2048 band). ANE and GPU are within noise at the
larger FFN shape (N=5632). The crossover lands around N≈4096 for this
input dim.

## Integration path

The CoreML C++ surface is Objective-C; we already use `.mm` files
elsewhere (`compiler/runtime/metal_runtime.mm`), so this is no new
language. The two pieces are:

1. **Offline compile.** `compiler/coreml/gen_matmul_model.py` is a
   templated MIL-builder script using coremltools. It runs at build time
   (or model-prep time) and writes a `.mlpackage` for each (K, N) shape
   the compiler wants on ANE. `coremltools` itself is Python and needs
   ≤ 3.13 — the binary proxy modules aren't packaged for 3.14 yet (we
   tried). `brew install python@3.12` is the workaround for now.

2. **Runtime dispatch.** `compiler/coreml/coreml_matmul_probe.mm` loads
   a compiled `.mlmodelc` via `MLModel` with
   `computeUnits = MLComputeUnitsCPUAndNeuralEngine` and runs
   `predictionFromFeatures:`. The MLX-side activation array (fp16
   contiguous) flows into an `MLMultiArray`; the result MLMultiArray's
   `dataPointer` can be wrapped back into an `mx::array` without a copy
   if the strides match. The same `.mm` will be the basis for the actual
   ANE backend in the next session.

The full lowering pass that routes mlc.matmul → CoreML model invocation
for `target_device = ane` is future work. S5 (the next commit) adds the
scheduling annotations the lowering will key off.

## Files

- `gen_matmul_model.py` — Python script that builds an `.mlpackage` for
  a given (M, K, N) fp16 matmul. Run inside a venv with coremltools.
- `coreml_matmul_probe.mm` — Objective-C++ binary that loads the
  `.mlpackage` and times `predictionFromFeatures:` calls on ANE.
- `mlx_matmul_probe.cpp` — Pure-C++ companion that times the same
  shape through `mx::matmul` on MLX-GPU.
- `CMakeLists.txt` — builds both probe binaries (macOS only).

## Caveats

- Per-call latency includes MLX/CoreML dispatch overhead, not just the
  kernel itself. ANE's much smaller dispatch cost is part of the win.
- The MLX probe does one `eval()` per matmul, which forces a sync; in
  the actual compiler path, MLX's lazy graph batches the entire forward
  before eval, hiding some of the GPU per-call overhead. So end-to-end
  the ratio at the *matmul level* is smaller than the table above
  suggests — but for shapes where ANE is 5x ahead, even at half the
  margin we'd come out clearly ahead on attention projections.
