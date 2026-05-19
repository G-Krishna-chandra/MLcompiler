#pragma once

#include "mlir/IR/BuiltinOps.h"

namespace mlir::mlc {

// Annotate each op's `target_device` attribute based on a single-batch
// matmul-shape heuristic backed by the S4 measurements:
//
//   matmul-style ops with output last-dim ≤ 4096 → ANE  (3-5x faster
//   than MLX-GPU on Apple silicon at these shapes for batch=1)
//   matmul-style ops with output last-dim > 4096 → GPU
//   attention, norm, add, embedding, lm_head, feedforward → GPU
//
// The actual dispatch through CoreML is future work; this pass is the
// scheduling skeleton the lowering keys off. Returns the count of ops
// whose annotation changed.
unsigned scheduleDevices(ModuleOp module);

} // namespace mlir::mlc
