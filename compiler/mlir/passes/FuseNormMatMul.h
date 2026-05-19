#pragma once

#include "mlir/IR/BuiltinOps.h"

namespace mlir::mlc {

// Fuse mlc.norm followed by mlc.matmul into mlc.fused_norm_matmul.
//
// Pattern: any mlc.matmul whose %x operand is defined by an mlc.norm op
// becomes a fused op carrying both. The original norm is recomputed inside
// the fused op (cheap relative to matmul). When all of a norm's matmul
// users have been fused, the norm becomes dead and is erased.
//
// Returns the number of matmul ops fused.
unsigned fuseNormMatMul(ModuleOp module);

} // namespace mlir::mlc
