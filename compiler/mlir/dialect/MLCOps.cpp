#include "compiler/mlir/dialect/MLCOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace mlc {

namespace {

ShapedType asShaped(Type t) {
  return ::llvm::dyn_cast<ShapedType>(t);
}

} // namespace

LogicalResult MatMulOp::verify() {
  auto xTy = asShaped(getX().getType());
  auto wTy = asShaped(getW().getType());
  auto yTy = asShaped(getY().getType());
  if (!xTy || !wTy || !yTy)
    return emitOpError("operands and result must be shaped types");
  if (xTy.getRank() < 2 || wTy.getRank() < 2)
    return emitOpError("x and w must each have rank >= 2");
  if (yTy.getRank() != xTy.getRank())
    return emitOpError("result rank must match x rank");
  int64_t xInner = xTy.getShape().back();
  int64_t wInner =
      getTransposeB() ? wTy.getShape().back() : wTy.getShape()[wTy.getRank() - 2];
  if (!ShapedType::isDynamic(xInner) && !ShapedType::isDynamic(wInner) &&
      xInner != wInner) {
    return emitOpError("contraction-dim mismatch: x inner=")
           << xInner << " vs w inner=" << wInner;
  }
  return success();
}

LogicalResult AttentionOp::verify() {
  auto qTy = asShaped(getQ().getType());
  auto kTy = asShaped(getK().getType());
  auto vTy = asShaped(getV().getType());
  if (!qTy || !kTy || !vTy)
    return emitOpError("q/k/v must be shaped types");
  int64_t hd = getHeadDim();
  int64_t nh = getNumHeads();
  int64_t nkv = getNumKvHeads();
  if (hd <= 0 || nh <= 0 || nkv <= 0)
    return emitOpError("num_heads, num_kv_heads, head_dim must be > 0");
  if (nh % nkv != 0)
    return emitOpError("num_heads must be a multiple of num_kv_heads");
  int64_t qLast = qTy.getShape().back();
  int64_t kLast = kTy.getShape().back();
  if (!ShapedType::isDynamic(qLast) && qLast != nh * hd)
    return emitOpError("q last dim (")
           << qLast << ") must equal num_heads*head_dim (" << nh * hd << ")";
  if (!ShapedType::isDynamic(kLast) && kLast != nkv * hd)
    return emitOpError("k last dim (")
           << kLast << ") must equal num_kv_heads*head_dim (" << nkv * hd << ")";
  return success();
}

LogicalResult FeedForwardOp::verify() {
  auto xTy = asShaped(getX().getType());
  auto wG = asShaped(getWGate().getType());
  auto wU = asShaped(getWUp().getType());
  auto wD = asShaped(getWDown().getType());
  if (!xTy || !wG || !wU || !wD)
    return emitOpError("operands must be shaped types");
  if (wG.getRank() != 2 || wU.getRank() != 2 || wD.getRank() != 2)
    return emitOpError("gate/up/down weights must be rank-2 matrices");
  if (!ShapedType::isDynamic(wG.getShape()[0]) &&
      !ShapedType::isDynamic(wU.getShape()[0]) &&
      wG.getShape()[0] != wU.getShape()[0]) {
    return emitOpError("gate and up weights must share the input dimension");
  }
  return success();
}

// Note: AddOp::inferReturnTypes is auto-emitted by TableGen because of
// SameOperandsAndResultType + DeclareOpInterfaceMethods<InferTypeOpInterface>.

} // namespace mlc
} // namespace mlir
