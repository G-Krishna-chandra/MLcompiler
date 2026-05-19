#ifndef MLC_OPS_TD
#define MLC_OPS_TD

include "MLCBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def MLC_MatMulOp : MLC_Op<"matmul", [Pure]> {
  let summary = "Batched (possibly quantized) matrix multiply: y = x @ w";
  let arguments = (ins
      MLC_FloatTensor:$x,
      MLC_QuantTensor:$w,
      MLC_DeviceAttr:$target_device,
      StrAttr:$quant_format,
      DefaultValuedAttr<BoolAttr, "false">:$transpose_b
  );
  let results = (outs MLC_FloatTensor:$y);
  let assemblyFormat = [{
      $x `,` $w attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def MLC_AttentionOp : MLC_Op<"attention", [Pure]> {
  let summary = "Paged flash attention (GQA)";
  let arguments = (ins
      MLC_FloatTensor:$q,
      MLC_FloatTensor:$k,
      MLC_FloatTensor:$v,
      MLC_IntTensor:$positions,
      MLC_DeviceAttr:$target_device,
      I64Attr:$num_heads,
      I64Attr:$num_kv_heads,
      I64Attr:$head_dim
  );
  let results = (outs MLC_FloatTensor:$out);
  let assemblyFormat = [{
      $q `,` $k `,` $v `,` $positions attr-dict
      `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def MLC_NormOp : MLC_Op<"norm", [Pure]> {
  let summary = "RMSNorm with fused gamma";
  let arguments = (ins
      MLC_FloatTensor:$x,
      MLC_FloatTensor:$gamma,
      MLC_DeviceAttr:$target_device,
      F32Attr:$epsilon
  );
  let results = (outs MLC_FloatTensor:$y);
  let assemblyFormat = [{
      $x `,` $gamma attr-dict `:` functional-type(operands, results)
  }];
}

def MLC_FeedForwardOp : MLC_Op<"feedforward", [Pure]> {
  let summary = "Fused SwiGLU FFN: down(silu(gate(x)) * up(x))";
  let arguments = (ins
      MLC_FloatTensor:$x,
      MLC_QuantTensor:$w_gate,
      MLC_QuantTensor:$w_up,
      MLC_QuantTensor:$w_down,
      MLC_DeviceAttr:$target_device,
      StrAttr:$quant_format
  );
  let results = (outs MLC_FloatTensor:$y);
  let assemblyFormat = [{
      $x `,` $w_gate `,` $w_up `,` $w_down attr-dict
      `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def MLC_AddOp : MLC_Op<"add",
    [Pure, SameOperandsAndResultType,
     DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Element-wise residual add";
  let arguments = (ins
      MLC_FloatTensor:$a,
      MLC_FloatTensor:$b,
      MLC_DeviceAttr:$target_device
  );
  let results = (outs MLC_FloatTensor:$y);
  let assemblyFormat = "$a `,` $b attr-dict `:` type($y)";
}

def MLC_EmbeddingOp : MLC_Op<"embedding", [Pure]> {
  let summary = "Token embedding table lookup";
  let arguments = (ins
      MLC_IntTensor:$ids,
      MLC_QuantTensor:$table,
      MLC_DeviceAttr:$target_device,
      StrAttr:$quant_format
  );
  let results = (outs MLC_FloatTensor:$y);
  let assemblyFormat = [{
      $ids `,` $table attr-dict `:` functional-type(operands, results)
  }];
}

def MLC_FusedNormMatMulOp : MLC_Op<"fused_norm_matmul", [Pure]> {
  let summary = "Fused RMSNorm + matmul: y = matmul(rms_norm(x, gamma, eps), w)";
  let description = [{
    Emitted by the norm+matmul fusion pass when one or more matmuls all
    consume the same RMSNorm result. The pass copies the norm into each
    fused op (the recomputation is cheap relative to the matmul); when all
    of a norm's matmul users have been fused, the original norm op is dead
    and can be DCE'd.
  }];
  let arguments = (ins
      MLC_FloatTensor:$x,
      MLC_FloatTensor:$gamma,
      MLC_QuantTensor:$w,
      MLC_DeviceAttr:$target_device,
      F32Attr:$epsilon,
      StrAttr:$quant_format,
      DefaultValuedAttr<BoolAttr, "false">:$transpose_b
  );
  let results = (outs MLC_FloatTensor:$y);
  let assemblyFormat = [{
      $x `,` $gamma `,` $w attr-dict
      `:` functional-type(operands, results)
  }];
}

def MLC_LMHeadOp : MLC_Op<"lm_head", [Pure]> {
  let summary = "Project hidden state to vocabulary logits";
  let arguments = (ins
      MLC_FloatTensor:$x,
      MLC_QuantTensor:$w,
      MLC_DeviceAttr:$target_device,
      StrAttr:$quant_format
  );
  let results = (outs MLC_FloatTensor:$logits);
  let assemblyFormat = [{
      $x `,` $w attr-dict `:` functional-type(operands, results)
  }];
}

#endif // MLC_OPS_TD
