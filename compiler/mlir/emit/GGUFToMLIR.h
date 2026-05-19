#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace mlc::frontend { class GGUFLoader; }

namespace mlir::mlc {

// Emit an MLIR module in the mlc dialect from a loaded GGUF model.
//
// Currently supports the `llama` architecture (covers TinyLlama, Llama 3.x).
// The emitted module contains one func.func @inference whose argument list is:
//
//   %ids        : tensor<?xi32>
//   %positions  : tensor<?xi32>
//   %token_embd : tensor<vocab x hidden x <quant>>
//   %output_norm: tensor<hidden x f32>      (RMSNorm gamma)
//   %output     : tensor<vocab x hidden x <quant>>   (lm_head weight)
//   then, per layer i in [0, n_layers):
//   %attn_norm  %attn_q  %attn_k  %attn_v  %attn_o
//   %ffn_norm   %ffn_gate %ffn_up %ffn_down
//
// The op chain inside the function follows the canonical pre-norm transformer
// pattern (norm → q/k/v matmul → attention → o matmul → add → norm → ffn → add)
// per layer, with mlc.embedding at entry and mlc.lm_head at exit.
//
// Throws std::runtime_error if the model lacks expected llama metadata or
// per-layer tensors.
OwningOpRef<ModuleOp> emitMLIR(MLIRContext &ctx,
                               const ::mlc::frontend::GGUFLoader &loader);

} // namespace mlir::mlc
