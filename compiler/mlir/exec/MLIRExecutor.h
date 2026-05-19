#pragma once

#include "compiler/frontends/gguf_loader.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <mlx/array.h>

#include <cstdint>
#include <vector>

namespace mlir::mlc::exec {

// Walks a mlc-dialect ModuleOp and executes it on MLX. Weight tensors come
// from a GGUFLoader (matched by the `mlc.name` arg attribute the emitter
// attaches). Result of the inference function is realized to a host fp32
// buffer.
class MLIRExecutor {
public:
  MLIRExecutor(::mlir::ModuleOp module,
               const ::mlc::frontend::GGUFLoader &loader);

  // Run the (single) func.func in the module. token_ids is the prompt,
  // positions is parallel (typically 0..seq-1 for prefill from start).
  // Returns the function's last result as a flat fp32 buffer + shape.
  struct RunResult {
    std::vector<float> data;
    std::vector<int> shape;
  };
  RunResult run(const std::vector<int32_t> &token_ids,
                const std::vector<int32_t> &positions);

private:
  ::mlir::ModuleOp module_;
  const ::mlc::frontend::GGUFLoader &loader_;
};

} // namespace mlir::mlc::exec
