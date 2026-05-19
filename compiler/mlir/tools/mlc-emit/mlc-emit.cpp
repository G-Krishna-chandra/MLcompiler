// mlc-emit: load a GGUF model and print its MLcompiler-dialect IR.
//
// Usage: mlc-emit <path/to/model.gguf>
//
// The printed IR has one func.func @inference taking ids, positions, and
// every weight tensor as arguments. The op chain matches the canonical
// pre-norm transformer block, repeated `block_count` times.

#include "compiler/frontends/gguf_loader.hpp"
#include "compiler/mlir/emit/GGUFToMLIR.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <stdexcept>
#include <string>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "Usage: %s <path/to/model.gguf>\n", argv[0]);
    return 2;
  }

  mlc::frontend::GGUFLoader loader(argv[1]);
  if (!loader.load()) {
    std::fprintf(stderr, "failed to load GGUF: %s\n", argv[1]);
    return 1;
  }

  // Note: emitMLIR throws std::runtime_error on a missing GGUF kv/tensor.
  // mlc-emit is compiled with -fno-exceptions (LLVM convention), so any
  // thrown exception will std::terminate — that's intentional: the GGUF is
  // either compatible or we fail loudly.
  mlir::MLIRContext ctx;
  auto module = mlir::mlc::emitMLIR(ctx, loader);
  if (mlir::failed(mlir::verify(*module))) {
    std::fprintf(stderr, "emitted module failed verification\n");
    return 1;
  }
  // Print in the registered (pretty) form. `assumeVerified` skips a second
  // verification pass during printing now that we've verified above.
  mlir::OpPrintingFlags flags;
  flags.assumeVerified();
  module->print(llvm::outs(), flags);
  llvm::outs() << "\n";
  return 0;
}
