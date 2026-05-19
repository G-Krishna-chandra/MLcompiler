// mlc-opt: an mlir-opt driver that registers the mlc dialect.
//
// Builds on top of MlirOptMain so we get every standard pass and dialect
// for free; we just preregister mlc so handwritten .mlir files using
// mlc.matmul, mlc.attention, etc. parse and round-trip.

#include "compiler/mlir/dialect/MLCDialect.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  ::mlir::DialectRegistry registry;
  ::mlir::registerAllDialects(registry);
  ::mlir::registerAllPasses();
  registry.insert<::mlir::mlc::MLCDialect>();
  return ::mlir::asMainReturnCode(
      ::mlir::MlirOptMain(argc, argv, "MLcompiler optimizer driver\n", registry));
}
