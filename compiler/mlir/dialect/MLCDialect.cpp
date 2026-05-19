#include "compiler/mlir/dialect/MLCDialect.h"
#include "compiler/mlir/dialect/MLCOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "compiler/mlir/dialect/MLCEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "compiler/mlir/dialect/MLCAttrs.cpp.inc"

#define GET_OP_CLASSES
#include "compiler/mlir/dialect/MLCOps.cpp.inc"

#include "compiler/mlir/dialect/MLCDialect.cpp.inc"

namespace mlir {
namespace mlc {

void MLCDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "compiler/mlir/dialect/MLCAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "compiler/mlir/dialect/MLCOps.cpp.inc"
      >();
}

} // namespace mlc
} // namespace mlir
