#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "compiler/mlir/dialect/MLCEnums.h.inc"
#include "compiler/mlir/dialect/MLCDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "compiler/mlir/dialect/MLCAttrs.h.inc"
