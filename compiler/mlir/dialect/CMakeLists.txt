# MLcompiler dialect: TableGen + library.
# This subdir is only added when MLIR is found at the top level.

set(LLVM_TARGET_DEFINITIONS MLCOps.td)
mlir_tablegen(MLCOps.h.inc -gen-op-decls)
mlir_tablegen(MLCOps.cpp.inc -gen-op-defs)
mlir_tablegen(MLCEnums.h.inc -gen-enum-decls)
mlir_tablegen(MLCEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(MLCAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=mlc)
mlir_tablegen(MLCAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=mlc)

set(LLVM_TARGET_DEFINITIONS MLCBase.td)
mlir_tablegen(MLCDialect.h.inc -gen-dialect-decls -dialect=mlc)
mlir_tablegen(MLCDialect.cpp.inc -gen-dialect-defs -dialect=mlc)

add_public_tablegen_target(MLCDialectIncGen)

add_mlir_dialect_library(MLIRMLCDialect
    MLCDialect.cpp
    MLCOps.cpp

    ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}

    DEPENDS
        MLCDialectIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRInferTypeOpInterface
        MLIRSideEffectInterfaces
)

# Generated .inc files land in this binary dir under the same relative path
# we use in our includes ("compiler/mlir/dialect/MLCDialect.h.inc"). Add the
# build root so those paths resolve.
target_include_directories(MLIRMLCDialect PUBLIC
    ${CMAKE_BINARY_DIR}
    ${CMAKE_SOURCE_DIR})
