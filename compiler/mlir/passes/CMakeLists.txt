# MLcompiler MLIR passes (Phase 2). Each pass is a free function; we don't
# use mlir::Pass machinery yet — overkill for our 2-3 transforms.

add_library(MLCPasses STATIC
    FuseNormMatMul.cpp
    FuseQKVMatMul.cpp)

target_include_directories(MLCPasses PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_BINARY_DIR})

target_link_libraries(MLCPasses PUBLIC
    MLIRMLCDialect
    MLIRFuncDialect
    MLIRIR
    MLIRSupport)
