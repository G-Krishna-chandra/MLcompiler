# GGUF → MLIR module emitter (Phase 2, P3).
# Links MLIRMLCDialect (the mlc dialect lib) and mlc_lib (for GGUFLoader).

add_library(MLCEmit STATIC
    GGUFToMLIR.cpp)

target_include_directories(MLCEmit PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_BINARY_DIR})

target_link_libraries(MLCEmit PUBLIC
    MLIRMLCDialect
    MLIRFuncDialect
    mlc_lib)
