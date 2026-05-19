# Compiler-path MLIR walker / MLX bridge.
# Requires both the GGUF loader (from mlc_lib) and MLX. C++20 for MLX headers.

if(NOT MLC_HAVE_MLX)
    message(STATUS "compiler/mlir/exec: MLX disabled, skipping")
    return()
endif()

add_library(MLCExec STATIC
    Quantize.cpp
    MLXBuilder.cpp)

set_target_properties(MLCExec PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)

target_include_directories(MLCExec PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${MLC_MLX_ROOT}/include)

target_link_libraries(MLCExec PUBLIC
    mlc_lib
    ${MLC_MLX_ROOT}/lib/libmlx.dylib)

if(APPLE)
    target_link_libraries(MLCExec PUBLIC
        "-framework Metal"
        "-framework MetalPerformanceShaders"
        "-framework Foundation"
        "-framework Accelerate")
endif()
