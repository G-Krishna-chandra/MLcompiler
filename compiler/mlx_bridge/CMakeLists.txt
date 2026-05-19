# MLX bridge: standalone smoke test that proves MLX is callable from our
# codebase. Lowering passes for the dialect will later emit MLX ops the same
# way this test does.

if(NOT EXISTS "${MLC_MLX_ROOT}/include/mlx/mlx.h")
    message(STATUS "MLX bridge: MLX not found at ${MLC_MLX_ROOT}, skipping")
    return()
endif()

add_executable(mlx_matmul_test mlx_matmul_test.cpp)

# MLX public headers use C++20 idioms (defaulted operator==, etc.). Keep the
# rest of the project at C++17 — only bump this target.
set_target_properties(mlx_matmul_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)

target_include_directories(mlx_matmul_test PRIVATE ${MLC_MLX_ROOT}/include)

target_link_libraries(mlx_matmul_test PRIVATE
    ${MLC_MLX_ROOT}/lib/libmlx.dylib)

if(APPLE)
    target_link_libraries(mlx_matmul_test PRIVATE
        "-framework Metal"
        "-framework MetalPerformanceShaders"
        "-framework Foundation"
        "-framework Accelerate")
endif()

# Wire into CTest so it runs alongside the existing test suite.
if(BUILD_TESTS)
    add_test(NAME mlx_matmul_smoke COMMAND mlx_matmul_test)
endif()
