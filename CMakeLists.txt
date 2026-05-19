cmake_minimum_required(VERSION 3.15)
project(MLCompiler VERSION 1.0.0 LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type defaults to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0)
    else()
        add_compile_options(-O3)
    endif()
endif()

# Include paths
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/compiler)

# Enable CTest early so add_test() calls from any subdir register. The
# `BUILD_TESTS` option below still gates whether the gtest suite is built.
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
endif()

# Add compiler subdirectories
add_subdirectory(compiler)

# MLX bridge — Phase-2 dependency for the future `mlc.matmul` GPU lowering.
# Built when MLX is installed (Homebrew default on Apple silicon).
option(MLC_ENABLE_MLX "Build the MLX bridge + smoke test" ON)
if(MLC_ENABLE_MLX)
    if(APPLE AND NOT MLC_MLX_ROOT AND EXISTS "/opt/homebrew/opt/mlx/include/mlx/mlx.h")
        set(MLC_MLX_ROOT "/opt/homebrew/opt/mlx" CACHE PATH "MLX install prefix")
    endif()
    if(MLC_MLX_ROOT AND EXISTS "${MLC_MLX_ROOT}/include/mlx/mlx.h")
        add_subdirectory(compiler/mlx_bridge)
        set(MLC_HAVE_MLX TRUE)
        message(STATUS "  MLX: ${MLC_MLX_ROOT}")
    else()
        message(STATUS "  MLX: not found, skipping mlx_bridge (set -DMLC_MLX_ROOT=...)")
        set(MLC_ENABLE_MLX OFF)
    endif()
endif()

# MLIR/LLVM integration. The Phase-2 compiler lives under compiler/mlir/.
# On macOS we expect Homebrew's llvm formula (ships MLIR). The user can
# override via -DCMAKE_PREFIX_PATH=/path/to/llvm. Configured BEFORE tests/
# so the mlc-opt target is visible when tests/CMakeLists.txt registers the
# round-trip ctest entry.
option(MLC_ENABLE_MLIR "Build the MLcompiler MLIR dialect + mlc-opt" ON)
if(MLC_ENABLE_MLIR)
    if(APPLE AND NOT MLIR_DIR AND NOT LLVM_DIR AND
       EXISTS "/opt/homebrew/opt/llvm/lib/cmake/mlir")
        set(MLIR_DIR "/opt/homebrew/opt/llvm/lib/cmake/mlir" CACHE PATH "")
        set(LLVM_DIR "/opt/homebrew/opt/llvm/lib/cmake/llvm" CACHE PATH "")
    endif()
    find_package(MLIR CONFIG)
    if(MLIR_FOUND)
        find_package(LLVM REQUIRED CONFIG)
        list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
        list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
        include(TableGen)
        include(AddLLVM)
        include(AddMLIR)
        include(HandleLLVMOptions)
        include_directories(${LLVM_INCLUDE_DIRS})
        include_directories(${MLIR_INCLUDE_DIRS})
        add_subdirectory(compiler/mlir)
        message(STATUS "  MLIR: ${MLIR_DIR}")
    else()
        message(STATUS "  MLIR: not found, skipping compiler/mlir (set -DMLIR_DIR=...)")
        set(MLC_ENABLE_MLIR OFF)
    endif()
endif()

# Add tests if enabled (BUILD_TESTS option was declared and enable_testing()
# called above so subdirs registered earlier can already add tests).
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Diagnostic binary for Metal availability
add_executable(metal_diag tools/metal_diag.mm)
target_link_libraries(metal_diag PRIVATE "-framework Foundation" "-framework Metal" "-framework CoreGraphics" "-framework QuartzCore")

# Standalone Metal hazard reproducer for the N1 single-encoder design work.
# Runs a sequence of single-encoder, multi-encoder, fence, and useResource
# variants of a "scale twice and verify the product" computation; tells us
# which sync primitives are actually necessary.
add_executable(metal_hazard_test tools/metal_hazard_test.mm)
target_link_libraries(metal_hazard_test PRIVATE "-framework Foundation" "-framework Metal")

# llama.cpp activation dumper for the parity harness.
# Builds only when libllama is present at the conventional sibling path.
set(LLAMA_CPP_ROOT ${CMAKE_SOURCE_DIR}/../llama.cpp)
set(LLAMA_CPP_LIB  ${LLAMA_CPP_ROOT}/build/bin/libllama.dylib)
if (EXISTS ${LLAMA_CPP_LIB})
    set(LLAMA_GGML_BASE ${LLAMA_CPP_ROOT}/build/bin/libggml-base.dylib)
    set(LLAMA_GGML      ${LLAMA_CPP_ROOT}/build/bin/libggml.dylib)
    add_executable(llamacpp_dump_activations tools/llamacpp_dump_activations.cpp)
    target_include_directories(llamacpp_dump_activations PRIVATE
        ${LLAMA_CPP_ROOT}/include
        ${LLAMA_CPP_ROOT}/ggml/include)
    target_link_libraries(llamacpp_dump_activations PRIVATE
        ${LLAMA_CPP_LIB}
        ${LLAMA_GGML}
        ${LLAMA_GGML_BASE})
endif()

# Diagnostic: dumps per-step CPU activations + final block-0 K/V cache to disk.
add_executable(mlc_dump_kv_cache tools/mlc_dump_kv_cache.cpp)
target_link_libraries(mlc_dump_kv_cache PRIVATE mlc_lib)
if(APPLE)
    target_link_libraries(mlc_dump_kv_cache PRIVATE
        "-framework Metal"
        "-framework MetalPerformanceShaders"
        "-framework Foundation"
        "-framework CoreGraphics"
        "-framework QuartzCore"
        "-framework Accelerate")
endif()

# Standalone benchmark: paged-flash vs sequential MPS attention.
add_executable(paged_flash_bench tools/paged_flash_bench.cpp)
target_link_libraries(paged_flash_bench PRIVATE mlc_lib)
if(APPLE)
    target_link_libraries(paged_flash_bench PRIVATE
        "-framework Metal"
        "-framework MetalPerformanceShaders"
        "-framework Foundation"
        "-framework CoreGraphics"
        "-framework QuartzCore"
        "-framework Accelerate")
endif()

message(STATUS "MLCompiler configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build tests: ${BUILD_TESTS}")
message(STATUS "  MLIR enabled: ${MLC_ENABLE_MLIR}")
message(STATUS "  MLX  enabled: ${MLC_ENABLE_MLX}")






