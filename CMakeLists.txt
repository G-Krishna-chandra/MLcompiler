cmake_minimum_required(VERSION 3.15)
project(MLCompiler VERSION 1.0.0 LANGUAGES CXX)

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

# Add compiler subdirectories
add_subdirectory(compiler)

# Add tests if enabled
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Diagnostic binary for Metal availability
add_executable(metal_diag tools/metal_diag.mm)
target_link_libraries(metal_diag PRIVATE "-framework Foundation" "-framework Metal" "-framework CoreGraphics" "-framework QuartzCore")

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

# Future: MLIR/LLVM integration point
# find_package(MLIR REQUIRED CONFIG)
# find_package(LLVM REQUIRED CONFIG)

message(STATUS "MLCompiler configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build tests: ${BUILD_TESTS}")






