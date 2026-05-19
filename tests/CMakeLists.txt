# GoogleTest setup using FetchContent
include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Test sources
set(TEST_SOURCES
    test_frontend.cpp
    test_gguf_loader.cpp
    test_gguf_values.cpp
    test_gguf_to_ir.cpp
    test_output_weight_transpose.cpp
    test_gguf_integration.cpp
    test_ir.cpp
    test_passes.cpp
    test_codegen.cpp
    test_runtime.cpp
    test_execution_graph.cpp
    test_ir_pipeline.cpp
    test_paged_kv.cpp
    test_batched_executor.cpp
    test_batched_elementwise.cpp
    test_scheduler.cpp
    test_scheduler_e2e.cpp
    test_paged_flash.cpp
    test_batched_paged_g1.cpp
    test_batched_walker_h1.cpp
)

# Create test executable
add_executable(mlc_tests ${TEST_SOURCES})

target_link_libraries(mlc_tests PRIVATE
    mlc_lib
    GTest::gtest
    GTest::gtest_main
    GTest::gmock
)
if(APPLE)
    target_link_libraries(mlc_tests PRIVATE
        "-framework Metal"
        "-framework MetalPerformanceShaders"
        "-framework Foundation"
        "-framework CoreGraphics"
        "-framework QuartzCore"
        "-framework Accelerate")
endif()

target_include_directories(mlc_tests PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/compiler
)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(mlc_tests)

# MLIR dialect round-trip test (only when MLIR is enabled).
if(MLC_ENABLE_MLIR AND TARGET mlc-opt)
    add_test(NAME mlir_dialect_roundtrip
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/mlir/test_dialect_roundtrip.sh
                $<TARGET_FILE:mlc-opt>
                ${CMAKE_CURRENT_SOURCE_DIR}/mlir/tinyllama_layer.mlir)
endif()

# GGUF → MLIR emission test on TinyLlama. Skips gracefully when the model
# isn't present (see the .sh script).
if(MLC_ENABLE_MLIR AND TARGET mlc-emit AND TARGET mlc-opt)
    add_test(NAME mlir_emit_tinyllama
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/mlir/test_emit_tinyllama.sh
                $<TARGET_FILE:mlc-emit>
                $<TARGET_FILE:mlc-opt>
                ${CMAKE_SOURCE_DIR}/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf)
endif()

# Compiler-path tests (require MLX + GGUFLoader + dialect). These live in a
# separate gtest binary because they need C++20 (MLX headers) — the main
# mlc_tests target is C++17 to keep the existing runtime build undisturbed.
if(MLC_ENABLE_MLIR AND TARGET MLCExec)
    add_executable(mlir_exec_tests
        mlir/test_mlx_matmul_lowering.cpp
        mlir/test_compiler_path_forward.cpp
        mlir/test_norm_matmul_fusion.cpp)
    set_target_properties(mlir_exec_tests PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF)
    target_link_libraries(mlir_exec_tests PRIVATE
        MLCExec
        MLCEmit
        GTest::gtest
        GTest::gtest_main)
    target_include_directories(mlir_exec_tests PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/compiler)
    add_test(NAME mlir_exec_q1_matmul COMMAND mlir_exec_tests)
endif()
