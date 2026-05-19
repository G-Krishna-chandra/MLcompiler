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
