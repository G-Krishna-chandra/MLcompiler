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
)

# Create test executable
add_executable(mlc_tests ${TEST_SOURCES})

target_link_libraries(mlc_tests PRIVATE
    mlc_lib
    GTest::gtest
    GTest::gtest_main
    GTest::gmock
)

target_include_directories(mlc_tests PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/compiler
)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(mlc_tests)
