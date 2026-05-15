# Compiler library sources
set(COMPILER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/frontends/frontend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/frontends/gguf_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/frontends/gguf_to_ir.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/frontends/gguf_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ir/ir.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ir/ir_builder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/passes/passes.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/passes/layout_normalization_pass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/passes/matmul_fusion_pass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/passes/architecture_hint_pass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/passes/tiling_pass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pipeline/ir_pipeline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen/cpu/cpu_codegen.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen/metal/metal_codegen.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/runtime.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/session.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/execution_graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/execution_plan_builder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/execution_executor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/execution_context.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_runtime.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/operator_backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/quantization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/quant_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/sampling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/tokenizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/attention_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/kernel_registry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/kernel_scheduler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/decode_runner.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/parity_harness.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/paged_kv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/batched_executor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/model_runner.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/util/cli_helpers.cpp
)

set(COMPILER_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/frontends/frontend.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/frontends/gguf_loader.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/frontends/gguf_to_ir.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/frontends/gguf_utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ir/ir.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ir/ir_builder.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/passes/passes.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/passes/layout_normalization_pass.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/passes/matmul_fusion_pass.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/passes/architecture_hint_pass.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/passes/tiling_pass.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pipeline/ir_pipeline.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen/cpu/cpu_codegen.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen/metal/metal_codegen.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/runtime.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/session.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/execution_graph.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/execution_plan_builder.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/execution_executor.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/execution_context.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_runtime.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/operator_backend.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/quantization.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/quant_utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/sampling.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/tokenizer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/attention_cpu.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/kernel_registry.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/kernel_scheduler.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/decode_runner.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/parity_harness.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/paged_kv.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/batched_executor.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/model_runner.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/float_convert.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/util/cli_helpers.hpp
)

# Create the mlc library
add_library(mlc_lib STATIC
    ${COMPILER_SOURCES}
    ${COMPILER_HEADERS}
)

target_include_directories(mlc_lib PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/compiler
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Hook llama.cpp tokenizer (vocab-only) when available next to the repo.
set(LLAMA_CPP_ROOT ${CMAKE_SOURCE_DIR}/../llama.cpp)
set(LLAMA_CPP_LIB ${LLAMA_CPP_ROOT}/build/bin/libllama.dylib)
if (EXISTS ${LLAMA_CPP_LIB})
    add_library(llama_cpp SHARED IMPORTED)
    set_target_properties(llama_cpp PROPERTIES IMPORTED_LOCATION ${LLAMA_CPP_LIB})
    target_include_directories(mlc_lib PRIVATE
        ${LLAMA_CPP_ROOT}/include
        ${LLAMA_CPP_ROOT}/ggml/include)
    target_compile_definitions(mlc_lib PUBLIC MLC_ENABLE_LLAMA_TOKENIZER=1)
    target_link_libraries(mlc_lib PRIVATE llama_cpp)
endif()

if(APPLE)
    target_link_libraries(mlc_lib PUBLIC
        "-framework Accelerate"
        "-framework Metal"
        "-framework MetalPerformanceShaders"
        "-framework Foundation"
        "-framework CoreGraphics"
        "-framework QuartzCore")
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_runtime.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
endif()

# Create the mlc executable
add_executable(mlc ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

target_link_libraries(mlc PRIVATE mlc_lib)
if (EXISTS ${LLAMA_CPP_LIB})
    target_include_directories(mlc PRIVATE
        ${LLAMA_CPP_ROOT}/include
        ${LLAMA_CPP_ROOT}/ggml/include)
    target_compile_definitions(mlc PRIVATE MLC_ENABLE_LLAMA_TOKENIZER=1)
endif()
if(APPLE)
    target_link_libraries(mlc PRIVATE
        "-framework Metal"
        "-framework MetalPerformanceShaders"
        "-framework Foundation"
        "-framework CoreGraphics"
        "-framework QuartzCore"
        "-framework Accelerate")
endif()

# Add subdirectories for organization
add_subdirectory(frontends)
add_subdirectory(ir)
add_subdirectory(passes)
add_subdirectory(codegen)
add_subdirectory(runtime)
