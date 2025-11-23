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
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/attention_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/kernel_registry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/kernel_scheduler.cpp
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
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/attention_cpu.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/kernel_registry.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/kernel_scheduler.hpp
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

if(APPLE)
    target_link_libraries(mlc_lib PUBLIC "-framework Accelerate" "-framework Metal" "-framework MetalPerformanceShaders" "-framework Foundation")
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/runtime/metal_runtime.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
endif()

# Create the mlc executable
add_executable(mlc ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

target_link_libraries(mlc PRIVATE mlc_lib)

# Add subdirectories for organization
add_subdirectory(frontends)
add_subdirectory(ir)
add_subdirectory(passes)
add_subdirectory(codegen)
add_subdirectory(runtime)
