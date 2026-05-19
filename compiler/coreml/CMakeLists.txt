# ANE/CoreML probe binary. Lives in its own subdir because it's
# Objective-C++ and pulls in the CoreML framework — neither needed by
# the rest of the project.

if(NOT APPLE)
    return()
endif()

add_executable(coreml_matmul_probe coreml_matmul_probe.mm)
set_target_properties(coreml_matmul_probe PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)
target_link_libraries(coreml_matmul_probe PRIVATE
    "-framework Foundation"
    "-framework CoreML")
set_source_files_properties(coreml_matmul_probe.mm
    PROPERTIES COMPILE_FLAGS "-fobjc-arc")

# Companion MLX-on-GPU matmul timer for apples-to-apples vs the CoreML
# probe above.
if(MLC_HAVE_MLX)
    add_executable(mlx_matmul_probe mlx_matmul_probe.cpp)
    set_target_properties(mlx_matmul_probe PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF)
    target_include_directories(mlx_matmul_probe PRIVATE ${MLC_MLX_ROOT}/include)
    target_link_libraries(mlx_matmul_probe PRIVATE
        ${MLC_MLX_ROOT}/lib/libmlx.dylib
        "-framework Metal"
        "-framework MetalPerformanceShaders"
        "-framework Foundation"
        "-framework Accelerate")
endif()
