if(NOT MLC_HAVE_MLX)
    return()
endif()

add_executable(mlc-kernel-bench mlc-kernel-bench.cpp)

set_target_properties(mlc-kernel-bench PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)

target_include_directories(mlc-kernel-bench PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}
    ${MLC_MLX_ROOT}/include)

target_link_libraries(mlc-kernel-bench PRIVATE
    MLCExec
    mlc_lib
    ${MLC_MLX_ROOT}/lib/libmlx.dylib)

if(APPLE)
    target_link_libraries(mlc-kernel-bench PRIVATE
        "-framework Metal"
        "-framework MetalPerformanceShaders"
        "-framework Foundation"
        "-framework Accelerate")
endif()
