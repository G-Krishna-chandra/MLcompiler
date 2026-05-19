add_executable(mlc-compile-run mlc-compile-run.cpp)

set_target_properties(mlc-compile-run PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)

target_link_libraries(mlc-compile-run PRIVATE
    MLCExec
    MLCEmit
    MLCPasses)
