add_executable(mlc-emit mlc-emit.cpp)

target_link_libraries(mlc-emit PRIVATE
    MLCEmit
    MLIRIR
    MLIRSupport
    MLIRFuncDialect)

# Regular add_executable auto-propagates rpath for IMPORTED shared libs
# (libllama.0.dylib) pulled in transitively via mlc_lib.
