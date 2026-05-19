get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_llvm_executable(mlc-opt mlc-opt.cpp)

llvm_update_compile_flags(mlc-opt)
target_link_libraries(mlc-opt PRIVATE
    MLIRMLCDialect
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    MLIROptLib
    MLIRRegisterAllDialects
    MLIRRegisterAllExtensions
    MLIRRegisterAllPasses)

mlir_check_all_link_libraries(mlc-opt)
