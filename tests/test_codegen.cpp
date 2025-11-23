#include <gtest/gtest.h>
#include "codegen/cpu/cpu_codegen.hpp"
#include "codegen/metal/metal_codegen.hpp"

TEST(CodeGenTest, CPUCodeGenInitialization) {
    mlc::codegen::cpu::CPUCodeGen codegen;
    EXPECT_NO_THROW(codegen.generateCode());
    EXPECT_NO_THROW(codegen.optimize());
}

TEST(CodeGenTest, MetalCodeGenInitialization) {
    mlc::codegen::metal::MetalCodeGen codegen;
    EXPECT_NO_THROW(codegen.generateCode());
    EXPECT_NO_THROW(codegen.optimize());
}






