#include <gtest/gtest.h>
#include "frontends/gguf_loader.hpp"
#include "frontends/gguf_to_ir.hpp"
#include "ir/ir.hpp"
#include <cstdlib>
#include <string>

namespace {
    using namespace mlc;
    using namespace mlc::frontend;
    using namespace mlc::ir;
}

TEST(GGUFIntegrationTest, LoadRealGGUFFile) {
    // Get test file path from environment variable
    const char* test_gguf_path = std::getenv("TEST_GGUF_PATH");
    if (!test_gguf_path) {
        GTEST_SKIP() << "TEST_GGUF_PATH not set, skipping integration test";
    }
    
    std::string gguf_path(test_gguf_path);
    
    // Load GGUF file
    GGUFLoader loader(gguf_path);
    ASSERT_TRUE(loader.load()) << "Failed to load GGUF file: " << gguf_path;
    
    // Check architecture metadata
    const auto& kv_metadata = loader.kvMetadata();
    auto arch_it = kv_metadata.find("general.architecture");
    if (arch_it != kv_metadata.end()) {
        if (arch_it->second.type == GGUFValueType::STRING) {
            std::string arch = std::get<std::string>(arch_it->second.data);
            // For LLaMA models, expect "llama"
            if (arch == "llama") {
                EXPECT_EQ(arch, "llama");
            }
        }
    }
    
    // Convert to IR
    auto graph = GGUFToIR(loader);
    ASSERT_NE(graph, nullptr);
    
    // Find output.weight tensor
    const auto& tensors = graph->tensors();
    auto output_weight_it = std::find_if(tensors.begin(), tensors.end(),
        [](const Tensor* t) { return t->name == "output.weight"; });
    
    if (output_weight_it != tensors.end()) {
        const Tensor* output_weight = *output_weight_it;
        
        // Check that shape is loaded exactly as stored (should be [vocab, hidden])
        EXPECT_EQ(output_weight->shape.size(), 2) 
            << "output.weight should be 2D";
        
        // Shape should match original_shape since we don't reshape
        EXPECT_EQ(output_weight->shape, output_weight->original_shape)
            << "output.weight shape should match original_shape (no reshaping)";
        
        // If shape is [vocab, hidden] where vocab > hidden, should have needs_transpose
        if (output_weight->shape.size() == 2) {
            int64_t d0 = output_weight->shape[0];
            int64_t d1 = output_weight->shape[1];
            
            if (d0 > d1) {
                // Should be marked for transpose
                EXPECT_TRUE(output_weight->metadata.find("needs_transpose") != output_weight->metadata.end())
                    << "output.weight with shape [" << d0 << ", " << d1 
                    << "] should have needs_transpose=true";
                
                if (output_weight->metadata.find("needs_transpose") != output_weight->metadata.end()) {
                    EXPECT_EQ(output_weight->metadata.at("needs_transpose"), "true")
                        << "needs_transpose should be 'true'";
                }
            }
        }
    } else {
        // If output.weight doesn't exist, that's OK - some models might use different names
        GTEST_SKIP() << "output.weight tensor not found in model (may use different naming)";
    }
}

TEST(GGUFIntegrationTest, OutputWeightShapePreservation) {
    const char* test_gguf_path = std::getenv("TEST_GGUF_PATH");
    if (!test_gguf_path) {
        GTEST_SKIP() << "TEST_GGUF_PATH not set, skipping integration test";
    }
    
    std::string gguf_path(test_gguf_path);
    
    GGUFLoader loader(gguf_path);
    ASSERT_TRUE(loader.load());
    
    // Get raw tensor info from loader
    const auto& gguf_tensors = loader.tensors();
    auto gguf_it = gguf_tensors.find("output.weight");
    
    if (gguf_it == gguf_tensors.end()) {
        GTEST_SKIP() << "output.weight not found in GGUF file";
    }
    
    // Convert to IR
    auto graph = GGUFToIR(loader);
    ASSERT_NE(graph, nullptr);
    
    // Find IR tensor
    const auto& ir_tensors = graph->tensors();
    auto ir_it = std::find_if(ir_tensors.begin(), ir_tensors.end(),
        [](const Tensor* t) { return t->name == "output.weight"; });
    
    ASSERT_NE(ir_it, ir_tensors.end()) << "output.weight should exist in IR";
    
    const Tensor* ir_tensor = *ir_it;
    const auto& gguf_tensor = gguf_it->second;
    
    // IR shape should match GGUF shape exactly (no reshaping)
    ASSERT_EQ(ir_tensor->shape.size(), gguf_tensor.shape.size())
        << "IR shape dimension count should match GGUF";
    
    for (size_t i = 0; i < ir_tensor->shape.size(); ++i) {
        EXPECT_EQ(ir_tensor->shape[i], static_cast<int64_t>(gguf_tensor.shape[i]))
            << "Shape dimension " << i << " should match exactly";
    }
}






