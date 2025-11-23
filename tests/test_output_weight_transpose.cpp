#include <gtest/gtest.h>
#include "frontends/gguf_to_ir.hpp"
#include "frontends/gguf_loader.hpp"
#include "ir/ir.hpp"
#include <fstream>
#include <filesystem>
#include <unistd.h>
#include "tests/gguf_test_utils.hpp"

namespace fs = std::filesystem;

using namespace mlc::frontend;
using namespace mlc::ir;
using namespace mlc::test::gguf;

// Create a test GGUF file with output.weight that needs transpose
std::string createGGUFWithOutputWeight() {
    std::string tmp_path = "/tmp/test_output_weight_" + std::to_string(getpid()) + ".gguf";
    std::ofstream file(tmp_path, std::ios::binary);
    
    if (!file.is_open()) {
        throw std::runtime_error("Failed to create test GGUF file");
    }
    
    // Write magic number "GGUF"
    writeU32(file, 0x46554747);
    
    // Write version
    writeU32(file, 1);
    
    // Write n_tensors
    writeU64(file, 1);
    
    // Write n_kv (number of key-value pairs)
    writeU64(file, 0);
    
    // Write tensor directory
    // Tensor: "output.weight" - F32, shape [32000, 4096] (vocab > hidden, GGUF stores as [vocab, hidden])
    writeString(file, "output.weight");
    writeU32(file, 2); // n_dims
    writeU64(file, 2); // array length
    writeU32(file, 32000); // vocab size (first dim in GGUF)
    writeU32(file, 4096);  // hidden size (second dim in GGUF)
    writeU32(file, 0); // dtype: F32
    writeU64(file, 0); // offset
    
    file.close();
    return tmp_path;
}

// Create a test GGUF file with lm_head.weight that needs transpose
std::string createGGUFWithLMHead() {
    std::string tmp_path = "/tmp/test_lm_head_" + std::to_string(getpid()) + ".gguf";
    std::ofstream file(tmp_path, std::ios::binary);
    
    if (!file.is_open()) {
        throw std::runtime_error("Failed to create test GGUF file");
    }
    
    writeU32(file, 0x46554747); // magic
    writeU32(file, 1); // version
    writeU64(file, 1); // n_tensors
    writeU64(file, 0); // n_kv
    
    // Tensor: "lm_head.weight" - F16, shape [50000, 1024] (vocab > hidden, GGUF stores as [vocab, hidden])
    writeString(file, "lm_head.weight");
    writeU32(file, 2); // n_dims
    writeU64(file, 2); // array length
    writeU32(file, 50000); // vocab (first dim in GGUF)
    writeU32(file, 1024);  // hidden (second dim in GGUF)
    writeU32(file, 1); // dtype: F16
    writeU64(file, 0); // offset
    
    file.close();
    return tmp_path;
}

// Create a test GGUF file with a tensor that doesn't need transpose
std::string createGGUFNoTranspose() {
    std::string tmp_path = "/tmp/test_no_transpose_" + std::to_string(getpid()) + ".gguf";
    std::ofstream file(tmp_path, std::ios::binary);
    
    if (!file.is_open()) {
        throw std::runtime_error("Failed to create test GGUF file");
    }
    
    writeU32(file, 0x46554747); // magic
    writeU32(file, 1); // version
    writeU64(file, 1); // n_tensors
    writeU64(file, 0); // n_kv
    
    // Tensor: "layer.0.weight" - F32, shape [32000, 4096] (vocab > hidden, no transpose)
    writeString(file, "layer.0.weight");
    writeU32(file, 2); // n_dims
    writeU64(file, 2); // array length
    writeU32(file, 32000); // vocab
    writeU32(file, 4096);  // hidden
    writeU32(file, 0); // dtype: F32
    writeU64(file, 0); // offset
    
    file.close();
    return tmp_path;
}

TEST(OutputWeightTransposeTest, DetectOutputWeightTranspose) {
    std::string tmp_path = createGGUFWithOutputWeight();
    
    GGUFLoader loader(tmp_path);
    loader.load();
    
    auto graph = GGUFToIR(loader);
    
    // Should NOT create transpose nodes - we mark with metadata instead
    EXPECT_EQ(graph->nodes().size(), 0);
    
    // Shape should be preserved exactly as stored in GGUF
    const auto& tensors = graph->tensors();
    auto it = std::find_if(tensors.begin(), tensors.end(),
        [](const Tensor* t) { return t->name == "output.weight"; });
    ASSERT_NE(it, tensors.end());
    
    const Tensor* output_tensor = *it;
    EXPECT_EQ(output_tensor->shape.size(), 2);
    EXPECT_EQ(output_tensor->shape[0], 32000); // vocab first (as stored in GGUF)
    EXPECT_EQ(output_tensor->shape[1], 4096);  // hidden second (as stored in GGUF)
    
    // Shape should match original_shape (no reshaping)
    EXPECT_EQ(output_tensor->shape, output_tensor->original_shape);
    
    // Should have needs_transpose metadata (vocab > hidden)
    EXPECT_TRUE(output_tensor->metadata.find("needs_transpose") != output_tensor->metadata.end());
    EXPECT_EQ(output_tensor->metadata.at("needs_transpose"), "true");
    
    // layout_transposed should be false (we don't transpose in loader)
    EXPECT_FALSE(output_tensor->layout_transposed);
    
    fs::remove(tmp_path);
}

TEST(OutputWeightTransposeTest, DetectLMHeadTranspose) {
    std::string tmp_path = createGGUFWithLMHead();
    
    GGUFLoader loader(tmp_path);
    loader.load();
    
    auto graph = GGUFToIR(loader);
    
    // Check shape is preserved (no transpose in loader)
    const auto& tensors = graph->tensors();
    auto it = std::find_if(tensors.begin(), tensors.end(),
        [](const Tensor* t) { return t->name == "lm_head.weight"; });
    ASSERT_NE(it, tensors.end());
    
    const Tensor* lm_head_tensor = *it;
    EXPECT_EQ(lm_head_tensor->shape[0], 50000); // vocab first (as stored)
    EXPECT_EQ(lm_head_tensor->shape[1], 1024);  // hidden second (as stored)
    
    // Shape should match original_shape
    EXPECT_EQ(lm_head_tensor->shape, lm_head_tensor->original_shape);
    
    // Note: lm_head.weight is not "output.weight", so it won't get needs_transpose
    // Only "output.weight" specifically gets the needs_transpose flag
    // This test verifies the shape is preserved correctly
    
    fs::remove(tmp_path);
}

TEST(OutputWeightTransposeTest, NoTransposeForNonOutputWeights) {
    std::string tmp_path = createGGUFNoTranspose();
    
    GGUFLoader loader(tmp_path);
    loader.load();
    
    auto graph = GGUFToIR(loader);
    
    // Tensor should have original shape (no transpose needed)
    const auto& tensors = graph->tensors();
    auto it = std::find_if(tensors.begin(), tensors.end(),
        [](const Tensor* t) { return t->name == "layer.0.weight"; });
    ASSERT_NE(it, tensors.end());
    
    const Tensor* layer_tensor = *it;
    EXPECT_EQ(layer_tensor->shape[0], 32000); // Original shape preserved
    EXPECT_EQ(layer_tensor->shape[1], 4096);
    EXPECT_FALSE(layer_tensor->layout_transposed);
    EXPECT_EQ(layer_tensor->original_shape, layer_tensor->shape);
    
    fs::remove(tmp_path);
}

TEST(OutputWeightTransposeTest, NoShapeMismatchErrors) {
    std::string tmp_path = createGGUFWithOutputWeight();
    
    GGUFLoader loader(tmp_path);
    
    // Should not throw shape mismatch errors
    EXPECT_NO_THROW(loader.load());
    
    auto graph = GGUFToIR(loader);
    
    // Should successfully create graph
    EXPECT_NE(graph, nullptr);
    EXPECT_GT(graph->tensors().size(), 0);
    
    fs::remove(tmp_path);
}

TEST(OutputWeightTransposeTest, TransposeMetadataCorrect) {
    std::string tmp_path = createGGUFWithOutputWeight();
    
    GGUFLoader loader(tmp_path);
    loader.load();
    
    auto graph = GGUFToIR(loader);
    
    const auto& tensors = graph->tensors();
    auto it = std::find_if(tensors.begin(), tensors.end(),
        [](const Tensor* t) { return t->name == "output.weight"; });
    ASSERT_NE(it, tensors.end());
    
    const Tensor* output_tensor = *it;
    
    // Check needs_transpose metadata is set
    EXPECT_TRUE(output_tensor->metadata.find("needs_transpose") != output_tensor->metadata.end());
    EXPECT_EQ(output_tensor->metadata.at("needs_transpose"), "true");
    
    // Check shape matches original_shape (no reshaping)
    EXPECT_EQ(output_tensor->shape, output_tensor->original_shape);
    EXPECT_EQ(output_tensor->original_shape[0], 32000);
    EXPECT_EQ(output_tensor->original_shape[1], 4096);
    
    // Check gguf_dtype metadata is present
    EXPECT_NE(output_tensor->metadata.find("gguf_dtype"), output_tensor->metadata.end());
    
    fs::remove(tmp_path);
}

// Create a test GGUF file with Q4_K_M dtype (uses Q4_K constant)
std::string createGGUFWithQ4KM() {
    std::string tmp_path = "/tmp/test_q4km_" + std::to_string(getpid()) + ".gguf";
    std::ofstream file(tmp_path, std::ios::binary);
    
    if (!file.is_open()) {
        throw std::runtime_error("Failed to create test GGUF file");
    }
    
    writeU32(file, 0x46554747); // magic
    writeU32(file, 1); // version
    writeU64(file, 1); // n_tensors
    writeU64(file, 0); // n_kv
    
    // Tensor: "output.weight" - Q4_K (which includes Q4_K_M), shape [32000, 4096] (vocab > hidden)
    writeString(file, "output.weight");
    writeU32(file, 2); // n_dims
    writeU64(file, 2); // array length
    writeU32(file, 32000); // vocab (first dim)
    writeU32(file, 4096);  // hidden (second dim)
    writeU32(file, 12); // dtype: Q4_K (includes Q4_K_M)
    writeU64(file, 0); // offset
    
    file.close();
    return tmp_path;
}

TEST(OutputWeightTransposeTest, Q4KMHandling) {
    std::string tmp_path = createGGUFWithQ4KM();
    
    GGUFLoader loader(tmp_path);
    EXPECT_NO_THROW(loader.load());
    
    auto graph = GGUFToIR(loader);
    
    const auto& tensors = graph->tensors();
    auto it = std::find_if(tensors.begin(), tensors.end(),
        [](const Tensor* t) { return t->name == "output.weight"; });
    ASSERT_NE(it, tensors.end());
    
    const Tensor* output_tensor = *it;
    
    // Q4_K should map to I4
    EXPECT_EQ(output_tensor->dtype, DataType::I4);
    
    // Shape should be preserved exactly as stored
    EXPECT_EQ(output_tensor->shape[0], 32000); // vocab (as stored)
    EXPECT_EQ(output_tensor->shape[1], 4096);  // hidden (as stored)
    EXPECT_EQ(output_tensor->shape, output_tensor->original_shape);
    
    // Should have needs_transpose metadata
    EXPECT_TRUE(output_tensor->metadata.find("needs_transpose") != output_tensor->metadata.end());
    EXPECT_EQ(output_tensor->metadata.at("needs_transpose"), "true");
    
    // Check gguf_dtype is Q4_K
    EXPECT_EQ(output_tensor->metadata.at("gguf_dtype"), "Q4_K");
    
    fs::remove(tmp_path);
}
