#include <gtest/gtest.h>
#include "frontends/gguf_to_ir.hpp"
#include "frontends/gguf_loader.hpp"
#include "ir/ir.hpp"
#include <fstream>
#include <filesystem>
#include <unistd.h>
#include <memory>
#include <algorithm>
#include "tests/gguf_test_utils.hpp"

namespace fs = std::filesystem;

using namespace mlc::frontend;
using namespace mlc::ir;
using namespace mlc::test::gguf;

// Create a test GGUF file with 3 tensors
std::string createTestGGUFFile() {
    std::string tmp_path = "/tmp/test_gguf_to_ir_" + std::to_string(getpid()) + ".gguf";
    std::ofstream file(tmp_path, std::ios::binary);
    
    if (!file.is_open()) {
        throw std::runtime_error("Failed to create test GGUF file");
    }
    
    // Write magic number "GGUF"
    writeU32(file, 0x46554747);
    
    // Write version
    writeU32(file, 1);
    
    // Write n_tensors
    writeU64(file, 3);
    
    // Write n_kv (number of key-value pairs)
    writeU64(file, 0);
    
    // Write tensor directory
    // Tensor 1: "layer0.weight" - F32, shape [784, 128]
    writeString(file, "layer0.weight");
    writeU32(file, 2); // n_dims
    writeU64(file, 2); // array length
    writeU32(file, 784);
    writeU32(file, 128);
    writeU32(file, 0); // dtype: F32
    writeU64(file, 0); // offset
    
    // Tensor 2: "layer0.bias" - F16, shape [128]
    writeString(file, "layer0.bias");
    writeU32(file, 1); // n_dims
    writeU64(file, 1); // array length
    writeU32(file, 128);
    writeU32(file, 1); // dtype: F16
    writeU64(file, 1000); // offset
    
    // Tensor 3: "embedding.weight" - Q8_0, shape [1000, 512]
    writeString(file, "embedding.weight");
    writeU32(file, 2); // n_dims
    writeU64(file, 2); // array length
    writeU32(file, 1000);
    writeU32(file, 512);
    writeU32(file, 8); // dtype: Q8_0
    writeU64(file, 2000); // offset
    
    file.close();
    return tmp_path;
}

TEST(GGUFToIRTest, BasicConversion) {
    std::string tmp_path = createTestGGUFFile();
    GGUFLoader loader(tmp_path);
    loader.load();
    
    auto graph = GGUFToIR(loader);
    
    ASSERT_NE(graph, nullptr);
    EXPECT_EQ(graph->tensors().size(), 3);
    EXPECT_EQ(graph->nodes().size(), 0); // No nodes should be created yet
    
    fs::remove(tmp_path);
}

TEST(GGUFToIRTest, TensorCountMatches) {
    std::string tmp_path = createTestGGUFFile();
    GGUFLoader loader(tmp_path);
    loader.load();
    
    auto graph = GGUFToIR(loader);
    
    const auto& gguf_tensors = loader.tensors();
    const auto& ir_tensors = graph->tensors();
    
    EXPECT_EQ(ir_tensors.size(), gguf_tensors.size());
    
    fs::remove(tmp_path);
}

TEST(GGUFToIRTest, TensorNamesPreserved) {
    std::string tmp_path = createTestGGUFFile();
    GGUFLoader loader(tmp_path);
    loader.load();
    
    auto graph = GGUFToIR(loader);
    
    const auto& ir_tensors = graph->tensors();
    
    // Check that all tensor names are preserved
    EXPECT_NE(std::find_if(ir_tensors.begin(), ir_tensors.end(),
        [](const Tensor* t) { return t->name == "layer0.weight"; }),
        ir_tensors.end());
    
    EXPECT_NE(std::find_if(ir_tensors.begin(), ir_tensors.end(),
        [](const Tensor* t) { return t->name == "layer0.bias"; }),
        ir_tensors.end());
    
    EXPECT_NE(std::find_if(ir_tensors.begin(), ir_tensors.end(),
        [](const Tensor* t) { return t->name == "embedding.weight"; }),
        ir_tensors.end());
    
    fs::remove(tmp_path);
}

TEST(GGUFToIRTest, ShapeConversion) {
    std::string tmp_path = createTestGGUFFile();
    GGUFLoader loader(tmp_path);
    loader.load();
    
    auto graph = GGUFToIR(loader);
    
    const auto& ir_tensors = graph->tensors();
    
    // Find layer0.weight tensor
    auto it = std::find_if(ir_tensors.begin(), ir_tensors.end(),
        [](const Tensor* t) { return t->name == "layer0.weight"; });
    ASSERT_NE(it, ir_tensors.end());
    
    const Tensor* tensor = *it;
    EXPECT_EQ(tensor->shape.size(), 2);
    EXPECT_EQ(tensor->shape[0], 784);
    EXPECT_EQ(tensor->shape[1], 128);
    
    // Check layer0.bias shape
    it = std::find_if(ir_tensors.begin(), ir_tensors.end(),
        [](const Tensor* t) { return t->name == "layer0.bias"; });
    ASSERT_NE(it, ir_tensors.end());
    
    tensor = *it;
    EXPECT_EQ(tensor->shape.size(), 1);
    EXPECT_EQ(tensor->shape[0], 128);
    
    fs::remove(tmp_path);
}

TEST(GGUFToIRTest, DtypeMappingF32) {
    std::string tmp_path = createTestGGUFFile();
    GGUFLoader loader(tmp_path);
    loader.load();
    
    auto graph = GGUFToIR(loader);
    
    const auto& ir_tensors = graph->tensors();
    
    // Find layer0.weight (F32)
    auto it = std::find_if(ir_tensors.begin(), ir_tensors.end(),
        [](const Tensor* t) { return t->name == "layer0.weight"; });
    ASSERT_NE(it, ir_tensors.end());
    
    EXPECT_EQ((*it)->dtype, DataType::F32);
    
    fs::remove(tmp_path);
}

TEST(GGUFToIRTest, DtypeMappingF16) {
    std::string tmp_path = createTestGGUFFile();
    GGUFLoader loader(tmp_path);
    loader.load();
    
    auto graph = GGUFToIR(loader);
    
    const auto& ir_tensors = graph->tensors();
    
    // Find layer0.bias (F16)
    auto it = std::find_if(ir_tensors.begin(), ir_tensors.end(),
        [](const Tensor* t) { return t->name == "layer0.bias"; });
    ASSERT_NE(it, ir_tensors.end());
    
    EXPECT_EQ((*it)->dtype, DataType::F16);
    
    fs::remove(tmp_path);
}

TEST(GGUFToIRTest, DtypeMappingQ8_0ToI8) {
    std::string tmp_path = createTestGGUFFile();
    GGUFLoader loader(tmp_path);
    loader.load();
    
    auto graph = GGUFToIR(loader);
    
    const auto& ir_tensors = graph->tensors();
    
    // Find embedding.weight (Q8_0 -> I8)
    auto it = std::find_if(ir_tensors.begin(), ir_tensors.end(),
        [](const Tensor* t) { return t->name == "embedding.weight"; });
    ASSERT_NE(it, ir_tensors.end());
    
    EXPECT_EQ((*it)->dtype, DataType::I8);
    
    fs::remove(tmp_path);
}

TEST(GGUFToIRTest, ByteOffsetPreserved) {
    std::string tmp_path = createTestGGUFFile();
    GGUFLoader loader(tmp_path);
    loader.load();
    
    auto graph = GGUFToIR(loader);
    
    const auto& ir_tensors = graph->tensors();
    
    // Find layer0.weight (offset 0)
    auto it = std::find_if(ir_tensors.begin(), ir_tensors.end(),
        [](const Tensor* t) { return t->name == "layer0.weight"; });
    ASSERT_NE(it, ir_tensors.end());
    EXPECT_EQ((*it)->byteOffset, 0);
    
    // Find layer0.bias (offset 1000)
    it = std::find_if(ir_tensors.begin(), ir_tensors.end(),
        [](const Tensor* t) { return t->name == "layer0.bias"; });
    ASSERT_NE(it, ir_tensors.end());
    EXPECT_EQ((*it)->byteOffset, 1000);
    
    // Find embedding.weight (offset 2000)
    it = std::find_if(ir_tensors.begin(), ir_tensors.end(),
        [](const Tensor* t) { return t->name == "embedding.weight"; });
    ASSERT_NE(it, ir_tensors.end());
    EXPECT_EQ((*it)->byteOffset, 2000);
    
    fs::remove(tmp_path);
}

TEST(GGUFToIRTest, NoNodesCreated) {
    std::string tmp_path = createTestGGUFFile();
    GGUFLoader loader(tmp_path);
    loader.load();
    
    auto graph = GGUFToIR(loader);
    
    // Verify no nodes are created (only tensors)
    EXPECT_EQ(graph->nodes().size(), 0);
    EXPECT_GT(graph->tensors().size(), 0);
    
    fs::remove(tmp_path);
}

TEST(GGUFToIRTest, DtypeMappingFunction) {
    // Test the mapping function directly
    EXPECT_EQ(mapGGUFDtypeToIR(0), DataType::F32);   // F32
    EXPECT_EQ(mapGGUFDtypeToIR(1), DataType::F16);   // F16
    EXPECT_EQ(mapGGUFDtypeToIR(30), DataType::BF16); // BF16
    EXPECT_EQ(mapGGUFDtypeToIR(8), DataType::I8);    // Q8_0
    EXPECT_EQ(mapGGUFDtypeToIR(2), DataType::I4);    // Q4_0
    EXPECT_EQ(mapGGUFDtypeToIR(24), DataType::I8);   // I8
}

TEST(GGUFToIRTest, EmptyLoader) {
    // Create an empty GGUF file
    std::string tmp_path = "/tmp/test_gguf_empty_" + std::to_string(getpid()) + ".gguf";
    std::ofstream file(tmp_path, std::ios::binary);
    
    // Write minimal valid GGUF header with 0 tensors
    writeU32(file, 0x46554747); // magic
    writeU32(file, 1); // version
    writeU64(file, 0); // n_tensors
    writeU64(file, 0); // n_kv
    file.close();
    
    GGUFLoader loader(tmp_path);
    loader.load();
    
    auto graph = GGUFToIR(loader);
    
    ASSERT_NE(graph, nullptr);
    EXPECT_EQ(graph->tensors().size(), 0);
    EXPECT_EQ(graph->nodes().size(), 0);
    
    fs::remove(tmp_path);
}
