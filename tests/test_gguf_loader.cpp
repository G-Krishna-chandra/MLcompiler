#include <gtest/gtest.h>
#include "frontends/gguf_loader.hpp"
#include <fstream>
#include <cstring>
#include <filesystem>
#include <unistd.h>
#include <vector>
#include <algorithm>
#include <limits>
#include "tests/gguf_test_utils.hpp"

namespace fs = std::filesystem;

namespace mlc {
namespace frontend {

using namespace mlc::test::gguf;

// Create a minimal valid GGUF file for testing
std::string createMockGGUFFile() {
    std::string tmp_path = "/tmp/test_gguf_" + std::to_string(getpid()) + ".gguf";
    std::ofstream file(tmp_path, std::ios::binary);
    
    if (!file.is_open()) {
        throw std::runtime_error("Failed to create test GGUF file");
    }
    
    // Write magic number "GGUF" (little-endian: 0x46554747)
    writeU32(file, 0x46554747);
    
    // Write version
    writeU32(file, 1);
    
    // Write n_tensors
    writeU64(file, 2);
    
    // Write n_kv (number of key-value pairs)
    writeU64(file, 1);
    
    // Write KV section
    // Key: "test.key"
    writeString(file, "test.key");
    // Value type: STRING (8)
    writeU32(file, 8); // GGUFValueType::STRING
    // Value: "test.value"
    writeString(file, "test.value");
    
    // Write tensor directory
    // Tensor 1: "tensor1"
    writeString(file, "tensor1");
    // n_dims
    writeU32(file, 2);
    // shape array
    writeU64(file, 2); // array length
    writeU32(file, 3); // dim 0
    writeU32(file, 4); // dim 1
    // dtype
    writeU32(file, 0); // F32 (dtype 0)
    // offset
    writeU64(file, 256); // data starts at offset 256
    
    // Tensor 2: "tensor2"
    writeString(file, "tensor2");
    // n_dims
    writeU32(file, 1);
    // shape array
    writeU64(file, 1); // array length
    writeU32(file, 10); // dim 0
    // dtype
    writeU32(file, 2); // F16
    // offset
    writeU64(file, 400); // data starts at offset 400
    
    // Write some dummy tensor data
    file.seekp(256, std::ios::beg);
    std::vector<float> tensor1_data(3 * 4, 1.5f);
    file.write(reinterpret_cast<const char*>(tensor1_data.data()), 
               tensor1_data.size() * sizeof(float));
    
    file.seekp(400, std::ios::beg);
    std::vector<uint16_t> tensor2_data(10, 0x3C00); // 1.0 in F16
    file.write(reinterpret_cast<const char*>(tensor2_data.data()), 
               tensor2_data.size() * sizeof(uint16_t));
    
    file.close();
    return tmp_path;
}

TEST(GGUFLoaderTest, FileNotFound) {
    GGUFLoader loader("/nonexistent/file.gguf");
    EXPECT_THROW(loader.load(), std::runtime_error);
}

TEST(GGUFLoaderTest, InvalidMagicNumber) {
    std::string tmp_path = "/tmp/test_invalid_magic_" + std::to_string(getpid()) + ".gguf";
    {
        std::ofstream file(tmp_path, std::ios::binary);
        writeU32(file, 0x12345678); // Invalid magic
        writeU32(file, 1);
    }
    
    GGUFLoader loader(tmp_path);
    EXPECT_THROW(loader.load(), std::runtime_error);
    
    fs::remove(tmp_path);
}

TEST(GGUFLoaderTest, HeaderParsing) {
    std::string tmp_path = createMockGGUFFile();
    
    GGUFLoader loader(tmp_path);
    EXPECT_TRUE(loader.load());
    
    // Verify we can access tensors (indirectly confirms header was parsed)
    const auto& tensors = loader.tensors();
    EXPECT_EQ(tensors.size(), 2);
    
    fs::remove(tmp_path);
}

TEST(GGUFLoaderTest, TensorDirectoryParsing) {
    std::string tmp_path = createMockGGUFFile();
    
    GGUFLoader loader(tmp_path);
    EXPECT_TRUE(loader.load());
    
    const auto& tensors = loader.tensors();
    
    // Check tensor1
    ASSERT_NE(tensors.find("tensor1"), tensors.end());
    const auto& t1 = tensors.at("tensor1");
    EXPECT_EQ(t1.name, "tensor1");
    EXPECT_EQ(t1.shape.size(), 2);
    EXPECT_EQ(t1.shape[0], 3);
    EXPECT_EQ(t1.shape[1], 4);
    EXPECT_EQ(t1.dtype, 0); // F32 is dtype 0
    EXPECT_EQ(t1.offset, 256);
    
    // Check tensor2
    ASSERT_NE(tensors.find("tensor2"), tensors.end());
    const auto& t2 = tensors.at("tensor2");
    EXPECT_EQ(t2.name, "tensor2");
    EXPECT_EQ(t2.shape.size(), 1);
    EXPECT_EQ(t2.shape[0], 10);
    EXPECT_EQ(t2.dtype, 2);
    EXPECT_EQ(t2.offset, 400);
    
    fs::remove(tmp_path);
}

TEST(GGUFLoaderTest, LoadTensorData) {
    std::string tmp_path = createMockGGUFFile();
    
    GGUFLoader loader(tmp_path);
    EXPECT_TRUE(loader.load());
    
    const auto& tensors = loader.tensors();
    const auto& t1 = tensors.at("tensor1");
    
    // Load tensor data
    auto data = loader.loadTensorData(t1);
    
    // Verify we got some data
    EXPECT_GT(data.size(), 0);
    
    // Verify the data matches what we wrote (first few bytes)
    // We wrote 3*4 floats = 12 floats = 48 bytes
    EXPECT_GE(data.size(), 48);
    
    // Check that the data contains our written values (1.5f = 0x3FC00000)
    // This is a basic check - in a real implementation we'd decode properly
    const float* float_data = reinterpret_cast<const float*>(data.data());
    if (data.size() >= 48) {
        EXPECT_FLOAT_EQ(float_data[0], 1.5f);
    }
    
    fs::remove(tmp_path);
}

TEST(GGUFLoaderTest, LoadTensorDataInvalidOffset) {
    std::string tmp_path = createMockGGUFFile();
    
    GGUFLoader loader(tmp_path);
    EXPECT_TRUE(loader.load());
    
    // Create a tensor with invalid offset
    GGUFTensorInfo invalid_tensor;
    invalid_tensor.name = "invalid";
    invalid_tensor.shape = {1};
    invalid_tensor.dtype = 1;
    invalid_tensor.offset = 999999999; // Way beyond file size
    
    // This should either throw or return empty data without crashing
    try {
        auto data = loader.loadTensorData(invalid_tensor);
        (void)data;
    } catch (const std::exception&) {
        // Expected path for offsets outside the file bounds
    }
    
    fs::remove(tmp_path);
}

TEST(GGUFLoaderTest, MultipleLoads) {
    std::string tmp_path = createMockGGUFFile();
    
    GGUFLoader loader(tmp_path);
    EXPECT_TRUE(loader.load());
    EXPECT_EQ(loader.tensors().size(), 2);
    
    // Load again - should work
    EXPECT_TRUE(loader.load());
    EXPECT_EQ(loader.tensors().size(), 2);
    
    fs::remove(tmp_path);
}

TEST(GGUFLoaderTest, SupportsRawValues) {
    std::string tmp_path = "/tmp/test_gguf_raw_" + std::to_string(getpid()) + ".gguf";
    std::ofstream file(tmp_path, std::ios::binary);
    ASSERT_TRUE(file.is_open());
    
    writeU32(file, 0x46554747); // magic
    writeU32(file, 1);          // version
    writeU64(file, 0);          // n_tensors
    writeU64(file, 1);          // n_kv
    
    writeString(file, "tokenizer.ggml.model");
    writeU32(file, 13); // RAW (v1 enum)
    std::vector<uint8_t> raw_payload = {0x01, 0x02, 0xAB, 0xCD, 0xEF};
    writeRawBlob(file, raw_payload);
    
    file.close();
    
    GGUFLoader loader(tmp_path);
    ASSERT_TRUE(loader.load());
    
    const auto& kv = loader.kvMetadata();
    auto it = kv.find("tokenizer.ggml.model");
    ASSERT_NE(it, kv.end());
    EXPECT_EQ(it->second.type, GGUFValueType::RAW);
    const auto& bytes = std::get<std::vector<uint8_t>>(it->second.data);
    EXPECT_EQ(bytes, raw_payload);
    
    fs::remove(tmp_path);
}

TEST(GGUFLoaderTest, SkipsCorruptKeyStrings) {
    constexpr uint64_t kCorruptLength = (1ULL << 20) + 16; // just beyond soft limit
    std::string tmp_path = "/tmp/test_gguf_corrupt_key_" + std::to_string(getpid()) + ".gguf";
    std::ofstream file(tmp_path, std::ios::binary);
    ASSERT_TRUE(file.is_open());
    
    writeU32(file, 0x46554747); // magic
    writeU32(file, 1);          // version
    writeU64(file, 0);          // n_tensors
    writeU64(file, 2);          // n_kv
    
    // Corrupt key entry
    writeU64(file, kCorruptLength);
    writeRepeatedByte(file, kCorruptLength, 0x41);
    writeU32(file, 0); // UINT8 type
    writeU8(file, 7);
    
    // Valid entry
    writeString(file, "valid.key");
    writeU32(file, 8); // STRING (v1 enum)
    writeString(file, "ok");
    
    file.close();
    
    GGUFLoader loader(tmp_path);
    ASSERT_TRUE(loader.load());
    
    const auto& kv = loader.kvMetadata();
    ASSERT_EQ(kv.size(), 1u);
    auto it = kv.find("valid.key");
    ASSERT_NE(it, kv.end());
    EXPECT_EQ(std::get<std::string>(it->second.data), "ok");
    
    fs::remove(tmp_path);
}

TEST(GGUFLoaderTest, DetectsTensorSizeOverflow) {
    std::string tmp_path = "/tmp/test_gguf_overflow_" + std::to_string(getpid()) + ".gguf";
    std::ofstream file(tmp_path, std::ios::binary);
    ASSERT_TRUE(file.is_open());
    
    writeU32(file, 0x46554747); // magic
    writeU32(file, 1);          // version
    writeU64(file, 1);          // n_tensors
    writeU64(file, 0);          // n_kv
    
    writeString(file, "huge.tensor");
    writeU32(file, 3); // n_dims
    writeU64(file, 3); // legacy array length prefix
    uint32_t dim = std::numeric_limits<uint32_t>::max();
    writeU32(file, dim);
    writeU32(file, dim);
    writeU32(file, dim);
    writeU32(file, 0); // dtype F32
    writeU64(file, 0); // offset
    
    file.close();
    
    GGUFLoader loader(tmp_path);
    EXPECT_THROW(loader.load(), std::runtime_error);
    
    fs::remove(tmp_path);
}

} // namespace frontend
} // namespace mlc
