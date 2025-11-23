#include <gtest/gtest.h>
#include "frontends/gguf_loader.hpp"
#include <fstream>
#include <filesystem>
#include <unistd.h>
#include <cstring>
#include "tests/gguf_test_utils.hpp"

namespace fs = std::filesystem;

using namespace mlc::frontend;
using namespace mlc::test::gguf;

// Create a test GGUF file with various value types in KV section
std::string createGGUFWithValueTypes() {
    std::string tmp_path = "/tmp/test_gguf_values_" + std::to_string(getpid()) + ".gguf";
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
    writeU64(file, 10);
    
    // KV pair 1: UINT8
    writeString(file, "test.uint8");
    writeU32(file, static_cast<uint32_t>(GGUFValueType::UINT8));
    writeU8(file, 42);
    
    // KV pair 2: INT8
    writeString(file, "test.int8");
    writeU32(file, static_cast<uint32_t>(GGUFValueType::INT8));
    writeI8(file, -42);
    
    // KV pair 3: UINT16
    writeString(file, "test.uint16");
    writeU32(file, static_cast<uint32_t>(GGUFValueType::UINT16));
    writeU16(file, 1234);
    
    // KV pair 4: INT16
    writeString(file, "test.int16");
    writeU32(file, static_cast<uint32_t>(GGUFValueType::INT16));
    writeI16(file, -1234);
    
    // KV pair 5: UINT32
    writeString(file, "test.uint32");
    writeU32(file, static_cast<uint32_t>(GGUFValueType::UINT32));
    writeU32(file, 12345678);
    
    // KV pair 6: INT32
    writeString(file, "test.int32");
    writeU32(file, static_cast<uint32_t>(GGUFValueType::INT32));
    writeI32(file, -12345678);
    
    // KV pair 7: UINT64 (v1 enum value = 10)
    writeString(file, "test.uint64");
    writeU32(file, 10); // v1 UINT64
    writeU64(file, 123456789012345ULL);
    
    // KV pair 8: INT64 (v1 enum value = 11)
    writeString(file, "test.int64");
    writeU32(file, 11); // v1 INT64
    writeI64(file, -123456789012345LL);
    
    // KV pair 9: FLOAT32 (v1 enum value = 6)
    writeString(file, "test.float32");
    writeU32(file, 6); // v1 FLOAT32
    writeF32(file, 3.14159f);
    
    // KV pair 10: FLOAT64 (v1 enum value = 12)
    writeString(file, "test.float64");
    writeU32(file, 12); // v1 FLOAT64
    writeF64(file, 3.141592653589793);
    
    // Write tensor directory (1 tensor)
    writeString(file, "tensor1");
    writeU32(file, 1); // n_dims
    writeU64(file, 1); // array length
    writeU32(file, 10); // dim
    writeU32(file, 0); // dtype F32
    writeU64(file, 0); // offset
    
    file.close();
    return tmp_path;
}

// Create a test GGUF file with STRING, BOOL, and ARRAY types
std::string createGGUFWithComplexTypes() {
    std::string tmp_path = "/tmp/test_gguf_complex_" + std::to_string(getpid()) + ".gguf";
    std::ofstream file(tmp_path, std::ios::binary);
    
    if (!file.is_open()) {
        throw std::runtime_error("Failed to create test GGUF file");
    }
    
    // Write magic number "GGUF"
    writeU32(file, 0x46554747);
    
    // Write version
    writeU32(file, 1);
    
    // Write n_tensors
    writeU64(file, 0);
    
    // Write n_kv
    writeU64(file, 5);
    
    // KV pair 1: STRING (v1 enum value = 8)
    writeString(file, "test.string");
    writeU32(file, 8); // v1 STRING
    writeString(file, "Hello, GGUF!");
    
    // KV pair 2: BOOL (true) (v1 enum value = 7)
    writeString(file, "test.bool_true");
    writeU32(file, 7); // v1 BOOL
    writeU8(file, 1);
    
    // KV pair 3: BOOL (false) (v1 enum value = 7)
    writeString(file, "test.bool_false");
    writeU32(file, 7); // v1 BOOL
    writeU8(file, 0);
    
    // KV pair 4: ARRAY of INT32 (v1 enum value = 9)
    writeString(file, "test.array_int32");
    writeU32(file, 9); // v1 ARRAY
    writeU32(file, 5); // v1 INT32 element type
    writeU64(file, 3); // element count
    writeI32(file, 10);
    writeI32(file, 20);
    writeI32(file, 30);
    
    // KV pair 5: ARRAY of STRING (v1 enum value = 9)
    writeString(file, "test.array_string");
    writeU32(file, 9); // v1 ARRAY
    writeU32(file, 8); // v1 STRING element type
    writeU64(file, 2); // element count
    writeString(file, "first");
    writeString(file, "second");
    
    file.close();
    return tmp_path;
}

// Create a test GGUF file with RAW bytes
std::string createGGUFWithRawBytes() {
    std::string tmp_path = "/tmp/test_gguf_raw_" + std::to_string(getpid()) + ".gguf";
    std::ofstream file(tmp_path, std::ios::binary);
    
    if (!file.is_open()) {
        throw std::runtime_error("Failed to create test GGUF file");
    }
    
    // Write magic number "GGUF"
    writeU32(file, 0x46554747);
    
    // Write version
    writeU32(file, 1);
    
    // Write n_tensors
    writeU64(file, 0);
    
    // Write n_kv
    writeU64(file, 1);
    
    // KV pair: RAW bytes
    writeString(file, "test.raw");
    // RAW type (13) is not in GGUF v3 spec, use ARRAY of UINT8 instead
    writeU32(file, static_cast<uint32_t>(GGUFValueType::ARRAY));
    writeU32(file, static_cast<uint32_t>(GGUFValueType::UINT8));
    std::vector<uint8_t> raw_data = {0x01, 0x02, 0x03, 0x04, 0x05};
    writeU64(file, raw_data.size());
    for (uint8_t b : raw_data) {
        writeU8(file, b);
    }
    
    file.close();
    return tmp_path;
}

TEST(GGUFValuesTest, ParseUINTTypes) {
    std::string tmp_path = createGGUFWithValueTypes();
    
    GGUFLoader loader(tmp_path);
    EXPECT_NO_THROW(loader.load());
    
    // Verify tensor was parsed correctly (indicates KV section was skipped properly)
    const auto& tensors = loader.tensors();
    EXPECT_EQ(tensors.size(), 1);
    
    fs::remove(tmp_path);
}

TEST(GGUFValuesTest, ParseINTTypes) {
    std::string tmp_path = createGGUFWithValueTypes();
    
    GGUFLoader loader(tmp_path);
    EXPECT_NO_THROW(loader.load());
    
    const auto& tensors = loader.tensors();
    EXPECT_EQ(tensors.size(), 1);
    
    fs::remove(tmp_path);
}

TEST(GGUFValuesTest, ParseFLOATTypes) {
    std::string tmp_path = createGGUFWithValueTypes();
    
    GGUFLoader loader(tmp_path);
    EXPECT_NO_THROW(loader.load());
    
    const auto& tensors = loader.tensors();
    EXPECT_EQ(tensors.size(), 1);
    
    fs::remove(tmp_path);
}

TEST(GGUFValuesTest, ParseString) {
    std::string tmp_path = createGGUFWithComplexTypes();
    
    GGUFLoader loader(tmp_path);
    EXPECT_NO_THROW(loader.load());
    
    const auto& tensors = loader.tensors();
    EXPECT_EQ(tensors.size(), 0);
    
    fs::remove(tmp_path);
}

TEST(GGUFValuesTest, ParseBool) {
    std::string tmp_path = createGGUFWithComplexTypes();
    
    GGUFLoader loader(tmp_path);
    EXPECT_NO_THROW(loader.load());
    
    const auto& tensors = loader.tensors();
    EXPECT_EQ(tensors.size(), 0);
    
    fs::remove(tmp_path);
}

TEST(GGUFValuesTest, ParseArrayOfInts) {
    std::string tmp_path = createGGUFWithComplexTypes();
    
    GGUFLoader loader(tmp_path);
    EXPECT_NO_THROW(loader.load());
    
    const auto& tensors = loader.tensors();
    EXPECT_EQ(tensors.size(), 0);
    
    fs::remove(tmp_path);
}

TEST(GGUFValuesTest, ParseArrayOfStrings) {
    std::string tmp_path = createGGUFWithComplexTypes();
    
    GGUFLoader loader(tmp_path);
    EXPECT_NO_THROW(loader.load());
    
    const auto& tensors = loader.tensors();
    EXPECT_EQ(tensors.size(), 0);
    
    fs::remove(tmp_path);
}

TEST(GGUFValuesTest, ParseRawBytes) {
    std::string tmp_path = createGGUFWithRawBytes();
    
    GGUFLoader loader(tmp_path);
    EXPECT_NO_THROW(loader.load());
    
    const auto& tensors = loader.tensors();
    EXPECT_EQ(tensors.size(), 0);
    
    fs::remove(tmp_path);
}

TEST(GGUFValuesTest, SkipUnneededKVPairs) {
    // Create a file with many KV pairs but we only care about tensors
    std::string tmp_path = "/tmp/test_gguf_skip_" + std::to_string(getpid()) + ".gguf";
    std::ofstream file(tmp_path, std::ios::binary);
    
    writeU32(file, 0x46554747); // magic
    writeU32(file, 1); // version
    writeU64(file, 1); // n_tensors
    writeU64(file, 20); // n_kv (many KV pairs)
    
    // Write 20 KV pairs with various types
    for (int i = 0; i < 20; ++i) {
        writeString(file, "key" + std::to_string(i));
        writeU32(file, static_cast<uint32_t>(GGUFValueType::UINT32));
        writeU32(file, i);
    }
    
    // Write tensor directory
    writeString(file, "tensor1");
    writeU32(file, 1);
    writeU64(file, 1);
    writeU32(file, 10);
    writeU32(file, 0);
    writeU64(file, 0);
    
    file.close();
    
    GGUFLoader loader(tmp_path);
    EXPECT_NO_THROW(loader.load());
    
    // Should still parse tensor correctly despite many KV pairs
    const auto& tensors = loader.tensors();
    EXPECT_EQ(tensors.size(), 1);
    EXPECT_NE(tensors.find("tensor1"), tensors.end());
    
    fs::remove(tmp_path);
}

TEST(GGUFValuesTest, NestedArrays) {
    std::string tmp_path = "/tmp/test_gguf_nested_" + std::to_string(getpid()) + ".gguf";
    std::ofstream file(tmp_path, std::ios::binary);
    
    writeU32(file, 0x46554747); // magic
    writeU32(file, 1); // version
    writeU64(file, 0); // n_tensors
    writeU64(file, 1); // n_kv
    
    // KV pair: ARRAY of ARRAY of INT32
    writeString(file, "test.nested_array");
    writeU32(file, static_cast<uint32_t>(GGUFValueType::ARRAY));
    writeU32(file, static_cast<uint32_t>(GGUFValueType::ARRAY)); // element type is ARRAY
    writeU64(file, 2); // 2 arrays
    
    // First nested array
    writeU32(file, static_cast<uint32_t>(GGUFValueType::INT32)); // nested element type
    writeU64(file, 2); // nested array length
    writeI32(file, 1);
    writeI32(file, 2);
    
    // Second nested array
    writeU32(file, static_cast<uint32_t>(GGUFValueType::INT32)); // nested element type
    writeU64(file, 2); // nested array length
    writeI32(file, 3);
    writeI32(file, 4);
    
    file.close();
    
    GGUFLoader loader(tmp_path);
    EXPECT_NO_THROW(loader.load());
    
    fs::remove(tmp_path);
}

TEST(GGUFValuesTest, ErrorOnUnsupportedType) {
    std::string tmp_path = "/tmp/test_gguf_error_" + std::to_string(getpid()) + ".gguf";
    std::ofstream file(tmp_path, std::ios::binary);
    
    writeU32(file, 0x46554747); // magic
    writeU32(file, 1); // version
    writeU64(file, 0); // n_tensors
    writeU64(file, 1); // n_kv
    
    // KV pair with unsupported type (99)
    writeString(file, "test.unsupported");
    writeU32(file, 99); // Invalid type
    
    file.close();
    
    GGUFLoader loader(tmp_path);
    EXPECT_THROW(loader.load(), std::runtime_error);
    
    fs::remove(tmp_path);
}

TEST(GGUFValuesTest, TensorDirectoryStillWorks) {
    std::string tmp_path = createGGUFWithValueTypes();
    
    GGUFLoader loader(tmp_path);
    EXPECT_NO_THROW(loader.load());
    
    const auto& tensors = loader.tensors();
    EXPECT_EQ(tensors.size(), 1);
    EXPECT_NE(tensors.find("tensor1"), tensors.end());
    
    const auto& tensor = tensors.at("tensor1");
    EXPECT_EQ(tensor.shape.size(), 1);
    EXPECT_EQ(tensor.shape[0], 10);
    
    fs::remove(tmp_path);
}
