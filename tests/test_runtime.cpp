#include <gtest/gtest.h>
#include "runtime/runtime.hpp"
#include "runtime/session.hpp"
#include "runtime/quantization.hpp"
#include "runtime/quant_utils.hpp"
#include "runtime/model_runner.hpp"
#include "runtime/execution_graph.hpp"
#include "runtime/execution_executor.hpp"
#include "runtime/kernel_registry.hpp"
#include "runtime/metal_runtime.hpp"
#include "runtime/execution_context.hpp"
#include "tests/gguf_test_utils.hpp"
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <unistd.h>
#include <cstring>
#include <vector>

namespace fs = std::filesystem;
using namespace mlc::test::gguf;

namespace {
uint16_t floatToFp16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFFu;

    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa = (mantissa | 0x800000u) >> (1 - exponent);
        return static_cast<uint16_t>(sign | ((mantissa + 0x1000u) >> 13));
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | ((mantissa + 0x1000u) >> 13));
}
} // namespace

TEST(RuntimeTest, BasicInitialization) {
    mlc::runtime::Runtime runtime;
    EXPECT_NO_THROW(runtime.execute());
}

TEST(RuntimeTest, InitializeShutdown) {
    mlc::runtime::Runtime runtime;
    EXPECT_NO_THROW(runtime.initialize());
    EXPECT_NO_THROW(runtime.shutdown());
}

namespace {
std::string createLinearGGUFFile() {
    std::string path = "/tmp/test_linear_" + std::to_string(getpid()) + ".gguf";
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to create GGUF file");
    }

    writeU32(file, 0x46554747);
    writeU32(file, 1);
    writeU64(file, 1); // n_tensors
    writeU64(file, 0); // n_kv

    writeString(file, "linear.weight");
    writeU32(file, 2); // dims
    writeU64(file, 2);
    writeU32(file, 2); // rows
    writeU32(file, 3); // cols
    writeU32(file, 0); // F32
    writeU64(file, 256); // data offset

    file.seekp(256, std::ios::beg);
    std::vector<float> weights = {
        1.0f, 2.0f, 3.0f,
        0.0f, 1.0f, 0.5f
    };
    std::vector<uint8_t> raw(weights.size() * sizeof(float));
    std::memcpy(raw.data(), weights.data(), raw.size());
    writeBytes(file, raw);

    file.close();
    return path;
}
} // namespace

std::string createRunnerGGUFFile() {
    std::string path = "/tmp/test_runner_" + std::to_string(getpid()) + ".gguf";
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to create GGUF file");
    }

    writeU32(file, 0x46554747);
    writeU32(file, 1);
    writeU64(file, 2); // tensors
    writeU64(file, 2); // kv pairs

    writeString(file, "llama.block_count");
    writeU32(file, 4); // UINT32
    writeU32(file, 2);

    writeString(file, "llama.embedding_length");
    writeU32(file, 4);
    writeU32(file, 3);

    writeString(file, "tok_embeddings.weight");
    writeU32(file, 2);
    writeU64(file, 2);
    writeU32(file, 4);
    writeU32(file, 3);
    writeU32(file, 0);
    writeU64(file, 512);

    writeString(file, "output.weight");
    writeU32(file, 2);
    writeU64(file, 2);
    writeU32(file, 2);
    writeU32(file, 3);
    writeU32(file, 0);
    writeU64(file, 560);

    file.seekp(512, std::ios::beg);
    std::vector<float> embeddings = {
        0.f, 0.1f, 0.2f,
        1.f, 1.1f, 1.2f,
        2.f, 2.1f, 2.2f,
        3.f, 3.1f, 3.2f
    };
    file.write(reinterpret_cast<const char*>(embeddings.data()),
               embeddings.size() * sizeof(float));

    file.seekp(560, std::ios::beg);
    std::vector<float> output_weight = {
        1.f, 0.f, 0.f,
        0.f, 1.f, 0.f
    };
    file.write(reinterpret_cast<const char*>(output_weight.data()),
               output_weight.size() * sizeof(float));

    file.close();
    return path;
}

std::string createTransposedEmbeddingGGUFFile() {
    std::string path = "/tmp/test_transposed_embed_" + std::to_string(getpid()) + ".gguf";
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to create GGUF file");
    }

    writeU32(file, 0x46554747);
    writeU32(file, 1);
    writeU64(file, 1);
    writeU64(file, 0);

    writeString(file, "token_embd.weight");
    writeU32(file, 2);
    writeU64(file, 2);
    writeU32(file, 3);
    writeU32(file, 4);
    writeU32(file, 24);
    writeU64(file, 256);

    file.seekp(256, std::ios::beg);
    std::vector<int8_t> weights = {
        10, 11, 12, 13,
        20, 21, 22, 23,
        30, 31, 32, 33
    };
    file.write(reinterpret_cast<const char*>(weights.data()), weights.size());
    file.close();
    return path;
}

std::string createTransposedOutputGGUFFile() {
    std::string path = "/tmp/test_transposed_output_" + std::to_string(getpid()) + ".gguf";
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to create GGUF file");
    }

    writeU32(file, 0x46554747);
    writeU32(file, 1);
    writeU64(file, 1);
    writeU64(file, 0);

    writeString(file, "output.weight");
    writeU32(file, 2);
    writeU64(file, 2);
    writeU32(file, 3);
    writeU32(file, 4);
    writeU32(file, 24);
    writeU64(file, 256);

    file.seekp(256, std::ios::beg);
    std::vector<int8_t> weights = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    };
    file.write(reinterpret_cast<const char*>(weights.data()), weights.size());
    file.close();
    return path;
}

std::string createTransposedRunnerGGUFFile() {
    std::string path = "/tmp/test_runner_transposed_" + std::to_string(getpid()) + ".gguf";
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to create GGUF file");
    }

    writeU32(file, 0x46554747);
    writeU32(file, 1);
    writeU64(file, 2);
    writeU64(file, 2);

    writeString(file, "llama.block_count");
    writeU32(file, 4);
    writeU32(file, 2);

    writeString(file, "llama.embedding_length");
    writeU32(file, 4);
    writeU32(file, 3);

    writeString(file, "tok_embeddings.weight");
    writeU32(file, 2);
    writeU64(file, 2);
    writeU32(file, 4);
    writeU32(file, 3);
    writeU32(file, 0);
    writeU64(file, 512);

    writeString(file, "output.weight");
    writeU32(file, 2);
    writeU64(file, 2);
    writeU32(file, 3);
    writeU32(file, 4);
    writeU32(file, 24);
    writeU64(file, 560);

    file.seekp(512, std::ios::beg);
    std::vector<float> embeddings = {
        0.f, 0.1f, 0.2f,
        1.f, 1.1f, 1.2f,
        2.f, 2.1f, 2.2f,
        3.f, 3.1f, 3.2f
    };
    file.write(reinterpret_cast<const char*>(embeddings.data()),
               embeddings.size() * sizeof(float));

    file.seekp(560, std::ios::beg);
    std::vector<int8_t> output_weight = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    };
    file.write(reinterpret_cast<const char*>(output_weight.data()), output_weight.size());
    file.close();
    return path;
}

TEST(RuntimeSessionTest, RunsLinearLayer) {
    std::string path = createLinearGGUFFile();
    mlc::runtime::Session session(path);

    std::vector<float> input = {1.0f, 2.0f, 3.0f};
    auto output = session.runLinear("linear.weight", input);

    ASSERT_EQ(output.size(), 2u);
    EXPECT_FLOAT_EQ(output[0], 1.0f * 1 + 2.0f * 2 + 3.0f * 3);
    EXPECT_FLOAT_EQ(output[1], 1.0f * 0 + 2.0f * 1 + 3.0f * 0.5f);

    fs::remove(path);
}

TEST(RuntimeSessionTest, ReturnsEmbeddingVector) {
    std::string path = createLinearGGUFFile();
    mlc::runtime::Session session(path);

    auto embedding = session.getEmbedding("linear.weight", 1);
    ASSERT_EQ(embedding.size(), 3u);
    EXPECT_FLOAT_EQ(embedding[0], 0.0f);
    EXPECT_FLOAT_EQ(embedding[1], 1.0f);
    EXPECT_FLOAT_EQ(embedding[2], 0.5f);

    fs::remove(path);
}

TEST(RuntimeSessionTest, EmbeddingHandlesTransposedTensor) {
    std::string path = createTransposedEmbeddingGGUFFile();
    mlc::runtime::Session session(path);

    auto embedding = session.getEmbedding("token_embd.weight", 2);
    ASSERT_EQ(embedding.size(), 3u);
    EXPECT_FLOAT_EQ(embedding[0], 12.0f);
    EXPECT_FLOAT_EQ(embedding[1], 22.0f);
    EXPECT_FLOAT_EQ(embedding[2], 32.0f);

    fs::remove(path);
}

TEST(RuntimeSessionTest, LinearHandlesTransposedQuantizedTensor) {
    std::string path = createTransposedOutputGGUFFile();
    mlc::runtime::Session session(path);

    std::vector<float> input = {1.0f, 2.0f, 3.0f};
    auto logits = session.runLinear("output.weight", input);

    ASSERT_EQ(logits.size(), 4u);
    EXPECT_FLOAT_EQ(logits[0], 38.0f);
    EXPECT_FLOAT_EQ(logits[1], 44.0f);
    EXPECT_FLOAT_EQ(logits[2], 50.0f);
    EXPECT_FLOAT_EQ(logits[3], 56.0f);

    fs::remove(path);
}

TEST(QuantizationTest, Q8_1DequantizationProducesBias) {
    using namespace mlc::runtime;
    constexpr size_t cols = 32;
    std::vector<uint8_t> buffer(q8_1RowSize(cols), 0);

    uint16_t* header = reinterpret_cast<uint16_t*>(buffer.data());
    header[0] = floatToFp16(2.0f);   // scale
    header[1] = floatToFp16(32.0f);  // bias accumulator -> bias 1.0f

    int8_t* qs = reinterpret_cast<int8_t*>(buffer.data() + 4);
    for (size_t i = 0; i < cols; ++i) {
        qs[i] = static_cast<int8_t>(i - 16);
    }

    std::vector<float> output(cols, 0.0f);
    dequantizeRowQ8_1(buffer.data(), cols, output.data());

    for (size_t i = 0; i < cols; ++i) {
        float expected = 2.0f * static_cast<float>(qs[i]) + 1.0f;
        EXPECT_FLOAT_EQ(output[i], expected);
    }
}

TEST(RotaryEmbeddingTest, CpuMatchesExpectedRotation) {
    std::vector<float> vec = {
        1.7640524f, 0.4001572f, 0.9787380f, 2.2408931f,
        1.8675580f, -0.9772779f, 0.95008844f, -0.1513572f
    };
    std::vector<float> cos = {
        0.96366274f, 0.3834415f, 0.79172504f, 0.5288949f
    };
    std::vector<float> sin = {
        0.56804454f, 0.92559665f, 0.07103606f, 0.0871293f
    };
    mlc::runtime::applyRotaryEmbedding(vec, cos, sin, vec.size(), cos.size() * 2);
    std::vector<float> expected = {
        1.4726444f, 1.3876770f, -1.6988745f, 1.7651681f,
        1.5480144f, -0.6410714f, 0.5156846f, 0.00272848f
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(vec[i], expected[i], 1e-5f);
    }
}

TEST(QuantizationTest, Q4_0DotProductMatchesDequant) {
    using namespace mlc::runtime;
    constexpr size_t cols = 32;
#if defined(_MSC_VER)
#pragma pack(push, 1)
#endif
    struct TestBlockQ4_0 {
        uint16_t d;
        uint8_t qs[cols / 2];
    }
#if defined(__clang__) || defined(__GNUC__)
    __attribute__((packed))
#endif
    ;
#if defined(_MSC_VER)
#pragma pack(pop)
#endif
    static_assert(sizeof(TestBlockQ4_0) == 18, "Unexpected Q4_0 block size");
    TestBlockQ4_0 block{};
    block.d = floatToFp16(0.5f);
    for (size_t i = 0; i < cols / 2; ++i) {
        uint8_t lo = static_cast<uint8_t>(i & 0xF);
        uint8_t hi = static_cast<uint8_t>((cols / 2 - i) & 0xF);
        block.qs[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    std::vector<uint8_t> row(sizeof(block));
    std::memcpy(row.data(), &block, sizeof(block));

    std::vector<float> vec(cols);
    for (size_t i = 0; i < cols; ++i) {
        vec[i] = 0.1f * static_cast<float>(i + 1);
    }

    std::vector<float> dequant(cols);
    dequantizeRowQ4_0(row.data(), cols, 1, dequant.data());
    float expected = 0.0f;
    for (size_t i = 0; i < cols; ++i) {
        expected += dequant[i] * vec[i];
    }

    float fast = dotProductRowQ4_0(row.data(), cols, 1, vec.data());
    EXPECT_NEAR(fast, expected, 1e-5f);
}

TEST(QuantizationTest, Q4_1DotProductMatchesDequant) {
    using namespace mlc::runtime;
    constexpr size_t cols = 32;
#if defined(_MSC_VER)
#pragma pack(push, 1)
#endif
    struct Block {
        uint16_t d;
        uint16_t m;
        uint8_t qs[cols / 2];
    }
#if defined(__clang__) || defined(__GNUC__)
    __attribute__((packed))
#endif
    ;
#if defined(_MSC_VER)
#pragma pack(pop)
#endif
    Block block{};
    block.d = floatToFp16(0.25f);
    block.m = floatToFp16(1.0f);
    for (size_t i = 0; i < cols / 2; ++i) {
        uint8_t lo = static_cast<uint8_t>(i & 0xF);
        uint8_t hi = static_cast<uint8_t>((i + 1) & 0xF);
        block.qs[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    std::vector<uint8_t> row(sizeof(block));
    std::memcpy(row.data(), &block, sizeof(block));
    std::vector<float> vec(cols);
    for (size_t i = 0; i < cols; ++i) vec[i] = 0.05f * static_cast<float>(i + 1);
    std::vector<float> dequant(cols);
    dequantizeRowQ4_1(row.data(), cols, dequant.data());
    float expected = 0.0f;
    for (size_t i = 0; i < cols; ++i) expected += dequant[i] * vec[i];
    float fast = dotProductRowQ4_1(row.data(), cols, vec.data());
    EXPECT_NEAR(fast, expected, 1e-5f);
}

TEST(QuantizationTest, Q5_0DotProductMatchesDequant) {
    using namespace mlc::runtime;
    constexpr size_t cols = 32;
#if defined(_MSC_VER)
#pragma pack(push, 1)
#endif
    struct Block {
        uint16_t d;
        uint8_t qh[4];
        uint8_t qs[cols / 2];
    }
#if defined(__clang__) || defined(__GNUC__)
    __attribute__((packed))
#endif
    ;
#if defined(_MSC_VER)
#pragma pack(pop)
#endif
    Block block{};
    block.d = floatToFp16(0.5f);
    std::memset(block.qh, 0, sizeof(block.qh));
    for (size_t i = 0; i < cols / 2; ++i) {
        uint8_t lo = static_cast<uint8_t>((i & 0xF));
        uint8_t hi = static_cast<uint8_t>(((cols / 2 - i) & 0xF));
        block.qs[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    std::vector<uint8_t> row(sizeof(block));
    std::memcpy(row.data(), &block, sizeof(block));
    std::vector<float> vec(cols);
    for (size_t i = 0; i < cols; ++i) vec[i] = 0.03f * static_cast<float>(i + 1);
    std::vector<float> dequant(cols);
    dequantizeRowQ5_0(row.data(), cols, dequant.data());
    float expected = 0.0f;
    for (size_t i = 0; i < cols; ++i) expected += dequant[i] * vec[i];
    float fast = dotProductRowQ5_0(row.data(), cols, vec.data());
    EXPECT_NEAR(fast, expected, 1e-5f);
}

TEST(QuantizationTest, Q5_1DotProductMatchesDequant) {
    using namespace mlc::runtime;
    constexpr size_t cols = 32;
#if defined(_MSC_VER)
#pragma pack(push, 1)
#endif
    struct Block {
        uint16_t d;
        uint16_t m;
        uint8_t qh[4];
        uint8_t qs[cols / 2];
    }
#if defined(__clang__) || defined(__GNUC__)
    __attribute__((packed))
#endif
    ;
#if defined(_MSC_VER)
#pragma pack(pop)
#endif
    Block block{};
    block.d = floatToFp16(0.5f);
    block.m = floatToFp16(0.25f);
    std::memset(block.qh, 0, sizeof(block.qh));
    for (size_t i = 0; i < cols / 2; ++i) {
        uint8_t lo = static_cast<uint8_t>((i + 1) & 0xF);
        uint8_t hi = static_cast<uint8_t>((i + 2) & 0xF);
        block.qs[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    std::vector<uint8_t> row(sizeof(block));
    std::memcpy(row.data(), &block, sizeof(block));
    std::vector<float> vec(cols);
    for (size_t i = 0; i < cols; ++i) vec[i] = -0.02f * static_cast<float>(i + 1);
    std::vector<float> dequant(cols);
    dequantizeRowQ5_1(row.data(), cols, dequant.data());
    float expected = 0.0f;
    for (size_t i = 0; i < cols; ++i) expected += dequant[i] * vec[i];
    float fast = dotProductRowQ5_1(row.data(), cols, vec.data());
    EXPECT_NEAR(fast, expected, 1e-5f);
}

TEST(QuantizationTest, Q8_0DotProductMatchesDequant) {
    using namespace mlc::runtime;
    constexpr size_t cols = 32;
#if defined(_MSC_VER)
#pragma pack(push, 1)
#endif
    struct Block {
        uint16_t d;
        int8_t qs[cols];
    }
#if defined(__clang__) || defined(__GNUC__)
    __attribute__((packed))
#endif
    ;
#if defined(_MSC_VER)
#pragma pack(pop)
#endif
    Block block{};
    block.d = floatToFp16(0.125f);
    for (size_t i = 0; i < cols; ++i) {
        block.qs[i] = static_cast<int8_t>(i - 16);
    }
    std::vector<uint8_t> row(sizeof(block));
    std::memcpy(row.data(), &block, sizeof(block));
    std::vector<float> vec(cols);
    for (size_t i = 0; i < cols; ++i) vec[i] = 0.01f * static_cast<float>(i + 1);
    std::vector<float> dequant(cols);
    dequantizeRowQ8_0(row.data(), cols, dequant.data());
    float expected = 0.0f;
    for (size_t i = 0; i < cols; ++i) expected += dequant[i] * vec[i];
    float fast = dotProductRowQ8_0(row.data(), cols, vec.data());
    EXPECT_NEAR(fast, expected, 1e-5f);
}

namespace {

template <typename QuantFunc, typename DequantFunc>
void RunRoundTripQuant(const QuantFunc& quant,
                       const DequantFunc& dequant,
                       size_t cols,
                       float tol) {
    std::vector<float> data(cols);
    for (size_t i = 0; i < cols; ++i) {
        data[i] = 0.1f * std::sin(static_cast<float>(i) * 0.23f) + 0.05f * std::cos(static_cast<float>(i) * 0.11f);
    }
    std::vector<uint8_t> quantized;
    quant(data.data(), cols, quantized);
    std::vector<float> restored(cols, 0.0f);
    dequant(quantized.data(), cols, restored.data());
    for (size_t i = 0; i < cols; ++i) {
        EXPECT_NEAR(data[i], restored[i], tol);
    }
}

} // namespace

TEST(QuantizationTest, Q4_0RoundTripQuantization) {
    using namespace mlc::runtime;
    constexpr size_t cols = 64;
    std::vector<float> data(cols);
    for (size_t i = 0; i < cols; ++i) {
        data[i] = std::sin(static_cast<float>(i));
    }
    std::vector<uint8_t> quant;
    quantizeRowQ4_0(data.data(), cols, 1, quant);
    std::vector<float> restored(cols);
    dequantizeRowQ4_0(quant.data(), cols, 1, restored.data());
    for (size_t i = 0; i < cols; ++i) {
        EXPECT_NEAR(data[i], restored[i], 1e-1f);
    }
}

TEST(QuantizationTest, Q4_1RoundTripQuantization) {
    using namespace mlc::runtime;
    constexpr size_t cols = 64;
    std::vector<float> data(cols);
    for (size_t i = 0; i < cols; ++i) {
        data[i] = std::cos(static_cast<float>(i) * 0.17f);
    }
    std::vector<uint8_t> quant;
    quantizeRowQ4_1(data.data(), cols, quant);
    std::vector<float> restored(cols);
    dequantizeRowQ4_1(quant.data(), cols, restored.data());
    for (size_t i = 0; i < cols; ++i) {
        EXPECT_NEAR(data[i], restored[i], 1.5e-1f);
    }
}

TEST(QuantizationTest, Q5_0RoundTripQuantization) {
    using namespace mlc::runtime;
    constexpr size_t cols = 64;
    std::vector<float> data(cols);
    for (size_t i = 0; i < cols; ++i) {
        data[i] = 0.25f * static_cast<float>(std::sin(i * 0.21f));
    }
    std::vector<uint8_t> quant;
    quantizeRowQ5_0(data.data(), cols, quant);
    std::vector<float> restored(cols);
    dequantizeRowQ5_0(quant.data(), cols, restored.data());
    for (size_t i = 0; i < cols; ++i) {
        EXPECT_NEAR(data[i], restored[i], 1.5e-1f);
    }
}

TEST(QuantizationTest, Q5_1RoundTripQuantization) {
    using namespace mlc::runtime;
    constexpr size_t cols = 64;
    std::vector<float> data(cols);
    for (size_t i = 0; i < cols; ++i) {
        data[i] = 0.4f * static_cast<float>(std::sin(i * 0.11f) + 0.2f);
    }
    std::vector<uint8_t> quant;
    quantizeRowQ5_1(data.data(), cols, quant);
    std::vector<float> restored(cols);
    dequantizeRowQ5_1(quant.data(), cols, restored.data());
    for (size_t i = 0; i < cols; ++i) {
        EXPECT_NEAR(data[i], restored[i], 1.5e-1f);
    }
}

TEST(QuantizationTest, Q8_0RoundTripQuantization) {
    using namespace mlc::runtime;
    constexpr size_t cols = 64;
    std::vector<float> data(cols);
    for (size_t i = 0; i < cols; ++i) {
        data[i] = 0.05f * static_cast<float>(std::sin(i * 0.31f));
    }
    std::vector<uint8_t> quant;
    quantizeRowQ8_0(data.data(), cols, quant);
    std::vector<float> restored(cols);
    dequantizeRowQ8_0(quant.data(), cols, restored.data());
    for (size_t i = 0; i < cols; ++i) {
        EXPECT_NEAR(data[i], restored[i], 5e-2f);
    }
}

TEST(QuantizationTest, Q8_1RoundTripQuantization) {
    using namespace mlc::runtime;
    constexpr size_t cols = 64;
    std::vector<float> data(cols);
    for (size_t i = 0; i < cols; ++i) {
        data[i] = 0.03f * static_cast<float>(std::cos(i * 0.19f)) + 0.01f;
    }
    std::vector<uint8_t> quant;
    quantizeRowQ8_1(data.data(), cols, quant);
    std::vector<float> restored(cols);
    dequantizeRowQ8_1(quant.data(), cols, restored.data());
    for (size_t i = 0; i < cols; ++i) {
        EXPECT_NEAR(data[i], restored[i], 7.5e-2f);
    }
}

TEST(QuantizationTest, Q2KRoundTripQuantization) {
    using namespace mlc::runtime;
    RunRoundTripQuant(quantizeRowQ2_K, dequantizeRowQ2_K, 256, 3.5e-1f);
}

TEST(QuantizationTest, Q3KRoundTripQuantization) {
    using namespace mlc::runtime;
    RunRoundTripQuant(quantizeRowQ3_K, dequantizeRowQ3_K, 256, 3.0e-1f);
}

TEST(QuantizationTest, Q4KRoundTripQuantization) {
    using namespace mlc::runtime;
    RunRoundTripQuant(quantizeRowQ4_K, dequantizeRowQ4_K, 256, 2.0e-1f);
}

TEST(QuantizationTest, Q5KRoundTripQuantization) {
    using namespace mlc::runtime;
    RunRoundTripQuant(quantizeRowQ5_K, dequantizeRowQ5_K, 256, 2.0e-1f);
}

TEST(QuantizationTest, Q6KRoundTripQuantization) {
    using namespace mlc::runtime;
    RunRoundTripQuant(quantizeRowQ6_K, dequantizeRowQ6_K, 256, 2.5e-1f);
}

TEST(QuantizationTest, Q8KRoundTripQuantization) {
    using namespace mlc::runtime;
    RunRoundTripQuant(quantizeRowQ8_K, dequantizeRowQ8_K, 256, 1.0e-1f);
}

TEST(KernelRegistryTest, SelectsQuantizedAttentionKernel) {
    using namespace mlc::runtime;
    KernelSelectionQuery query;
    query.op = ExecOpType::Attention;
    query.preferred_backend = BackendKind::Metal;
    query.activation_dtype = mlc::ir::DataType::F32;
    query.weight_dtype = mlc::ir::DataType::I8;
    query.activation_quantized = true;
    query.weight_quantized = true;
    query.grouped_query = true;
    query.head_dim = 128;
    const auto& registry = KernelDescriptorRegistry::Instance();
    const KernelDescriptor* descriptor = registry.select(query);
    ASSERT_NE(descriptor, nullptr);
    EXPECT_EQ(descriptor->backend, BackendKind::Metal);
    EXPECT_TRUE(descriptor->supports_quant_weight);
    EXPECT_FALSE(descriptor->id.empty());
}

TEST(ExecutionExecutorTest, TraceCapturesKernelNotes) {
    using namespace mlc::runtime;
    ExecutionGraph graph;
    auto& input = graph.addTensor("input", {1}, mlc::ir::DataType::F32);
    input.is_state = true;
    graph.addTensor("output", {1}, mlc::ir::DataType::F32);
    auto& node = graph.addNode("dummy",
                               ExecOpType::MatMul,
                               {"input"},
                               {"output"},
                               BackendKind::CPU);
    node.kernel_id = "cpu.matmul.f32";
    ExecutionExecutor executor(graph);
    auto result = executor.run();
    ASSERT_EQ(result.trace.size(), 1u);
    bool found_kernel = false;
    for (const auto& note : result.trace[0].notes) {
        if (note.find("kernel=") != std::string::npos) {
            found_kernel = true;
            break;
        }
    }
    EXPECT_TRUE(found_kernel);
}

TEST(QuantizationTest, Q2KSimplePatternProducesUniformValues) {
    using namespace mlc::runtime;
    constexpr size_t cols = 256; // one Q2_K block
    size_t row_bytes = q2_kRowSize(cols);
    std::vector<uint8_t> buffer(row_bytes, 0);

    // scales occupy first QK_K/16 bytes => 16 bytes for Q2_K
    std::fill(buffer.begin(), buffer.begin() + 16, 0x01); // scale=1, min=0
    // qs occupies next QK_K/4 bytes => 64 bytes
    std::fill(buffer.begin() + 16, buffer.begin() + 16 + 64, 0x55); // 0b01010101 so every 2-bit value == 1
    // d located after scales+qs
    uint16_t* d_ptr = reinterpret_cast<uint16_t*>(buffer.data() + 16 + 64);
    d_ptr[0] = floatToFp16(0.5f); // d
    d_ptr[1] = floatToFp16(0.0f); // dmin

    std::vector<float> output(cols, 0.0f);
    dequantizeRowQ2_K(buffer.data(), cols, output.data());

    for (float v : output) {
        EXPECT_FLOAT_EQ(v, 0.5f);
    }
}

static void testQkDotProduct(const std::vector<uint8_t>& buffer,
                             size_t cols,
                             void (*dequant)(const uint8_t*, size_t, float*),
                             float (*dot)(const uint8_t*, size_t, const float*)) {
    std::vector<float> vec(cols);
    for (size_t i = 0; i < cols; ++i) {
        vec[i] = 0.0025f * static_cast<float>((i % 17) + 1);
    }
    std::vector<float> deq(cols);
    dequant(buffer.data(), cols, deq.data());
    float expected = 0.0f;
    for (size_t i = 0; i < cols; ++i) {
        expected += deq[i] * vec[i];
    }
    float fast = dot(buffer.data(), cols, vec.data());
    EXPECT_NEAR(fast, expected, 1e-4f);
}

TEST(QuantizationTest, QKDotProductsMatchDequant) {
    using namespace mlc::runtime;
    constexpr size_t cols = 256;
    {
        size_t row_bytes = q2_kRowSize(cols);
        std::vector<uint8_t> buffer(row_bytes, 0x11);
        uint16_t* d_ptr = reinterpret_cast<uint16_t*>(buffer.data() + 16 + 64);
        d_ptr[0] = floatToFp16(0.8f);
        d_ptr[1] = floatToFp16(0.2f);
        testQkDotProduct(buffer, cols, dequantizeRowQ2_K, dotProductRowQ2_K);
    }
    {
        size_t row_bytes = q3_kRowSize(cols);
        std::vector<uint8_t> buffer(row_bytes, 0x22);
        uint16_t* d_ptr = reinterpret_cast<uint16_t*>(buffer.data() + 80);
        d_ptr[0] = floatToFp16(0.6f);
        testQkDotProduct(buffer, cols, dequantizeRowQ3_K, dotProductRowQ3_K);
    }
    {
        size_t row_bytes = q4_kRowSize(cols);
        std::vector<uint8_t> buffer(row_bytes, 0x33);
        uint16_t* d_ptr = reinterpret_cast<uint16_t*>(buffer.data() + 144 - 4);
        d_ptr[0] = floatToFp16(0.5f);
        d_ptr[1] = floatToFp16(0.1f);
        testQkDotProduct(buffer, cols, dequantizeRowQ4_K, dotProductRowQ4_K);
    }
    {
        size_t row_bytes = q5_kRowSize(cols);
        std::vector<uint8_t> buffer(row_bytes, 0x44);
        uint16_t* d_ptr = reinterpret_cast<uint16_t*>(buffer.data() + 176 - 4);
        d_ptr[0] = floatToFp16(0.4f);
        d_ptr[1] = floatToFp16(0.05f);
        testQkDotProduct(buffer, cols, dequantizeRowQ5_K, dotProductRowQ5_K);
    }
    {
        size_t row_bytes = q6_kRowSize(cols);
        std::vector<uint8_t> buffer(row_bytes, 0x55);
        uint16_t* d_ptr = reinterpret_cast<uint16_t*>(buffer.data() + 210 - 2);
        d_ptr[0] = floatToFp16(0.3f);
        testQkDotProduct(buffer, cols, dequantizeRowQ6_K, dotProductRowQ6_K);
    }
    {
        size_t row_bytes = q8_kRowSize(cols);
        std::vector<uint8_t> buffer(row_bytes, 0x66);
        float* d_ptr = reinterpret_cast<float*>(buffer.data());
        d_ptr[0] = 0.2f;
        testQkDotProduct(buffer, cols, dequantizeRowQ8_K, dotProductRowQ8_K);
    }
}

TEST(ModelRunnerTest, DryRunProducesEmbeddingAndLogits) {
    std::string path = createRunnerGGUFFile();
    mlc::runtime::ModelRunner runner(path);
    mlc::runtime::RunConfig config;
    config.token_id = 1;
    config.preview_length = 3;

    auto report = runner.dryRun(config);
    EXPECT_EQ(report.embedding_tensor, "tok_embeddings.weight");
    ASSERT_EQ(report.embedding_preview.size(), 3u);
    EXPECT_FLOAT_EQ(report.embedding_preview[0], 1.0f);
    EXPECT_FLOAT_EQ(report.embedding_preview[1], 1.1f);
    EXPECT_FLOAT_EQ(report.embedding_preview[2], 1.2f);
    EXPECT_EQ(report.embedding_dim, 3u);

    EXPECT_EQ(report.logits_tensor, "output.weight");
    ASSERT_EQ(report.logits_preview.size(), 2u);
    EXPECT_FLOAT_EQ(report.logits_preview[0], 1.0f);
    EXPECT_FLOAT_EQ(report.logits_preview[1], 1.1f);
    EXPECT_TRUE(report.logits_error.empty());
    EXPECT_EQ(report.schedule.size(), report.plan.nodes().size());

    fs::remove(path);
}

TEST(ModelRunnerTest, DryRunHandlesTransposedQuantizedHeadTensor) {
    std::string path = createTransposedRunnerGGUFFile();
    mlc::runtime::ModelRunner runner(path);
    mlc::runtime::RunConfig config;
    config.token_id = 1;
    config.preview_length = 3;

    auto report = runner.dryRun(config);
    EXPECT_EQ(report.embedding_tensor, "tok_embeddings.weight");
    EXPECT_EQ(report.logits_tensor, "output.weight");
    EXPECT_TRUE(report.logits_error.empty());
    ASSERT_EQ(report.logits_preview.size(), 3u);
    EXPECT_NEAR(report.logits_preview[0], 17.3f, 1e-5f);
    EXPECT_NEAR(report.logits_preview[1], 20.6f, 1e-5f);
    EXPECT_NEAR(report.logits_preview[2], 23.9f, 1e-5f);
    EXPECT_EQ(report.logits_dim, 4u);

    fs::remove(path);
}

TEST(ExecutionRuntimeTest, ExecutesEmbeddingAndLinear) {
    std::string path = createRunnerGGUFFile();
    mlc::runtime::Session session(path);

    mlc::runtime::ExecutionGraph graph;
    graph.addTensor("token_ids", {1}, mlc::ir::DataType::I4);
    graph.addTensor("embedding_out", {3}, mlc::ir::DataType::F32);
    auto& embed_node = graph.addNode("embedding_lookup",
                                     mlc::runtime::ExecOpType::Embedding,
                                     {"token_ids"},
                                     {"embedding_out"},
                                     mlc::runtime::BackendKind::CPU);
    embed_node.annotations["weight"] = "tok_embeddings.weight";

    graph.addTensor("logits", {2}, mlc::ir::DataType::F32);
    auto& linear_node = graph.addNode("lm_head",
                                      mlc::runtime::ExecOpType::Linear,
                                      {"embedding_out"},
                                      {"logits"},
                                      mlc::runtime::BackendKind::CPU);
    linear_node.annotations["weight"] = "output.weight";

    mlc::runtime::ExecutionContext context(session, &graph);
    context.setToken(1);
    context.setTensor("token_ids", {1.0f});

    mlc::runtime::BackendRegistry registry;
    mlc::runtime::ExecutionExecutor executor(graph, &registry, &context);
    auto status = executor.run();

    EXPECT_TRUE(status.success);

    const auto* logits = context.getTensor("logits");
    ASSERT_NE(logits, nullptr);

    auto expected_embedding = session.getEmbedding("tok_embeddings.weight", 1);
    auto expected_logits = session.runLinear("output.weight", expected_embedding);

    ASSERT_EQ(logits->size(), expected_logits.size());
    for (size_t i = 0; i < expected_logits.size(); ++i) {
        EXPECT_FLOAT_EQ((*logits)[i], expected_logits[i]);
    }

    fs::remove(path);
}

TEST(ExecutionRuntimeTest, AttentionUpdatesKvCache) {
    std::string path = createRunnerGGUFFile();
    mlc::runtime::Session session(path);

    mlc::runtime::ExecutionGraph graph;
    auto& q_tensor = graph.addTensor("q", {2}, mlc::ir::DataType::F32);
    auto& k_tensor = graph.addTensor("k_in", {2}, mlc::ir::DataType::F32);
    auto& v_tensor = graph.addTensor("v_in", {2}, mlc::ir::DataType::F32);
    (void)q_tensor;
    (void)k_tensor;
    (void)v_tensor;

    auto& k_cache = graph.addTensor("kv_k", std::vector<int64_t>{1, 4, 2}, mlc::ir::DataType::F32);
    k_cache.is_state = true;
    auto& v_cache = graph.addTensor("kv_v", std::vector<int64_t>{1, 4, 2}, mlc::ir::DataType::F32);
    v_cache.is_state = true;

    graph.addTensor("attn_out", {2}, mlc::ir::DataType::F32);

    auto& attn = graph.addNode("attention",
                               mlc::runtime::ExecOpType::Attention,
                               {"q", "k_in", "v_in"},
                               {"attn_out", "kv_k", "kv_v"},
                               mlc::runtime::BackendKind::CPU);
    attn.attributes["heads"] = 1.0f;
    attn.attributes["kv_heads"] = 1.0f;
    attn.attributes["head_dim"] = 2.0f;
    attn.annotations["kv_cache_k"] = "kv_k";
    attn.annotations["kv_cache_v"] = "kv_v";

    mlc::runtime::ExecutionContext context(session, &graph);
    context.setSequencePosition(0);
    context.setTensor("q", {1.0f, 0.0f});
    context.setTensor("k_in", {0.25f, 0.0f});
    context.setTensor("v_in", {0.5f, 1.0f});

    ASSERT_FALSE(context.hasTensor("kv_k"));
    mlc::runtime::ExecutionExecutor executor(graph, nullptr, &context);
    ASSERT_TRUE(context.hasTensor("kv_k"));
    ASSERT_TRUE(context.hasTensor("kv_v"));
    auto result = executor.run();

    EXPECT_TRUE(result.success);
    const auto* attn_out = context.getTensor("attn_out");
    ASSERT_NE(attn_out, nullptr);
    ASSERT_EQ(attn_out->size(), 2u);
    EXPECT_NEAR((*attn_out)[0], 0.5f, 1e-5f);
    EXPECT_NEAR((*attn_out)[1], 1.0f, 1e-5f);

    const auto* cache_k_ptr = context.getTensor("kv_k");
    ASSERT_NE(cache_k_ptr, nullptr);
    EXPECT_FLOAT_EQ((*cache_k_ptr)[0], 0.25f);
    EXPECT_FLOAT_EQ((*cache_k_ptr)[1], 0.0f);

    const auto* cache_v_ptr = context.getTensor("kv_v");
    ASSERT_NE(cache_v_ptr, nullptr);
    EXPECT_FLOAT_EQ((*cache_v_ptr)[0], 0.5f);
    EXPECT_FLOAT_EQ((*cache_v_ptr)[1], 1.0f);

    fs::remove(path);
}

#if defined(__APPLE__)
TEST(MetalRuntimeTest, MatMulMatchesCPUWhenAvailable) {
    auto& executor = mlc::runtime::MetalExecutor::Instance();
    if (!executor.isAvailable()) {
        GTEST_SKIP() << "Metal device unavailable";
    }

    const size_t rows = 4;
    const size_t cols = 4;
    std::vector<float> weights = {
        1.0f, 0.5f, -0.25f, 2.0f,
        -1.0f, 0.0f, 0.75f, 0.5f,
        0.3f, 1.1f, -0.6f, 0.2f,
        2.2f, -0.4f, 0.8f, 1.0f
    };
    std::vector<float> input = {0.2f, -0.5f, 1.0f, 0.7f};

    std::vector<float> gpu_output;
    ASSERT_TRUE(executor.runMatMul(weights, input, rows, cols, false, gpu_output));
    ASSERT_EQ(gpu_output.size(), rows);

    std::vector<float> expected(rows, 0.0f);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            expected[r] += weights[r * cols + c] * input[c];
        }
    }

    for (size_t i = 0; i < rows; ++i) {
        EXPECT_NEAR(gpu_output[i], expected[i], 1e-4f);
    }
}

TEST(MetalRuntimeTest, MatMulWithBiasMatchesCPUWhenAvailable) {
    auto& executor = mlc::runtime::MetalExecutor::Instance();
    if (!executor.isAvailable()) {
        GTEST_SKIP() << "Metal device unavailable";
    }
    const size_t rows = 3;
    const size_t cols = 3;
    std::vector<float> weights = {
        0.5f, -0.2f, 1.0f,
        -1.5f, 0.75f, 0.25f,
        0.3f, 0.8f, -0.6f
    };
    std::vector<float> input = {1.0f, -0.5f, 0.25f};
    std::vector<float> bias = {0.1f, -0.2f, 0.05f};
    std::vector<float> expected(rows, 0.0f);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            expected[r] += weights[r * cols + c] * input[c];
        }
        expected[r] += bias[r];
    }
    std::vector<float> gpu_output;
    ASSERT_TRUE(executor.runMatMul(weights, input, rows, cols, false, gpu_output, &bias));
    ASSERT_EQ(gpu_output.size(), rows);
    for (size_t i = 0; i < rows; ++i) {
        EXPECT_NEAR(gpu_output[i], expected[i], 1e-4f);
    }
}

TEST(MetalRuntimeTest, MatMulQ4_0MatchesCPUWhenAvailable) {
    auto& executor = mlc::runtime::MetalExecutor::Instance();
    if (!executor.isAvailable()) {
        GTEST_SKIP() << "Metal device unavailable";
    }
    constexpr size_t rows = 2;
    constexpr size_t cols = 32;
    std::vector<uint8_t> weights(rows * mlc::runtime::q4_0RowSize(cols, 1));
    for (size_t r = 0; r < rows; ++r) {
        uint8_t* row = weights.data() + r * mlc::runtime::q4_0RowSize(cols, 1);
        uint16_t* d_ptr = reinterpret_cast<uint16_t*>(row);
        d_ptr[0] = floatToFp16(0.5f + 0.1f * r);
        uint8_t* qs = row + 2;
        for (size_t j = 0; j < cols / 2; ++j) {
            uint8_t lo = static_cast<uint8_t>((j + r) & 0xF);
            uint8_t hi = static_cast<uint8_t>(((cols / 2 - j) + r) & 0xF);
            qs[j] = static_cast<uint8_t>((hi << 4) | lo);
        }
    }
    std::vector<float> input(cols);
    for (size_t i = 0; i < cols; ++i) input[i] = 0.01f * static_cast<float>(i + 1);
    std::vector<float> expected(rows);
    for (size_t r = 0; r < rows; ++r) {
        const uint8_t* row = weights.data() + r * mlc::runtime::q4_0RowSize(cols, 1);
        expected[r] = mlc::runtime::dotProductRowQ4_0(row, cols, 1, input.data());
    }
    std::vector<float> gpu_output;
    if (!executor.runMatMulQ4_0(weights,
                                input,
                                rows,
                                cols,
                                mlc::runtime::q4_0RowSize(cols, 1),
                                1,
                                gpu_output)) {
        GTEST_SKIP() << "Metal Q4_0 matmul unavailable";
    }
    ASSERT_EQ(gpu_output.size(), expected.size());
    for (size_t i = 0; i < rows; ++i) {
        EXPECT_NEAR(gpu_output[i], expected[i], 1e-3f);
    }
}

TEST(MetalRuntimeTest, MatMulQ4_0WithBiasMatchesCPUWhenAvailable) {
    auto& executor = mlc::runtime::MetalExecutor::Instance();
    if (!executor.isAvailable()) {
        GTEST_SKIP() << "Metal device unavailable";
    }
    constexpr size_t rows = 2;
    constexpr size_t cols = 32;
    std::vector<uint8_t> weights(rows * mlc::runtime::q4_0RowSize(cols, 1));
    for (size_t r = 0; r < rows; ++r) {
        uint8_t* row = weights.data() + r * mlc::runtime::q4_0RowSize(cols, 1);
        uint16_t* d_ptr = reinterpret_cast<uint16_t*>(row);
        d_ptr[0] = floatToFp16(0.5f + 0.1f * r);
        uint8_t* qs = row + 2;
        for (size_t j = 0; j < cols / 2; ++j) {
            uint8_t lo = static_cast<uint8_t>((j + r) & 0xF);
            uint8_t hi = static_cast<uint8_t>(((cols / 2 - j) + r) & 0xF);
            qs[j] = static_cast<uint8_t>((hi << 4) | lo);
        }
    }
    std::vector<float> input(cols);
    for (size_t i = 0; i < cols; ++i) input[i] = 0.01f * static_cast<float>(i + 1);
    std::vector<float> bias = {0.25f, -0.15f};
    std::vector<float> expected(rows);
    for (size_t r = 0; r < rows; ++r) {
        const uint8_t* row = weights.data() + r * mlc::runtime::q4_0RowSize(cols, 1);
        expected[r] = mlc::runtime::dotProductRowQ4_0(row, cols, 1, input.data()) + bias[r];
    }
    std::vector<float> gpu_output;
    if (!executor.runMatMulQ4_0(weights,
                                input,
                                rows,
                                cols,
                                mlc::runtime::q4_0RowSize(cols, 1),
                                1,
                                gpu_output,
                                &bias)) {
        GTEST_SKIP() << "Metal Q4_0 matmul unavailable";
    }
    ASSERT_EQ(gpu_output.size(), expected.size());
    for (size_t i = 0; i < rows; ++i) {
        EXPECT_NEAR(gpu_output[i], expected[i], 1e-3f);
    }
}

#if defined(__APPLE__)
TEST(MetalRuntimeTest, AttentionUsesSharedKvBuffersWhenAvailable) {
    auto& executor = mlc::runtime::MetalExecutor::Instance();
    if (!executor.isAvailable()) {
        GTEST_SKIP() << "Metal device unavailable";
    }

    const size_t num_heads = 1;
    const size_t kv_heads = 1;
    const size_t head_dim = 2;
    const size_t context_length = 4;

    std::vector<float> q = {0.2f, -0.1f};
    std::vector<float> k = {0.5f, 0.25f};
    std::vector<float> v = {1.0f, -2.0f};
    std::vector<float> mask;

    std::vector<float> kv_cache_k(kv_heads * context_length * head_dim, 0.0f);
    std::vector<float> kv_cache_v(kv_heads * context_length * head_dim, 0.0f);
    mlc::runtime::MetalBufferHandle k_handle;
    mlc::runtime::MetalBufferHandle v_handle;
    ASSERT_TRUE(executor.ensureSharedBuffer(kv_cache_k, k_handle));
    ASSERT_TRUE(executor.ensureSharedBuffer(kv_cache_v, v_handle));

    std::vector<float> output;
    ASSERT_TRUE(executor.runAttention(q,
                                      k,
                                      v,
                                      num_heads,
                                      kv_heads,
                                      head_dim,
                                      context_length,
                                      mask,
                                      0,
                                      0,
                                      10000.0f,
                                      1.0f,
                                      kv_cache_k,
                                      kv_cache_v,
                                      &k_handle,
                                      &v_handle,
                                      output));

    ASSERT_EQ(kv_cache_k[0], k[0]);
    ASSERT_EQ(kv_cache_k[1], k[1]);
    ASSERT_EQ(kv_cache_v[0], v[0]);
    ASSERT_EQ(kv_cache_v[1], v[1]);
    ASSERT_EQ(output.size(), q.size());

    executor.releaseBuffer(k_handle);
    executor.releaseBuffer(v_handle);
}

TEST(MetalRuntimeTest, ScatterKvCacheWritesBuffer) {
    auto& executor = mlc::runtime::MetalExecutor::Instance();
    if (!executor.isAvailable()) {
        GTEST_SKIP() << "Metal device unavailable";
    }
    std::vector<float> cache = {0.0f, 0.0f, 0.0f, 0.0f};
    mlc::runtime::MetalBufferHandle handle;
    ASSERT_TRUE(executor.ensureSharedBuffer(cache, handle));
    std::vector<float> src = {1.0f, 2.0f, 3.0f, 4.0f};
    ASSERT_TRUE(executor.scatterKVCache(src, handle,
                                        /*kv_heads=*/1,
                                        /*tokens=*/2,
                                        /*head_dim=*/2,
                                        /*context_length=*/2,
                                        /*base_position=*/0));
    EXPECT_FLOAT_EQ(cache[0], 1.0f);
    EXPECT_FLOAT_EQ(cache[1], 2.0f);
    EXPECT_FLOAT_EQ(cache[2], 3.0f);
    EXPECT_FLOAT_EQ(cache[3], 4.0f);
    executor.releaseBuffer(handle);
}
#endif

TEST(MetalRuntimeTest, MatMulQ4_1MatchesCPUWhenAvailable) {
    auto& executor = mlc::runtime::MetalExecutor::Instance();
    if (!executor.isAvailable()) {
        GTEST_SKIP() << "Metal device unavailable";
    }
    constexpr size_t rows = 2;
    constexpr size_t cols = 32;
    std::vector<uint8_t> weights(rows * mlc::runtime::q4_1RowSize(cols));
    for (size_t r = 0; r < rows; ++r) {
        uint8_t* row = weights.data() + r * mlc::runtime::q4_1RowSize(cols);
        uint16_t* d_ptr = reinterpret_cast<uint16_t*>(row);
        d_ptr[0] = floatToFp16(0.25f + 0.05f * r);
        d_ptr[1] = floatToFp16(0.1f * r);
        uint8_t* qs = row + 4;
        for (size_t j = 0; j < cols / 2; ++j) {
            uint8_t lo = static_cast<uint8_t>((j + r) & 0xF);
            uint8_t hi = static_cast<uint8_t>(((rows + j) & 0xF));
            qs[j] = static_cast<uint8_t>((hi << 4) | lo);
        }
    }
    std::vector<float> input(cols);
    for (size_t i = 0; i < cols; ++i) input[i] = 0.02f * static_cast<float>(i + 1);
    std::vector<float> expected(rows);
    for (size_t r = 0; r < rows; ++r) {
        const uint8_t* row = weights.data() + r * mlc::runtime::q4_1RowSize(cols);
        expected[r] = mlc::runtime::dotProductRowQ4_1(row, cols, input.data());
    }
    std::vector<float> gpu_output;
    if (!executor.runMatMulQ4_1(weights,
                                input,
                                rows,
                                cols,
                                mlc::runtime::q4_1RowSize(cols),
                                gpu_output)) {
        GTEST_SKIP() << "Metal Q4_1 matmul unavailable";
    }
    ASSERT_EQ(gpu_output.size(), expected.size());
    for (size_t i = 0; i < rows; ++i) {
        EXPECT_NEAR(gpu_output[i], expected[i], 1e-3f);
    }
}

TEST(MetalRuntimeTest, MatMulQ5MatchesCPUWhenAvailable) {
    auto& executor = mlc::runtime::MetalExecutor::Instance();
    if (!executor.isAvailable()) {
        GTEST_SKIP() << "Metal device unavailable";
    }
    constexpr size_t cols = 32;
    auto makeRow = [&](float scale, float offset) {
        std::vector<uint8_t> row(offset == 0.0f ? mlc::runtime::q5_0RowSize(cols)
                                                : mlc::runtime::q5_1RowSize(cols));
        uint8_t* ptr = row.data();
        uint16_t* header = reinterpret_cast<uint16_t*>(ptr);
        header[0] = floatToFp16(scale);
        size_t qh_offset = (offset == 0.0f) ? 2 : 4;
        if (offset != 0.0f) {
            header[1] = floatToFp16(offset);
        }
        uint8_t* qh = ptr + qh_offset;
        uint8_t* qs = ptr + qh_offset + 4;
        for (size_t j = 0; j < 4; ++j) qh[j] = static_cast<uint8_t>(0xAA + j);
        for (size_t j = 0; j < cols / 2; ++j) {
            uint8_t lo = static_cast<uint8_t>((j + 1) & 0xF);
            uint8_t hi = static_cast<uint8_t>(((cols / 2 - j) & 0xF));
            qs[j] = static_cast<uint8_t>((hi << 4) | lo);
        }
        return row;
    };
    std::vector<uint8_t> row_q5_0 = makeRow(0.3f, 0.0f);
    std::vector<uint8_t> row_q5_1 = makeRow(0.4f, 0.1f);
    std::vector<uint8_t> weights(row_q5_0.size() + row_q5_1.size());
    std::memcpy(weights.data(), row_q5_0.data(), row_q5_0.size());
    std::memcpy(weights.data() + row_q5_0.size(), row_q5_1.data(), row_q5_1.size());

    std::vector<float> input(cols);
    for (size_t i = 0; i < cols; ++i) input[i] = -0.03f * static_cast<float>(i + 1);

    std::vector<float> expected = {
        mlc::runtime::dotProductRowQ5_0(row_q5_0.data(), cols, input.data()),
        mlc::runtime::dotProductRowQ5_1(row_q5_1.data(), cols, input.data())
    };

    std::vector<float> gpu_q5_0;
    if (!executor.runMatMulQ5_0(row_q5_0,
                                input,
                                1,
                                cols,
                                mlc::runtime::q5_0RowSize(cols),
                                gpu_q5_0)) {
        GTEST_SKIP() << "Metal Q5_0 matmul unavailable";
    }
    std::vector<float> gpu_q5_1;
    if (!executor.runMatMulQ5_1(row_q5_1,
                                input,
                                1,
                                cols,
                                mlc::runtime::q5_1RowSize(cols),
                                gpu_q5_1)) {
        GTEST_SKIP() << "Metal Q5_1 matmul unavailable";
    }
    EXPECT_NEAR(gpu_q5_0[0], expected[0], 1e-3f);
    EXPECT_NEAR(gpu_q5_1[0], expected[1], 1e-3f);
}

TEST(MetalRuntimeTest, MatMulQ4KMatchesCPUWhenAvailable) {
    auto& executor = mlc::runtime::MetalExecutor::Instance();
    if (!executor.isAvailable()) {
        GTEST_SKIP() << "Metal device unavailable";
    }
    constexpr size_t cols = 256;
    size_t stride = mlc::runtime::q4_kRowSize(cols);
    std::vector<uint8_t> row(stride, 0x5A);
    uint16_t* d_ptr = reinterpret_cast<uint16_t*>(row.data() + 12 + 128);
    d_ptr[0] = floatToFp16(0.45f);
    d_ptr[1] = floatToFp16(0.05f);
    std::vector<float> input(cols);
    for (size_t i = 0; i < cols; ++i) input[i] = 0.002f * static_cast<float>(i + 1);
    float expected = mlc::runtime::dotProductRowQ4_K(row.data(), cols, input.data());
    std::vector<float> gpu_out;
    if (!executor.runMatMulQ4K(row, input, 1, cols, stride, gpu_out)) {
        GTEST_SKIP() << "Metal Q4_K matmul unavailable";
    }
    EXPECT_NEAR(gpu_out[0], expected, 1e-3f);
}

TEST(MetalRuntimeTest, MatMulQ5KMatchesCPUWhenAvailable) {
    auto& executor = mlc::runtime::MetalExecutor::Instance();
    if (!executor.isAvailable()) {
        GTEST_SKIP() << "Metal device unavailable";
    }
    constexpr size_t cols = 256;
    size_t stride = mlc::runtime::q5_kRowSize(cols);
    std::vector<uint8_t> row(stride, 0x6C);
    uint16_t* d_ptr = reinterpret_cast<uint16_t*>(row.data() + 12 + 32 + 128);
    d_ptr[0] = floatToFp16(0.35f);
    d_ptr[1] = floatToFp16(0.02f);
    std::vector<float> input(cols);
    for (size_t i = 0; i < cols; ++i) input[i] = -0.0015f * static_cast<float>(i + 1);
    float expected = mlc::runtime::dotProductRowQ5_K(row.data(), cols, input.data());
    std::vector<float> gpu_out;
    if (!executor.runMatMulQ5K(row, input, 1, cols, stride, gpu_out)) {
        GTEST_SKIP() << "Metal Q5_K matmul unavailable";
    }
    EXPECT_NEAR(gpu_out[0], expected, 1e-3f);
}

TEST(MetalRuntimeTest, MatMulQ6KMatchesCPUWhenAvailable) {
    auto& executor = mlc::runtime::MetalExecutor::Instance();
    if (!executor.isAvailable()) {
        GTEST_SKIP() << "Metal device unavailable";
    }
    constexpr size_t cols = 256;
    size_t stride = mlc::runtime::q6_kRowSize(cols);
    std::vector<uint8_t> row(stride, 0x77);
    uint16_t* d_ptr = reinterpret_cast<uint16_t*>(row.data() + 128 + 64 + 32);
    d_ptr[0] = floatToFp16(0.25f);
    std::vector<float> input(cols);
    for (size_t i = 0; i < cols; ++i) input[i] = 0.0005f * static_cast<float>(i + 1);
    float expected = mlc::runtime::dotProductRowQ6_K(row.data(), cols, input.data());
    std::vector<float> gpu_out;
    if (!executor.runMatMulQ6K(row, input, 1, cols, stride, gpu_out)) {
        GTEST_SKIP() << "Metal Q6_K matmul unavailable";
    }
    EXPECT_NEAR(gpu_out[0], expected, 1e-3f);
}

TEST(MetalRuntimeTest, MatMulQ8KMatchesCPUWhenAvailable) {
    auto& executor = mlc::runtime::MetalExecutor::Instance();
    if (!executor.isAvailable()) {
        GTEST_SKIP() << "Metal device unavailable";
    }
    constexpr size_t cols = 256;
    size_t stride = mlc::runtime::q8_kRowSize(cols);
    std::vector<uint8_t> row(stride, 0x88);
    float* d_ptr = reinterpret_cast<float*>(row.data());
    d_ptr[0] = 0.15f;
    std::vector<float> input(cols);
    for (size_t i = 0; i < cols; ++i) input[i] = -0.0008f * static_cast<float>(i + 1);
    float expected = mlc::runtime::dotProductRowQ8_K(row.data(), cols, input.data());
    std::vector<float> gpu_out;
    if (!executor.runMatMulQ8K(row, input, 1, cols, stride, gpu_out)) {
        GTEST_SKIP() << "Metal Q8_K matmul unavailable";
    }
    EXPECT_NEAR(gpu_out[0], expected, 1e-3f);
}

TEST(MetalRuntimeTest, MatMulQ8_0MatchesCPUWhenAvailable) {
    auto& executor = mlc::runtime::MetalExecutor::Instance();
    if (!executor.isAvailable()) {
        GTEST_SKIP() << "Metal device unavailable";
    }
    constexpr size_t cols = 32;
    size_t stride = mlc::runtime::q8_0RowSize(cols);
    std::vector<uint8_t> row(stride, 0);
    uint16_t* header = reinterpret_cast<uint16_t*>(row.data());
    header[0] = floatToFp16(0.3f);
    int8_t* qs = reinterpret_cast<int8_t*>(row.data() + 2);
    for (size_t i = 0; i < cols; ++i) {
        qs[i] = static_cast<int8_t>(i - 16);
    }
    std::vector<float> input(cols);
    for (size_t i = 0; i < cols; ++i) input[i] = 0.05f * static_cast<float>(i + 1);
    float expected = mlc::runtime::dotProductRowQ8_0(row.data(), cols, input.data());
    std::vector<float> gpu_out;
    if (!executor.runMatMulQ8_0(row, input, 1, cols, stride, gpu_out)) {
        GTEST_SKIP() << "Metal Q8_0 matmul unavailable";
    }
    EXPECT_NEAR(gpu_out[0], expected, 1e-3f);
}

TEST(MetalRuntimeTest, MatMulQ8_1MatchesCPUWhenAvailable) {
    auto& executor = mlc::runtime::MetalExecutor::Instance();
    if (!executor.isAvailable()) {
        GTEST_SKIP() << "Metal device unavailable";
    }
    constexpr size_t cols = 32;
    size_t stride = mlc::runtime::q8_1RowSize(cols);
    std::vector<uint8_t> row(stride, 0);
    uint16_t* header = reinterpret_cast<uint16_t*>(row.data());
    header[0] = floatToFp16(0.2f);
    header[1] = floatToFp16(4.0f);
    int8_t* qs = reinterpret_cast<int8_t*>(row.data() + 4);
    for (size_t i = 0; i < cols; ++i) {
        qs[i] = static_cast<int8_t>(15 - static_cast<int>(i));
    }
    std::vector<float> input(cols);
    for (size_t i = 0; i < cols; ++i) input[i] = -0.04f * static_cast<float>(i + 1);
    float expected = mlc::runtime::dotProductRowQ8_1(row.data(), cols, input.data());
    std::vector<float> gpu_out;
    if (!executor.runMatMulQ8_1(row, input, 1, cols, stride, gpu_out)) {
        GTEST_SKIP() << "Metal Q8_1 matmul unavailable";
    }
    EXPECT_NEAR(gpu_out[0], expected, 1e-3f);
}

TEST(MetalRuntimeTest, MatMulQ2KMatchesCPUWhenAvailable) {
    auto& executor = mlc::runtime::MetalExecutor::Instance();
    if (!executor.isAvailable()) {
        GTEST_SKIP() << "Metal device unavailable";
    }
    constexpr size_t rows = 1;
    constexpr size_t cols = 256;
    size_t stride = mlc::runtime::q2_kRowSize(cols);
    std::vector<uint8_t> weights(stride * rows, 0x13);
    uint16_t* d_ptr = reinterpret_cast<uint16_t*>(weights.data() + 16 + 64);
    d_ptr[0] = floatToFp16(0.6f);
    d_ptr[1] = floatToFp16(0.2f);
    std::vector<float> input(cols);
    for (size_t i = 0; i < cols; ++i) input[i] = 0.001f * static_cast<float>(i + 1);
    float expected = mlc::runtime::dotProductRowQ2_K(weights.data(), cols, input.data());
    std::vector<float> gpu_out;
    if (!executor.runMatMulQ2K(weights, input, rows, cols, stride, gpu_out)) {
        GTEST_SKIP() << "Metal Q2_K matmul unavailable";
    }
    EXPECT_NEAR(gpu_out[0], expected, 1e-3f);
}

TEST(MetalRuntimeTest, MatMulQ3KMatchesCPUWhenAvailable) {
    auto& executor = mlc::runtime::MetalExecutor::Instance();
    if (!executor.isAvailable()) {
        GTEST_SKIP() << "Metal device unavailable";
    }
    constexpr size_t rows = 1;
    constexpr size_t cols = 256;
    size_t stride = mlc::runtime::q3_kRowSize(cols);
    std::vector<uint8_t> weights(stride * rows, 0x22);
    uint16_t* d_ptr = reinterpret_cast<uint16_t*>(weights.data() + 64 + 16);
    d_ptr[0] = floatToFp16(0.5f);
    std::vector<float> input(cols);
    for (size_t i = 0; i < cols; ++i) input[i] = -0.002f * static_cast<float>(i + 1);
    float expected = mlc::runtime::dotProductRowQ3_K(weights.data(), cols, input.data());
    std::vector<float> gpu_out;
    if (!executor.runMatMulQ3K(weights, input, rows, cols, stride, gpu_out)) {
        GTEST_SKIP() << "Metal Q3_K matmul unavailable";
    }
    EXPECT_NEAR(gpu_out[0], expected, 1e-3f);
}

TEST(MetalRuntimeTest, FeedForwardMatchesCPUWhenAvailable) {
    auto& executor = mlc::runtime::MetalExecutor::Instance();
    if (!executor.isAvailable()) {
        GTEST_SKIP() << "Metal device unavailable";
    }
    std::vector<float> gate = {-2.0f, -0.5f, 0.0f, 0.25f, 0.75f, 1.5f, 2.5f};
    std::vector<float> up = {0.5f, -1.0f, 2.0f, -0.75f, 1.25f, 0.3f, -0.1f};
    std::vector<float> expected(gate.size());
    for (size_t i = 0; i < gate.size(); ++i) {
        float g = gate[i];
        float silu = g / (1.0f + std::exp(-g));
        expected[i] = silu * up[i];
    }
    std::vector<float> gpu_output;
    if (!executor.runFeedForward(gate, up, gpu_output)) {
        GTEST_SKIP() << "FeedForward Metal kernel unavailable";
    }
    ASSERT_EQ(gpu_output.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(gpu_output[i], expected[i], 1e-4f);
    }
}

TEST(MetalRuntimeTest, AddMatchesCPUWhenAvailable) {
    auto& executor = mlc::runtime::MetalExecutor::Instance();
    if (!executor.isAvailable()) {
        GTEST_SKIP() << "Metal device unavailable";
    }
    std::vector<float> a = {0.1f, -0.2f, 0.3f, -0.4f, 0.5f};
    std::vector<float> b = {1.0f, 1.5f, -1.0f, -1.5f, 0.0f};
    std::vector<float> expected(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        expected[i] = a[i] + b[i];
    }
    std::vector<float> gpu_output;
    if (!executor.runAdd(a, b, gpu_output)) {
        GTEST_SKIP() << "Metal add kernel unavailable";
    }
    ASSERT_EQ(gpu_output.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(gpu_output[i], expected[i], 1e-5f);
    }
}

TEST(MetalRuntimeTest, RmsNormMatchesCPUWhenAvailable) {
    auto& executor = mlc::runtime::MetalExecutor::Instance();
    if (!executor.isAvailable()) {
        GTEST_SKIP() << "Metal device unavailable";
    }
    std::vector<float> input = {0.25f, -0.5f, 0.75f, -1.0f};
    std::vector<float> weight = {1.0f, 1.1f, 0.9f, 1.0f};
    const float epsilon = 1e-5f;
    float sum = 0.0f;
    for (float v : input) sum += v * v;
    float mean_sq = sum / static_cast<float>(input.size());
    float inv = 1.0f / std::sqrt(mean_sq + epsilon);
    std::vector<float> expected(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        expected[i] = input[i] * inv * weight[i];
    }
    std::vector<float> gpu_output;
    if (!executor.runRmsNorm(input, weight, epsilon, gpu_output)) {
        GTEST_SKIP() << "Metal RMS norm kernel unavailable";
    }
    ASSERT_EQ(gpu_output.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(gpu_output[i], expected[i], 1e-4f);
    }
}

TEST(MetalRuntimeTest, SoftmaxMatchesCPUWhenAvailable) {
    auto& executor = mlc::runtime::MetalExecutor::Instance();
    if (!executor.isAvailable()) {
        GTEST_SKIP() << "Metal device unavailable";
    }
    std::vector<float> input = {1.0f, 2.0f, -1.0f, 0.5f};
    std::vector<float> expected(input.size());
    float max_val = *std::max_element(input.begin(), input.end());
    float sum = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        expected[i] = std::exp(input[i] - max_val);
        sum += expected[i];
    }
    for (float& v : expected) {
        v /= sum;
    }
    std::vector<float> gpu_output;
    if (!executor.runSoftmax(input, gpu_output)) {
        GTEST_SKIP() << "Metal softmax kernel unavailable";
    }
    ASSERT_EQ(gpu_output.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(gpu_output[i], expected[i], 1e-5f);
    }
}
TEST(MetalRuntimeTest, AttentionMatchesCPUForMultiTokensWhenAvailable) {
    auto& executor = mlc::runtime::MetalExecutor::Instance();
    if (!executor.isAvailable()) {
        GTEST_SKIP() << "Metal device unavailable";
    }

    std::string path = createRunnerGGUFFile();
    mlc::runtime::Session session(path);

    const size_t num_heads = 4;
    const size_t kv_heads = 2;
    const size_t head_dim = 2;
    const size_t context_length = 4;
    const size_t tokens_to_write = 3;

    mlc::runtime::ExecutionGraph graph;
    graph.addTensor("q", {static_cast<int64_t>(num_heads * head_dim)}, mlc::ir::DataType::F32);
    graph.addTensor("k_in", {static_cast<int64_t>(kv_heads * head_dim)}, mlc::ir::DataType::F32);
    graph.addTensor("v_in", {static_cast<int64_t>(kv_heads * head_dim)}, mlc::ir::DataType::F32);
    graph.addTensor("attn_mask", {static_cast<int64_t>(tokens_to_write)}, mlc::ir::DataType::F32);

    auto& k_cache = graph.addTensor("kv_k_multi_gpu",
                                    std::vector<int64_t>{static_cast<int64_t>(kv_heads),
                                                         static_cast<int64_t>(context_length),
                                                         static_cast<int64_t>(head_dim)},
                                    mlc::ir::DataType::F32);
    k_cache.is_state = true;
    auto& v_cache = graph.addTensor("kv_v_multi_gpu",
                                    std::vector<int64_t>{static_cast<int64_t>(kv_heads),
                                                         static_cast<int64_t>(context_length),
                                                         static_cast<int64_t>(head_dim)},
                                    mlc::ir::DataType::F32);
    v_cache.is_state = true;

    graph.addTensor("attn_multi_out_gpu",
                    {static_cast<int64_t>(num_heads * head_dim)},
                    mlc::ir::DataType::F32);

    auto& attn = graph.addNode("multi_attention_gpu",
                               mlc::runtime::ExecOpType::Attention,
                               {"q", "k_in", "v_in"},
                               {"attn_multi_out_gpu", "kv_k_multi_gpu", "kv_v_multi_gpu"},
                               mlc::runtime::BackendKind::CPU);
    attn.attributes["heads"] = static_cast<float>(num_heads);
    attn.attributes["kv_heads"] = static_cast<float>(kv_heads);
    attn.attributes["head_dim"] = static_cast<float>(head_dim);
    attn.annotations["kv_cache_k"] = "kv_k_multi_gpu";
    attn.annotations["kv_cache_v"] = "kv_v_multi_gpu";
    attn.annotations["attention_mask"] = "attn_mask";
    mlc::runtime::ModelConfig cfg;
    cfg.head_count = num_heads;
    cfg.kv_head_count = kv_heads;
    cfg.head_dim = head_dim;
    cfg.rotary_dim = head_dim;
    cfg.rope_freq_base = 10000.0f;
    cfg.rope_freq_scale = 1.0f;
    graph.setModelConfig(cfg);

    std::vector<float> q_values = {
        0.05f, 0.15f, 0.25f, 0.35f,
        0.45f, 0.55f, 0.65f, 0.75f
    };
    std::vector<float> k_values = {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f,
        0.9f, 1.0f, 1.1f, 1.2f
    };
    std::vector<float> v_values = {
        1.0f, 1.5f, 2.0f, 2.5f,
        3.0f, 3.5f, 4.0f, 4.5f,
        5.0f, 5.5f, 6.0f, 6.5f
    };
    std::vector<float> mask_values = {0.0f, -0.75f, -1.5f};

    mlc::runtime::ExecutionContext context(session, &graph);
    context.setSequencePosition(0);
    context.setTensor("q", q_values);
    context.setTensor("k_in", k_values);
    context.setTensor("v_in", v_values);
    context.setTensor("attn_mask", mask_values);

    mlc::runtime::ExecutionExecutor cpu_executor(graph, nullptr, &context);
    auto status = cpu_executor.run();
    ASSERT_TRUE(status.success);

    const auto* cpu_out = context.getTensor("attn_multi_out_gpu");
    ASSERT_NE(cpu_out, nullptr);
    auto expected_output = *cpu_out;
    const auto* cpu_k = context.getTensor("kv_k_multi_gpu");
    const auto* cpu_v = context.getTensor("kv_v_multi_gpu");
    ASSERT_NE(cpu_k, nullptr);
    ASSERT_NE(cpu_v, nullptr);
    auto expected_k = *cpu_k;
    auto expected_v = *cpu_v;

    std::vector<float> gpu_cache_k(expected_k.size(), 0.0f);
    std::vector<float> gpu_cache_v(expected_v.size(), 0.0f);
    std::vector<float> gpu_output;
    ASSERT_TRUE(executor.runAttention(q_values,
                                      k_values,
                                      v_values,
                                      num_heads,
                                      kv_heads,
                                      head_dim,
                                      context_length,
                                      mask_values,
                                      0,
                                      cfg.rotary_dim,
                                      cfg.rope_freq_base,
                                      cfg.rope_freq_scale,
                                      gpu_cache_k,
                                      gpu_cache_v,
                                      nullptr,
                                      nullptr,
                                      gpu_output));

    ASSERT_EQ(gpu_output.size(), expected_output.size());
    for (size_t i = 0; i < expected_output.size(); ++i) {
        EXPECT_NEAR(gpu_output[i], expected_output[i], 1e-4f);
    }
    ASSERT_EQ(gpu_cache_k.size(), expected_k.size());
    ASSERT_EQ(gpu_cache_v.size(), expected_v.size());
    for (size_t i = 0; i < expected_k.size(); ++i) {
        EXPECT_NEAR(gpu_cache_k[i], expected_k[i], 1e-5f);
        EXPECT_NEAR(gpu_cache_v[i], expected_v[i], 1e-5f);
    }

    fs::remove(path);
}
#endif

TEST(ExecutionRuntimeTest, AttentionHandlesMultipleTokensAndHeads) {
    std::string path = createRunnerGGUFFile();
    mlc::runtime::Session session(path);

    mlc::runtime::ExecutionGraph graph;
    graph.addTensor("q", {2}, mlc::ir::DataType::F32);
    graph.addTensor("k_in", {2}, mlc::ir::DataType::F32);
    graph.addTensor("v_in", {2}, mlc::ir::DataType::F32);

    auto& k_cache = graph.addTensor("kv_k_multi", std::vector<int64_t>{1, 4, 1}, mlc::ir::DataType::F32);
    k_cache.is_state = true;
    auto& v_cache = graph.addTensor("kv_v_multi", std::vector<int64_t>{1, 4, 1}, mlc::ir::DataType::F32);
    v_cache.is_state = true;

    graph.addTensor("attn_multi_out", {2}, mlc::ir::DataType::F32);
    graph.addTensor("attn_mask", {2}, mlc::ir::DataType::F32);

    auto& attn = graph.addNode("multi_attention",
                               mlc::runtime::ExecOpType::Attention,
                               {"q", "k_in", "v_in"},
                               {"attn_multi_out", "kv_k_multi", "kv_v_multi"},
                               mlc::runtime::BackendKind::CPU);
    attn.attributes["heads"] = 2.0f;
    attn.attributes["kv_heads"] = 1.0f;
    attn.attributes["head_dim"] = 1.0f;
    attn.annotations["kv_cache_k"] = "kv_k_multi";
    attn.annotations["kv_cache_v"] = "kv_v_multi";
    attn.annotations["attention_mask"] = "attn_mask";

    mlc::runtime::ExecutionContext context(session, &graph);
    context.setSequencePosition(0);
    context.setTensor("q", {1.0f, 1.0f});
    context.setTensor("k_in", {1.0f, 2.0f});
    context.setTensor("v_in", {10.0f, 20.0f});
    context.setTensor("attn_mask", {0.0f, -1e9f});

    mlc::runtime::ExecutionExecutor executor(graph, nullptr, &context);
    auto status = executor.run();
    EXPECT_TRUE(status.success);

    const auto* attn_out = context.getTensor("attn_multi_out");
    ASSERT_NE(attn_out, nullptr);
    ASSERT_EQ(attn_out->size(), 2u);
    float expected = 10.0f; // mask suppresses second token
    EXPECT_NEAR((*attn_out)[0], expected, 1e-4f);
    EXPECT_NEAR((*attn_out)[1], expected, 1e-4f);

    const auto* cache_k_ptr = context.getTensor("kv_k_multi");
    ASSERT_NE(cache_k_ptr, nullptr);
    EXPECT_FLOAT_EQ((*cache_k_ptr)[0], 1.0f);
    EXPECT_FLOAT_EQ((*cache_k_ptr)[1], 2.0f);

    const auto* cache_v_ptr = context.getTensor("kv_v_multi");
    ASSERT_NE(cache_v_ptr, nullptr);
    EXPECT_FLOAT_EQ((*cache_v_ptr)[0], 10.0f);
    EXPECT_FLOAT_EQ((*cache_v_ptr)[1], 20.0f);

    fs::remove(path);
}

TEST(ExecutionRuntimeTest, AttentionSupportsQuantizedCaches) {
    std::string path = createRunnerGGUFFile();
    mlc::runtime::Session session(path);

    mlc::runtime::ExecutionGraph graph;
    graph.addTensor("q", {2}, mlc::ir::DataType::F32);
    graph.addTensor("k_in", {2}, mlc::ir::DataType::F32);
    graph.addTensor("v_in", {2}, mlc::ir::DataType::F32);

    auto& k_cache = graph.addTensor("kv_k_quant", std::vector<int64_t>{1, 4, 1}, mlc::ir::DataType::F32);
    k_cache.is_state = true;
    k_cache.ggml_dtype = mlc::frontend::GGML_TYPE_Q4_0;
    k_cache.has_ggml_dtype = true;
    k_cache.quantized = true;
    k_cache.quant_version = 1;

    auto& v_cache = graph.addTensor("kv_v_quant", std::vector<int64_t>{1, 4, 1}, mlc::ir::DataType::F32);
    v_cache.is_state = true;
    v_cache.ggml_dtype = mlc::frontend::GGML_TYPE_Q4_0;
    v_cache.has_ggml_dtype = true;
    v_cache.quantized = true;
    v_cache.quant_version = 1;

    graph.addTensor("attn_out_quant", {2}, mlc::ir::DataType::F32);

    auto& attn = graph.addNode("quant_attention",
                               mlc::runtime::ExecOpType::Attention,
                               {"q", "k_in", "v_in"},
                               {"attn_out_quant", "kv_k_quant", "kv_v_quant"},
                               mlc::runtime::BackendKind::CPU);
    attn.attributes["heads"] = 1.0f;
    attn.attributes["kv_heads"] = 1.0f;
    attn.attributes["head_dim"] = 2.0f;
    attn.annotations["kv_cache_k"] = "kv_k_quant";
    attn.annotations["kv_cache_v"] = "kv_v_quant";

    mlc::runtime::ExecutionContext context(session, &graph);
    context.setSequencePosition(0);
    context.setTensor("q", {1.0f, 2.0f});
    context.setTensor("k_in", {0.5f, -0.5f});
    context.setTensor("v_in", {3.0f, 4.0f});

    mlc::runtime::ExecutionExecutor executor(graph, nullptr, &context);
    auto status = executor.run();
    ASSERT_TRUE(status.success);

    const auto* out = context.getTensor("attn_out_quant");
    ASSERT_NE(out, nullptr);
    ASSERT_EQ(out->size(), 2u);
    EXPECT_NEAR((*out)[0], 3.0f, 1e-3f);
    EXPECT_NEAR((*out)[1], 4.0f, 1e-3f);

    auto* cache_storage = context.tensorStorage("kv_k_quant");
    ASSERT_NE(cache_storage, nullptr);
    ASSERT_EQ(cache_storage->dtype, mlc::frontend::GGML_TYPE_Q4_0);
    std::vector<float> decoded(2, 0.0f);
    mlc::runtime::dequantizeRowTo(cache_storage->raw_data.data(),
                                  cache_storage->dtype,
                                  2,
                                  cache_storage->quant_version,
                                  decoded.data());
    EXPECT_NEAR(decoded[0], 0.5f, 5e-2f);
    EXPECT_NEAR(decoded[1], -0.5f, 5e-2f);

    fs::remove(path);
}

