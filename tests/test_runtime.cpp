#include <gtest/gtest.h>
#include "runtime/runtime.hpp"
#include "runtime/session.hpp"
#include "runtime/quantization.hpp"
#include "runtime/float_convert.hpp"
#include "runtime/quant_utils.hpp"
#include "runtime/model_runner.hpp"
#include "runtime/execution_graph.hpp"
#include "runtime/execution_executor.hpp"
#include "runtime/kernel_registry.hpp"
#include "runtime/metal_runtime.hpp"
#include "runtime/execution_context.hpp"
#include "runtime/tokenizer.hpp"
#include "runtime/decode_runner.hpp"
#include "tests/gguf_test_utils.hpp"
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <unistd.h>
#include <cstring>
#include <tuple>
#include <random>
#include <vector>

namespace fs = std::filesystem;
using namespace mlc::test::gguf;
using mlc::runtime::MetalExecutor;

namespace {
uint16_t testFloatToFp16(float value) {
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

TEST(FloatConvertTest, Fp16Subnormal) {
    constexpr uint16_t kMinSubnormal = 0x0001;
    constexpr uint16_t kNextSubnormal = 0x0002;
    float v0 = mlc::runtime::fp16ToFloat(kMinSubnormal);
    float v1 = mlc::runtime::fp16ToFloat(kNextSubnormal);
    EXPECT_NEAR(v0, std::ldexp(1.0f, -24), 1e-10f);
    EXPECT_NEAR(v1, std::ldexp(1.0f, -23), 1e-10f);
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

    // Canonical GGML: shape = [ne0=cols=3 (input dim), ne1=rows=2 (output dim)].
    // Storage is row-major: 2 rows of 3 floats each. Row 0 = [1,2,3], row 1 = [0,1,0.5].
    // Test runs Session::runLinear with input.size()=3 → matches_cols → output.size()=2.
    writeString(file, "linear.weight");
    writeU32(file, 2); // dims
    writeU64(file, 2);
    writeU32(file, 3); // ne0 = cols (input dim)
    writeU32(file, 2); // ne1 = rows (output dim)
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

    // Canonical GGML: tok_embeddings.weight = [ne0=hidden=3, ne1=vocab=4].
    // Embedding for token i is row i (4 rows × 3 floats each).
    writeString(file, "tok_embeddings.weight");
    writeU32(file, 2);
    writeU64(file, 2);
    writeU32(file, 3); // ne0 = hidden_dim (cols)
    writeU32(file, 4); // ne1 = vocab_size (rows)
    writeU32(file, 0);
    writeU64(file, 512);

    // Canonical GGML: output.weight = [ne0=hidden=3 (input dim), ne1=vocab=2 (output dim)].
    writeString(file, "output.weight");
    writeU32(file, 2);
    writeU64(file, 2);
    writeU32(file, 3); // ne0 = cols = input dim
    writeU32(file, 2); // ne1 = rows = output dim
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

    // Canonical GGML: token_embd.weight = [ne0=vocab=4, ne1=hidden=3], dtype I8.
    // Storage is row-major: 3 rows of 4 bytes each. Because rows < cols and the name
    // matches "token_embd", Session::getEmbedding triggers tokens_as_rows=false (column-
    // major embedding lookup). For token i, output[r] = data[r*cols + i] for r in 0..rows-1.
    // Token 2 -> [data[2], data[6], data[10]] = [12, 22, 32]. Output size = rows = 3.
    writeString(file, "token_embd.weight");
    writeU32(file, 2);
    writeU64(file, 2);
    writeU32(file, 4); // ne0 = cols = vocab (4 tokens)
    writeU32(file, 3); // ne1 = rows = hidden (3 dims)
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

    // Canonical GGML: output.weight = [ne0=cols=3 (input dim), ne1=rows=4 (output dim)],
    // dtype I8. Test calls Session::runLinear with input.size()=3 → matches_cols →
    // non-transpose path → output.size()=4.
    // Storage row-major: 3 rows of 4 bytes each? No — wait, with cols=3 the storage is
    // 4 rows of 3 bytes each. Re-derive: data has 12 bytes, rows=4, row_stride=3.
    // Row r = bytes [3r .. 3r+2]. Then per session::runLinear matches_rows path... actually
    // input.size()=3==cols so we take matches_cols. output[r] = sum_c W[r,c]*input[c].
    // Row 0=[1,2,3], output[0]=1+4+9=14.
    // BUT the test expects [38, 44, 50, 56] which corresponds to the matches_rows branch
    // (input dotting columns of a 3x4 layout). To preserve the test's intent of exercising
    // the transposed path, we keep the dimensions such that input.size()==rows triggers it:
    // ne0=4=cols (output dim), ne1=3=rows (input dim) → input.size()=3==rows → transpose.
    writeString(file, "output.weight");
    writeU32(file, 2);
    writeU64(file, 2);
    writeU32(file, 4); // ne0 = cols (output dim under matches_rows path)
    writeU32(file, 3); // ne1 = rows (input dim under matches_rows path)
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

std::string createQuantFFNGGUFFile(uint32_t dtype = 2 /*Q4_0*/, size_t cols = 32) {
    std::string path = "/tmp/test_quant_ffn_" + std::to_string(getpid()) + ".gguf";
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to create GGUF file");
    }

    writeU32(file, 0x46554747);
    writeU32(file, 1);
    writeU64(file, 1); // tensors
    writeU64(file, 1); // kv

    writeString(file, "general.quantization_version");
    writeU32(file, 4); // UINT32
    writeU32(file, 1);

    // Canonical GGML: q_weight = [ne0=cols (input dim), ne1=rows=2 (output dim)].
    // Two quantized rows, each `cols` elements wide. Test runs fused FFN with input.size()=cols
    // → matches_cols → output.size()=2.
    writeString(file, "q_weight");
    writeU32(file, 2);  // dims
    writeU32(file, static_cast<uint32_t>(cols)); // ne0 = cols (input dim)
    writeU32(file, 2);                           // ne1 = rows (output dim)
    writeU32(file, dtype);
    writeU64(file, 256);

    // Two rows, 32 cols each.
    std::vector<float> row0(cols, 0.0f);
    row0[0] = 1.0f;
    std::vector<float> row1(cols, 0.0f);
    row1[1] = 1.0f;
    std::vector<uint8_t> buf;
    uint32_t quant_version = 1;
    size_t row_size = mlc::runtime::ggmlRowSizeBytes(dtype, row0.size(), quant_version);
    std::vector<uint8_t> rowbuf0(row_size);
    std::vector<uint8_t> rowbuf1(row_size);
    mlc::runtime::quantizeRowFrom(row0.data(), dtype, row0.size(), quant_version, rowbuf0.data());
    mlc::runtime::quantizeRowFrom(row1.data(), dtype, row1.size(), quant_version, rowbuf1.data());
    buf.reserve(rowbuf0.size() + rowbuf1.size());
    buf.insert(buf.end(), rowbuf0.begin(), rowbuf0.end());
    buf.insert(buf.end(), rowbuf1.begin(), rowbuf1.end());
    file.seekp(256, std::ios::beg);
    file.write(reinterpret_cast<const char*>(buf.data()), buf.size());
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

    // Canonical GGML: tok_embeddings.weight = [ne0=hidden=3, ne1=vocab=4].
    writeString(file, "tok_embeddings.weight");
    writeU32(file, 2);
    writeU64(file, 2);
    writeU32(file, 3); // ne0 = hidden_dim
    writeU32(file, 4); // ne1 = vocab_size
    writeU32(file, 0);
    writeU64(file, 512);

    // Canonical GGML: output.weight = [ne0=cols=4 (output dim), ne1=rows=3 (input dim)],
    // dtype I8. Input size 3 (from embedding) matches rows → matches_rows path → output size = cols = 4.
    writeString(file, "output.weight");
    writeU32(file, 2);
    writeU64(file, 2);
    writeU32(file, 4); // ne0 = cols (output dim under transposed dispatch)
    writeU32(file, 3); // ne1 = rows (input dim)
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

std::string createGemmaLikeGGUFFile() {
    std::string path = "/tmp/test_gemma_" + std::to_string(getpid()) + ".gguf";
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to create GGUF file");
    }

    writeU32(file, 0x46554747);
    writeU32(file, 1);
    writeU64(file, 2); // tensors
    writeU64(file, 2); // kv pairs

    writeString(file, "gemma.block_count");
    writeU32(file, 4);
    writeU32(file, 1);

    writeString(file, "gemma.embedding_length");
    writeU32(file, 4);
    writeU32(file, 4);

    writeString(file, "token_embd.weight");
    writeU32(file, 2);
    writeU64(file, 2);
    writeU32(file, 4);
    writeU32(file, 4);
    writeU32(file, 0);
    writeU64(file, 512);

    writeString(file, "head.weight");
    writeU32(file, 2);
    writeU64(file, 2);
    writeU32(file, 4);
    writeU32(file, 4);
    writeU32(file, 0);
    writeU64(file, 560);

    file.seekp(512, std::ios::beg);
    std::vector<float> embeddings = {
        1.f, 1.f, 1.f, 1.f,
        2.f, 2.f, 2.f, 2.f,
        3.f, 3.f, 3.f, 3.f,
        4.f, 4.f, 4.f, 4.f
    };
    file.write(reinterpret_cast<const char*>(embeddings.data()),
               embeddings.size() * sizeof(float));

    file.seekp(560, std::ios::beg);
    std::vector<float> head = {
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 0.f,
        0.f, 0.f, 0.f, 1.f
    };
    file.write(reinterpret_cast<const char*>(head.data()), head.size() * sizeof(float));
    file.close();
    return path;
}

std::string createMistralLikeGGUFFile() {
    std::string path = "/tmp/test_mistral_" + std::to_string(getpid()) + ".gguf";
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to create GGUF file");
    }

    writeU32(file, 0x46554747);
    writeU32(file, 1);
    writeU64(file, 2); // tensors
    writeU64(file, 3); // kv pairs

    writeString(file, "mistral.block_count");
    writeU32(file, 4);
    writeU32(file, 1);

    writeString(file, "mistral.embedding_length");
    writeU32(file, 4);
    writeU32(file, 4);

    writeString(file, "attention.sliding_window");
    writeU32(file, 4);
    writeU32(file, 64);

    writeString(file, "token_embd.weight");
    writeU32(file, 2);
    writeU64(file, 2);
    writeU32(file, 4);
    writeU32(file, 4);
    writeU32(file, 0);
    writeU64(file, 512);

    writeString(file, "head.weight");
    writeU32(file, 2);
    writeU64(file, 2);
    writeU32(file, 4);
    writeU32(file, 4);
    writeU32(file, 0);
    writeU64(file, 560);

    file.seekp(512, std::ios::beg);
    std::vector<float> embeddings = {
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 0.f,
        0.f, 0.f, 0.f, 1.f
    };
    file.write(reinterpret_cast<const char*>(embeddings.data()),
               embeddings.size() * sizeof(float));

    file.seekp(560, std::ios::beg);
    std::vector<float> head = {
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 0.f,
        0.f, 0.f, 0.f, 1.f
    };
    file.write(reinterpret_cast<const char*>(head.data()), head.size() * sizeof(float));
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

TEST(ExecutionRuntimeTest, FusedFeedForwardUsesAnnotatedWeights) {
    std::string path = createLinearGGUFFile();
    mlc::runtime::Session session(path);

    mlc::runtime::ExecutionGraph graph;
    graph.addTensor("x", {2}, mlc::ir::DataType::F32);
    graph.addTensor("ffn_out", {2}, mlc::ir::DataType::F32);

    auto& ffn = graph.addNode("fused_ffn",
                              mlc::runtime::ExecOpType::FeedForward,
                              {"x"},
                              {"ffn_out"},
                              mlc::runtime::BackendKind::CPU);
    // Reuse the single weight tensor for gate/up; skip down projection.
    ffn.annotations["param0"] = "linear.weight";
    ffn.annotations["param1"] = "linear.weight";

    mlc::runtime::ModelConfig cfg;
    cfg.activation = "silu";
    graph.setModelConfig(cfg);

    mlc::runtime::ExecutionContext ctx(session, &graph);
    ctx.setTensor("x", {1.0f, 2.0f, 3.0f});

    mlc::runtime::ExecutionExecutor executor(graph, nullptr, &ctx);
    auto status = executor.run();
    ASSERT_GT(status.trace.size(), 0u);
    ASSERT_TRUE(status.success);

    const auto* out = ctx.getTensor("ffn_out");
    ASSERT_NE(out, nullptr);
    // For linear.weight = [[1,2,3],[0,1,0.5]]:
    // gate=up = [14, 3.5]; silu(gate) ~ [13.999, 3.398]; mix ~ [195.99, 11.89]
    EXPECT_NEAR((*out)[0], 196.0f, 1.0f);
    EXPECT_NEAR((*out)[1], 11.9f, 0.2f);

    fs::remove(path);
}

TEST(ExecutionRuntimeTest, FusedFeedForwardSupportsQuantizedWeights) {
    std::string path = createQuantFFNGGUFFile();
    mlc::runtime::Session session(path);

    mlc::runtime::ExecutionGraph graph;
    graph.addTensor("x", {32}, mlc::ir::DataType::F32);
    graph.addTensor("ffn_out_q", {2}, mlc::ir::DataType::F32);

    auto& ffn = graph.addNode("fused_ffn_q",
                              mlc::runtime::ExecOpType::FeedForward,
                              {"x"},
                              {"ffn_out_q"},
                              mlc::runtime::BackendKind::CPU);
    ffn.annotations["param0"] = "q_weight";
    ffn.annotations["param1"] = "q_weight";

    mlc::runtime::ModelConfig cfg;
    cfg.activation = "silu";
    graph.setModelConfig(cfg);

    // Build expected from dequantizing q_weight manually.
    const auto& tensors = session.loader().tensors();
    auto it = tensors.find("q_weight");
    ASSERT_NE(it, tensors.end());
    auto raw = session.loader().loadTensorData(it->second);
    ASSERT_FALSE(raw.empty());
    // Canonical GGML: shape[1] = rows = number of output rows; row_stride = total / rows.
    size_t stride = raw.size() / static_cast<size_t>(it->second.shape[1]);
    ASSERT_GT(stride, 0u);
    std::vector<float> deq_row(32);
    std::vector<float> gate_vals(2);
    std::vector<float> up_vals(2);
    uint32_t qv = session.loader().quantizationVersion();
    mlc::runtime::dequantizeRowQ4_0(raw.data(), 32, qv, deq_row.data());
    gate_vals[0] = 0.0f;
    for (size_t i = 0; i < 32; ++i) gate_vals[0] += deq_row[i] * (i == 0 ? 1.0f : (i == 1 ? 2.0f : 0.0f));
    mlc::runtime::dequantizeRowQ4_0(raw.data() + stride, 32, qv, deq_row.data());
    gate_vals[1] = 0.0f;
    for (size_t i = 0; i < 32; ++i) gate_vals[1] += deq_row[i] * (i == 0 ? 1.0f : (i == 1 ? 2.0f : 0.0f));
    up_vals = gate_vals;

    std::vector<float> input(32, 0.0f);
    input[0] = 1.0f;
    input[1] = 2.0f;

    mlc::runtime::ExecutionContext ctx(session, &graph);
    ctx.setTensor("x", std::move(input));

    mlc::runtime::ExecutionExecutor executor(graph, nullptr, &ctx);
    auto status = executor.run();
    ASSERT_TRUE(status.success);

    const auto* out = ctx.getTensor("ffn_out_q");
    ASSERT_NE(out, nullptr);
    // gate/up from quantized weight should match manual dequant dot.
    auto silu_fn = [](float x) { return x / (1.0f + std::exp(-x)); };
    float mix0 = silu_fn(gate_vals[0]) * up_vals[0];
    float mix1 = silu_fn(gate_vals[1]) * up_vals[1];
    EXPECT_NEAR((*out)[0], mix0, 1e-2f);
    EXPECT_NEAR((*out)[1], mix1, 1e-2f);

    fs::remove(path);
}

TEST(ExecutionRuntimeTest, FusedFeedForwardSupportsQuantizedWeightsKSeries) {
    // Use Q4_K to validate K-format support in fused FFN computeLinear.
    constexpr size_t cols = 256;
    std::string path = createQuantFFNGGUFFile(mlc::frontend::GGML_TYPE_Q4_K, cols);
    mlc::runtime::Session session(path);

    mlc::runtime::ExecutionGraph graph;
    graph.addTensor("x", {static_cast<int64_t>(cols)}, mlc::ir::DataType::F32);
    graph.addTensor("ffn_out_qk", {2}, mlc::ir::DataType::F32);

    auto& ffn = graph.addNode("fused_ffn_qk",
                              mlc::runtime::ExecOpType::FeedForward,
                              {"x"},
                              {"ffn_out_qk"},
                              mlc::runtime::BackendKind::CPU);
    ffn.annotations["param0"] = "q_weight";
    ffn.annotations["param1"] = "q_weight";

    mlc::runtime::ModelConfig cfg;
    cfg.activation = "silu";
    graph.setModelConfig(cfg);

    const auto& tensors = session.loader().tensors();
    auto it = tensors.find("q_weight");
    ASSERT_NE(it, tensors.end());
    auto raw = session.loader().loadTensorData(it->second);
    // Canonical GGML: shape[1] = rows = number of output rows; row_stride = total / rows.
    size_t stride = raw.size() / static_cast<size_t>(it->second.shape[1]);
    std::vector<float> deq_row(cols);
    std::vector<float> gate_vals(2);
    std::vector<float> up_vals(2);
    uint32_t qv = session.loader().quantizationVersion();
    mlc::runtime::dequantizeRowTo(raw.data(), it->second.dtype, cols, qv, deq_row.data());
    gate_vals[0] = 0.0f;
    for (size_t i = 0; i < cols; ++i) gate_vals[0] += deq_row[i] * (i == 0 ? 1.0f : (i == 1 ? 2.0f : 0.0f));
    mlc::runtime::dequantizeRowTo(raw.data() + stride, it->second.dtype, cols, qv, deq_row.data());
    gate_vals[1] = 0.0f;
    for (size_t i = 0; i < cols; ++i) gate_vals[1] += deq_row[i] * (i == 0 ? 1.0f : (i == 1 ? 2.0f : 0.0f));
    up_vals = gate_vals;

    std::vector<float> input(cols, 0.0f);
    input[0] = 1.0f;
    input[1] = 2.0f;

    mlc::runtime::ExecutionContext ctx(session, &graph);
    ctx.setTensor("x", std::move(input));

    mlc::runtime::ExecutionExecutor executor(graph, nullptr, &ctx);
    auto status = executor.run();
    ASSERT_FALSE(status.trace.empty());
    const auto& last = status.trace.back();
    EXPECT_TRUE(last.success) << mlc::runtime::formatTraceEntry(last);
    ASSERT_TRUE(status.success);

    const auto* out = ctx.getTensor("ffn_out_qk");
    ASSERT_NE(out, nullptr);
    auto silu_fn = [](float x) { return x / (1.0f + std::exp(-x)); };
    float mix0 = silu_fn(gate_vals[0]) * up_vals[0];
    float mix1 = silu_fn(gate_vals[1]) * up_vals[1];
    EXPECT_NEAR((*out)[0], mix0, 5e-2f);
    EXPECT_NEAR((*out)[1], mix1, 5e-2f);

    fs::remove(path);
}

TEST(ExecutionRuntimeTest, FusedFeedForwardDownProjectionWithBias) {
    // Build a small GGUF with F32 weights for gate/up/down and validate bias handling.
    std::string path = "/tmp/test_fused_ffn_down_" + std::to_string(getpid()) + ".gguf";
    {
        std::ofstream file(path, std::ios::binary);
        ASSERT_TRUE(file.is_open());
        writeU32(file, 0x46554747);
        writeU32(file, 1);  // version
        writeU64(file, 2);  // tensors
        writeU64(file, 0);  // kv

        // Canonical GGML: w_gate = [ne0=cols=3 (input dim), ne1=rows=2 (output dim)] F32.
        writeString(file, "w_gate");
        writeU32(file, 2);
        writeU32(file, 3); // ne0 = cols (input dim)
        writeU32(file, 2); // ne1 = rows (output dim)
        writeU32(file, 0);  // F32
        writeU64(file, 256);

        // Canonical GGML: w_down = [ne0=cols=2 (input dim), ne1=rows=2 (output dim)] F32 — square.
        writeString(file, "w_down");
        writeU32(file, 2);
        writeU32(file, 2); // ne0 = cols
        writeU32(file, 2); // ne1 = rows
        writeU32(file, 0);  // F32
        writeU64(file, 256 + sizeof(float) * 6);

        file.seekp(256, std::ios::beg);
        // w_gate values.
        std::vector<float> w_gate = {1.f, 2.f, 3.f,
                                     0.f, 1.f, 0.5f};
        file.write(reinterpret_cast<const char*>(w_gate.data()), w_gate.size() * sizeof(float));
        // w_down simple projection.
        std::vector<float> w_down = {1.f, 0.f,
                                     0.f, 1.f};
        file.write(reinterpret_cast<const char*>(w_down.data()), w_down.size() * sizeof(float));
    }

    mlc::runtime::Session session(path);
    mlc::runtime::ExecutionGraph graph;
    graph.addTensor("x", {3}, mlc::ir::DataType::F32);
    graph.addTensor("ffn_out_down", {2}, mlc::ir::DataType::F32);

    auto& ffn = graph.addNode("fused_ffn_down",
                              mlc::runtime::ExecOpType::FeedForward,
                              {"x"},
                              {"ffn_out_down"},
                              mlc::runtime::BackendKind::CPU);
    ffn.annotations["param0"] = "w_gate";
    ffn.annotations["param1"] = "w_gate";
    ffn.annotations["param2"] = "w_down";
    ffn.annotations["bias"] = "bias_down";

    mlc::runtime::ModelConfig cfg;
    cfg.activation = "silu";
    graph.setModelConfig(cfg);

    mlc::runtime::ExecutionContext ctx(session, &graph);
    ctx.setTensor("x", {1.0f, 2.0f, 3.0f});
    ctx.setTensor("bias_down", {0.5f, -1.0f});

    mlc::runtime::ExecutionExecutor executor(graph, nullptr, &ctx);
    auto status = executor.run();
    ASSERT_TRUE(status.success);

    const auto* out = ctx.getTensor("ffn_out_down");
    ASSERT_NE(out, nullptr);
    // gate/up = [14, 3.5]; silu -> ~[14, 3.398]; mix ~ [196, 11.89]
    // down = identity 2x2, plus bias.
    EXPECT_NEAR((*out)[0], 196.5f, 1.0f);
    EXPECT_NEAR((*out)[1], 10.9f, 0.5f);

    fs::remove(path);
}

TEST(QuantizationTest, Q8_1DequantizationProducesBias) {
    using namespace mlc::runtime;
    constexpr size_t cols = 32;
    std::vector<uint8_t> buffer(q8_1RowSize(cols), 0);

    uint16_t* header = reinterpret_cast<uint16_t*>(buffer.data());
    header[0] = testFloatToFp16(2.0f);   // scale
    header[1] = testFloatToFp16(32.0f);  // bias accumulator -> bias 1.0f

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
    block.d = testFloatToFp16(0.5f);
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
    block.d = testFloatToFp16(0.25f);
    block.m = testFloatToFp16(1.0f);
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
    block.d = testFloatToFp16(0.5f);
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
    block.d = testFloatToFp16(0.5f);
    block.m = testFloatToFp16(0.25f);
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
    block.d = testFloatToFp16(0.125f);
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

// TODO(quant-precision): This test fails because the CPU Q4_0 quantize routine in
// quantization.cpp produces ~6 elements (out of 64) outside the 1e-1 tolerance for
// sin(0..63) input. Pure quantize/dequant precision issue — does NOT call the Metal
// kernel and is unrelated to Bug A (split-half indexing) or Bug B (operator_backend
// rows/cols swap). Pre-dates the GGML-canonical test fixture cleanup. Leaving in
// place as a flag; revisit when tightening CPU quant routines.
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
    d_ptr[0] = testFloatToFp16(0.5f); // d
    d_ptr[1] = testFloatToFp16(0.0f); // dmin

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
        SCOPED_TRACE("q2_k");
        size_t row_bytes = q2_kRowSize(cols);
        std::vector<uint8_t> buffer(row_bytes, 0x11);
        uint16_t* d_ptr = reinterpret_cast<uint16_t*>(buffer.data() + 16 + 64);
        d_ptr[0] = testFloatToFp16(0.8f);
        d_ptr[1] = testFloatToFp16(0.2f);
        testQkDotProduct(buffer, cols, dequantizeRowQ2_K, dotProductRowQ2_K);
    }
    {
        SCOPED_TRACE("q3_k");
        size_t row_bytes = q3_kRowSize(cols);
        std::vector<uint8_t> buffer(row_bytes, 0x22);
        uint16_t* d_ptr = reinterpret_cast<uint16_t*>(buffer.data() + 80);
        d_ptr[0] = testFloatToFp16(0.6f);
        testQkDotProduct(buffer, cols, dequantizeRowQ3_K, dotProductRowQ3_K);
    }
    {
        SCOPED_TRACE("q4_k");
        size_t row_bytes = q4_kRowSize(cols);
        std::vector<uint8_t> buffer(row_bytes, 0x33);
        uint16_t* d_ptr = reinterpret_cast<uint16_t*>(buffer.data() + 144 - 4);
        d_ptr[0] = testFloatToFp16(0.5f);
        d_ptr[1] = testFloatToFp16(0.1f);
        testQkDotProduct(buffer, cols, dequantizeRowQ4_K, dotProductRowQ4_K);
    }
    {
        SCOPED_TRACE("q5_k");
        size_t row_bytes = q5_kRowSize(cols);
        std::vector<uint8_t> buffer(row_bytes, 0x44);
        uint16_t* d_ptr = reinterpret_cast<uint16_t*>(buffer.data() + 176 - 4);
        d_ptr[0] = testFloatToFp16(0.4f);
        d_ptr[1] = testFloatToFp16(0.05f);
        testQkDotProduct(buffer, cols, dequantizeRowQ5_K, dotProductRowQ5_K);
    }
    {
        SCOPED_TRACE("q6_k");
        size_t row_bytes = q6_kRowSize(cols);
        std::vector<uint8_t> buffer(row_bytes, 0x55);
        uint16_t* d_ptr = reinterpret_cast<uint16_t*>(buffer.data() + 210 - 2);
        d_ptr[0] = testFloatToFp16(0.3f);
        testQkDotProduct(buffer, cols, dequantizeRowQ6_K, dotProductRowQ6_K);
    }
    {
        SCOPED_TRACE("q8_k");
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

TEST(ModelRunnerTest, DryRunWorksForGemmaStyleMetadataAndNames) {
    std::string path = createGemmaLikeGGUFFile();
    mlc::runtime::ModelRunner runner(path);
    mlc::runtime::RunConfig config;
    config.token_id = 2;
    config.preview_length = 2;

    auto report = runner.dryRun(config);
    EXPECT_EQ(report.embedding_tensor, "token_embd.weight");
    EXPECT_EQ(report.logits_tensor, "head.weight");
    ASSERT_EQ(report.embedding_preview.size(), 2u);
    EXPECT_FLOAT_EQ(report.embedding_preview[0], 3.0f);
    EXPECT_FLOAT_EQ(report.embedding_preview[1], 3.0f);
    ASSERT_EQ(report.logits_preview.size(), 2u);
    // Head is identity; embedding row for token 2 is all 3s, so logits should be 3s.
    EXPECT_FLOAT_EQ(report.logits_preview[0], 3.0f);
    EXPECT_FLOAT_EQ(report.logits_preview[1], 3.0f);
    fs::remove(path);
}

TEST(ModelRunnerTest, DryRunHandlesMistralMetadataAndSlidingWindow) {
    std::string path = createMistralLikeGGUFFile();
    mlc::runtime::ModelRunner runner(path);
    mlc::runtime::RunConfig config;
    config.token_id = 0;
    config.preview_length = 2;

    auto report = runner.dryRun(config);
    EXPECT_EQ(report.embedding_tensor, "token_embd.weight");
    EXPECT_EQ(report.logits_tensor, "head.weight");
    ASSERT_EQ(report.embedding_preview.size(), 2u);
    EXPECT_FLOAT_EQ(report.embedding_preview[0], 1.0f);
    EXPECT_FLOAT_EQ(report.embedding_preview[1], 0.0f);
    ASSERT_EQ(report.logits_preview.size(), 2u);
    EXPECT_FLOAT_EQ(report.logits_preview[0], 1.0f);
    EXPECT_FLOAT_EQ(report.logits_preview[1], 0.0f);
    // Sliding window metadata present in kv; ensure it didn't block dryRun.
    EXPECT_TRUE(report.logits_error.empty());
    fs::remove(path);
}

TEST(ExecutionRuntimeTest, AttentionRespectsSlidingWindowMask) {
    std::string path = createRunnerGGUFFile();
    mlc::runtime::Session session(path);

    mlc::runtime::ExecutionGraph graph;
    graph.addTensor("q", {2}, mlc::ir::DataType::F32);
    graph.addTensor("k_in", {4}, mlc::ir::DataType::F32); // 2 tokens * kv_heads(1) * head_dim(2)
    graph.addTensor("v_in", {4}, mlc::ir::DataType::F32);

    auto& k_cache = graph.addTensor("kv_k_window", std::vector<int64_t>{1, 3, 2}, mlc::ir::DataType::F32);
    k_cache.is_state = true;
    auto& v_cache = graph.addTensor("kv_v_window", std::vector<int64_t>{1, 3, 2}, mlc::ir::DataType::F32);
    v_cache.is_state = true;

    graph.addTensor("attn_out_window", {2}, mlc::ir::DataType::F32);
    graph.addTensor("attn_mask_window", {3}, mlc::ir::DataType::F32);

    auto& attn = graph.addNode("sliding_attention",
                               mlc::runtime::ExecOpType::Attention,
                               {"q", "k_in", "v_in"},
                               {"attn_out_window", "kv_k_window", "kv_v_window"},
                               mlc::runtime::BackendKind::CPU);
    attn.attributes["heads"] = 1.0f;
    attn.attributes["kv_heads"] = 1.0f;
    attn.attributes["head_dim"] = 2.0f;
    attn.annotations["kv_cache_k"] = "kv_k_window";
    attn.annotations["kv_cache_v"] = "kv_v_window";
    attn.annotations["attention_mask"] = "attn_mask_window";

    // Set sliding window to 0 so only the current position (2) is visible.
    mlc::runtime::ModelConfig cfg;
    cfg.sliding_window = 0;
    cfg.context_length = 3;
    cfg.head_count = 1;
    cfg.kv_head_count = 1;
    cfg.head_dim = 2;
    graph.setModelConfig(cfg);

    mlc::runtime::ExecutionContext context(session, &graph);
    context.setSequencePosition(2); // simulate generation at position 2 (0-based)
    context.setTensor("q", {1.0f, 0.0f});               // query
    context.setTensor("k_in", {1.0f, 0.0f, 0.0f, 1.0f}); // k for positions 0 and 1
    context.setTensor("v_in", {10.0f, 0.0f, 0.0f, 20.0f}); // v for positions 0 and 1
    // Explicit sliding mask: only position 2 visible (positions 0,1 masked).
    context.setTensor("attn_mask_window", {-1e9f, -1e9f, 0.0f});

    mlc::runtime::ExecutionExecutor executor(graph, nullptr, &context);
    auto status = executor.run();
    ASSERT_TRUE(status.success);

    const auto* out = context.getTensor("attn_out_window");
    ASSERT_NE(out, nullptr);
    // Sliding window 0 means only position 2 is visible; we wrote token0 into pos2.
    EXPECT_NEAR((*out)[0], 10.0f, 1e-4f);
    EXPECT_NEAR((*out)[1], 0.0f, 1e-4f);

    fs::remove(path);
}

TEST(ExecutionRuntimeTest, GemmaFeedForwardUsesGeGLU) {
    std::string path = createRunnerGGUFFile();
    mlc::runtime::Session session(path);
    mlc::runtime::ExecutionGraph graph;
    graph.addTensor("gate", {2}, mlc::ir::DataType::F32);
    graph.addTensor("up", {2}, mlc::ir::DataType::F32);
    graph.addTensor("out", {2}, mlc::ir::DataType::F32);

    auto& ffn = graph.addNode("ffn",
                              mlc::runtime::ExecOpType::FeedForward,
                              {"gate", "up"},
                              {"out"},
                              mlc::runtime::BackendKind::CPU);
    ffn.annotations["activation"] = "geglu"; // should be respected

    mlc::runtime::ModelConfig cfg;
    cfg.family = mlc::runtime::ArchitectureFamily::Gemma;
    cfg.activation = ""; // force default path
    graph.setModelConfig(cfg);

    mlc::runtime::ExecutionContext ctx(session, &graph);
    ctx.setTensor("gate", {1.0f, -1.0f});
    ctx.setTensor("up", {2.0f, 3.0f});

    mlc::runtime::ExecutionExecutor exec(graph, nullptr, &ctx);
    auto status = exec.run();
    ASSERT_TRUE(status.success);

    const auto* out = ctx.getTensor("out");
    ASSERT_NE(out, nullptr);
    auto gelu = [](float x) {
        const float kAlpha = std::sqrt(2.0f / static_cast<float>(M_PI));
        return 0.5f * x * (1.0f + std::tanh(kAlpha * (x + 0.044715f * x * x * x)));
    };
    // geGLU: gelu(gate) * up
    EXPECT_NEAR((*out)[0], gelu(1.0f) * 2.0f, 1e-3f);
    EXPECT_NEAR((*out)[1], gelu(-1.0f) * 3.0f, 1e-3f);

    fs::remove(path);
}

TEST(ExecutionRuntimeTest, MistralSlidingWindowAutoMask) {
    std::string path = createRunnerGGUFFile();
    mlc::runtime::Session session(path);

    auto run_with_window = [&](size_t sliding_window) {
        mlc::runtime::ExecutionGraph graph;
        graph.addTensor("q", {2}, mlc::ir::DataType::F32);
        graph.addTensor("k_in", {4}, mlc::ir::DataType::F32); // two tokens
        graph.addTensor("v_in", {4}, mlc::ir::DataType::F32);

        auto& k_cache = graph.addTensor("kv_k_window_auto", std::vector<int64_t>{1, 3, 2}, mlc::ir::DataType::F32);
        k_cache.is_state = true;
        auto& v_cache = graph.addTensor("kv_v_window_auto", std::vector<int64_t>{1, 3, 2}, mlc::ir::DataType::F32);
        v_cache.is_state = true;

        graph.addTensor("attn_out_window_auto", {2}, mlc::ir::DataType::F32);

        auto& attn = graph.addNode("sliding_attention_auto",
                                   mlc::runtime::ExecOpType::Attention,
                                   {"q", "k_in", "v_in"},
                                   {"attn_out_window_auto", "kv_k_window_auto", "kv_v_window_auto"},
                                   mlc::runtime::BackendKind::CPU);
        attn.attributes["heads"] = 1.0f;
        attn.attributes["kv_heads"] = 1.0f;
        attn.attributes["head_dim"] = 2.0f;
        attn.annotations["kv_cache_k"] = "kv_k_window_auto";
        attn.annotations["kv_cache_v"] = "kv_v_window_auto";

        mlc::runtime::ModelConfig cfg;
        cfg.sliding_window = sliding_window;
        cfg.context_length = 3;
        cfg.head_count = 1;
        cfg.kv_head_count = 1;
        cfg.head_dim = 2;
        cfg.family = mlc::runtime::ArchitectureFamily::Mistral;
        graph.setModelConfig(cfg);

        mlc::runtime::ExecutionContext context(session, &graph);
        context.setSequencePosition(2); // simulate generation at position 2
        context.setTensor("q", {1.0f, 0.0f});                 // query
        context.setTensor("k_in", {1.0f, 0.0f, 0.0f, 1.0f});   // tokens 0 and 1
        context.setTensor("v_in", {5.0f, 0.0f, 0.0f, 10.0f});  // v for positions 0 and 1

        mlc::runtime::ExecutionExecutor executor(graph, nullptr, &context);
        auto status = executor.run();
        EXPECT_TRUE(status.success);

        const auto* out = context.getTensor("attn_out_window_auto");
        EXPECT_NE(out, nullptr);
        return out ? (*out)[0] : 0.0f;
    };

    float unmasked = run_with_window(0); // no sliding mask applied
    float masked = run_with_window(1);   // masks token0 at position2
    EXPECT_GT(masked, unmasked);

    fs::remove(path);
}

TEST(ExecutionRuntimeTest, AttentionAppliesPerHeadAlibiMaskShape) {
    std::string path = createRunnerGGUFFile();
    mlc::runtime::Session session(path);

    mlc::runtime::ExecutionGraph graph;
    graph.addTensor("q", {2}, mlc::ir::DataType::F32);   // 2 heads * dim1
    graph.addTensor("k_in", {2}, mlc::ir::DataType::F32); // 2 tokens * dim1
    graph.addTensor("v_in", {2}, mlc::ir::DataType::F32);

    auto& k_cache = graph.addTensor("kv_k_alibi", std::vector<int64_t>{1, 2, 1}, mlc::ir::DataType::F32);
    k_cache.is_state = true;
    auto& v_cache = graph.addTensor("kv_v_alibi", std::vector<int64_t>{1, 2, 1}, mlc::ir::DataType::F32);
    v_cache.is_state = true;

    graph.addTensor("attn_out_alibi", {2}, mlc::ir::DataType::F32);
    graph.addTensor("attn_mask_alibi", {4}, mlc::ir::DataType::F32); // head-major mask

    auto& attn = graph.addNode("alibi_attention",
                               mlc::runtime::ExecOpType::Attention,
                               {"q", "k_in", "v_in"},
                               {"attn_out_alibi", "kv_k_alibi", "kv_v_alibi"},
                               mlc::runtime::BackendKind::CPU);
    attn.attributes["heads"] = 2.0f;
    attn.attributes["kv_heads"] = 1.0f;
    attn.attributes["head_dim"] = 1.0f;
    attn.annotations["kv_cache_k"] = "kv_k_alibi";
    attn.annotations["kv_cache_v"] = "kv_v_alibi";
    attn.annotations["attention_mask"] = "attn_mask_alibi";

    mlc::runtime::ModelConfig cfg;
    cfg.context_length = 2;
    cfg.head_count = 2;
    cfg.kv_head_count = 1;
    cfg.head_dim = 1;
    graph.setModelConfig(cfg);

    mlc::runtime::ExecutionContext context(session, &graph);
    context.setSequencePosition(0);
    // Uniform q/k so mask decides.
    context.setTensor("q", {1.0f, 1.0f});
    context.setTensor("k_in", {1.0f, 1.0f});   // two tokens
    context.setTensor("v_in", {1.0f, 2.0f});   // distinguish tokens
    // Head-major mask: head0 no bias, head1 heavily penalizes token0.
    context.setTensor("attn_mask_alibi", {0.0f, 0.0f, -5.0f, 0.0f});

    mlc::runtime::ExecutionExecutor executor(graph, nullptr, &context);
    auto status = executor.run();
    ASSERT_TRUE(status.success);

    const auto* out = context.getTensor("attn_out_alibi");
    ASSERT_NE(out, nullptr);
    // Head0 roughly averages tokens -> (1+2)/2 = 1.5
    EXPECT_NEAR((*out)[0], 1.5f, 1e-2f);
    // Head1 should focus on token1 (value 2.0) due to mask bias.
    EXPECT_NEAR((*out)[1], 2.0f, 1e-2f);

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
    ASSERT_TRUE(executor.runMatMul(std::string{}, weights, input, rows, cols, false, gpu_output));
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
    ASSERT_TRUE(executor.runMatMul(std::string{}, weights, input, rows, cols, false, gpu_output, &bias));
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
        d_ptr[0] = testFloatToFp16(0.5f + 0.1f * r);
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
    if (!executor.runMatMulQ4_0(std::string{}, weights,
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
        d_ptr[0] = testFloatToFp16(0.5f + 0.1f * r);
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
    if (!executor.runMatMulQ4_0(std::string{}, weights,
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
    MetalExecutor::CacheDescriptor cache_k{
        mlc::frontend::GGML_TYPE_F32,
        1,
        head_dim * sizeof(float),
        &k_handle,
        nullptr,
        &kv_cache_k};
    MetalExecutor::CacheDescriptor cache_v{
        mlc::frontend::GGML_TYPE_F32,
        1,
        head_dim * sizeof(float),
        &v_handle,
        nullptr,
        &kv_cache_v};
    ASSERT_TRUE(executor.runAttention(q,
                                      k,
                                      v,
                                      num_heads,
                                      kv_heads,
                                      head_dim,
                                      context_length,
                                      mask,
                                      nullptr,
                                      0,
                                      0,
                                      10000.0f,
                                      1.0f,
                                      cache_k,
                                      cache_v,
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

// TODO(metal-q4_1-split-half): Q4_1 Metal kernel has the same interleaved-vs-split-half
// indexing bug that Q4_0 had pre-fix (commit d9720c7). q4_1_matmul at metal_runtime.mm:233
// walks `col_index++` instead of computing (j, j+16) from the canonical block layout.
// Test compares Metal Q4_1 output to dotProductRowQ4_1 (CPU reference) — CPU is correct,
// Metal is broken, hence the divergence. Same fix as Bug A but for Q4_1; out of scope
// for the current Bug B-focused round.
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
        d_ptr[0] = testFloatToFp16(0.25f + 0.05f * r);
        d_ptr[1] = testFloatToFp16(0.1f * r);
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
    if (!executor.runMatMulQ4_1(std::string{}, weights,
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

// TODO(metal-q5-split-half): Q5_0/Q5_1 Metal kernels have the same interleaved-vs-
// split-half indexing bug as Q4_0 had pre-fix (commit d9720c7). q5_0_matmul at
// metal_runtime.mm:264+ walks `col_index++` instead of canonical (j, j+16) indexing.
// Same fix pattern as Bug A; out of scope for the current Bug B-focused round.
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
        header[0] = testFloatToFp16(scale);
        size_t qh_offset = (offset == 0.0f) ? 2 : 4;
        if (offset != 0.0f) {
            header[1] = testFloatToFp16(offset);
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
    if (!executor.runMatMulQ5_0(std::string{}, row_q5_0,
                                input,
                                1,
                                cols,
                                mlc::runtime::q5_0RowSize(cols),
                                gpu_q5_0)) {
        GTEST_SKIP() << "Metal Q5_0 matmul unavailable";
    }
    std::vector<float> gpu_q5_1;
    if (!executor.runMatMulQ5_1(std::string{}, row_q5_1,
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
    d_ptr[0] = testFloatToFp16(0.45f);
    d_ptr[1] = testFloatToFp16(0.05f);
    std::vector<float> input(cols);
    for (size_t i = 0; i < cols; ++i) input[i] = 0.002f * static_cast<float>(i + 1);
    float expected = mlc::runtime::dotProductRowQ4_K(row.data(), cols, input.data());
    std::vector<float> gpu_out;
    if (!executor.runMatMulQ4K(std::string{}, row, input, 1, cols, stride, gpu_out)) {
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
    d_ptr[0] = testFloatToFp16(0.35f);
    d_ptr[1] = testFloatToFp16(0.02f);
    std::vector<float> input(cols);
    for (size_t i = 0; i < cols; ++i) input[i] = -0.0015f * static_cast<float>(i + 1);
    float expected = mlc::runtime::dotProductRowQ5_K(row.data(), cols, input.data());
    std::vector<float> gpu_out;
    if (!executor.runMatMulQ5K(std::string{}, row, input, 1, cols, stride, gpu_out)) {
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
    // The BlockQ6_K layout puts the fp16 scale at the end of the block.
    uint16_t* d_ptr = reinterpret_cast<uint16_t*>(row.data() + stride - sizeof(uint16_t));
    d_ptr[0] = testFloatToFp16(0.25f);
    std::vector<float> input(cols);
    for (size_t i = 0; i < cols; ++i) input[i] = 0.0005f * static_cast<float>(i + 1);
    float expected = mlc::runtime::dotProductRowQ6_K(row.data(), cols, input.data());
    std::vector<float> gpu_out;
    if (!executor.runMatMulQ6K(std::string{}, row, input, 1, cols, stride, gpu_out)) {
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
    if (!executor.runMatMulQ8K(std::string{}, row, input, 1, cols, stride, gpu_out)) {
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
    header[0] = testFloatToFp16(0.3f);
    int8_t* qs = reinterpret_cast<int8_t*>(row.data() + 2);
    for (size_t i = 0; i < cols; ++i) {
        qs[i] = static_cast<int8_t>(i - 16);
    }
    std::vector<float> input(cols);
    for (size_t i = 0; i < cols; ++i) input[i] = 0.05f * static_cast<float>(i + 1);
    float expected = mlc::runtime::dotProductRowQ8_0(row.data(), cols, input.data());
    std::vector<float> gpu_out;
    if (!executor.runMatMulQ8_0(std::string{}, row, input, 1, cols, stride, gpu_out)) {
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
    header[0] = testFloatToFp16(0.2f);
    header[1] = testFloatToFp16(4.0f);
    int8_t* qs = reinterpret_cast<int8_t*>(row.data() + 4);
    for (size_t i = 0; i < cols; ++i) {
        qs[i] = static_cast<int8_t>(15 - static_cast<int>(i));
    }
    std::vector<float> input(cols);
    for (size_t i = 0; i < cols; ++i) input[i] = -0.04f * static_cast<float>(i + 1);
    float expected = mlc::runtime::dotProductRowQ8_1(row.data(), cols, input.data());
    std::vector<float> gpu_out;
    if (!executor.runMatMulQ8_1(std::string{}, row, input, 1, cols, stride, gpu_out)) {
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
    d_ptr[0] = testFloatToFp16(0.6f);
    d_ptr[1] = testFloatToFp16(0.2f);
    std::vector<float> input(cols);
    for (size_t i = 0; i < cols; ++i) input[i] = 0.001f * static_cast<float>(i + 1);
    float expected = mlc::runtime::dotProductRowQ2_K(weights.data(), cols, input.data());
    std::vector<float> gpu_out;
    if (!executor.runMatMulQ2K(std::string{}, weights, input, rows, cols, stride, gpu_out)) {
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
    d_ptr[0] = testFloatToFp16(0.5f);
    std::vector<float> input(cols);
    for (size_t i = 0; i < cols; ++i) input[i] = -0.002f * static_cast<float>(i + 1);
    float expected = mlc::runtime::dotProductRowQ3_K(weights.data(), cols, input.data());
    std::vector<float> gpu_out;
    if (!executor.runMatMulQ3K(std::string{}, weights, input, rows, cols, stride, gpu_out)) {
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

TEST(MetalRuntimeTest, AddResidualBiasMatchesCPUWhenAvailable) {
    auto& executor = mlc::runtime::MetalExecutor::Instance();
    if (!executor.isAvailable()) {
        GTEST_SKIP() << "Metal device unavailable";
    }
    std::vector<float> a = {1.0f, -2.0f, 3.5f, 0.25f};
    std::vector<float> b = {0.5f, 0.5f, -0.5f, 1.0f};
    std::vector<float> bias = {0.1f, -0.2f, 0.3f, 0.0f};
    std::vector<float> expected(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        expected[i] = a[i] + b[i] + bias[i];
    }
    std::vector<float> gpu_output;
    if (!executor.runAdd(a, b, gpu_output, &bias)) {
        GTEST_SKIP() << "Metal fused add+residual+bias unavailable";
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

TEST(MetalRuntimeTest, LayerNormMatchesCPUWhenAvailable) {
    auto& executor = mlc::runtime::MetalExecutor::Instance();
    if (!executor.isAvailable()) {
        GTEST_SKIP() << "Metal device unavailable";
    }
    std::vector<float> input = {0.5f, -1.0f, 2.0f, 0.25f};
    std::vector<float> weight = {1.0f, 0.9f, 1.1f, 0.8f};
    std::vector<float> bias = {0.1f, -0.2f, 0.3f, 0.0f};
    const float epsilon = 1e-5f;
    float sum = 0.0f;
    for (float v : input) sum += v;
    float mean = sum / static_cast<float>(input.size());
    float var = 0.0f;
    for (float v : input) {
        float d = v - mean;
        var += d * d;
    }
    float inv = 1.0f / std::sqrt(var / static_cast<float>(input.size()) + epsilon);
    std::vector<float> expected(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        expected[i] = (input[i] - mean) * inv * weight[i] + bias[i];
    }
    std::vector<float> gpu_output;
    if (!executor.runLayerNorm(input, weight, &bias, epsilon, gpu_output)) {
        GTEST_SKIP() << "Metal layer norm kernel unavailable";
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
    MetalExecutor::CacheDescriptor cache_k_desc{
        mlc::frontend::GGML_TYPE_F32,
        1,
        head_dim * sizeof(float),
        nullptr,
        nullptr,
        &gpu_cache_k};
    MetalExecutor::CacheDescriptor cache_v_desc{
        mlc::frontend::GGML_TYPE_F32,
        1,
        head_dim * sizeof(float),
        nullptr,
        nullptr,
        &gpu_cache_v};
    ASSERT_TRUE(executor.runAttention(q_values,
                                      k_values,
                                      v_values,
                                      num_heads,
                                      kv_heads,
                                      head_dim,
                                      context_length,
                                      mask_values,
                                      nullptr,
                                      0,
                                      cfg.rotary_dim,
                                      cfg.rope_freq_base,
                                      cfg.rope_freq_scale,
                                      cache_k_desc,
                                      cache_v_desc,
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

TEST(MetalRuntimeTest, AttentionQuantizedCachesStayOnGPU) {
    auto& executor = mlc::runtime::MetalExecutor::Instance();
    if (!executor.isAvailable()) {
        GTEST_SKIP() << "Metal device unavailable";
    }

    const size_t num_heads = 1;
    const size_t kv_heads = 1;
    const size_t head_dim = 2;
    const size_t context_length = 2;
    const size_t rows = kv_heads * context_length;

    // Single-token attention; expected output equals v_new.
    std::vector<float> q = {1.0f, 0.0f};
    std::vector<float> k_new = {1.0f, 0.0f};
    std::vector<float> v_new = {3.0f, 4.0f};
    std::vector<float> mask;

    size_t row_stride = mlc::runtime::q4_0RowSize(head_dim, 1);
    std::vector<uint8_t> raw_k(rows * row_stride, 0);
    std::vector<uint8_t> raw_v(rows * row_stride, 0);
    std::vector<float> kv_cache_k(rows * head_dim, 0.0f);
    std::vector<float> kv_cache_v(rows * head_dim, 0.0f);

    mlc::runtime::MetalBufferHandle k_handle;
    mlc::runtime::MetalBufferHandle v_handle;
    ASSERT_TRUE(executor.ensureSharedBuffer(kv_cache_k, k_handle));
    ASSERT_TRUE(executor.ensureSharedBuffer(kv_cache_v, v_handle));

    MetalExecutor::CacheDescriptor cache_k{
        mlc::frontend::GGML_TYPE_Q4_0,
        1,
        row_stride,
        &k_handle,
        &raw_k,
        &kv_cache_k};
    MetalExecutor::CacheDescriptor cache_v{
        mlc::frontend::GGML_TYPE_Q4_0,
        1,
        row_stride,
        &v_handle,
        &raw_v,
        &kv_cache_v};

    std::vector<float> output;
    ASSERT_TRUE(executor.runAttention(q,
                                      k_new,
                                      v_new,
                                      num_heads,
                                      kv_heads,
                                      head_dim,
                                      context_length,
                                      mask,
                                      nullptr,
                                      0,
                                      0,
                                      10000.0f,
                                      1.0f,
                                      cache_k,
                                      cache_v,
                                      output));

    ASSERT_EQ(output.size(), v_new.size());
    EXPECT_NEAR(output[0], v_new[0], 1e-5f);
    EXPECT_NEAR(output[1], v_new[1], 1e-5f);

    // Scatter should have updated the shared cache buffers.
    ASSERT_EQ(kv_cache_k[0], k_new[0]);
    ASSERT_EQ(kv_cache_k[1], k_new[1]);
    ASSERT_EQ(kv_cache_v[0], v_new[0]);
    ASSERT_EQ(kv_cache_v[1], v_new[1]);

    executor.releaseBuffer(k_handle);
    executor.releaseBuffer(v_handle);
}

TEST(MetalRuntimeTest, AttentionQuantizedKFormatCachesStayOnGPU_Row2) {
    auto& executor = mlc::runtime::MetalExecutor::Instance();
    if (!executor.isAvailable()) {
        GTEST_SKIP() << "Metal device unavailable";
    }

    const size_t num_heads = 1;
    const size_t kv_heads = 1;
    const size_t head_dim = 2;
    const size_t context_length = 2;
    const size_t rows = kv_heads * context_length;

    // Single-token attention; expected output equals v_new.
    std::vector<float> q = {0.5f, -0.5f};
    std::vector<float> k_new = {1.0f, 0.0f};
    std::vector<float> v_new = {2.0f, 3.0f};
    std::vector<float> mask;

    size_t row_stride = mlc::runtime::q4_kRowSize(head_dim);
    std::vector<uint8_t> raw_k(rows * row_stride, 0);
    std::vector<uint8_t> raw_v(rows * row_stride, 0);
    // Initialize raw buffers to something valid using quantize helper.
    std::vector<float> init_vals(head_dim, 0.0f);
    std::vector<uint8_t> tmp;
    mlc::runtime::quantizeRowQ4_K(init_vals.data(), head_dim, tmp);
    if (tmp.size() == row_stride) {
        for (size_t r = 0; r < rows; ++r) {
            std::copy(tmp.begin(), tmp.end(), raw_k.begin() + r * row_stride);
            std::copy(tmp.begin(), tmp.end(), raw_v.begin() + r * row_stride);
        }
    }

    std::vector<float> kv_cache_k(rows * head_dim, 0.0f);
    std::vector<float> kv_cache_v(rows * head_dim, 0.0f);

    mlc::runtime::MetalBufferHandle k_handle;
    mlc::runtime::MetalBufferHandle v_handle;
    ASSERT_TRUE(executor.ensureSharedBuffer(kv_cache_k, k_handle));
    ASSERT_TRUE(executor.ensureSharedBuffer(kv_cache_v, v_handle));

    MetalExecutor::CacheDescriptor cache_k{
        mlc::frontend::GGML_TYPE_Q4_K,
        1,
        row_stride,
        &k_handle,
        &raw_k,
        &kv_cache_k};
    MetalExecutor::CacheDescriptor cache_v{
        mlc::frontend::GGML_TYPE_Q4_K,
        1,
        row_stride,
        &v_handle,
        &raw_v,
        &kv_cache_v};

    std::vector<float> output;
    ASSERT_TRUE(executor.runAttention(q,
                                      k_new,
                                      v_new,
                                      num_heads,
                                      kv_heads,
                                      head_dim,
                                      context_length,
                                      mask,
                                      nullptr,
                                      0,
                                      0,
                                      10000.0f,
                                      1.0f,
                                      cache_k,
                                      cache_v,
                                      output));

    ASSERT_EQ(output.size(), v_new.size());
    EXPECT_NEAR(output[0], v_new[0], 1e-5f);
    EXPECT_NEAR(output[1], v_new[1], 1e-5f);

    ASSERT_EQ(kv_cache_k[0], k_new[0]);
    ASSERT_EQ(kv_cache_k[1], k_new[1]);
    ASSERT_EQ(kv_cache_v[0], v_new[0]);
    ASSERT_EQ(kv_cache_v[1], v_new[1]);

    executor.releaseBuffer(k_handle);
    executor.releaseBuffer(v_handle);
}

TEST(MetalRuntimeTest, AttentionQuantizedKFormatCachesStayOnGPU_Row4) {
    auto& executor = mlc::runtime::MetalExecutor::Instance();
    if (!executor.isAvailable()) {
        GTEST_SKIP() << "Metal device unavailable";
    }

    const size_t num_heads = 1;
    const size_t kv_heads = 1;
    const size_t head_dim = 4; // use full 4 slots to exercise packing
    const size_t context_length = 3;
    const size_t rows = kv_heads * context_length;

    std::vector<float> q = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> k_new = {0.5f, -0.5f, 0.25f, -0.25f};
    std::vector<float> v_new = {2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> mask;

    size_t row_stride = mlc::runtime::q4_kRowSize(head_dim);
    std::vector<uint8_t> raw_k(rows * row_stride, 0);
    std::vector<uint8_t> raw_v(rows * row_stride, 0);

    // Quantize the new K/V row into the raw buffers for the first position.
    std::vector<uint8_t> tmp_k;
    mlc::runtime::quantizeRowQ4_K(k_new.data(), head_dim, tmp_k);
    ASSERT_EQ(tmp_k.size(), row_stride);
    std::copy(tmp_k.begin(), tmp_k.end(), raw_k.begin());
    std::copy(tmp_k.begin(), tmp_k.end(), raw_v.begin());

    std::vector<float> kv_cache_k(rows * head_dim, 0.0f);
    std::vector<float> kv_cache_v(rows * head_dim, 0.0f);

    mlc::runtime::MetalBufferHandle k_handle;
    mlc::runtime::MetalBufferHandle v_handle;
    ASSERT_TRUE(executor.ensureSharedBuffer(kv_cache_k, k_handle));
    ASSERT_TRUE(executor.ensureSharedBuffer(kv_cache_v, v_handle));

    MetalExecutor::CacheDescriptor cache_k{
        mlc::frontend::GGML_TYPE_Q4_K,
        1,
        row_stride,
        &k_handle,
        &raw_k,
        &kv_cache_k};
    MetalExecutor::CacheDescriptor cache_v{
        mlc::frontend::GGML_TYPE_Q4_K,
        1,
        row_stride,
        &v_handle,
        &raw_v,
        &kv_cache_v};

    std::vector<float> output;
    ASSERT_TRUE(executor.runAttention(q,
                                      k_new,
                                      v_new,
                                      num_heads,
                                      kv_heads,
                                      head_dim,
                                      context_length,
                                      mask,
                                      nullptr,
                                      0,
                                      0,
                                      10000.0f,
                                      1.0f,
                                      cache_k,
                                      cache_v,
                                      output));

    ASSERT_EQ(output.size(), v_new.size());
    for (size_t i = 0; i < v_new.size(); ++i) {
        EXPECT_NEAR(output[i], v_new[i], 1e-4f);
    }

    // Shared cache buffers should reflect the written row.
    for (size_t i = 0; i < head_dim; ++i) {
        EXPECT_NEAR(kv_cache_k[i], k_new[i], 1e-4f);
        EXPECT_NEAR(kv_cache_v[i], v_new[i], 1e-4f);
    }

    executor.releaseBuffer(k_handle);
    executor.releaseBuffer(v_handle);
}

TEST(MetalRuntimeTest, AttentionQuantizedKFormatCachesStayOnGPU_Q5K) {
    auto& executor = mlc::runtime::MetalExecutor::Instance();
    if (!executor.isAvailable()) {
        GTEST_SKIP() << "Metal device unavailable";
    }

    const size_t num_heads = 1;
    const size_t kv_heads = 1;
    const size_t head_dim = 4;
    const size_t context_length = 2;
    const size_t rows = kv_heads * context_length;

    std::vector<float> q = {0.25f, -0.5f, 0.75f, -0.25f};
    std::vector<float> k_new = {0.5f, 0.5f, -0.5f, -0.5f};
    std::vector<float> v_new = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> mask;

    size_t row_stride = mlc::runtime::q5_kRowSize(head_dim);
    std::vector<uint8_t> raw_k(rows * row_stride, 0);
    std::vector<uint8_t> raw_v(rows * row_stride, 0);
    std::vector<uint8_t> tmp_k;
    mlc::runtime::quantizeRowQ5_K(k_new.data(), head_dim, tmp_k);
    ASSERT_EQ(tmp_k.size(), row_stride);
    std::copy(tmp_k.begin(), tmp_k.end(), raw_k.begin());
    std::copy(tmp_k.begin(), tmp_k.end(), raw_v.begin());

    std::vector<float> kv_cache_k(rows * head_dim, 0.0f);
    std::vector<float> kv_cache_v(rows * head_dim, 0.0f);

    mlc::runtime::MetalBufferHandle k_handle;
    mlc::runtime::MetalBufferHandle v_handle;
    ASSERT_TRUE(executor.ensureSharedBuffer(kv_cache_k, k_handle));
    ASSERT_TRUE(executor.ensureSharedBuffer(kv_cache_v, v_handle));

    MetalExecutor::CacheDescriptor cache_k{
        mlc::frontend::GGML_TYPE_Q5_K,
        1,
        row_stride,
        &k_handle,
        &raw_k,
        &kv_cache_k};
    MetalExecutor::CacheDescriptor cache_v{
        mlc::frontend::GGML_TYPE_Q5_K,
        1,
        row_stride,
        &v_handle,
        &raw_v,
        &kv_cache_v};

    std::vector<float> output;
    ASSERT_TRUE(executor.runAttention(q,
                                      k_new,
                                      v_new,
                                      num_heads,
                                      kv_heads,
                                      head_dim,
                                      context_length,
                                      mask,
                                      nullptr,
                                      0,
                                      0,
                                      10000.0f,
                                      1.0f,
                                      cache_k,
                                      cache_v,
                                      output));

    ASSERT_EQ(output.size(), v_new.size());
    for (size_t i = 0; i < v_new.size(); ++i) {
        EXPECT_NEAR(output[i], v_new[i], 1e-4f);
    }

    for (size_t i = 0; i < head_dim; ++i) {
        EXPECT_NEAR(kv_cache_k[i], k_new[i], 1e-4f);
        EXPECT_NEAR(kv_cache_v[i], v_new[i], 1e-4f);
    }

    executor.releaseBuffer(k_handle);
    executor.releaseBuffer(v_handle);
}

TEST(MetalRuntimeTest, AttentionQuantizedClassicQ5CachesStayOnGPU) {
    auto& executor = mlc::runtime::MetalExecutor::Instance();
    if (!executor.isAvailable()) {
        GTEST_SKIP() << "Metal device unavailable";
    }

    const size_t num_heads = 1;
    const size_t kv_heads = 1;
    const size_t head_dim = 4;
    const size_t context_length = 2;
    const size_t rows = kv_heads * context_length;

    std::vector<float> q = {0.1f, 0.2f, 0.3f, 0.4f};
    std::vector<float> k_new = {0.5f, -0.5f, 0.25f, -0.25f};
    std::vector<float> v_new = {2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> mask;

    size_t row_stride = mlc::runtime::q5_1RowSize(head_dim);
    std::vector<uint8_t> raw_k(rows * row_stride, 0);
    std::vector<uint8_t> raw_v(rows * row_stride, 0);
    std::vector<uint8_t> tmp_q;
    mlc::runtime::quantizeRowQ5_1(k_new.data(), head_dim, tmp_q);
    ASSERT_EQ(tmp_q.size(), row_stride);
    std::copy(tmp_q.begin(), tmp_q.end(), raw_k.begin());
    std::copy(tmp_q.begin(), tmp_q.end(), raw_v.begin());

    std::vector<float> kv_cache_k(rows * head_dim, 0.0f);
    std::vector<float> kv_cache_v(rows * head_dim, 0.0f);

    mlc::runtime::MetalBufferHandle k_handle;
    mlc::runtime::MetalBufferHandle v_handle;
    ASSERT_TRUE(executor.ensureSharedBuffer(kv_cache_k, k_handle));
    ASSERT_TRUE(executor.ensureSharedBuffer(kv_cache_v, v_handle));

    MetalExecutor::CacheDescriptor cache_k{
        mlc::frontend::GGML_TYPE_Q5_1,
        1,
        row_stride,
        &k_handle,
        &raw_k,
        &kv_cache_k};
    MetalExecutor::CacheDescriptor cache_v{
        mlc::frontend::GGML_TYPE_Q5_1,
        1,
        row_stride,
        &v_handle,
        &raw_v,
        &kv_cache_v};

    std::vector<float> output;
    ASSERT_TRUE(executor.runAttention(q,
                                      k_new,
                                      v_new,
                                      num_heads,
                                      kv_heads,
                                      head_dim,
                                      context_length,
                                      mask,
                                      nullptr,
                                      0,
                                      0,
                                      10000.0f,
                                      1.0f,
                                      cache_k,
                                      cache_v,
                                      output));

    ASSERT_EQ(output.size(), v_new.size());
    for (size_t i = 0; i < v_new.size(); ++i) {
        EXPECT_NEAR(output[i], v_new[i], 1e-4f);
    }

    for (size_t i = 0; i < head_dim; ++i) {
        EXPECT_NEAR(kv_cache_k[i], k_new[i], 1e-4f);
        EXPECT_NEAR(kv_cache_v[i], v_new[i], 1e-4f);
    }

    executor.releaseBuffer(k_handle);
    executor.releaseBuffer(v_handle);
}

TEST(MetalRuntimeTest, AttentionMultiHeadMaskedMatchesCPUWhenAvailable) {
    auto& executor = mlc::runtime::MetalExecutor::Instance();
    if (!executor.isAvailable()) {
        GTEST_SKIP() << "Metal device unavailable";
    }

    const size_t num_heads = 2;
    const size_t kv_heads = 1; // GQA style
    const size_t head_dim = 2;
    const size_t context_length = 2;
    const size_t kv_span = kv_heads * head_dim;
    // Two tokens of K/V; mask will zero out the second token.
    std::vector<float> q = {
        1.0f, 0.0f,  // head 0
        0.5f, 0.5f   // head 1
    };
    std::vector<float> k_new = {
        1.0f, 0.0f,  // token 0
        0.0f, 1.0f   // token 1
    };
    std::vector<float> v_new = {
        10.0f, 20.0f, // token 0
        30.0f, 40.0f  // token 1
    };
    std::vector<float> mask = {0.0f, -1e9f};

    // Use F32 caches to keep focus on attention math/masking.
    std::vector<float> kv_cache_k(context_length * kv_span, 0.0f);
    std::vector<float> kv_cache_v(context_length * kv_span, 0.0f);

    mlc::runtime::MetalBufferHandle k_handle;
    mlc::runtime::MetalBufferHandle v_handle;
    ASSERT_TRUE(executor.ensureSharedBuffer(kv_cache_k, k_handle));
    ASSERT_TRUE(executor.ensureSharedBuffer(kv_cache_v, v_handle));

    MetalExecutor::CacheDescriptor cache_k{
        mlc::frontend::GGML_TYPE_F32,
        1,
        static_cast<size_t>(kv_span * sizeof(float)),
        &k_handle,
        nullptr,
        &kv_cache_k};
    MetalExecutor::CacheDescriptor cache_v{
        mlc::frontend::GGML_TYPE_F32,
        1,
        static_cast<size_t>(kv_span * sizeof(float)),
        &v_handle,
        nullptr,
        &kv_cache_v};

    std::vector<float> output;
    ASSERT_TRUE(executor.runAttention(q,
                                      k_new,
                                      v_new,
                                      num_heads,
                                      kv_heads,
                                      head_dim,
                                      context_length,
                                      mask,
                                      nullptr,
                                      0,
                                      0,
                                      10000.0f,
                                      1.0f,
                                      cache_k,
                                      cache_v,
                                      output));

    ASSERT_EQ(output.size(), num_heads * head_dim);
    // Mask removes token 1, so each head should output token 0's V.
    EXPECT_NEAR(output[0], v_new[0], 1e-5f);
    EXPECT_NEAR(output[1], v_new[1], 1e-5f);
    EXPECT_NEAR(output[2], v_new[0], 1e-5f);
    EXPECT_NEAR(output[3], v_new[1], 1e-5f);

    executor.releaseBuffer(k_handle);
    executor.releaseBuffer(v_handle);
}

class MetalAttentionEquivalenceTest
    : public ::testing::TestWithParam<std::tuple<int, int, int, int, bool>> {};

TEST_P(MetalAttentionEquivalenceTest, MatchesCPUAcrossShapesAndMasks) {
    auto& executor = mlc::runtime::MetalExecutor::Instance();
    if (!executor.isAvailable()) {
        GTEST_SKIP() << "Metal device unavailable";
    }
    int heads = std::get<0>(GetParam());
    int kv_heads = std::get<1>(GetParam());
    int head_dim = std::get<2>(GetParam());
    int tokens = std::get<3>(GetParam());
    bool apply_mask = std::get<4>(GetParam());
    kv_heads = std::max(1, kv_heads);
    heads = std::max(1, heads);
    head_dim = std::max(2, head_dim);
    tokens = std::max(1, tokens);

    size_t kv_span = kv_heads * head_dim;
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> q(heads * head_dim);
    std::vector<float> k(tokens * kv_span);
    std::vector<float> v(tokens * kv_span);
    for (float& f : q) f = dist(rng);
    for (float& f : k) f = dist(rng);
    for (float& f : v) f = dist(rng);

    std::vector<float> mask;
    if (apply_mask) {
        mask.resize(tokens);
        for (int i = 0; i < tokens; ++i) {
            mask[i] = (i == tokens - 1) ? -1e9f : 0.0f;
        }
    }

    // Build CPU cache and run attention (using Session/ExecutionGraph).
    mlc::runtime::ExecutionGraph graph;
    graph.addTensor("q", {static_cast<int64_t>(q.size())}, mlc::ir::DataType::F32);
    graph.addTensor("k_in", {static_cast<int64_t>(k.size())}, mlc::ir::DataType::F32);
    graph.addTensor("v_in", {static_cast<int64_t>(v.size())}, mlc::ir::DataType::F32);
    auto& k_cache = graph.addTensor("kv_k", std::vector<int64_t>{kv_heads, tokens, head_dim}, mlc::ir::DataType::F32);
    k_cache.is_state = true;
    auto& v_cache = graph.addTensor("kv_v", std::vector<int64_t>{kv_heads, tokens, head_dim}, mlc::ir::DataType::F32);
    v_cache.is_state = true;
    graph.addTensor("attn_out", {static_cast<int64_t>(heads * head_dim)}, mlc::ir::DataType::F32);
    if (apply_mask) {
        graph.addTensor("attn_mask", {static_cast<int64_t>(mask.size())}, mlc::ir::DataType::F32);
    }
    auto& attn = graph.addNode("attn",
                               mlc::runtime::ExecOpType::Attention,
                               {"q", "k_in", "v_in"},
                               {"attn_out", "kv_k", "kv_v"},
                               mlc::runtime::BackendKind::CPU);
    attn.attributes["heads"] = static_cast<float>(heads);
    attn.attributes["kv_heads"] = static_cast<float>(kv_heads);
    attn.attributes["head_dim"] = static_cast<float>(head_dim);
    attn.annotations["kv_cache_k"] = "kv_k";
    attn.annotations["kv_cache_v"] = "kv_v";
    if (apply_mask) attn.annotations["attention_mask"] = "attn_mask";

    mlc::runtime::Session session(createRunnerGGUFFile());
    mlc::runtime::ExecutionContext ctx(session, &graph);
    ctx.setSequencePosition(0);
    ctx.setTensor("q", q);
    ctx.setTensor("k_in", k);
    ctx.setTensor("v_in", v);
    if (apply_mask) ctx.setTensor("attn_mask", mask);

    mlc::runtime::ExecutionExecutor cpu_exec(graph, nullptr, &ctx);
    auto status = cpu_exec.run();
    ASSERT_TRUE(status.success);
    const auto* cpu_out = ctx.getTensor("attn_out");
    ASSERT_NE(cpu_out, nullptr);

    // Prepare Metal caches.
    std::vector<float> kv_cache_k(tokens * kv_span, 0.0f);
    std::vector<float> kv_cache_v(tokens * kv_span, 0.0f);
    mlc::runtime::MetalBufferHandle k_handle;
    mlc::runtime::MetalBufferHandle v_handle;
    ASSERT_TRUE(executor.ensureSharedBuffer(kv_cache_k, k_handle));
    ASSERT_TRUE(executor.ensureSharedBuffer(kv_cache_v, v_handle));

    MetalExecutor::CacheDescriptor cache_k{
        mlc::frontend::GGML_TYPE_F32,
        1,
        static_cast<size_t>(kv_span * sizeof(float)),
        &k_handle,
        nullptr,
        &kv_cache_k};
    MetalExecutor::CacheDescriptor cache_v{
        mlc::frontend::GGML_TYPE_F32,
        1,
        static_cast<size_t>(kv_span * sizeof(float)),
        &v_handle,
        nullptr,
        &kv_cache_v};

    std::vector<float> gpu_out;
    ASSERT_TRUE(executor.runAttention(q,
                                      k,
                                      v,
                                      heads,
                                      kv_heads,
                                      head_dim,
                                      tokens,
                                      mask,
                                      nullptr,
                                      0,
                                      0,
                                      10000.0f,
                                      1.0f,
                                      cache_k,
                                      cache_v,
                                      gpu_out));

    ASSERT_EQ(gpu_out.size(), cpu_out->size());
    for (size_t i = 0; i < gpu_out.size(); ++i) {
        EXPECT_NEAR(gpu_out[i], (*cpu_out)[i], 5e-3f) << "idx=" << i;
    }

    executor.releaseBuffer(k_handle);
    executor.releaseBuffer(v_handle);
}

INSTANTIATE_TEST_SUITE_P(
    MetalAttentionSweeps,
    MetalAttentionEquivalenceTest,
    ::testing::Values(
        std::make_tuple(1, 1, 32, 2, false),
        std::make_tuple(2, 1, 64, 3, true),
        std::make_tuple(4, 2, 32, 4, false),
        std::make_tuple(8, 4, 32, 2, true)));

TEST(MetalRuntimeTest, Integration_GGUFMatMulMatchesCPUWhenAvailable) {
    auto& executor = mlc::runtime::MetalExecutor::Instance();
    if (!executor.isAvailable()) {
        GTEST_SKIP() << "Metal device unavailable";
    }

    std::string path = createRunnerGGUFFile();
    mlc::runtime::Session session(path);
    const auto& tensors = session.loader().tensors();
    auto it = tensors.find("output.weight");
    ASSERT_NE(it, tensors.end());
    const auto& tensor = it->second;
    ASSERT_EQ(tensor.dtype, mlc::frontend::GGML_TYPE_F32);
    ASSERT_EQ(tensor.shape.size(), 2u);
    // Canonical GGML: shape[0]=ne0=cols (input dim), shape[1]=ne1=rows (output dim).
    size_t cols = static_cast<size_t>(tensor.shape[0]);
    size_t rows = static_cast<size_t>(tensor.shape[1]);
    auto raw = session.loader().loadTensorData(tensor);
    ASSERT_EQ(raw.size(), rows * cols * sizeof(float));
    std::vector<float> weights(rows * cols);
    std::memcpy(weights.data(), raw.data(), raw.size());

    // Use embedding for token 1 from the same file. Embedding dim = cols (input dim).
    std::vector<float> input = session.getEmbedding("tok_embeddings.weight", 1);
    ASSERT_EQ(input.size(), cols);

    // CPU expected.
    std::vector<float> expected(rows, 0.0f);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            expected[r] += weights[r * cols + c] * input[c];
        }
    }

    std::vector<float> gpu_out;
    ASSERT_TRUE(executor.runMatMul(std::string{}, weights, input, rows, cols, false, gpu_out));
    ASSERT_EQ(gpu_out.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(gpu_out[i], expected[i], 1e-4f);
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

// TODO(quant-kv-cpu-attention): Pre-existing failure unrelated to Bug A or Bug B.
// CPU attention path with Q4_0-flagged KV-cache state tensors fails inside executor.run()
// (status.success comes back false). Probably a path that decodeCacheTensor or the CPU
// attention kernel doesn't fully handle for the inline-graph (non-IR-builder) test setup.
// Needs its own investigation; not on the Bug B critical path.
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
    EXPECT_NEAR((*out)[0], 3.0f, 2e-1f);
    EXPECT_NEAR((*out)[1], 4.0f, 2e-1f);

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

TEST(DecodeRunnerTest, CacheReportReflectsStateStorage) {
    std::string path = createRunnerGGUFFile();
    mlc::runtime::Session session(path);

    mlc::runtime::ExecutionGraph graph;
    auto& cache = graph.addTensor("kv_cache_report",
                                  std::vector<int64_t>{2, 3, 4},
                                  mlc::ir::DataType::F32);
    cache.is_state = true;
    cache.ggml_dtype = mlc::frontend::GGML_TYPE_Q4_0;
    cache.has_ggml_dtype = true;
    cache.quant_version = 1;

    mlc::runtime::ExecutionContext context(session, &graph);
    context.ensureStateTensor(cache);

    auto* storage = context.tensorStorage("kv_cache_report");
    ASSERT_NE(storage, nullptr);
    EXPECT_EQ(storage->dtype, mlc::frontend::GGML_TYPE_Q4_0);

    auto report = mlc::runtime::BuildCacheReport(graph, context, session.loader());
    ASSERT_EQ(report.size(), 1u);
    const auto& entry = report[0];
    EXPECT_EQ(entry.name, "kv_cache_report");
    EXPECT_EQ(entry.dtype, mlc::frontend::GGML_TYPE_Q4_0);
    EXPECT_EQ(entry.quant_version, 1u);
    EXPECT_GT(entry.row_stride_bytes, 0u);
    EXPECT_GT(entry.byte_size, 0u);

    fs::remove(path);
}

// TODO(internal-bpe): Test exercises the internal BPE fallback in tokenizer.cpp by
// constructing a synthetic GGUF that llama.cpp's vocab loader cannot consume (no
// tokenizer.ggml.model field, missing scores), so use_llama_cpp_ stays false.
// Production (mlc chat on TinyLlama) goes through the llama.cpp delegation, never
// touches this code path. The internal BPE merge logic fails to merge "world" into
// token id 18 — emits "w","o","r","l","d" as separate tokens. Real bug, but in
// dead-for-production code; revisit if/when we drop the llama.cpp tokenizer dep.
TEST(TokenizerTest, EncodesAndDecodesText) {
    std::string path = "/tmp/test_tokenizer_" + std::to_string(getpid()) + ".gguf";
    std::ofstream file(path, std::ios::binary);
    ASSERT_TRUE(file.is_open());

    // Header: version 1, no tensors, 6 KV entries.
    writeU32(file, 0x46554747);
    writeU32(file, 1);
    writeU64(file, 0); // n_tensors
    writeU64(file, 6); // n_kv

    // tokenizer.ggml.tokens (array of strings)
    writeString(file, "tokenizer.ggml.tokens");
    writeU32(file, 9);  // ARRAY (v1 enum)
    writeU32(file, 8);  // element type STRING
    std::vector<std::string> toks = {
        "<s>", "<eos>", "<unk>",
        "h", "e", "l", "o", " ", "w", "r", "d",
        "he", "hel", "hell", "hello",
        "wo", "wor", "worl", "world"
    };
    writeU64(file, toks.size());
    for (const auto& t : toks) writeString(file, t);

    // bos/eos/unk ids
    writeString(file, "tokenizer.ggml.bos_token_id");
    writeU32(file, 4); // UINT32
    writeU32(file, 0);

    writeString(file, "tokenizer.ggml.eos_token_id");
    writeU32(file, 4);
    writeU32(file, 1);

    writeString(file, "tokenizer.ggml.unk_token_id");
    writeU32(file, 4);
    writeU32(file, 2);

    // scores
    writeString(file, "tokenizer.ggml.scores");
    writeU32(file, 9); // ARRAY
    writeU32(file, 6); // element type FLOAT32 (v1 enum index for FLOAT32 is 6)
    std::vector<float> scores = {
        0.0f, 0.0f, -10.0f, // <s>, <eos>, <unk>
        -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, // single letters
        1.0f, 2.0f, 3.0f, 4.0f, // he, hel, hell, hello
        1.0f, 2.0f, 3.0f, 4.0f  // wo, wor, worl, world
    };
    writeU64(file, scores.size());
    for (float s : scores) writeF32(file, s);

    // merges
    writeString(file, "tokenizer.ggml.merges");
    writeU32(file, 9); // ARRAY
    writeU32(file, 8); // STRING
    std::vector<std::string> merges = {
        "h e",
        "he l",
        "hel l",
        "hell o",
        "  w",
        "wo r",
        "wor l",
        "worl d",
        "hello  "
    };
    writeU64(file, merges.size());
    for (const auto& m : merges) writeString(file, m);

    file.close();

    mlc::frontend::GGUFLoader loader(path);
    ASSERT_TRUE(loader.load());
    mlc::runtime::Tokenizer tokenizer(loader);
    ASSERT_TRUE(tokenizer.valid());

    mlc::runtime::TokenizerConfig cfg;
    cfg.add_bos = true;
    cfg.add_eos = true;

    auto ids = tokenizer.encode("hello world", cfg);
    // Expected tokens: <s>, hello, " ", world, <eos>
    std::vector<uint64_t> expected = {0, 14, 7, 18, 1};
    ASSERT_EQ(ids, expected);

    auto decoded = tokenizer.decode(ids);
    EXPECT_EQ(decoded, "hello world");

    fs::remove(path);
}

#include "runtime/parity_harness.hpp"

TEST(ParityHarnessTest, EmbeddingOutputMatchesBetweenMetalAndCpu) {
    // Real model needed — synthetic GGUFs in this file don't carry full weights.
    // Skip when the model isn't on disk so CI without the asset still passes.
    const char* env_path = std::getenv("MLC_PARITY_MODEL");
    std::string path = env_path ? env_path
                                : "models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
    if (!fs::exists(path)) {
        GTEST_SKIP() << "Parity model not found at " << path
                     << " (set MLC_PARITY_MODEL to override)";
    }

    mlc::runtime::parity::CompareOptions opts;
    opts.gguf_path = path;
    opts.prompt = "Hi";
    // Only need the embedding boundary for this assertion — keeps the test fast.
    opts.tap_patterns = {"hidden_state_0"};

    auto report = mlc::runtime::parity::compareMetalVsCpu(opts);
    ASSERT_FALSE(report.layers.empty());

    const mlc::runtime::parity::LayerComparison* embed = nullptr;
    for (const auto& l : report.layers) {
        if (l.name == "hidden_state_0") {
            embed = &l;
            break;
        }
    }
    ASSERT_NE(embed, nullptr) << "hidden_state_0 not captured by either side";
    EXPECT_TRUE(embed->present_a);
    EXPECT_TRUE(embed->present_b);
    EXPECT_GT(embed->element_count, 0u);
    EXPECT_GT(embed->cosine_sim, 0.9999f)
        << "embedding cosine_sim=" << embed->cosine_sim
        << " (max_abs=" << embed->max_abs_diff
        << " rms=" << embed->rms << ")";
}
