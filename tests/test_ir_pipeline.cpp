#include <gtest/gtest.h>

#include "frontends/gguf_loader.hpp"
#include "pipeline/ir_pipeline.hpp"
#include "tests/gguf_test_utils.hpp"

#include <filesystem>
#include <fstream>
#include <unistd.h>

using namespace mlc::test::gguf;

namespace {

std::string createPipelineGGUFFile() {
    std::string path = "/tmp/test_ir_pipeline_" + std::to_string(getpid()) + ".gguf";
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open pipeline GGUF file");
    }

    constexpr uint32_t magic = 0x46554747;
    writeU32(file, magic);
    writeU32(file, 1); // version
    writeU64(file, 10); // tensors
    writeU64(file, 3);  // kv pairs

    writeString(file, "llama.block_count");
    writeU32(file, 4);
    writeU32(file, 1);

    writeString(file, "llama.embedding_length");
    writeU32(file, 4);
    writeU32(file, 4);

    writeString(file, "llama.attention.head_count");
    writeU32(file, 4);
    writeU32(file, 2);

    auto writeTensor = [&](const std::string& name,
                           uint32_t rows,
                           uint32_t cols,
                           uint64_t offset) {
        writeString(file, name);
        writeU32(file, 2);
        writeU64(file, 2);
        writeU32(file, rows);
        writeU32(file, cols);
        writeU32(file, 0); // F32
        writeU64(file, offset);
    };

    const uint64_t base = 512;
    const uint64_t stride = 256;
    writeTensor("tok_embeddings.weight", 4, 4, base + stride * 0);
    writeTensor("blk.0.attn_q.weight", 4, 4, base + stride * 1);
    writeTensor("blk.0.attn_k.weight", 4, 4, base + stride * 2);
    writeTensor("blk.0.attn_v.weight", 4, 4, base + stride * 3);
    writeTensor("blk.0.attn_output.weight", 4, 4, base + stride * 4);
    writeTensor("blk.0.ffn_gate.weight", 4, 4, base + stride * 5);
    writeTensor("blk.0.ffn_up.weight", 4, 4, base + stride * 6);
    writeTensor("blk.0.ffn_down.weight", 4, 4, base + stride * 7);
    writeTensor("blk.0.attn_norm.weight", 4, 4, base + stride * 8);
    writeTensor("output.weight", 8, 4, base + stride * 9);

    auto writeTensorData = [&](uint64_t offset, size_t elements) {
        file.seekp(offset, std::ios::beg);
        std::vector<float> values(elements, 0.1f);
        file.write(reinterpret_cast<const char*>(values.data()),
                   values.size() * sizeof(float));
    };

    writeTensorData(base + stride * 0, 16);
    writeTensorData(base + stride * 1, 16);
    writeTensorData(base + stride * 2, 16);
    writeTensorData(base + stride * 3, 16);
    writeTensorData(base + stride * 4, 16);
    writeTensorData(base + stride * 5, 16);
    writeTensorData(base + stride * 6, 16);
    writeTensorData(base + stride * 7, 16);
    writeTensorData(base + stride * 8, 16);
    writeTensorData(base + stride * 9, 32);
    file.close();
    return path;
}

} // namespace

TEST(IRPipelineTest, BuildsAndSchedulesGraph) {
    std::string path = createPipelineGGUFFile();
    mlc::frontend::GGUFLoader loader(path);
    ASSERT_TRUE(loader.load());

    mlc::pipeline::IRPipeline pipeline;
    auto result = pipeline.Run(loader);

    ASSERT_TRUE(result.ir_graph);
    EXPECT_GT(result.ir_graph->nodes().size(), 5u);
    EXPECT_GT(result.exec_graph.nodes().size(), 5u);

    bool saw_cpu = false;
    bool saw_metal = false;
    for (const auto& node : result.exec_graph.nodes()) {
        if (node.backend == mlc::runtime::BackendKind::CPU) saw_cpu = true;
        if (node.backend == mlc::runtime::BackendKind::Metal) saw_metal = true;
    }
    EXPECT_TRUE(saw_cpu);
    EXPECT_TRUE(saw_metal);

    std::filesystem::remove(path);
}
