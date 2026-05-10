#include <gtest/gtest.h>
#include "passes/passes.hpp"
#include "passes/architecture_hint_pass.hpp"
#include "runtime/execution_graph.hpp"
#include "ir/ir.hpp"

namespace {
class DummyPass : public mlc::passes::Pass {
public:
    explicit DummyPass(bool change) : change_(change) {}
    const char* name() const override { return "DummyPass"; }
    bool Run(mlc::ir::Graph&) override { return change_; }
private:
    bool change_;
};
} // namespace

TEST(PassesTest, BasicInitialization) {
    mlc::passes::PassManager manager;
    mlc::ir::Graph graph;
    EXPECT_EQ(manager.run(graph), 0u);
}

TEST(PassesTest, AddPass) {
    mlc::passes::PassManager manager;
    mlc::ir::Graph graph;
    manager.addPass(std::make_unique<DummyPass>(true));
    manager.addPass(std::make_unique<DummyPass>(false));
    EXPECT_EQ(manager.run(graph), 1u);
}

TEST(PassesTest, ArchitectureHintPassAnnotatesGraph) {
    mlc::runtime::ModelConfig config;
    config.family = mlc::runtime::ArchitectureFamily::Gemma;
    config.head_count = 8;
    config.kv_head_count = 4;
    config.hidden_size = 512;
    config.head_dim = 64;
    config.sliding_window = 4096;
    config.grouped_query_attention = true;
    config.activation = "geglu";

    mlc::ir::Graph graph;
    // Attention node
    auto* attn_out = graph.addTensor("attn_out", {4, 64}, mlc::ir::DataType::F32);
    auto* attn = graph.addNode(mlc::ir::OpKind::Attention, "attn");
    attn->outputs.push_back(attn_out);
    // FeedForward node
    auto* ffn_out = graph.addTensor("ffn_out", {4, 512}, mlc::ir::DataType::F32);
    auto* ffn = graph.addNode(mlc::ir::OpKind::FeedForward, "ffn");
    ffn->outputs.push_back(ffn_out);
    // MatMul node
    auto* mm_out = graph.addTensor("mm_out", {512, 512}, mlc::ir::DataType::F32);
    auto* mm = graph.addNode(mlc::ir::OpKind::MatMul, "mm");
    mm->outputs.push_back(mm_out);

    mlc::passes::ArchitectureHintPass pass(config);
    EXPECT_TRUE(pass.Run(graph));

    // Family annotation
    EXPECT_EQ(attn->metadata["architecture_family"], std::string("gemma"));
    // Attention hints
    EXPECT_FLOAT_EQ(attn->attributes["heads"], 8.0f);
    EXPECT_FLOAT_EQ(attn->attributes["kv_heads"], 4.0f);
    EXPECT_FLOAT_EQ(attn->attributes["head_dim"], 64.0f);
    EXPECT_EQ(attn->metadata["grouped_query"], std::string("true"));
    EXPECT_EQ(attn->metadata["sliding_window"], std::to_string(config.sliding_window));
    EXPECT_EQ(attn->metadata["use_alibi"], std::string()); // default false -> not set
    EXPECT_TRUE(attn->attributes.count("tile_m") > 0);
    // FeedForward activation metadata
    EXPECT_EQ(ffn->metadata["activation"], std::string("geglu"));
    // MatMul tiling should be set from hidden_size
    EXPECT_TRUE(mm->attributes.count("tile_m") > 0);
    EXPECT_TRUE(mm->attributes.count("tile_n") > 0);
}
