#include <gtest/gtest.h>
#include "runtime/execution_graph.hpp"
#include "runtime/execution_plan_builder.hpp"
#include "runtime/execution_executor.hpp"

using namespace mlc::runtime;
using mlc::ir::DataType;

TEST(ExecutionGraphTest, SimpleTopologicalOrder) {
    ExecutionGraph graph;
    graph.addTensor("tokens", {1}, DataType::I4);
    graph.addTensor("hidden_0", {4}, DataType::F32);
    graph.addNode("embedding", ExecOpType::Embedding, {"tokens"}, {"hidden_0"});
    graph.addTensor("hidden_1", {4}, DataType::F32);
    graph.addNode("attention", ExecOpType::Attention, {"hidden_0"}, {"hidden_1"});
    graph.addNode("lm_head", ExecOpType::Output, {"hidden_1"}, {"logits"});

    auto order = graph.topologicalOrder();
    ASSERT_EQ(order.size(), 3u);
    EXPECT_EQ(order.front(), "embedding");
    EXPECT_EQ(order.back(), "lm_head");

    auto dump = graph.dump();
    EXPECT_NE(dump.find("attention"), std::string::npos);
}

TEST(ExecutionPlanBuilderTest, ToyPlanGeneratesLayers) {
    auto graph = ExecutionPlanBuilder::BuildToy(3, 4096);
    ASSERT_FALSE(graph.nodes().empty());
    EXPECT_EQ(graph.nodes().size(), 1 + 3 * 2 + 2); // embedding + 3*(attn+ffn) + norm + head

    auto order = graph.topologicalOrder();
    EXPECT_EQ(order.size(), graph.nodes().size());
    EXPECT_EQ(order.front(), "embedding_lookup");
    EXPECT_EQ(order.back(), "lm_head");

    const auto& cfg = graph.modelConfig();
    EXPECT_EQ(cfg.num_layers, 3u);
    EXPECT_EQ(cfg.hidden_size, 4096u);
    EXPECT_GT(cfg.head_count, 0u);

    const auto& tensors = graph.tensors();
    auto k_it = tensors.find("layer_0_kv_cache_k");
    ASSERT_NE(k_it, tensors.end());
    EXPECT_TRUE(k_it->second.is_state);
    EXPECT_EQ(k_it->second.metadata.at("kv_kind"), "k");
    EXPECT_EQ(k_it->second.metadata.at("layer"), "0");

    auto v_it = tensors.find("layer_0_kv_cache_v");
    ASSERT_NE(v_it, tensors.end());
    EXPECT_TRUE(v_it->second.is_state);
    EXPECT_EQ(v_it->second.metadata.at("kv_kind"), "v");
}

TEST(ExecutionExecutorTest, SimulatesPlanInTopoOrder) {
    auto graph = ExecutionPlanBuilder::BuildToy(2, 128);
    ExecutionExecutor executor(graph);
    auto result = executor.run();
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.executed_nodes, graph.nodes().size());
    ASSERT_FALSE(result.trace.empty());
    EXPECT_EQ(result.trace.front().node, "embedding_lookup");
    ASSERT_FALSE(result.trace.front().notes.empty());
    EXPECT_NE(result.trace.front().notes[0].find("backend"), std::string::npos);
    bool found_attn = false;
    for (const auto& entry : result.trace) {
        if (entry.op == ExecOpType::Attention) {
            found_attn = true;
            EXPECT_FALSE(entry.notes.empty());
        }
    }
    EXPECT_TRUE(found_attn);
}
