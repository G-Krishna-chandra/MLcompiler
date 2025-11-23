#include <gtest/gtest.h>
#include "passes/passes.hpp"

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

