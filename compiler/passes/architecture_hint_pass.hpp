#pragma once

#include "passes/passes.hpp"
#include "runtime/execution_graph.hpp"

namespace mlc {
namespace passes {

// Annotate the IR with architecture-specific hints (family, head dims, kv heads, sliding window).
class ArchitectureHintPass : public Pass {
public:
    explicit ArchitectureHintPass(const runtime::ModelConfig& config) : config_(config) {}
    const char* name() const override { return "ArchitectureHintPass"; }
    bool Run(ir::Graph& graph) override;

private:
    runtime::ModelConfig config_;
};

} // namespace passes
} // namespace mlc
