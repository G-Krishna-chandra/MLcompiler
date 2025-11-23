#pragma once

#include "passes.hpp"

namespace mlc {
namespace passes {

class MatMulFusionPass : public Pass {
public:
    const char* name() const override { return "MatMulFusion"; }
    bool Run(ir::Graph& graph) override;
};

} // namespace passes
} // namespace mlc
