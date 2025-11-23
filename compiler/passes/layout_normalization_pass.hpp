#pragma once

#include "passes.hpp"

namespace mlc {
namespace passes {

class LayoutNormalizationPass : public Pass {
public:
    const char* name() const override { return "LayoutNormalization"; }
    bool Run(ir::Graph& graph) override;
};

} // namespace passes
} // namespace mlc
