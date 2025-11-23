#pragma once

#include "passes.hpp"

namespace mlc {
namespace passes {

class TilingPass : public Pass {
public:
    const char* name() const override { return "Tiling"; }
    bool Run(ir::Graph& graph) override;
};

} // namespace passes
} // namespace mlc
