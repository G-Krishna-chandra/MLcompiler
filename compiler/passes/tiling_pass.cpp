#include "passes/tiling_pass.hpp"

#include <algorithm>

namespace mlc {
namespace passes {

bool TilingPass::Run(ir::Graph& graph) {
    bool changed = false;
    for (ir::Node* node : graph.nodes()) {
        if (!node) continue;
        switch (node->kind) {
            case ir::OpKind::MatMul:
            case ir::OpKind::Linear:
            case ir::OpKind::Attention:
            case ir::OpKind::FeedForward:
                break;
            default:
                continue;
        }
        if (node->outputs.empty() || !node->outputs[0]) continue;
        const auto& shape = node->outputs[0]->shape;
        if (shape.empty()) continue;
        float m = static_cast<float>(std::abs(shape[0]));
        float n = static_cast<float>(shape.size() > 1 ? std::abs(shape[1]) : shape[0]);
        float tile_m = std::clamp(m / 4.0f, 8.0f, 256.0f);
        float tile_n = std::clamp(n / 4.0f, 8.0f, 256.0f);
        if (node->attributes["tile_m"] != tile_m ||
            node->attributes["tile_n"] != tile_n) {
            node->attributes["tile_m"] = tile_m;
            node->attributes["tile_n"] = tile_n;
            changed = true;
        }
    }
    return changed;
}

} // namespace passes
} // namespace mlc
