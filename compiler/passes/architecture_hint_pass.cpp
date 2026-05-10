#include "passes/architecture_hint_pass.hpp"

#include <algorithm>
#include <string>

#include "runtime/execution_graph.hpp"

namespace mlc {
namespace passes {

bool ArchitectureHintPass::Run(ir::Graph& graph) {
    bool changed = false;
    const std::string family_str = runtime::toString(config_.family);
    const float head_dim = config_.head_dim > 0
                               ? static_cast<float>(config_.head_dim)
                               : (config_.hidden_size > 0 && config_.head_count > 0
                                      ? static_cast<float>(config_.hidden_size / config_.head_count)
                                      : 0.0f);

    for (ir::Node* node : graph.nodes()) {
        if (!node) continue;
        // Tag every node with family for downstream scheduling/codegen.
        node->metadata["architecture_family"] = family_str;

        if (node->kind == ir::OpKind::Attention) {
            if (config_.head_count > 0) node->attributes["heads"] = static_cast<float>(config_.head_count);
            if (config_.kv_head_count > 0) node->attributes["kv_heads"] = static_cast<float>(config_.kv_head_count);
            if (head_dim > 0.0f) node->attributes["head_dim"] = head_dim;
            if (config_.sliding_window > 0) {
                node->metadata["sliding_window"] = std::to_string(config_.sliding_window);
            }
            if (config_.use_alibi) {
                node->metadata["use_alibi"] = "true";
            }
            if (config_.rope_freq_base > 0.0f) {
                node->attributes["rope_freq_base"] = config_.rope_freq_base;
                node->attributes["rope_freq_scale"] = config_.rope_freq_scale;
            }
            if (config_.grouped_query_attention) {
                node->metadata["grouped_query"] = "true";
            }
            // Provide tiling hints tuned per head size.
            if (head_dim > 0.0f && !node->attributes.count("tile_m")) {
                float tile = std::clamp(head_dim * 2.0f, 16.0f, 256.0f);
                node->attributes["tile_m"] = tile;
                node->attributes["tile_n"] = tile;
                changed = true;
            }
        } else if (node->kind == ir::OpKind::FeedForward) {
            if (!config_.activation.empty()) {
                node->metadata["activation"] = config_.activation;
            }
        } else if (node->kind == ir::OpKind::Embedding) {
            // Propagate embedding layout hints if present on weights.
            for (auto* t : node->tensor_inputs) {
                if (!t) continue;
                auto it = t->metadata.find("tokens_column_major");
                if (it != t->metadata.end() && it->second == "true") {
                    node->metadata["embedding_layout"] = "column_major";
                    break;
                }
            }
        } else if (node->kind == ir::OpKind::MatMul || node->kind == ir::OpKind::Linear) {
            // Encourage heavier tiling for large hidden dims.
            if (config_.hidden_size > 0 && !node->attributes.count("tile_m")) {
                float tile = std::clamp(static_cast<float>(config_.hidden_size) / 8.0f, 16.0f, 512.0f);
                node->attributes["tile_m"] = tile;
                node->attributes["tile_n"] = tile;
                changed = true;
            }
        }
    }
    return changed;
}

} // namespace passes
} // namespace mlc
