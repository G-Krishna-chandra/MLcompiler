#include "passes/matmul_fusion_pass.hpp"

namespace mlc {
namespace passes {

bool MatMulFusionPass::Run(ir::Graph& graph) {
    bool changed = false;
    for (ir::Node* node : graph.nodes()) {
        if (!node) continue;
        if (node->kind != ir::OpKind::MatMul) continue;
        auto it = node->metadata.find("role");
        if (it == node->metadata.end()) continue;
        const std::string& role = it->second;
        if (role == "attn_out" || role == "ffn_down") {
            node->kind = ir::OpKind::Linear;
            if (!node->metadata.count("fused_bias")) {
                node->metadata["fused_bias"] = node->metadata.count("bias") ? "true" : "false";
            }
            changed = true;
        } else if (role == "attn_q" || role == "attn_k" || role == "attn_v") {
            node->metadata["fused_into"] = "attention";
            changed = true;
        }
    }
    return changed;
}

} // namespace passes
} // namespace mlc
