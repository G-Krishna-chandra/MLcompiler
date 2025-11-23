#include "passes/layout_normalization_pass.hpp"

#include <unordered_map>

namespace mlc {
namespace passes {

bool LayoutNormalizationPass::Run(ir::Graph& graph) {
    std::unordered_map<ir::Tensor*, ir::Tensor*> normalized;
    bool changed = false;
    for (ir::Tensor* tensor : graph.tensors()) {
        if (!tensor) continue;
        auto it = tensor->metadata.find("needs_transpose");
        if (it == tensor->metadata.end() || it->second != "true") continue;
        if (tensor->shape.size() != 2) continue;
        std::vector<int64_t> swapped = {tensor->shape[1], tensor->shape[0]};
        std::string norm_name = tensor->name + ".normalized";
        ir::Tensor* normalized_tensor = graph.addTensor(norm_name, swapped, tensor->dtype);
        ir::Node* transpose = graph.addNode(ir::OpKind::Transpose, tensor->name + ".transpose");
        transpose->tensor_inputs.push_back(tensor);
        transpose->outputs.push_back(normalized_tensor);
        normalized[tensor] = normalized_tensor;
        tensor->metadata["layout_transposed"] = "true";
        changed = true;
    }

    if (!changed) return false;

    for (ir::Node* node : graph.nodes()) {
        if (!node) continue;
        for (ir::Tensor*& param : node->tensor_inputs) {
            if (!param) continue;
            auto it = normalized.find(param);
            if (it == normalized.end()) continue;
            param = it->second;
            node->metadata["normalized_inputs"] = "true";
        }
    }
    return true;
}

} // namespace passes
} // namespace mlc
