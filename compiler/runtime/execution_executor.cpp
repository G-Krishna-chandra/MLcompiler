#include "runtime/execution_executor.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "runtime/kernel_registry.hpp"

namespace mlc {
namespace runtime {

namespace {
size_t elementCount(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 0;
    size_t total = 1;
    for (int64_t dim : shape) {
        total *= static_cast<size_t>(std::max<int64_t>(1, dim));
    }
    return total;
}
}

ExecutionExecutor::ExecutionExecutor(const ExecutionGraph& graph,
                                     const BackendRegistry* registry,
                                     ExecutionContext* context)
    : graph_(graph),
      registry_(registry ? registry : &BackendRegistry::Default()),
      context_(context) {
    if (context_) {
        context_->setGraph(&graph_);
        for (const auto& [name, tensor] : graph_.tensors()) {
            if (!tensor.is_state) continue;
            context_->ensureStateTensor(tensor);
        }
    }
}

ExecutionExecutor::Result ExecutionExecutor::run(size_t max_nodes) const {
    Result result;
    auto order = graph_.topologicalOrder();
    if (max_nodes > 0 && max_nodes < order.size()) {
        order.resize(max_nodes);
    }

    std::unordered_map<std::string, const ExecutionNode*> node_lookup;
    for (const auto& node : graph_.nodes()) {
        node_lookup[node.name] = &node;
    }

    std::unordered_map<std::string, int> producer_count;
    std::unordered_set<std::string> available;
    for (const auto& [name, tensor] : graph_.tensors()) {
        producer_count[name] = 0;
        if (tensor.is_state) {
            available.insert(name); // states can be treated as persistent inputs
        }
    }
    for (const auto& node : graph_.nodes()) {
        for (const auto& output : node.outputs) {
            producer_count[output]++;
        }
    }
    for (const auto& [name, count] : producer_count) {
        if (count == 0) {
            available.insert(name);
        }
    }
    if (context_) {
        for (const auto& name : context_->tensorNames()) {
            available.insert(name);
        }
    }

    for (const auto& node_name : order) {
        auto it = node_lookup.find(node_name);
        if (it == node_lookup.end()) continue;
        const ExecutionNode* node = it->second;
        ExecutionTraceEntry entry;
        entry.node = node->name;
        entry.op = node->op;
        entry.backend = node->backend;

        for (const auto& input : node->inputs) {
            if (input.empty()) continue;
            if (!available.count(input)) {
                entry.success = false;
                entry.missing_inputs.push_back(input);
            }
        }

        if (!entry.success) {
            result.success = false;
        }

        const KernelDescriptor* descriptor = nullptr;
        if (!node->kernel_id.empty()) {
            descriptor = KernelDescriptorRegistry::Instance().findById(node->kernel_id);
        }
        const auto& backend = registry_->backendFor(node->backend);
        auto backend_result = backend.execute(*node, context_, descriptor);
        if (!backend_result.message.empty()) {
            entry.notes.push_back(backend_result.message);
        }
        if (!backend_result.kernel_id.empty()) {
            entry.notes.push_back("kernel=" + backend_result.kernel_id);
        }
        entry.success &= backend_result.success;

        for (const auto& output : node->outputs) {
            available.insert(output);
            auto tensor_it = graph_.tensors().find(output);
            if (tensor_it != graph_.tensors().end() && tensor_it->second.is_state) {
                std::ostringstream oss;
                oss << "updates state tensor '" << output << "'";
                entry.notes.push_back(oss.str());
            }
        }

        if (node->annotations.count("kv_cache_k") || node->annotations.count("kv_cache_v")) {
            std::ostringstream oss;
            oss << "kv-cache";
            if (node->annotations.count("kv_cache_k")) {
                oss << " K=" << node->annotations.at("kv_cache_k");
            }
            if (node->annotations.count("kv_cache_v")) {
                oss << " V=" << node->annotations.at("kv_cache_v");
            }
            entry.notes.push_back(oss.str());
        }
        entry.notes.push_back("backend=" + toString(node->backend));

        result.trace.push_back(entry);
    }

    result.executed_nodes = result.trace.size();
    return result;
}

std::string formatTraceEntry(const ExecutionTraceEntry& entry) {
    std::ostringstream oss;
    oss << entry.node << " (" << toString(entry.op) << ")";
    if (!entry.success) {
        oss << " [missing inputs: ";
        for (size_t i = 0; i < entry.missing_inputs.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << entry.missing_inputs[i];
        }
        oss << "]";
    }
    if (!entry.notes.empty()) {
        oss << " - ";
        for (size_t i = 0; i < entry.notes.size(); ++i) {
            if (i > 0) oss << "; ";
            oss << entry.notes[i];
        }
    }
    return oss.str();
}

} // namespace runtime
} // namespace mlc
