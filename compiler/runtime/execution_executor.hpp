#pragma once

#include "runtime/execution_graph.hpp"
#include "runtime/operator_backend.hpp"
#include "runtime/execution_context.hpp"

#include <string>
#include <vector>

namespace mlc {
namespace runtime {

struct ExecutionTraceEntry {
    std::string node;
    ExecOpType op = ExecOpType::Unknown;
    BackendKind backend = BackendKind::Auto;
    bool success = true;
    std::vector<std::string> missing_inputs;
    std::vector<std::string> notes;
};

class ExecutionExecutor {
public:
    explicit ExecutionExecutor(const ExecutionGraph& graph,
                               const BackendRegistry* registry = nullptr,
                               ExecutionContext* context = nullptr);

    struct Result {
        std::vector<ExecutionTraceEntry> trace;
        size_t executed_nodes = 0;
        bool success = true;
    };

    Result run(size_t max_nodes = 0) const;

private:
    const ExecutionGraph& graph_;
    const BackendRegistry* registry_;
    ExecutionContext* context_;
};

std::string formatTraceEntry(const ExecutionTraceEntry& entry);

} // namespace runtime
} // namespace mlc
