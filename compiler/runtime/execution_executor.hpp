#pragma once

#include "runtime/execution_graph.hpp"
#include "runtime/operator_backend.hpp"
#include "runtime/execution_context.hpp"

#include <string>
#include <unordered_map>
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

    // Phase F2 (continuous batching v2) — chunked execution. Run nodes in
    // topological-order range [start_node, start_node + node_count). When
    // node_count == 0, run to the end. start_node == 0 + node_count == 0 is
    // equivalent to run(0). Used by BatchedExecutor to interleave per-
    // request graph walks with batched paged-flash attention dispatches at
    // the right node boundaries.
    Result run_range(size_t start_node, size_t node_count) const;

    // Topological order accessor for callers (e.g. BatchedExecutor) that
    // need to identify node indices for chunked execution. Returns the same
    // order run() / run_range() iterate.
    std::vector<std::string> topologicalOrder() const { return graph_.topologicalOrder(); }

    // Coarse per-op-type profiling. Accumulates total ms and call count across
    // every backend.execute() invocation while MLC_PROFILE_NODES is set in the
    // environment. Static so callers don't need a back-channel to the executor
    // instance. Use clearNodeProfile() at the start of a measurement window and
    // dumpNodeProfile() to read it out.
    struct OpProfileEntry {
        double total_ms = 0.0;
        size_t calls = 0;
    };
    static const std::unordered_map<ExecOpType, OpProfileEntry>& nodeProfile();
    static void clearNodeProfile();

private:
    const ExecutionGraph& graph_;
    const BackendRegistry* registry_;
    ExecutionContext* context_;
};

std::string formatTraceEntry(const ExecutionTraceEntry& entry);

} // namespace runtime
} // namespace mlc
