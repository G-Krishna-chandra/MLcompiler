#pragma once

#include <vector>

#include "runtime/execution_graph.hpp"
#include "runtime/execution_context.hpp"

namespace mlc {
namespace runtime {

struct BackendExecutionResult;

BackendExecutionResult RunAttentionCPU(const ExecutionNode& node,
                                       ExecutionContext* context,
                                       const std::vector<float>& q,
                                       const std::vector<float>& k_new,
                                       const std::vector<float>& v_new);

} // namespace runtime
} // namespace mlc
