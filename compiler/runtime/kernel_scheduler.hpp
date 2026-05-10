#pragma once

#include "frontends/gguf_loader.hpp"
#include "ir/ir.hpp"
#include "runtime/execution_graph.hpp"

namespace mlc {
namespace runtime {

class KernelScheduler {
public:
    // Build a model configuration from GGUF metadata without scheduling.
    static ModelConfig BuildModelConfig(const frontend::GGUFLoader& loader);

    static ExecutionGraph Schedule(const ir::Graph& graph,
                                   const frontend::GGUFLoader& loader);
};

} // namespace runtime
} // namespace mlc
