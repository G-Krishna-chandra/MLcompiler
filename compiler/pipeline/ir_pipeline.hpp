#pragma once

#include <memory>

#include "frontends/gguf_loader.hpp"
#include "ir/ir.hpp"
#include "runtime/execution_graph.hpp"

namespace mlc {
namespace pipeline {

struct PipelineResult {
    std::unique_ptr<ir::Graph> ir_graph;
    runtime::ExecutionGraph exec_graph;
};

class IRPipeline {
public:
    PipelineResult Run(const frontend::GGUFLoader& loader);
};

} // namespace pipeline
} // namespace mlc
