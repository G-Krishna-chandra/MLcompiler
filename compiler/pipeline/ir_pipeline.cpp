#include "pipeline/ir_pipeline.hpp"

#include "ir/ir_builder.hpp"
#include "passes/layout_normalization_pass.hpp"
#include "passes/matmul_fusion_pass.hpp"
#include "passes/passes.hpp"
#include "passes/tiling_pass.hpp"
#include "runtime/kernel_scheduler.hpp"

namespace mlc {
namespace pipeline {

PipelineResult IRPipeline::Run(const frontend::GGUFLoader& loader) {
    PipelineResult result;
    result.ir_graph = ir::IRBuilder::BuildFromLoader(loader);

    passes::PassManager manager;
    manager.addPass(std::make_unique<passes::LayoutNormalizationPass>());
    manager.addPass(std::make_unique<passes::MatMulFusionPass>());
    manager.addPass(std::make_unique<passes::TilingPass>());
    manager.run(*result.ir_graph);

    result.exec_graph = runtime::KernelScheduler::Schedule(*result.ir_graph, loader);
    return result;
}

} // namespace pipeline
} // namespace mlc
