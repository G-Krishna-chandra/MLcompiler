#include "passes.hpp"

#include <cstdio>

namespace mlc {
namespace passes {

void PassManager::addPass(std::unique_ptr<Pass> pass) {
    if (pass) {
        passes_.push_back(std::move(pass));
    }
}

size_t PassManager::run(ir::Graph& graph) {
    size_t count = 0;
    for (auto& pass : passes_) {
        if (!pass) continue;
        bool changed = pass->Run(graph);
        if (std::getenv("MLC_VERBOSE")) {
            std::fprintf(stderr, "[PassManager] ran %s -> %s\n",
                         pass->name(),
                         changed ? "changed" : "no-change");
        }
        if (changed) ++count;
    }
    return count;
}

} // namespace passes
} // namespace mlc

