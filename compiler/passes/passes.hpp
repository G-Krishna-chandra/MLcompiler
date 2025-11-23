#pragma once

#pragma once

#include "ir/ir.hpp"

#include <memory>
#include <string>
#include <vector>

namespace mlc {
namespace passes {

class Pass {
public:
    virtual ~Pass() = default;
    virtual const char* name() const = 0;
    virtual bool Run(ir::Graph& graph) = 0;
};

class PassManager {
public:
    PassManager() = default;
    ~PassManager() = default;

    void addPass(std::unique_ptr<Pass> pass);
    size_t run(ir::Graph& graph);
    const std::vector<std::unique_ptr<Pass>>& passes() const { return passes_; }

private:
    std::vector<std::unique_ptr<Pass>> passes_;
};

} // namespace passes
} // namespace mlc

