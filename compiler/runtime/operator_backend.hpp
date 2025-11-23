#pragma once

#include "runtime/execution_graph.hpp"
#include "runtime/execution_context.hpp"
#include "runtime/kernel_registry.hpp"

#include <memory>
#include <string>

namespace mlc {
namespace runtime {

struct BackendExecutionResult {
    bool success = true;
    std::string message;
    std::string kernel_id;
};

class ExecutionBackend {
public:
    virtual ~ExecutionBackend() = default;
    virtual BackendExecutionResult execute(const ExecutionNode& node,
                                           ExecutionContext* context,
                                           const KernelDescriptor* descriptor) const = 0;
};

class CpuExecutionBackend : public ExecutionBackend {
public:
    BackendExecutionResult execute(const ExecutionNode& node,
                                   ExecutionContext* context,
                                   const KernelDescriptor* descriptor) const override;
};

class MetalExecutionBackend : public ExecutionBackend {
public:
    BackendExecutionResult execute(const ExecutionNode& node,
                                   ExecutionContext* context,
                                   const KernelDescriptor* descriptor) const override;
};

class BackendRegistry {
public:
    BackendRegistry();

    const ExecutionBackend& backendFor(BackendKind kind) const;

    static const BackendRegistry& Default();

private:
    CpuExecutionBackend cpu_backend_;
    MetalExecutionBackend metal_backend_;
};

} // namespace runtime
} // namespace mlc
