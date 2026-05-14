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
    // Backend that actually executed the kernel. May differ from
    // node.backend or from the receiving ExecutionBackend subclass when an
    // op falls back (e.g. Metal kernel unavailable for a shape, or
    // shouldUseFor() returned false). Used by the parity harness to verify
    // that force-CPU runs really ran on CPU.
    BackendKind actual_backend = BackendKind::CPU;
    std::string message;
    std::string kernel_id;
};

class ExecutionBackend {
public:
    virtual ~ExecutionBackend() = default;
    virtual BackendExecutionResult execute(const ExecutionNode& node,
                                           ExecutionContext* context,
                                           const KernelDescriptor* descriptor) const = 0;
    // Deferred-commit dispatch. A backend that supports fusion encodes work
    // onto MetalExecutor's open fusion command buffer and does NOT commit;
    // the executor drives the commit at window-flush time. The default impl
    // falls back to synchronous execute(); only MetalExecutionBackend
    // overrides for ops it has an encodeX entry point for. Callers that want
    // the deferred semantics must check the returned actual_backend /
    // success and treat a non-overridden default as "not fusable, fall back
    // to execute() after a window flush."
    virtual BackendExecutionResult encode(const ExecutionNode& node,
                                          ExecutionContext* context,
                                          const KernelDescriptor* descriptor) const {
        return execute(node, context, descriptor);
    }
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
    BackendExecutionResult encode(const ExecutionNode& node,
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
