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
    // CB-batching (B4): a non-fusable Metal op that encoded onto the open
    // forward-pass CB and deferred its result download writes its (fp32)
    // result buffer pointer + element count here. The executor inserts
    // these into pass_outputs so subsequent fusable FromBuffer encodes
    // chain through GPU memory instead of via the empty host slot. nullptr
    // / 0 means the result is already on host (synchronous mode).
    void* gpu_output_buffer = nullptr;       // id<MTLBuffer>, opaque
    size_t gpu_output_element_count = 0;
};

// Bundle of buffer-residency state the executor passes into encode() so the
// backend can route to FromHost vs FromBuffer kernel variants without
// having to query the executor's window map directly. Defaults describe
// "no GPU-resident inputs, drain output to host on flush" — i.e. equivalent
// to the previous single-variant encode() behavior.
struct FusionInputs {
    void* primary_input_buffer = nullptr;    // id<MTLBuffer> if non-null
    size_t primary_input_count = 0;          // valid when primary_input_buffer != null
    void* secondary_input_buffer = nullptr;  // Add only
    size_t secondary_input_count = 0;
    void* output_buffer = nullptr;           // pool-checked-out; always provided in fusion mode
    std::vector<float>* host_dst = nullptr;  // stable pointer from allocateTensor; drain target
    bool needs_host_output = true;
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
                                          const KernelDescriptor* descriptor,
                                          const FusionInputs& fusion) const {
        (void)fusion;
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
                                  const KernelDescriptor* descriptor,
                                  const FusionInputs& fusion) const override;
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
