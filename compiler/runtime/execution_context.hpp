#pragma once

#include "runtime/session.hpp"
#include "runtime/metal_runtime.hpp"
#include "runtime/execution_graph.hpp"
#include "runtime/tensor_storage.hpp"
#include "runtime/quantization.hpp"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mlc {
namespace runtime {

class ExecutionContext {
public:
    explicit ExecutionContext(const Session& session,
                              const ExecutionGraph* graph = nullptr);

    const Session& session() const { return session_; }
    void setGraph(const ExecutionGraph* graph) { graph_ = graph; }
    const ExecutionGraph* graph() const { return graph_; }

    void setToken(uint64_t token_id);
    uint64_t token() const { return token_id_; }

    void setSequencePosition(size_t position) { sequence_position_ = position; }
    size_t sequencePosition() const { return sequence_position_; }

    void setTensor(const std::string& name, std::vector<float> values);
    bool hasTensor(const std::string& name) const;
    const std::vector<float>* getTensor(const std::string& name) const;
    std::vector<float>* mutableTensor(const std::string& name);
    std::vector<std::string> tensorNames() const;

    // Utilities to reason about tensor storage/shape.
    size_t tensorElementCount(const ExecutionTensor& tensor) const;
    void clearStateTensors();

    std::vector<float>& allocateTensor(const std::string& name,
                                       size_t elements,
                                       bool zero_initialize = true);

    const ExecutionTensor* tensorInfo(const std::string& name) const;
    TensorStorage* tensorStorage(const std::string& name);
    const TensorStorage* tensorStorage(const std::string& name) const;
    void ensureStateTensor(const ExecutionTensor& tensor_info);

    const std::vector<float>& getParameter(const std::string& tensor_name) const;

    // Tap mechanism: register tensor names to capture into host buffers as the
    // executor produces them. The latest value of each tapped tensor is kept;
    // when the executor runs prefill multiple times, the tap reflects the most
    // recent step. tapsEmpty() lets callers (the executor) skip work fast when
    // no taps are registered.
    void registerTap(const std::string& tensor_name);
    void clearTaps();
    bool tapsEmpty() const { return tap_names_.empty(); }
    bool isTapped(const std::string& tensor_name) const;
    // Called by the executor after a node produces an output. No-op if not tapped.
    void captureTapIfRegistered(const std::string& tensor_name);
    const std::unordered_map<std::string, std::vector<float>>& tapData() const { return tap_data_; }

#if defined(__APPLE__)
    MetalBufferHandle* ensureMetalBuffer(const std::string& name,
                                         std::vector<float>& data);
    void invalidateMetalBuffer(const std::string& name);
    void markMetalBufferModified(const std::string& name,
                                 size_t element_offset,
                                 size_t element_count);
    MetalBufferHandle* metalBuffer(const std::string& name);
#endif

private:
    const Session& session_;
    const ExecutionGraph* graph_ = nullptr;
    uint64_t token_id_ = 0;
    size_t sequence_position_ = 0;
    std::unordered_map<std::string, TensorStorage> tensors_;
    mutable std::unordered_map<std::string, TensorStorage> parameter_cache_;
    std::unordered_set<std::string> tap_names_;
    std::unordered_map<std::string, std::vector<float>> tap_data_;
#if defined(__APPLE__)
    std::unordered_map<std::string, MetalBufferHandle> metal_buffers_;
#endif
};

} // namespace runtime
} // namespace mlc
