#pragma once

#include "runtime/session.hpp"
#include "runtime/metal_runtime.hpp"
#include "runtime/execution_graph.hpp"
#include "runtime/tensor_storage.hpp"
#include "runtime/quantization.hpp"

#include <string>
#include <unordered_map>
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

    std::vector<float>& allocateTensor(const std::string& name,
                                       size_t elements,
                                       bool zero_initialize = true);

    const ExecutionTensor* tensorInfo(const std::string& name) const;
    TensorStorage* tensorStorage(const std::string& name);
    const TensorStorage* tensorStorage(const std::string& name) const;
    void ensureStateTensor(const ExecutionTensor& tensor_info);

    const std::vector<float>& getParameter(const std::string& tensor_name) const;

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
#if defined(__APPLE__)
    std::unordered_map<std::string, MetalBufferHandle> metal_buffers_;
#endif
};

} // namespace runtime
} // namespace mlc
