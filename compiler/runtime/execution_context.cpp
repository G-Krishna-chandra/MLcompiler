#include "runtime/execution_context.hpp"
#include "frontends/ggml_types.hpp"
#include "runtime/quant_utils.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace mlc {
namespace runtime {

ExecutionContext::ExecutionContext(const Session& session,
                                   const ExecutionGraph* graph)
    : session_(session), graph_(graph) {
}

void ExecutionContext::setToken(uint64_t token_id) {
    token_id_ = token_id;
    tokens_.assign(1, token_id);
    seq_len_ = 1;
}

void ExecutionContext::setTensor(const std::string& name,
                                 std::vector<float> values) {
    tensors_[name] = TensorStorage::FromFloatVector(std::move(values));
#if defined(__APPLE__)
    invalidateMetalBuffer(name);
#endif
}

bool ExecutionContext::hasTensor(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

const std::vector<float>* ExecutionContext::getTensor(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return nullptr;
    return it->second.tryFloatData();
}

std::vector<float>* ExecutionContext::mutableTensor(const std::string& name) {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return nullptr;
#if defined(__APPLE__)
    invalidateMetalBuffer(name);
#endif
    return it->second.tryMutableFloatData();
}

std::vector<std::string> ExecutionContext::tensorNames() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const auto& kv : tensors_) {
        names.push_back(kv.first);
    }
    return names;
}

void ExecutionContext::registerTap(const std::string& tensor_name) {
    if (tensor_name.empty()) return;
    tap_names_.insert(tensor_name);
}

void ExecutionContext::clearTaps() {
    tap_names_.clear();
    tap_data_.clear();
}

bool ExecutionContext::isTapped(const std::string& tensor_name) const {
    return tap_names_.count(tensor_name) > 0;
}

void ExecutionContext::recordDispatch(const std::string& node_name, BackendKind backend) {
    if (node_name.empty()) return;
    dispatch_trace_[node_name] = backend;
}

void ExecutionContext::captureTapIfRegistered(const std::string& tensor_name) {
    if (tap_names_.empty()) return;
    if (!tap_names_.count(tensor_name)) return;
    auto it = tensors_.find(tensor_name);
    if (it == tensors_.end()) return;
    const std::vector<float>* data = it->second.tryFloatData();
    if (!data) return;
    tap_data_[tensor_name] = *data;
}

std::vector<float>& ExecutionContext::allocateTensor(const std::string& name,
                                                     size_t elements,
                                                     bool zero_initialize) {
    std::vector<float> values(elements);
    if (!zero_initialize) {
        // leave uninitialized but still reserve size
    }
    tensors_[name] = TensorStorage::FromFloatVector(std::move(values));
#if defined(__APPLE__)
    invalidateMetalBuffer(name);
#endif
    return tensors_[name].float_data;
}

const ExecutionTensor* ExecutionContext::tensorInfo(const std::string& name) const {
    if (!graph_) return nullptr;
    const auto& tensors = graph_->tensors();
    auto it = tensors.find(name);
    if (it == tensors.end()) return nullptr;
    return &it->second;
}

TensorStorage* ExecutionContext::tensorStorage(const std::string& name) {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return nullptr;
    return &it->second;
}

const TensorStorage* ExecutionContext::tensorStorage(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return nullptr;
    return &it->second;
}

size_t ExecutionContext::tensorElementCount(const ExecutionTensor& tensor) const {
    if (tensor.shape.empty()) return 0;
    size_t total = 1;
    for (int64_t dim : tensor.shape) {
        total *= static_cast<size_t>(std::max<int64_t>(1, dim));
    }
    return total;
}

void ExecutionContext::ensureStateTensor(const ExecutionTensor& tensor_info) {
    auto it = tensors_.find(tensor_info.name);
    if (it != tensors_.end()) return;
    uint32_t dtype = tensor_info.has_ggml_dtype ? tensor_info.ggml_dtype : frontend::GGML_TYPE_F32;
    uint32_t quant_version = tensor_info.quant_version;
    if (quant_version == 0) quant_version = 1;
    size_t head_dim = 1;
    if (!tensor_info.shape.empty()) {
        head_dim = static_cast<size_t>(std::max<int64_t>(1, tensor_info.shape.back()));
    }
    size_t elements = tensorElementCount(tensor_info);
    size_t rows = head_dim > 0 ? elements / head_dim : 0;
    TensorStorage storage;
    if (dtype == frontend::GGML_TYPE_F32) {
        storage = TensorStorage::FromFloatVector(std::vector<float>(rows * head_dim, 0.0f));
        storage.row_stride_bytes = head_dim * sizeof(float);
    } else {
        size_t row_bytes = ggmlRowSizeBytes(dtype, head_dim, quant_version);
        storage.dtype = dtype;
        storage.quant_version = quant_version;
        storage.row_stride_bytes = row_bytes;
        storage.raw_data.assign(rows * row_bytes, 0);
    }
    tensors_[tensor_info.name] = std::move(storage);
}

void ExecutionContext::clearStateTensors() {
    if (!graph_) return;
    for (const auto& kv : graph_->tensors()) {
        const ExecutionTensor& info = kv.second;
        if (!info.is_state) continue;
        ensureStateTensor(info);
        auto* storage = tensorStorage(info.name);
        if (!storage) continue;
        if (!storage->float_data.empty()) {
            std::fill(storage->float_data.begin(), storage->float_data.end(), 0.0f);
        } else if (!storage->raw_data.empty()) {
            std::fill(storage->raw_data.begin(), storage->raw_data.end(), 0);
        } else if (storage->row_stride_bytes > 0) {
            size_t bytes = tensorElementCount(info) * storage->row_stride_bytes;
            storage->raw_data.assign(bytes, 0);
        }
#if defined(__APPLE__)
        invalidateMetalBuffer(info.name);
#endif
    }
}

const std::vector<float>& ExecutionContext::getParameter(const std::string& tensor_name) const {
    auto it = parameter_cache_.find(tensor_name);
    if (it != parameter_cache_.end()) return it->second.float_data;

    const auto& tensors = session_.loader().tensors();
    auto tensor_it = tensors.find(tensor_name);
    if (tensor_it == tensors.end()) {
        throw std::runtime_error("Parameter tensor '" + tensor_name + "' not found");
    }
    if (tensor_it->second.dtype != frontend::GGML_TYPE_F32) {
        throw std::runtime_error("Parameter tensor '" + tensor_name + "' must be F32 for direct access");
    }
    const auto& raw = session_.tensorData(tensor_it->second);
    if (raw.empty()) {
        return parameter_cache_.emplace(tensor_name,
                                        TensorStorage::FromFloatVector(std::vector<float>{}))
            .first->second.float_data;
    }
    size_t count = raw.size() / sizeof(float);
    std::vector<float> values(count);
    std::memcpy(values.data(), raw.data(), raw.size());
    auto inserted =
        parameter_cache_.emplace(tensor_name, TensorStorage::FromFloatVector(std::move(values)));
    return inserted.first->second.float_data;
}

#if defined(__APPLE__)
MetalBufferHandle* ExecutionContext::ensureMetalBuffer(const std::string& name,
                                                       std::vector<float>& data) {
    auto& handle = metal_buffers_[name];
    if (!MetalExecutor::Instance().ensureSharedBuffer(data, handle)) {
        MetalExecutor::Instance().releaseBuffer(handle);
        metal_buffers_.erase(name);
        return nullptr;
    }
    return &handle;
}

void ExecutionContext::invalidateMetalBuffer(const std::string& name) {
    auto it = metal_buffers_.find(name);
    if (it != metal_buffers_.end()) {
        MetalExecutor::Instance().releaseBuffer(it->second);
        metal_buffers_.erase(it);
    }
}

void ExecutionContext::markMetalBufferModified(const std::string& name,
                                               size_t element_offset,
                                               size_t element_count) {
    auto it = metal_buffers_.find(name);
    if (it == metal_buffers_.end()) return;
    size_t offset_bytes = element_offset * sizeof(float);
    size_t length_bytes = element_count * sizeof(float);
    MetalExecutor::Instance().markHostModified(it->second, offset_bytes, length_bytes);
}

MetalBufferHandle* ExecutionContext::metalBuffer(const std::string& name) {
    auto it = metal_buffers_.find(name);
    if (it == metal_buffers_.end()) return nullptr;
    return &it->second;
}
#endif

} // namespace runtime
} // namespace mlc
