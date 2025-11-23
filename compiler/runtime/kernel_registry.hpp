#pragma once

#include "runtime/execution_graph.hpp"

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mlc {
namespace runtime {

struct KernelDescriptor {
    std::string id;
    ExecOpType op = ExecOpType::Unknown;
    BackendKind backend = BackendKind::Auto;
    ir::DataType activation_dtype = ir::DataType::F32;
    ir::DataType weight_dtype = ir::DataType::F32;
    bool supports_quant_input = false;
    bool supports_quant_weight = false;
    bool supports_grouped_query = false;
    size_t tile_m = 0;
    size_t tile_n = 0;
    size_t tile_k = 0;
    size_t min_batch = 1;
    size_t min_head_dim = 1;
};

struct KernelSelectionQuery {
    ExecOpType op = ExecOpType::Unknown;
    BackendKind preferred_backend = BackendKind::Auto;
    ir::DataType activation_dtype = ir::DataType::F32;
    ir::DataType weight_dtype = ir::DataType::F32;
    bool activation_quantized = false;
    bool weight_quantized = false;
    size_t batch = 1;
    size_t context = 1;
    size_t head_count = 1;
    size_t kv_head_count = 1;
    size_t head_dim = 1;
    bool grouped_query = false;
};

class KernelDescriptorRegistry {
public:
    static const KernelDescriptorRegistry& Instance();

    const KernelDescriptor* select(const KernelSelectionQuery& query) const;
    const KernelDescriptor* findById(const std::string& id) const;

private:
    KernelDescriptorRegistry();

    std::vector<KernelDescriptor> kernels_;
    std::unordered_map<std::string, size_t> index_by_id_;
};

} // namespace runtime
} // namespace mlc
