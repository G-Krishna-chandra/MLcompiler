#include "runtime/kernel_registry.hpp"

#include <limits>

namespace mlc {
namespace runtime {

namespace {

int scoreKernel(const KernelDescriptor& kernel,
                const KernelSelectionQuery& query) {
    int score = 0;
    if (kernel.backend == query.preferred_backend) {
        score += 40;
    } else if (query.preferred_backend == BackendKind::Auto) {
        score += 10;
    } else if (kernel.backend == BackendKind::Auto) {
        score += 5;
    } else {
        return std::numeric_limits<int>::min();
    }

    if (kernel.activation_dtype == query.activation_dtype) {
        score += 10;
    } else {
        return std::numeric_limits<int>::min();
    }

    if (query.weight_dtype == kernel.weight_dtype) {
        score += 5;
    }

    if (query.activation_quantized && !kernel.supports_quant_input) {
        return std::numeric_limits<int>::min();
    }
    if (query.weight_quantized && !kernel.supports_quant_weight) {
        return std::numeric_limits<int>::min();
    }

    if (query.grouped_query && !kernel.supports_grouped_query) {
        return std::numeric_limits<int>::min();
    }

    if (query.batch >= kernel.min_batch) {
        score += 5;
    }

    if (query.head_dim >= kernel.min_head_dim) {
        score += 2;
    }

    score += static_cast<int>((kernel.tile_m + kernel.tile_n + kernel.tile_k) / 32);
    return score;
}

} // namespace

KernelDescriptorRegistry::KernelDescriptorRegistry() {
    auto add_kernel = [&](KernelDescriptor desc) {
        index_by_id_[desc.id] = kernels_.size();
        kernels_.push_back(std::move(desc));
    };

    add_kernel({"cpu.embedding.f32.scalar",
                        ExecOpType::Embedding,
                        BackendKind::CPU,
                        ir::DataType::F32,
                        ir::DataType::F32,
                        false,
                        false,
                        false,
                        8,
                        8,
                        8,
                        1,
                        1});

    add_kernel({"metal.embedding.f32.tile64",
                        ExecOpType::Embedding,
                        BackendKind::Metal,
                        ir::DataType::F32,
                        ir::DataType::F32,
                        false,
                        true,
                        false,
                        64,
                        64,
                        32,
                        1,
                        1});

    add_kernel({"cpu.attention.f32",
                        ExecOpType::Attention,
                        BackendKind::CPU,
                        ir::DataType::F32,
                        ir::DataType::F32,
                        false,
                        false,
                        true,
                        32,
                        32,
                        32,
                        1,
                        32});

    add_kernel({"metal.attention.f32.batch",
                        ExecOpType::Attention,
                        BackendKind::Metal,
                        ir::DataType::F32,
                        ir::DataType::F32,
                        false,
                        true,
                        true,
                        128,
                        128,
                        64,
                        1,
                        32});

    add_kernel({"metal.attention.quant.qkv",
                        ExecOpType::Attention,
                        BackendKind::Metal,
                        ir::DataType::F32,
                        ir::DataType::I8,
                        true,
                        true,
                        true,
                        128,
                        128,
                        64,
                        1,
                        32});

    add_kernel({"cpu.matmul.f32",
                        ExecOpType::MatMul,
                        BackendKind::CPU,
                        ir::DataType::F32,
                        ir::DataType::F32,
                        false,
                        false,
                        false,
                        64,
                        64,
                        32,
                        1,
                        1});

    add_kernel({"cpu.matmul.quant",
                        ExecOpType::MatMul,
                        BackendKind::CPU,
                        ir::DataType::F32,
                        ir::DataType::I8,
                        false,
                        true,
                        false,
                        64,
                        64,
                        32,
                        1,
                        1});

    add_kernel({"metal.matmul.f32",
                        ExecOpType::MatMul,
                        BackendKind::Metal,
                        ir::DataType::F32,
                        ir::DataType::F32,
                        false,
                        false,
                        false,
                        128,
                        128,
                        64,
                        1,
                        1});

    add_kernel({"metal.matmul.quant",
                        ExecOpType::MatMul,
                        BackendKind::Metal,
                        ir::DataType::F32,
                        ir::DataType::I8,
                        false,
                        true,
                        false,
                        128,
                        128,
                        64,
                        1,
                        1});

    add_kernel({"metal.ffn.f32",
                        ExecOpType::FeedForward,
                        BackendKind::Metal,
                        ir::DataType::F32,
                        ir::DataType::F32,
                        false,
                        false,
                        false,
                        64,
                        64,
                        32,
                        1,
                        1});

    add_kernel({"cpu.norm.f32",
                        ExecOpType::Norm,
                        BackendKind::CPU,
                        ir::DataType::F32,
                        ir::DataType::F32,
                        false,
                        false,
                        false,
                        32,
                        32,
                        32,
                        1,
                        1});
}

const KernelDescriptorRegistry& KernelDescriptorRegistry::Instance() {
    static KernelDescriptorRegistry registry;
    return registry;
}

const KernelDescriptor* KernelDescriptorRegistry::select(const KernelSelectionQuery& query) const {
    const KernelDescriptor* best = nullptr;
    int best_score = std::numeric_limits<int>::min();
    for (const auto& kernel : kernels_) {
        if (kernel.op != query.op) continue;
        int score = scoreKernel(kernel, query);
        if (score > best_score) {
            best_score = score;
            best = &kernel;
        }
    }
    return best;
}

const KernelDescriptor* KernelDescriptorRegistry::findById(const std::string& id) const {
    if (id.empty()) return nullptr;
    auto it = index_by_id_.find(id);
    if (it == index_by_id_.end()) return nullptr;
    return &kernels_[it->second];
}

} // namespace runtime
} // namespace mlc
