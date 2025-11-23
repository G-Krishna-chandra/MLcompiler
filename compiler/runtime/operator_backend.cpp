#include "runtime/operator_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <vector>
#include "runtime/attention_cpu.hpp"
#include "runtime/metal_runtime.hpp"
#include "runtime/quant_utils.hpp"
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif
#include "runtime/quantization.hpp"
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

#include "frontends/ggml_types.hpp"

namespace mlc {
namespace runtime {

namespace {

std::string getAnnotation(const ExecutionNode& node, const std::string& key) {
    auto it = node.annotations.find(key);
    if (it == node.annotations.end()) return {};
    return it->second;
}

float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

#if defined(__APPLE__)
bool siluAccelerate(const std::vector<float>& input,
                    std::vector<float>& output) {
    size_t n = input.size();
    if (n == 0) return false;
    if (output.size() != n) output.resize(n);
    std::vector<float> tmp(n);
    float minus_one = -1.0f;
    vDSP_vsmul(input.data(), 1, &minus_one, tmp.data(), 1, n);
    int count = static_cast<int>(n);
    vvexpf(tmp.data(), tmp.data(), &count);
    float one = 1.0f;
    vDSP_vsadd(tmp.data(), 1, &one, tmp.data(), 1, n);
    vDSP_svdiv(&one, tmp.data(), 1, tmp.data(), 1, n);
    vDSP_vmul(tmp.data(), 1, input.data(), 1, output.data(), 1, n);
    return true;
}
#endif

void softmax(std::vector<float>& values) {
    if (values.empty()) return;
    float max_val = *std::max_element(values.begin(), values.end());
    float sum = 0.0f;
    for (float& v : values) {
        v = std::exp(v - max_val);
        sum += v;
    }
    if (sum <= 0.0f) sum = 1.0f;
    for (float& v : values) {
        v /= sum;
    }
}

void rmsNorm(const std::vector<float>& input,
             const std::vector<float>& weight,
             std::vector<float>& output,
             float epsilon = 1e-5f) {
    size_t n = input.size();
    if (output.size() != n) {
        output.assign(n, 0.0f);
    }
    float mean_sq = 0.0f;
    for (float v : input) {
        mean_sq += v * v;
    }
    mean_sq = mean_sq / static_cast<float>(std::max<size_t>(1, n));
    float inv = 1.0f / std::sqrt(mean_sq + epsilon);
    for (size_t i = 0; i < n; ++i) {
        float gamma = i < weight.size() ? weight[i] : 1.0f;
        output[i] = input[i] * inv * gamma;
    }
}

const std::vector<float>* fetchInput(const ExecutionContext& context,
                                     const ExecutionNode& node,
                                     size_t index,
                                     BackendExecutionResult& result) {
    if (index >= node.inputs.size()) {
        result.success = false;
        std::ostringstream oss;
        oss << "Node '" << node.name << "' missing input index " << index;
        result.message = oss.str();
        return nullptr;
    }
    const std::string& name = node.inputs[index];
    const std::vector<float>* tensor = context.getTensor(name);
    if (!tensor) {
        result.success = false;
        std::ostringstream oss;
        oss << "Input tensor '" << name << "' unavailable for node '" << node.name << "'";
        result.message = oss.str();
        return nullptr;
    }
    return tensor;
}

size_t tensorElementCount(const ExecutionTensor& tensor) {
    if (tensor.shape.empty()) return 0;
    size_t total = 1;
    for (int64_t dim : tensor.shape) {
        total *= static_cast<size_t>(std::max<int64_t>(1, dim));
    }
    return total;
}

size_t inferHeadDim(const ExecutionTensor* tensor, size_t fallback) {
    if (fallback > 0) return fallback;
    if (!tensor || tensor->shape.empty()) return 0;
    return static_cast<size_t>(std::max<int64_t>(1, tensor->shape.back()));
}

struct CacheDecodeResult {
    std::vector<float> buffer;
    std::vector<float>* data = nullptr;
    size_t head_dim = 0;
    bool owns_buffer = false;
};

bool decodeCacheTensor(const ExecutionTensor* info,
                       TensorStorage* storage,
                       size_t head_dim_attr,
                       CacheDecodeResult& result) {
    if (!info || !storage) return false;
    result.head_dim = inferHeadDim(info, head_dim_attr);
    if (result.head_dim == 0) return false;
    size_t elements = tensorElementCount(*info);
    if (elements == 0) return false;
    size_t rows = elements / result.head_dim;
    if (storage->dtype == frontend::GGML_TYPE_F32) {
        if (storage->float_data.size() != elements) {
            storage->float_data.resize(elements, 0.0f);
        }
        result.data = &storage->float_data;
        result.owns_buffer = false;
        return true;
    }
    size_t stride = storage->row_stride_bytes;
    if (stride == 0) {
        stride = ggmlRowSizeBytes(storage->dtype, result.head_dim, storage->quant_version);
    }
    size_t required = rows * stride;
    if (storage->raw_data.size() != required) {
        storage->raw_data.resize(required, 0);
    }
    result.buffer.resize(elements);
    const uint8_t* ptr = storage->raw_data.data();
    for (size_t row = 0; row < rows; ++row) {
        dequantizeRowTo(ptr + row * stride,
                        storage->dtype,
                        result.head_dim,
                        storage->quant_version,
                        result.buffer.data() + row * result.head_dim);
    }
    result.data = &result.buffer;
    result.owns_buffer = true;
    return true;
}

void encodeCacheTensor(const ExecutionTensor* info,
                       TensorStorage* storage,
                       const CacheDecodeResult& decoded) {
    if (!info || !storage) return;
    if (!decoded.owns_buffer || !decoded.data) return;
    if (storage->dtype == frontend::GGML_TYPE_F32) {
        storage->float_data = *decoded.data;
        storage->row_stride_bytes = decoded.head_dim * sizeof(float);
        return;
    }
    size_t elements = decoded.data->size();
    if (elements == 0 || decoded.head_dim == 0) return;
    size_t rows = elements / decoded.head_dim;
    size_t stride = storage->row_stride_bytes;
    if (stride == 0) {
        stride = ggmlRowSizeBytes(storage->dtype, decoded.head_dim, storage->quant_version);
    }
    storage->raw_data.resize(rows * stride);
    uint8_t* dst = storage->raw_data.data();
    for (size_t row = 0; row < rows; ++row) {
        quantizeRowFrom(decoded.data->data() + row * decoded.head_dim,
                        storage->dtype,
                        decoded.head_dim,
                        storage->quant_version,
                        dst + row * stride);
    }
}

} // namespace

BackendExecutionResult CpuExecutionBackend::execute(const ExecutionNode& node,
                                                    ExecutionContext* context,
                                                    const KernelDescriptor* descriptor) const {
    BackendExecutionResult result;
    if (descriptor) {
        result.kernel_id = descriptor->id;
    } else if (!node.kernel_id.empty()) {
        result.kernel_id = node.kernel_id;
    }
    struct Finalize {
        BackendExecutionResult& res;
        const KernelDescriptor* desc;
        ~Finalize() {
            if (desc && res.message.empty()) {
                res.message = "kernel=" + desc->id;
            }
        }
    } finalize{result, descriptor};
    if (!context) {
        std::ostringstream oss;
        oss << "CPU backend simulation for op " << toString(node.op);
        result.message = oss.str();
        return result;
    }

    switch (node.op) {
        case ExecOpType::Embedding: {
            std::string weight = getAnnotation(node, "weight");
            if (weight.empty()) {
                result.success = false;
                result.message = "Embedding node missing weight annotation";
                return result;
            }
            try {
                auto embedding = context->session().getEmbedding(weight, context->token());
                if (!node.outputs.empty()) {
                    context->setTensor(node.outputs[0], std::move(embedding));
                }
                result.message = "embedding lookup";
            } catch (const std::exception& e) {
                result.success = false;
                result.message = e.what();
            }
            return result;
        }
        case ExecOpType::MatMul:
        case ExecOpType::Linear: {
            const std::vector<float>* input = fetchInput(*context, node, 0, result);
            if (!result.success) return result;
            std::string weight = getAnnotation(node, "weight");
            if (weight.empty()) {
                result.success = false;
                result.message = "MatMul node missing weight annotation";
                return result;
            }
            try {
                auto output = context->session().runLinear(weight, *input);
                std::string bias_name = getAnnotation(node, "bias");
                if (!bias_name.empty()) {
                    const auto& bias = context->getParameter(bias_name);
                    if (bias.size() != output.size()) {
                        result.success = false;
                        std::ostringstream oss;
                        oss << "Bias tensor '" << bias_name << "' size mismatch";
                        result.message = oss.str();
                        return result;
                    }
                    for (size_t i = 0; i < output.size(); ++i) {
                        output[i] += bias[i];
                    }
                }
                if (!node.outputs.empty()) {
                    context->setTensor(node.outputs[0], std::move(output));
                }
                result.message = "matmul";
            } catch (const std::exception& e) {
                result.success = false;
                result.message = e.what();
            }
            return result;
        }
        case ExecOpType::Norm: {
            const std::vector<float>* input = fetchInput(*context, node, 0, result);
            if (!result.success) return result;
            std::string weight_name = getAnnotation(node, "weight");
            if (weight_name.empty()) {
                if (!node.outputs.empty()) context->setTensor(node.outputs[0], *input);
                result.message = "norm-pass";
                return result;
            }
            try {
                const auto& weight = context->getParameter(weight_name);
                std::vector<float> output(input->size(), 0.0f);
#if defined(__APPLE__)
                if (input->size() >= 32) {
                    float mean_sq = 0.0f;
                    vDSP_measqv(input->data(), 1, &mean_sq, input->size());
                    float inv = 1.0f / std::sqrt(mean_sq + 1e-5f);
                    vDSP_vsmul(input->data(), 1, &inv, output.data(), 1, input->size());
                    for (size_t i = 0; i < output.size(); ++i) {
                        float gamma = i < weight.size() ? weight[i] : 1.0f;
                        output[i] *= gamma;
                    }
                    if (!node.outputs.empty()) {
                        context->setTensor(node.outputs[0], std::move(output));
                    }
                    result.message = "rms-norm-accelerate";
                    return result;
                }
#endif
                rmsNorm(*input, weight, output);
                if (!node.outputs.empty()) {
                    context->setTensor(node.outputs[0], std::move(output));
                }
                result.message = "rms-norm";
            } catch (const std::exception& e) {
                result.success = false;
                result.message = e.what();
            }
            return result;
        }
        case ExecOpType::FeedForward: {
            if (node.inputs.size() < 2) {
                result.success = false;
                result.message = "FeedForward node missing gate/up inputs";
                return result;
            }
            const std::vector<float>* gate = fetchInput(*context, node, 0, result);
            if (!result.success) return result;
            const std::vector<float>* up = fetchInput(*context, node, 1, result);
            if (!result.success) return result;
            if (gate->size() != up->size()) {
                result.success = false;
                result.message = "FeedForward input size mismatch";
                return result;
            }
            std::vector<float> mix(gate->size(), 0.0f);
#if defined(__APPLE__)
            bool used_accelerate = false;
#endif
#if defined(__APPLE__)
            if (gate->size() >= 32) {
                std::vector<float> silu_values(gate->size());
                if (siluAccelerate(*gate, silu_values)) {
                    vDSP_vmul(silu_values.data(), 1, up->data(), 1, mix.data(), 1, gate->size());
                    used_accelerate = true;
                }
            }
            if (!used_accelerate) {
#endif
                for (size_t i = 0; i < mix.size(); ++i) {
                    mix[i] = silu((*gate)[i]) * (*up)[i];
                }
#if defined(__APPLE__)
            }
#endif
            if (!node.outputs.empty()) {
                context->setTensor(node.outputs[0], std::move(mix));
            }
            result.message =
#if defined(__APPLE__)
                (used_accelerate ? "ffn-gated-accelerate" : "ffn-gated");
#else
                "ffn-gated";
#endif
            return result;
        }
        case ExecOpType::Attention: {
            bool use_gpu = (node.backend == BackendKind::Metal && MetalExecutor::Instance().isAvailable());
            if (node.inputs.size() < 3) {
                result.success = false;
                result.message = "Attention node missing qkv inputs";
                return result;
            }
            const std::vector<float>* q = fetchInput(*context, node, 0, result);
            if (!result.success) return result;
            const std::vector<float>* k_new = fetchInput(*context, node, 1, result);
            if (!result.success) return result;
            const std::vector<float>* v_new = fetchInput(*context, node, 2, result);
            if (!result.success) return result;

            if (use_gpu && context) {
                std::string cache_k_name = getAnnotation(node, "kv_cache_k");
                std::string cache_v_name = getAnnotation(node, "kv_cache_v");
                TensorStorage* cache_k_storage = context->tensorStorage(cache_k_name);
                TensorStorage* cache_v_storage = context->tensorStorage(cache_v_name);
                const ExecutionTensor* cache_k_info = context->tensorInfo(cache_k_name);
                const ExecutionTensor* cache_v_info = context->tensorInfo(cache_v_name);
                CacheDecodeResult cache_k_decoded;
                CacheDecodeResult cache_v_decoded;
                size_t head_dim_attr = static_cast<size_t>(node.attributes.count("head_dim") ? node.attributes.at("head_dim") : 0.0f);
                if (!decodeCacheTensor(cache_k_info, cache_k_storage, head_dim_attr, cache_k_decoded) ||
                    !decodeCacheTensor(cache_v_info, cache_v_storage, head_dim_attr, cache_v_decoded)) {
                    use_gpu = false;
                } else {
                    size_t num_heads = static_cast<size_t>(node.attributes.count("heads") ? node.attributes.at("heads") : 0.0f);
                    size_t head_dim = static_cast<size_t>(node.attributes.count("head_dim") ? node.attributes.at("head_dim") : 0.0f);
                    size_t kv_heads = static_cast<size_t>(node.attributes.count("kv_heads") ? node.attributes.at("kv_heads") : num_heads);
                    const ExecutionTensor* cache_info = context->tensorInfo(cache_k_name);
                    size_t context_length = cache_info && cache_info->shape.size() >= 2
                                                ? static_cast<size_t>(cache_info->shape[1])
                                                : 0;
                    size_t position = context->sequencePosition();
                    size_t rotary_dim = 0;
                    float rope_freq_base = 10000.0f;
                    float rope_freq_scale = 1.0f;
                    if (context && context->graph()) {
                        const auto& cfg = context->graph()->modelConfig();
                        if (cfg.rotary_dim > 0 && head_dim > 0) {
                            rotary_dim = std::min(cfg.rotary_dim, head_dim);
                            if (cfg.rope_freq_base > 0.0f) {
                                rope_freq_base = cfg.rope_freq_base;
                            }
                            if (cfg.rope_freq_scale > 0.0f) {
                                rope_freq_scale = cfg.rope_freq_scale;
                            }
                        }
                    }
                    MetalBufferHandle* cache_k_handle = nullptr;
                    MetalBufferHandle* cache_v_handle = nullptr;
#if defined(__APPLE__)
                    if (!cache_k_decoded.owns_buffer) {
                        cache_k_handle = context->ensureMetalBuffer(cache_k_name, *cache_k_decoded.data);
                    }
                    if (!cache_v_decoded.owns_buffer) {
                        cache_v_handle = context->ensureMetalBuffer(cache_v_name, *cache_v_decoded.data);
                    }
#endif
                    if (context_length > 0 && num_heads > 0 && head_dim > 0) {
                        std::vector<float> mask;
                        auto mask_it = node.annotations.find("attention_mask");
                        if (mask_it != node.annotations.end()) {
                            const auto* mask_tensor = context->getTensor(mask_it->second);
                            if (mask_tensor) mask = *mask_tensor;
                        }
                        std::vector<float> gpu_out;
                        if (MetalExecutor::Instance().runAttention(
                                *q,
                                *k_new,
                                *v_new,
                                num_heads,
                                kv_heads,
                                head_dim,
                                context_length,
                                mask,
                                position,
                                rotary_dim,
                                rope_freq_base,
                                rope_freq_scale,
                                *cache_k_decoded.data,
                                *cache_v_decoded.data,
                                cache_k_handle,
                                cache_v_handle,
                                gpu_out)) {
                            if (!node.outputs.empty()) {
                                context->setTensor(node.outputs[0], std::move(gpu_out));
                            }
                            encodeCacheTensor(cache_k_info, cache_k_storage, cache_k_decoded);
                            encodeCacheTensor(cache_v_info, cache_v_storage, cache_v_decoded);
                            result.message = "metal-attention";
                            return result;
                        }
                    }
                }
            }
            return RunAttentionCPU(node, context, *q, *k_new, *v_new);
        }
        case ExecOpType::Add: {
            if (node.inputs.size() < 2) {
                result.success = false;
                result.message = "Add op missing operands";
                return result;
            }
            const std::vector<float>* a = fetchInput(*context, node, 0, result);
            if (!result.success) return result;
            const std::vector<float>* b = fetchInput(*context, node, 1, result);
            if (!result.success) return result;
            if (a->size() != b->size()) {
                result.success = false;
                result.message = "Add operand size mismatch";
                return result;
            }
            std::vector<float> sum(a->size());
#if defined(__APPLE__)
            bool used_accelerate = false;
            if (a->size() >= 32) {
                vDSP_vadd(a->data(), 1, b->data(), 1, sum.data(), 1, a->size());
                used_accelerate = true;
            } else {
                for (size_t i = 0; i < sum.size(); ++i) {
                    sum[i] = (*a)[i] + (*b)[i];
                }
            }
#else
            bool used_accelerate = false;
            for (size_t i = 0; i < sum.size(); ++i) {
                sum[i] = (*a)[i] + (*b)[i];
            }
#endif
            if (!node.outputs.empty()) {
                context->setTensor(node.outputs[0], std::move(sum));
            }
            result.message =
#if defined(__APPLE__)
                (used_accelerate ? "add-accelerate" : "add");
#else
                "add";
#endif
            return result;
        }
        case ExecOpType::Softmax: {
            const std::vector<float>* input = fetchInput(*context, node, 0, result);
            if (!result.success) return result;
            std::vector<float> tmp = *input;
#if defined(__APPLE__)
            if (tmp.size() >= 4) {
                int count = static_cast<int>(tmp.size());
                float* raw = tmp.data();
                float max_val = *std::max_element(tmp.begin(), tmp.end());
                for (float& v : tmp) {
                    v -= max_val;
                }
                vvexpf(raw, raw, &count);
                float sum = 0.0f;
                vDSP_sve(raw, 1, &sum, count);
                if (sum > 0.0f) {
                    float inv = 1.0f / sum;
                    vDSP_vsmul(raw, 1, &inv, raw, 1, count);
                }
            } else {
                softmax(tmp);
            }
#else
            softmax(tmp);
#endif
            if (!node.outputs.empty()) context->setTensor(node.outputs[0], std::move(tmp));
            result.message = "softmax";
            return result;
        }
        default:
            result.success = false;
            std::ostringstream oss;
            oss << "CPU backend unsupported op " << toString(node.op);
            result.message = oss.str();
            return result;
    }
}

BackendExecutionResult MetalExecutionBackend::execute(const ExecutionNode& node,
                                                      ExecutionContext* context,
                                                      const KernelDescriptor* descriptor) const {
    BackendExecutionResult result;
    if (descriptor) {
        result.kernel_id = descriptor->id;
    } else if (!node.kernel_id.empty()) {
        result.kernel_id = node.kernel_id;
    }
    struct Finalize {
        BackendExecutionResult& res;
        const KernelDescriptor* desc;
        ~Finalize() {
            if (desc && res.message.empty()) {
                res.message = "kernel=" + desc->id;
            }
        }
    } finalize{result, descriptor};
    if (!context || !MetalExecutor::Instance().isAvailable()) {
        CpuExecutionBackend cpu;
        return cpu.execute(node, context, descriptor);
    }

    switch (node.op) {
        case ExecOpType::MatMul:
        case ExecOpType::Linear: {
            const std::vector<float>* input = fetchInput(*context, node, 0, result);
            if (!result.success) return result;
            std::string weight = getAnnotation(node, "weight");
            if (weight.empty()) {
                result.success = false;
                result.message = "Metal matmul missing weight annotation";
                return result;
            }
            const auto& tensors = context->session().loader().tensors();
            auto w_it = tensors.find(weight);
            if (w_it == tensors.end() || w_it->second.shape.size() != 2) {
                CpuExecutionBackend cpu;
                return cpu.execute(node, context, descriptor);
            }
            if (w_it->second.shape[1] != input->size()) {
                CpuExecutionBackend cpu;
                return cpu.execute(node, context, descriptor);
            }
            size_t rows = static_cast<size_t>(w_it->second.shape[0]);
            const std::vector<float>* bias_tensor = nullptr;
            std::string bias_name = getAnnotation(node, "bias");
            if (!bias_name.empty()) {
                try {
                    const auto& bias = context->getParameter(bias_name);
                    if (bias.size() != rows) {
                        CpuExecutionBackend cpu;
                        return cpu.execute(node, context, descriptor);
                    }
                    bias_tensor = &bias;
                } catch (const std::exception&) {
                    CpuExecutionBackend cpu;
                    return cpu.execute(node, context, descriptor);
                }
            }
            std::vector<float> output;
            bool ok = false;
            auto raw = context->session().loader().loadTensorData(w_it->second);
            if (w_it->second.dtype == frontend::GGML_TYPE_F32) {
                std::vector<float> weights(raw.size() / sizeof(float));
                std::memcpy(weights.data(), raw.data(), raw.size());
                ok = MetalExecutor::Instance().runMatMul(weights,
                                                         *input,
                                                         rows,
                                                         static_cast<size_t>(w_it->second.shape[1]),
                                                         false,
                                                         output,
                                                         bias_tensor);
            } else {
                size_t row_stride = raw.size() / rows;
                switch (w_it->second.dtype) {
                    case frontend::GGML_TYPE_Q4_0:
                        ok = MetalExecutor::Instance().runMatMulQ4_0(raw,
                                                                     *input,
                                                                     rows,
                                                                     static_cast<size_t>(w_it->second.shape[1]),
                                                                     row_stride,
                                                                     context->session().loader().quantizationVersion(),
                                                                     output,
                                                                     bias_tensor);
                        break;
                    case frontend::GGML_TYPE_Q4_1:
                        ok = MetalExecutor::Instance().runMatMulQ4_1(raw,
                                                                     *input,
                                                                     rows,
                                                                     static_cast<size_t>(w_it->second.shape[1]),
                                                                     row_stride,
                                                                     output,
                                                                     bias_tensor);
                        break;
                    case frontend::GGML_TYPE_Q5_0:
                        ok = MetalExecutor::Instance().runMatMulQ5_0(raw,
                                                                     *input,
                                                                     rows,
                                                                     static_cast<size_t>(w_it->second.shape[1]),
                                                                     row_stride,
                                                                     output,
                                                                     bias_tensor);
                        break;
                    case frontend::GGML_TYPE_Q5_1:
                        ok = MetalExecutor::Instance().runMatMulQ5_1(raw,
                                                                     *input,
                                                                     rows,
                                                                     static_cast<size_t>(w_it->second.shape[1]),
                                                                     row_stride,
                                                                     output,
                                                                     bias_tensor);
                        break;
                    case frontend::GGML_TYPE_Q2_K:
                        ok = MetalExecutor::Instance().runMatMulQ2K(raw,
                                                                    *input,
                                                                    rows,
                                                                    static_cast<size_t>(w_it->second.shape[1]),
                                                                    row_stride,
                                                                    output,
                                                                    bias_tensor);
                        break;
                    case frontend::GGML_TYPE_Q3_K:
                        ok = MetalExecutor::Instance().runMatMulQ3K(raw,
                                                                    *input,
                                                                    rows,
                                                                    static_cast<size_t>(w_it->second.shape[1]),
                                                                    row_stride,
                                                                    output,
                                                                    bias_tensor);
                        break;
                    case frontend::GGML_TYPE_Q4_K:
                        ok = MetalExecutor::Instance().runMatMulQ4K(raw,
                                                                    *input,
                                                                    rows,
                                                                    static_cast<size_t>(w_it->second.shape[1]),
                                                                    row_stride,
                                                                    output,
                                                                    bias_tensor);
                        break;
                    case frontend::GGML_TYPE_Q5_K:
                        ok = MetalExecutor::Instance().runMatMulQ5K(raw,
                                                                    *input,
                                                                    rows,
                                                                    static_cast<size_t>(w_it->second.shape[1]),
                                                                    row_stride,
                                                                    output,
                                                                    bias_tensor);
                        break;
                    case frontend::GGML_TYPE_Q6_K:
                        ok = MetalExecutor::Instance().runMatMulQ6K(raw,
                                                                    *input,
                                                                    rows,
                                                                    static_cast<size_t>(w_it->second.shape[1]),
                                                                    row_stride,
                                                                    output,
                                                                    bias_tensor);
                        break;
                    case frontend::GGML_TYPE_Q8_K:
                        ok = MetalExecutor::Instance().runMatMulQ8K(raw,
                                                                    *input,
                                                                    rows,
                                                                    static_cast<size_t>(w_it->second.shape[1]),
                                                                    row_stride,
                                                                    output,
                                                                    bias_tensor);
                        break;
                    case frontend::GGML_TYPE_Q8_0:
                        ok = MetalExecutor::Instance().runMatMulQ8_0(raw,
                                                                     *input,
                                                                     rows,
                                                                     static_cast<size_t>(w_it->second.shape[1]),
                                                                     row_stride,
                                                                     output,
                                                                     bias_tensor);
                        break;
                    case frontend::GGML_TYPE_Q8_1:
                        ok = MetalExecutor::Instance().runMatMulQ8_1(raw,
                                                                     *input,
                                                                     rows,
                                                                     static_cast<size_t>(w_it->second.shape[1]),
                                                                     row_stride,
                                                                     output,
                                                                     bias_tensor);
                        break;
                    default:
                        ok = false;
                        break;
                }
            }
            if (!ok) {
                CpuExecutionBackend cpu;
                return cpu.execute(node, context, descriptor);
            }
            if (!node.outputs.empty()) {
                context->setTensor(node.outputs[0], std::move(output));
            }
            result.message = "metal-matmul";
            return result;
        }
        case ExecOpType::FeedForward: {
            const std::vector<float>* gate = fetchInput(*context, node, 0, result);
            if (!result.success) return result;
            const std::vector<float>* up = fetchInput(*context, node, 1, result);
            if (!result.success) return result;
            if (gate->size() != up->size()) {
                CpuExecutionBackend cpu;
                return cpu.execute(node, context, descriptor);
            }
            std::vector<float> output;
            if (!MetalExecutor::Instance().runFeedForward(*gate, *up, output)) {
                CpuExecutionBackend cpu;
                return cpu.execute(node, context, descriptor);
            }
            if (!node.outputs.empty()) {
                context->setTensor(node.outputs[0], std::move(output));
            }
            result.message = "metal-ffn";
            return result;
        }
        case ExecOpType::Norm: {
            const std::vector<float>* input = fetchInput(*context, node, 0, result);
            if (!result.success) return result;
            std::string weight_name = getAnnotation(node, "weight");
            if (weight_name.empty()) {
                CpuExecutionBackend cpu;
                return cpu.execute(node, context, descriptor);
            }
            try {
                const auto& weight = context->getParameter(weight_name);
                std::vector<float> output;
                if (!MetalExecutor::Instance().runRmsNorm(*input, weight, 1e-5f, output)) {
                    CpuExecutionBackend cpu;
                    return cpu.execute(node, context, descriptor);
                }
                if (!node.outputs.empty()) {
                    context->setTensor(node.outputs[0], std::move(output));
                }
                result.message = "metal-norm";
                return result;
            } catch (const std::exception&) {
                CpuExecutionBackend cpu;
                return cpu.execute(node, context, descriptor);
            }
        }
        case ExecOpType::Add: {
            const std::vector<float>* a = fetchInput(*context, node, 0, result);
            if (!result.success) return result;
            const std::vector<float>* b = fetchInput(*context, node, 1, result);
            if (!result.success) return result;
            if (a->size() != b->size()) {
                CpuExecutionBackend cpu;
                return cpu.execute(node, context, descriptor);
            }
            std::vector<float> output;
            if (!MetalExecutor::Instance().runAdd(*a, *b, output)) {
                CpuExecutionBackend cpu;
                return cpu.execute(node, context, descriptor);
            }
            if (!node.outputs.empty()) {
                context->setTensor(node.outputs[0], std::move(output));
            }
            result.message = "metal-add";
            return result;
        }
        case ExecOpType::Softmax: {
            const std::vector<float>* input = fetchInput(*context, node, 0, result);
            if (!result.success) return result;
            std::vector<float> output;
            if (!MetalExecutor::Instance().runSoftmax(*input, output)) {
                CpuExecutionBackend cpu;
                return cpu.execute(node, context, descriptor);
            }
            if (!node.outputs.empty()) {
                context->setTensor(node.outputs[0], std::move(output));
            }
            result.message = "metal-softmax";
            return result;
        }
        default: {
            CpuExecutionBackend cpu;
            return cpu.execute(node, context, descriptor);
        }
    }
}

BackendRegistry::BackendRegistry() = default;

const ExecutionBackend& BackendRegistry::backendFor(BackendKind kind) const {
    switch (kind) {
        case BackendKind::CPU:
            return cpu_backend_;
        case BackendKind::Metal:
            return metal_backend_;
        case BackendKind::Auto:
        default:
            return cpu_backend_;
    }
}

const BackendRegistry& BackendRegistry::Default() {
    static BackendRegistry registry;
    return registry;
}

} // namespace runtime
} // namespace mlc
#if defined(__AVX2__) || defined(__AVX512F__)
float dotProductSimd(const float* lhs, const float* rhs, size_t dim) {
#if defined(__AVX512F__)
    __m512 acc = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 a = _mm512_loadu_ps(lhs + i);
        __m512 b = _mm512_loadu_ps(rhs + i);
        acc = _mm512_fmadd_ps(a, b, acc);
    }
    float sum = _mm512_reduce_add_ps(acc);
    for (; i < dim; ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
#elif defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 a = _mm256_loadu_ps(lhs + i);
        __m256 b = _mm256_loadu_ps(rhs + i);
        acc = _mm256_fmadd_ps(a, b, acc);
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < dim; ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
#endif
}
#endif
