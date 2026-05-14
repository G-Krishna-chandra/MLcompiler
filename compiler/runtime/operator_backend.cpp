#include "runtime/operator_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <vector>
#include "runtime/attention_cpu.hpp"
#include "runtime/execution_context.hpp"
#include "runtime/kernel_registry.hpp"
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

bool loadMask(const ExecutionNode& node,
              ExecutionContext* context,
              std::vector<float>& mask) {
    auto it = node.annotations.find("attention_mask");
    if (it == node.annotations.end()) return false;
    const auto* tensor = context ? context->getTensor(it->second) : nullptr;
    if (!tensor) return false;
    mask = *tensor;
    return true;
}

std::string getAnnotation(const ExecutionNode& node, const std::string& key) {
    auto it = node.annotations.find(key);
    if (it == node.annotations.end()) return {};
    return it->second;
}

float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

float gelu(float x) {
    const float kAlpha = std::sqrt(2.0f / static_cast<float>(M_PI));
    return 0.5f * x * (1.0f + std::tanh(kAlpha * (x + 0.044715f * x * x * x)));
}

// Loop a single-token execute() over the multi-token input by slicing each
// listed input into per-token chunks, calling the backend with seq_len
// temporarily forced to 1, and stacking the per-token output back into
// node.outputs[0]. Restores tensors and seq_len on exit. Used by Metal
// MatMul/Linear/Norm/Add execute paths to multi-token-prefill correctly
// without rewriting any kernels.
template <typename BackendT>
BackendExecutionResult LoopExecutePerToken(const BackendT& backend,
                                            const ExecutionNode& node,
                                            ExecutionContext* context,
                                            const KernelDescriptor* descriptor,
                                            std::initializer_list<size_t> input_indices) {
    BackendExecutionResult result;
    result.success = true;
    if (!context) {
        result.success = false;
        result.message = "LoopExecutePerToken needs context";
        return result;
    }
    size_t seq_len = context->seqLen();
    if (seq_len <= 1) {
        return backend.execute(node, context, descriptor);
    }
    struct SavedInput { size_t index; std::vector<float> data; size_t per_token; };
    std::vector<SavedInput> saved;
    saved.reserve(input_indices.size());
    for (size_t idx : input_indices) {
        if (idx >= node.inputs.size()) continue;
        const auto* t = context->getTensor(node.inputs[idx]);
        if (!t || t->size() % seq_len != 0) {
            result.success = false;
            result.message = "LoopExecutePerToken: input size not divisible by seq_len";
            return result;
        }
        saved.push_back({idx, *t, t->size() / seq_len});
    }
    std::string out_name = node.outputs.empty() ? std::string() : node.outputs[0];
    std::vector<float> stacked;
    size_t per_token_out = 0;
    BackendExecutionResult last;
    last.success = true;
    last.actual_backend = BackendKind::Metal;
    for (size_t t = 0; t < seq_len; ++t) {
        for (const auto& s : saved) {
            std::vector<float> sub(s.data.begin() + t * s.per_token,
                                   s.data.begin() + (t + 1) * s.per_token);
            context->setTensor(node.inputs[s.index], std::move(sub));
        }
        context->setSeqLen(1);
        last = backend.execute(node, context, descriptor);
        context->setSeqLen(seq_len);
        if (!last.success) break;
        if (!out_name.empty()) {
            const auto* out_t = context->getTensor(out_name);
            if (out_t) {
                if (t == 0) {
                    per_token_out = out_t->size();
                    stacked.resize(seq_len * per_token_out, 0.0f);
                }
                if (out_t->size() == per_token_out) {
                    std::copy(out_t->begin(), out_t->end(),
                              stacked.begin() + t * per_token_out);
                }
            }
        }
    }
    for (auto& s : saved) {
        context->setTensor(node.inputs[s.index], std::move(s.data));
    }
    if (last.success && !out_name.empty()) {
        context->setTensor(out_name, std::move(stacked));
    }
    return last;
}

bool buildAlibiMask(const ExecutionContext* context,
                    const ExecutionNode& node,
                    std::vector<float>& mask) {
    if (!context || !context->graph()) return false;
    const auto& cfg = context->graph()->modelConfig();
    if (cfg.sliding_window > 0) return false; // sliding window handled elsewhere
    std::string cache_k_name = getAnnotation(node, "kv_cache_k");
    const ExecutionTensor* cache_info = context->tensorInfo(cache_k_name);
    size_t context_length = cache_info && cache_info->shape.size() >= 2
                                ? static_cast<size_t>(cache_info->shape[1])
                                : 0;
    if (context_length == 0) return false;
    if (cfg.alibi_slopes.empty()) return false;
    size_t heads = cfg.head_count > 0 ? cfg.head_count : cfg.alibi_slopes.size();
    mask.assign(heads * context_length, 0.0f);
    for (size_t h = 0; h < heads && h < cfg.alibi_slopes.size(); ++h) {
        float slope = cfg.alibi_slopes[h];
        for (size_t t = 0; t < context_length; ++t) {
            mask[h * context_length + t] = slope * static_cast<float>(t);
        }
    }
    return true;
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

void layerNorm(const std::vector<float>& input,
               const std::vector<float>& weight,
               std::vector<float>& output,
               float epsilon = 1e-5f) {
    size_t n = input.size();
    if (n == 0) {
        output.clear();
        return;
    }
    if (output.size() != n) output.assign(n, 0.0f);
    float sum = 0.0f;
    for (float v : input) sum += v;
    float mean = sum / static_cast<float>(n);
    float var = 0.0f;
    for (float v : input) {
        float d = v - mean;
        var += d * d;
    }
    var /= static_cast<float>(std::max<size_t>(1, n));
    float inv_std = 1.0f / std::sqrt(var + epsilon);
    for (size_t i = 0; i < n; ++i) {
        float gamma = i < weight.size() ? weight[i] : 1.0f;
        output[i] = (input[i] - mean) * inv_std * gamma;
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
    std::vector<float>* data = nullptr;
    size_t head_dim = 0;
    bool owns_buffer = false;  // unused now; data lives in storage->dequant_shadow or float_data
    // Rows beyond this position-per-head were never written and stay at the
    // value the buffer was zero-initialized to. Setters honor this so the
    // dequant + re-quant per attention call is O(active context) rather than
    // O(full context_length). 0 = "all rows".
    size_t valid_positions_per_head = 0;
};

bool decodeCacheTensor(const ExecutionNode& node,
                       const ExecutionTensor* info,
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
    // Persistent shadow buffer in storage avoids per-call 2 MB alloc + zero-init.
    // First-use sizes it; subsequent calls reuse.
    if (storage->dequant_shadow.size() != elements) {
        storage->dequant_shadow.assign(elements, 0.0f);
    }
    std::vector<float>& shadow = storage->dequant_shadow;
#if defined(__APPLE__)
    // Dequant via Metal only when the owning node would itself run on Metal —
    // routes through MetalExecutor::shouldUseFor so force-CPU is honored.
    if (MetalExecutor::Instance().shouldUseFor(node)) {
        bool ok = false;
        const size_t cols = result.head_dim * rows;
        switch (storage->dtype) {
        case frontend::GGML_TYPE_Q4_0:
            ok = MetalExecutor::Instance().dequantQ4Block(storage->raw_data, cols, shadow);
            break;
        case frontend::GGML_TYPE_Q4_1:
            ok = MetalExecutor::Instance().dequantQ4_1Block(storage->raw_data, cols, shadow);
            break;
        case frontend::GGML_TYPE_Q5_0:
            ok = MetalExecutor::Instance().dequantQ5_0Block(storage->raw_data, cols, shadow);
            break;
        case frontend::GGML_TYPE_Q5_1:
            ok = MetalExecutor::Instance().dequantQ5_1Block(storage->raw_data, cols, shadow);
            break;
        case frontend::GGML_TYPE_Q8_0:
            ok = MetalExecutor::Instance().dequantQ8Block(storage->raw_data, cols, shadow);
            break;
        case frontend::GGML_TYPE_Q8_1:
            ok = MetalExecutor::Instance().dequantQ8_1Block(storage->raw_data, cols, shadow);
            break;
        default:
            break;
        }
        if (ok) {
            result.data = &shadow;
            result.owns_buffer = true;
            return true;
        }
    }
#endif
    const uint8_t* ptr = storage->raw_data.data();
    // Per-kv-head row range: when `valid_positions_per_head` is non-zero, only
    // dequant the first `valid_positions` rows per kv_head — the rest of the
    // cache has never been written and remains at the buffer's zero-init.
    // For TinyLlama at decode position 80 / context 2048 that's a 25× cut.
    size_t valid_per_head = result.valid_positions_per_head;
    if (valid_per_head == 0 || result.head_dim == 0) {
        for (size_t row = 0; row < rows; ++row) {
            dequantizeRowTo(ptr + row * stride,
                            storage->dtype,
                            result.head_dim,
                            storage->quant_version,
                            shadow.data() + row * result.head_dim);
        }
    } else {
        size_t context_length = result.head_dim ? rows : 0;
        size_t kv_heads = 0;
        if (info && info->shape.size() >= 3) {
            kv_heads = static_cast<size_t>(std::max<int64_t>(1, info->shape[0]));
            context_length = static_cast<size_t>(std::max<int64_t>(1, info->shape[1]));
        }
        if (kv_heads == 0 || context_length == 0) {
            // Shape unknown; fall back to full dequant.
            for (size_t row = 0; row < rows; ++row) {
                dequantizeRowTo(ptr + row * stride,
                                storage->dtype,
                                result.head_dim,
                                storage->quant_version,
                                shadow.data() + row * result.head_dim);
            }
        } else {
            size_t cap = std::min(valid_per_head, context_length);
            for (size_t kvh = 0; kvh < kv_heads; ++kvh) {
                size_t base_row = kvh * context_length;
                for (size_t r = 0; r < cap; ++r) {
                    size_t row = base_row + r;
                    dequantizeRowTo(ptr + row * stride,
                                    storage->dtype,
                                    result.head_dim,
                                    storage->quant_version,
                                    shadow.data() + row * result.head_dim);
                }
            }
        }
    }
    result.data = &shadow;
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
    // Mirror the per-kv-head row range from decodeCacheTensor — rows that
    // were never read also weren't written, so they stay zero in the storage
    // (raw_data starts zero-init from ensureStateTensor) and don't need
    // re-encoding. This is the encode-side half of the active-rows
    // optimization that keeps fp16 cache perf-neutral at low context.
    size_t valid_per_head = decoded.valid_positions_per_head;
    if (valid_per_head == 0) {
        for (size_t row = 0; row < rows; ++row) {
            quantizeRowFrom(decoded.data->data() + row * decoded.head_dim,
                            storage->dtype,
                            decoded.head_dim,
                            storage->quant_version,
                            dst + row * stride);
        }
        return;
    }
    size_t context_length = rows;
    size_t kv_heads = 1;
    if (info->shape.size() >= 3) {
        kv_heads = static_cast<size_t>(std::max<int64_t>(1, info->shape[0]));
        context_length = static_cast<size_t>(std::max<int64_t>(1, info->shape[1]));
    } else {
        for (size_t row = 0; row < rows; ++row) {
            quantizeRowFrom(decoded.data->data() + row * decoded.head_dim,
                            storage->dtype,
                            decoded.head_dim,
                            storage->quant_version,
                            dst + row * stride);
        }
        return;
    }
    size_t cap = std::min(valid_per_head, context_length);
    for (size_t kvh = 0; kvh < kv_heads; ++kvh) {
        size_t base_row = kvh * context_length;
        for (size_t r = 0; r < cap; ++r) {
            size_t row = base_row + r;
            quantizeRowFrom(decoded.data->data() + row * decoded.head_dim,
                            storage->dtype,
                            decoded.head_dim,
                            storage->quant_version,
                            dst + row * stride);
        }
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
                size_t seq_len = context->seqLen();
                std::vector<float> stacked;
                if (seq_len <= 1) {
                    stacked = context->session().getEmbedding(weight, context->token());
                } else {
                    const auto& toks = context->tokens();
                    for (size_t t = 0; t < seq_len; ++t) {
                        uint64_t tok = t < toks.size() ? toks[t] : context->token();
                        auto e = context->session().getEmbedding(weight, tok);
                        if (t == 0) stacked.reserve(seq_len * e.size());
                        stacked.insert(stacked.end(), e.begin(), e.end());
                    }
                }
                if (!node.outputs.empty()) {
                    context->setTensor(node.outputs[0], std::move(stacked));
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
                size_t seq_len = context->seqLen();
                std::vector<float> output;
                if (seq_len <= 1 || input->size() % seq_len != 0) {
                    output = context->session().runLinear(weight, *input);
                } else {
                    size_t cols = input->size() / seq_len;
                    std::vector<float> sub(cols);
                    for (size_t t = 0; t < seq_len; ++t) {
                        std::copy(input->begin() + t * cols,
                                  input->begin() + (t + 1) * cols,
                                  sub.begin());
                        auto row_out = context->session().runLinear(weight, sub);
                        if (t == 0) output.reserve(seq_len * row_out.size());
                        output.insert(output.end(), row_out.begin(), row_out.end());
                    }
                }
                std::string bias_name = getAnnotation(node, "bias");
                if (!bias_name.empty()) {
                    const auto& bias = context->getParameter(bias_name);
                    size_t per = bias.size();
                    if (per == 0 || output.size() % per != 0) {
                        result.success = false;
                        std::ostringstream oss;
                        oss << "Bias tensor '" << bias_name << "' size mismatch";
                        result.message = oss.str();
                        return result;
                    }
                    for (size_t i = 0; i < output.size(); ++i) {
                        output[i] += bias[i % per];
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
            std::string bias_name = getAnnotation(node, "bias");
            std::string norm_kind = getAnnotation(node, "norm_kind");
            if (weight_name.empty()) {
                if (!node.outputs.empty()) context->setTensor(node.outputs[0], *input);
                result.message = "norm-pass";
                return result;
            }
            try {
                const auto& weight = context->getParameter(weight_name);
                size_t seq_len = context->seqLen();
                if (seq_len < 1) seq_len = 1;
                size_t per_token = input->size() / seq_len;
                if (per_token == 0) per_token = input->size();
                const std::vector<float>* bias_ptr = nullptr;
                if (!bias_name.empty()) {
                    bias_ptr = &context->getParameter(bias_name);
                    if (bias_ptr->size() != per_token) {
                        result.success = false;
                        result.message = "LayerNorm bias size mismatch";
                        return result;
                    }
                }
                std::vector<float> output(input->size(), 0.0f);
                std::vector<float> tok_in(per_token);
                std::vector<float> tok_out(per_token);
                auto applyOne = [&](size_t off) {
                    std::copy(input->begin() + off,
                              input->begin() + off + per_token,
                              tok_in.begin());
#if defined(__APPLE__)
                    if (norm_kind != "layer" && per_token >= 32) {
                        float mean_sq = 0.0f;
                        vDSP_measqv(tok_in.data(), 1, &mean_sq, per_token);
                        float inv = 1.0f / std::sqrt(mean_sq + 1e-5f);
                        vDSP_vsmul(tok_in.data(), 1, &inv, tok_out.data(), 1, per_token);
                        for (size_t i = 0; i < per_token; ++i) {
                            float gamma = i < weight.size() ? weight[i] : 1.0f;
                            tok_out[i] *= gamma;
                        }
                        return;
                    }
#endif
                    if (norm_kind == "layer") {
                        layerNorm(tok_in, weight, tok_out);
                        if (bias_ptr) {
                            for (size_t i = 0; i < per_token; ++i) {
                                tok_out[i] += (*bias_ptr)[i];
                            }
                        }
                    } else {
                        rmsNorm(tok_in, weight, tok_out);
                    }
                };
                for (size_t t = 0; t < seq_len; ++t) {
                    applyOne(t * per_token);
                    std::copy(tok_out.begin(), tok_out.end(), output.begin() + t * per_token);
                }
                if (!node.outputs.empty()) {
                    context->setTensor(node.outputs[0], std::move(output));
                }
                result.message = (norm_kind == "layer") ? "layer-norm" : "rms-norm";
            } catch (const std::exception& e) {
                result.success = false;
                result.message = e.what();
            }
            return result;
        }
        case ExecOpType::FeedForward: {
            // Two paths:
            // - Legacy: gate/up provided as inputs.
            // - Fused: single activation input and weights annotated as param0/param1(/param2).
            std::vector<float> gate_storage;
            std::vector<float> up_storage;
            const std::vector<float>* gate = nullptr;
            const std::vector<float>* up = nullptr;

            auto computeLinear = [&](const std::string& weight_name,
                                     const std::vector<float>& input) -> std::vector<float> {
                const auto& loader = context->session().loader();
                const auto& tensors = loader.tensors();
                auto it = tensors.find(weight_name);
                if (it == tensors.end()) {
                    throw std::runtime_error("Weight tensor '" + weight_name + "' not found");
                }
                const auto& t = it->second;
                if (t.shape.size() != 2) {
                    throw std::runtime_error("Weight tensor '" + weight_name + "' must be 2D");
                }
                // GGML convention: shape[0]=ne0=cols (input dim), shape[1]=ne1=rows (output dim).
                // Same convention as Session::runLinear; mirrors the dispatcher fix at :963-964.
                size_t cols = static_cast<size_t>(t.shape[0]);
                size_t rows = static_cast<size_t>(t.shape[1]);
                if (cols != input.size()) {
                    throw std::runtime_error("Input size mismatch for '" + weight_name + "'");
                }
                const auto& raw = context->session().tensorData(t);
                if (raw.empty()) {
                    throw std::runtime_error("Weight tensor '" + weight_name + "' has no data");
                }
                std::vector<float> out(rows, 0.0f);
                uint32_t qv = loader.quantizationVersion();
                size_t stride = ggmlRowSizeBytes(t.dtype, cols, qv);
                if (stride == 0) {
                    stride = raw.size() / rows;
                }
                if (stride == 0 || raw.size() < stride * rows) {
                    throw std::runtime_error("Weight tensor '" + weight_name + "' has inconsistent size");
                }
                // Fast dot products per row, using quant-specific kernels when available.
                // For F32 we tile over columns to improve cache locality.
                constexpr size_t kBlock = 128;
                std::vector<float> tile_buffer;
                tile_buffer.reserve(kBlock);
                for (size_t r = 0; r < rows; ++r) {
                    const uint8_t* row_ptr = raw.data() + r * stride;
                    switch (t.dtype) {
                        case frontend::GGML_TYPE_F32: {
                            const float* w = reinterpret_cast<const float*>(row_ptr);
                            float acc = 0.0f;
                            size_t c = 0;
                            for (; c + kBlock <= cols; c += kBlock) {
                                // Manually unroll a small block to stay cache hot.
                                for (size_t i = 0; i < kBlock; ++i) {
                                    acc += w[c + i] * input[c + i];
                                }
                            }
                            for (; c < cols; ++c) {
                                acc += w[c] * input[c];
                            }
                            out[r] = acc;
                            break;
                        }
                        case frontend::GGML_TYPE_Q4_0:
                            out[r] = dotProductRowQ4_0(row_ptr, cols, qv, input.data());
                            break;
                        case frontend::GGML_TYPE_Q4_1:
                            out[r] = dotProductRowQ4_1(row_ptr, cols, input.data());
                            break;
                        case frontend::GGML_TYPE_Q5_0:
                            out[r] = dotProductRowQ5_0(row_ptr, cols, input.data());
                            break;
                        case frontend::GGML_TYPE_Q5_1:
                            out[r] = dotProductRowQ5_1(row_ptr, cols, input.data());
                            break;
                        case frontend::GGML_TYPE_Q8_0:
                            out[r] = dotProductRowQ8_0(row_ptr, cols, input.data());
                            break;
                        case frontend::GGML_TYPE_Q8_1:
                            out[r] = dotProductRowQ8_1(row_ptr, cols, input.data());
                            break;
                        case frontend::GGML_TYPE_Q2_K:
                            out[r] = dotProductRowQ2_K(row_ptr, cols, input.data());
                            break;
                        case frontend::GGML_TYPE_Q3_K:
                            out[r] = dotProductRowQ3_K(row_ptr, cols, input.data());
                            break;
                        case frontend::GGML_TYPE_Q4_K:
                            out[r] = dotProductRowQ4_K(row_ptr, cols, input.data());
                            break;
                        case frontend::GGML_TYPE_Q5_K:
                            out[r] = dotProductRowQ5_K(row_ptr, cols, input.data());
                            break;
                        case frontend::GGML_TYPE_Q6_K:
                            out[r] = dotProductRowQ6_K(row_ptr, cols, input.data());
                            break;
                        case frontend::GGML_TYPE_Q8_K:
                            out[r] = dotProductRowQ8_K(row_ptr, cols, input.data());
                            break;
                        default:
                            throw std::runtime_error("Unsupported dtype in fused FFN");
                    }
                }
                return out;
            };

            auto fetchBias = [&](const std::string& bias_name, size_t expected) -> std::vector<float> {
                if (bias_name.empty()) return {};
                // First look for a tensor already in the context (allows test injection).
                if (context) {
                    if (const auto* t = context->getTensor(bias_name)) {
                        if (t->size() == expected) return *t;
                        throw std::runtime_error("Bias tensor '" + bias_name + "' size mismatch");
                    }
                }
                // Then try to load as a parameter (must be F32).
                try {
                    const auto& b = context->getParameter(bias_name);
                    if (b.size() != expected) {
                        throw std::runtime_error("Bias tensor '" + bias_name + "' size mismatch");
                    }
                    return b;
                } catch (const std::exception&) {
                    // Not found or wrong dtype; treat as absent.
                    return {};
                }
            };

            if (node.inputs.size() >= 2) {
                gate = fetchInput(*context, node, 0, result);
                if (!result.success) return result;
                up = fetchInput(*context, node, 1, result);
                if (!result.success) return result;
            } else if (node.inputs.size() == 1 && context) {
                const std::vector<float>* x = fetchInput(*context, node, 0, result);
                if (!result.success) return result;
                std::string w_gate = getAnnotation(node, "param0");
                std::string w_up = getAnnotation(node, "param1");
                std::string w_down = getAnnotation(node, "param2");
                if (w_gate.empty() || w_up.empty()) {
                    result.success = false;
                    result.message = "FeedForward missing weight annotations for fused path";
                    return result;
                }
                try {
                    gate_storage = computeLinear(w_gate, *x);
                    up_storage = computeLinear(w_up, *x);
                    gate = &gate_storage;
                    up = &up_storage;
                } catch (const std::exception& e) {
                    result.success = false;
                    result.message = e.what();
                    return result;
                }
            } else {
                result.success = false;
                result.message = "FeedForward node missing gate/up inputs";
                return result;
            }

            if (!gate || !up || gate->size() != up->size()) {
                result.success = false;
                result.message = "FeedForward input size mismatch";
                return result;
            }
            std::string activation = "silu";
            if (context && context->graph()) {
                const auto& cfg = context->graph()->modelConfig();
                activation = cfg.activation;
                if (activation.empty()) {
                    // Defaults per family.
                    if (cfg.family == ArchitectureFamily::Gemma) {
                        activation = "geglu";
                    } else {
                        activation = "silu";
                    }
                }
                // normalize
                std::transform(activation.begin(), activation.end(), activation.begin(), [](unsigned char c) {
                    return static_cast<char>(std::tolower(c));
                });
            }
            std::vector<float> mix(gate->size(), 0.0f);
#if defined(__APPLE__)
            bool used_accelerate = false;
#endif
#if defined(__APPLE__)
            if (activation == "silu" && gate->size() >= 32) {
                std::vector<float> silu_values(gate->size());
                if (siluAccelerate(*gate, silu_values)) {
                    vDSP_vmul(silu_values.data(), 1, up->data(), 1, mix.data(), 1, gate->size());
                    used_accelerate = true;
                }
            }
            if (!used_accelerate) {
#endif
                if (activation == "geglu" || activation == "gelu") {
                    for (size_t i = 0; i < mix.size(); ++i) {
                        mix[i] = gelu((*gate)[i]) * (*up)[i];
                    }
                } else { // default silu
                    for (size_t i = 0; i < mix.size(); ++i) {
                        mix[i] = silu((*gate)[i]) * (*up)[i];
                    }
            }
#if defined(__APPLE__)
            }
#endif
            if (!node.outputs.empty()) {
                std::string w_down = getAnnotation(node, "param2");
                std::string bias_name = getAnnotation(node, "bias");
                if (!w_down.empty() && context) {
                    try {
                        auto down = computeLinear(w_down, mix);
                        if (!bias_name.empty()) {
                            auto bias = fetchBias(bias_name, down.size());
                            if (!bias.empty()) {
                                for (size_t i = 0; i < down.size(); ++i) {
                                    down[i] += bias[i];
                                }
                            }
                        }
                        context->setTensor(node.outputs[0], std::move(down));
                    } catch (const std::exception& e) {
                        result.success = false;
                        result.message = e.what();
                        return result;
                    }
                } else {
                    context->setTensor(node.outputs[0], std::move(mix));
                }
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
            bool use_gpu = MetalExecutor::Instance().shouldUseFor(node);
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

            // Multi-token path (item 3 wedge-prep): loop single-token attention
            // seq_len times. Each iteration writes one token's K/V into the
            // cache at base_position + t and computes its attention output.
            // Single-token (decode) skips this entirely and falls through to
            // the existing path below.
            if (context->seqLen() > 1) {
                size_t seq_len = context->seqLen();
                size_t num_heads = static_cast<size_t>(node.attributes.count("heads") ? node.attributes.at("heads") : 0.0f);
                size_t head_dim = static_cast<size_t>(node.attributes.count("head_dim") ? node.attributes.at("head_dim") : 0.0f);
                size_t kv_heads = static_cast<size_t>(node.attributes.count("kv_heads") ? node.attributes.at("kv_heads") : num_heads);
                size_t q_per = num_heads * head_dim;
                size_t kv_per = (kv_heads > 0 ? kv_heads : num_heads) * head_dim;
                if (q_per == 0 || kv_per == 0 || q->size() != seq_len * q_per ||
                    k_new->size() != seq_len * kv_per || v_new->size() != seq_len * kv_per) {
                    result.success = false;
                    result.message = "Attention multi-token size mismatch";
                    return result;
                }
                size_t base_pos = context->sequencePosition();
                std::string out_name = node.outputs.empty() ? std::string() : node.outputs[0];
                std::vector<float> stacked(seq_len * q_per, 0.0f);
                // Take a local copy of the activations (setTensor below would
                // invalidate q/k_new/v_new pointers under reentry).
                std::vector<float> q_full = *q;
                std::vector<float> k_full = *k_new;
                std::vector<float> v_full = *v_new;
                BackendExecutionResult last;
                last.success = true;
                last.actual_backend = BackendKind::CPU;
                for (size_t t = 0; t < seq_len; ++t) {
                    std::vector<float> q_t(q_full.begin() + t * q_per,
                                           q_full.begin() + (t + 1) * q_per);
                    std::vector<float> k_t(k_full.begin() + t * kv_per,
                                           k_full.begin() + (t + 1) * kv_per);
                    std::vector<float> v_t(v_full.begin() + t * kv_per,
                                           v_full.begin() + (t + 1) * kv_per);
                    context->setTensor(node.inputs[0], std::move(q_t));
                    context->setTensor(node.inputs[1], std::move(k_t));
                    context->setTensor(node.inputs[2], std::move(v_t));
                    context->setSeqLen(1);
                    context->setSequencePosition(base_pos + t);
                    last = execute(node, context, descriptor);
                    if (!last.success) {
                        context->setSeqLen(seq_len);
                        context->setSequencePosition(base_pos);
                        return last;
                    }
                    if (!out_name.empty()) {
                        const auto* out_t = context->getTensor(out_name);
                        if (out_t && out_t->size() == q_per) {
                            std::copy(out_t->begin(), out_t->end(),
                                      stacked.begin() + t * q_per);
                        }
                    }
                }
                context->setSeqLen(seq_len);
                context->setSequencePosition(base_pos);
                context->setTensor(node.inputs[0], std::move(q_full));
                context->setTensor(node.inputs[1], std::move(k_full));
                context->setTensor(node.inputs[2], std::move(v_full));
                if (!out_name.empty()) {
                    context->setTensor(out_name, std::move(stacked));
                }
                result.actual_backend = last.actual_backend;
                result.kernel_id = last.kernel_id;
                result.message = "attention-multi-loop";
                return result;
            }

            std::vector<float> mask;
            bool has_mask = loadMask(node, context, mask);
            if (!has_mask && context && context->graph()) {
                const auto& cfg = context->graph()->modelConfig();
                if (cfg.sliding_window > 0) {
                    // Build a sliding-window mask: positions older than window get -inf.
                    std::string cache_k_name = getAnnotation(node, "kv_cache_k");
                    const ExecutionTensor* cache_info = context->tensorInfo(cache_k_name);
                    size_t context_length = cache_info && cache_info->shape.size() >= 2
                                                ? static_cast<size_t>(cache_info->shape[1])
                                                : 0;
                    size_t base_position = context->sequencePosition();
                    if (context_length > 0) {
                        mask.assign(context_length, 0.0f);
                        const float neg_inf = -1e9f;
                        for (size_t t = 0; t < context_length; ++t) {
                            size_t dist = (base_position >= t) ? (base_position - t) : (context_length);
                            if (dist > cfg.sliding_window) {
                                mask[t] = neg_inf;
                            }
                        }
                        has_mask = true;
                    }
                } else if (cfg.use_alibi || cfg.family == ArchitectureFamily::Mistral || cfg.family == ArchitectureFamily::Gemma) {
                    has_mask = buildAlibiMask(context, node, mask);
                }
            }

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
                // Active-rows hint: the cache rows beyond
                // (sequencePosition + 1) per kv_head have never been written
                // and stay at zero. Plumbed only when the cache storage is
                // non-F32 (decode/encode are no-ops at F32 because the
                // float_data IS the storage); for F16 it's the bulk of the
                // commit-1 perf preservation.
                size_t pre_position = context->sequencePosition();
                if (cache_k_storage && cache_k_storage->dtype != frontend::GGML_TYPE_F32) {
                    cache_k_decoded.valid_positions_per_head = pre_position + 1;
                }
                if (cache_v_storage && cache_v_storage->dtype != frontend::GGML_TYPE_F32) {
                    cache_v_decoded.valid_positions_per_head = pre_position + 1;
                }
                if (!decodeCacheTensor(node, cache_k_info, cache_k_storage, head_dim_attr, cache_k_decoded) ||
                    !decodeCacheTensor(node, cache_v_info, cache_v_storage, head_dim_attr, cache_v_decoded)) {
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
                    // decodeCacheTensor has already produced fp32 buffers in
                    // cache_*_decoded.data (regardless of underlying storage
                    // dtype). Tell runAttention's prepareCache the descriptor
                    // is fp32 so it short-circuits the redundant per-row dequant.
                    // Underlying storage dtype is preserved on the storage
                    // pointer for encodeCacheTensor's re-encode at the end.
                    MetalExecutor::CacheDescriptor cache_k_desc;
                    cache_k_desc.dtype = frontend::GGML_TYPE_F32;
                    cache_k_desc.quant_version = cache_k_storage ? cache_k_storage->quant_version : 1;
                    cache_k_desc.row_stride_bytes = cache_k_decoded.head_dim * sizeof(float);
                    cache_k_desc.raw_quant = cache_k_storage ? &cache_k_storage->raw_data : nullptr;
                    cache_k_desc.float_data = cache_k_decoded.data;
                    MetalExecutor::CacheDescriptor cache_v_desc;
                    cache_v_desc.dtype = frontend::GGML_TYPE_F32;
                    cache_v_desc.quant_version = cache_v_storage ? cache_v_storage->quant_version : 1;
                    cache_v_desc.row_stride_bytes = cache_v_decoded.head_dim * sizeof(float);
                    cache_v_desc.raw_quant = cache_v_storage ? &cache_v_storage->raw_data : nullptr;
                    cache_v_desc.float_data = cache_v_decoded.data;
#if defined(__APPLE__)
                    // Only wrap a persistent MTLBuffer when the cache lives in
                    // its own stable float_data (F32 storage). For non-F32
                    // storage (e.g., F16 with MLC_FP16_KVCACHE) the dequant
                    // produces a buffer in cache_*_decoded that goes out of
                    // scope at the end of this attention call, so caching the
                    // wrap by name would store a stale pointer. The dead GPU
                    // cache path (sharedCacheValid force-pinned false) doesn't
                    // consume the handle today, so this is defensive.
                    if (cache_k_storage && cache_k_storage->dtype == frontend::GGML_TYPE_F32) {
                        cache_k_desc.handle = context->ensureMetalBuffer(cache_k_name, *cache_k_decoded.data);
                    }
                    if (cache_v_storage && cache_v_storage->dtype == frontend::GGML_TYPE_F32) {
                        cache_v_desc.handle = context->ensureMetalBuffer(cache_v_name, *cache_v_decoded.data);
                    }
#endif
                    if (context_length > 0 && num_heads > 0 && head_dim > 0) {
                        std::vector<float> mask;
                        auto mask_it = node.annotations.find("attention_mask");
                        if (mask_it != node.annotations.end()) {
                            const auto* mask_tensor = context->getTensor(mask_it->second);
                            if (mask_tensor) mask = *mask_tensor;
                        }
                        const std::vector<float>* alibi_slopes = nullptr;
                        if (mask.empty() && context && context->graph()) {
                            const auto& cfg = context->graph()->modelConfig();
                            if (cfg.use_alibi && !cfg.alibi_slopes.empty()) {
                                alibi_slopes = &cfg.alibi_slopes;
                            }
                        }
                        // Pre-allocate the output tensor in the context so
                        // its host slot has a stable name for re-resolution
                        // at flush time. runAttention dual-modes: with an
                        // open forward-pass CB it defers result download
                        // and the readback writes into the slot via the
                        // resolver in flushForwardPassCB. Without an open
                        // CB it fills the slot synchronously.
                        std::string out_name = node.outputs.empty() ? std::string() : node.outputs[0];
                        std::vector<float> tmp_out;
                        std::vector<float>* attn_out = &tmp_out;
                        if (!out_name.empty()) {
                            attn_out = &context->allocateTensor(out_name, num_heads * head_dim, /*zero=*/false);
                        }
                        if (MetalExecutor::Instance().runAttention(
                                *q,
                                *k_new,
                                *v_new,
                                num_heads,
                                kv_heads,
                                head_dim,
                                context_length,
                                mask,
                                alibi_slopes,
                                position,
                                rotary_dim,
                                rope_freq_base,
                                rope_freq_scale,
                                cache_k_desc,
                                cache_v_desc,
                                *attn_out,
                                out_name)) {
                            encodeCacheTensor(cache_k_info, cache_k_storage, cache_k_decoded);
                            encodeCacheTensor(cache_v_info, cache_v_storage, cache_v_decoded);
                            result.actual_backend = BackendKind::Metal;
                            result.message = "metal-attention";
                            // If the call deferred onto the open CB, expose
                            // its (fp32) result buffer so the executor can
                            // chain a FromBuffer consumer (residual_1 Add).
                            result.gpu_output_buffer = MetalExecutor::Instance().lastDeferredOutputBuffer();
                            result.gpu_output_element_count =
                                MetalExecutor::Instance().lastDeferredOutputElementCount();
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
            std::string bias_name = getAnnotation(node, "bias");
            const std::vector<float>* bias = nullptr;
            if (!bias_name.empty()) {
                bias = &context->getParameter(bias_name);
                if (bias->size() != a->size()) {
                    result.success = false;
                    result.message = "Add bias size mismatch";
                    return result;
                }
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
            if (bias) {
                for (size_t i = 0; i < sum.size(); ++i) {
                    sum[i] += (*bias)[i];
                }
            }
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
        case ExecOpType::Slice: {
            const std::vector<float>* input = fetchInput(*context, node, 0, result);
            if (!result.success) return result;
            auto attr_off = node.attributes.find("slice_offset");
            auto attr_len = node.attributes.find("slice_length");
            size_t offset = attr_off != node.attributes.end() ? static_cast<size_t>(attr_off->second) : 0;
            size_t length = attr_len != node.attributes.end() ? static_cast<size_t>(attr_len->second) : input->size();
            size_t seq_len = context->seqLen();
            if (seq_len < 1) seq_len = 1;
            size_t per_token_in = input->size() / seq_len;
            if (per_token_in == 0) per_token_in = input->size();
            if (offset + length > per_token_in) {
                result.success = false;
                result.message = "Slice out of range";
                return result;
            }
            std::vector<float> out(seq_len * length);
            for (size_t t = 0; t < seq_len; ++t) {
                std::copy(input->begin() + t * per_token_in + offset,
                          input->begin() + t * per_token_in + offset + length,
                          out.begin() + t * length);
            }
            if (!node.outputs.empty()) {
                context->setTensor(node.outputs[0], std::move(out));
            }
            result.message = "slice";
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
    // Default for the Metal backend: success paths in this function will leave
    // this in place. Any CPU-fallback path returns CpuExecutionBackend::execute()
    // directly, whose result already carries actual_backend = CPU.
    result.actual_backend = BackendKind::Metal;
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
            if (context->seqLen() > 1) {
                return LoopExecutePerToken(*this, node, context, descriptor, {0});
            }
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
            // GGML convention: shape[0]=ne0=cols (input dim), shape[1]=ne1=rows (output dim).
            // Same convention as Session::runLinear in session.cpp:194-195.
            size_t cols = static_cast<size_t>(w_it->second.shape[0]);
            size_t rows = static_cast<size_t>(w_it->second.shape[1]);
            bool transpose_weight = false;
            if (cols != input->size()) {
                if (rows == input->size()) {
                    transpose_weight = true;
                } else {
                    CpuExecutionBackend cpu;
                    return cpu.execute(node, context, descriptor);
                }
            }
            size_t out_rows = transpose_weight ? cols : rows;
            const std::vector<float>* bias_tensor = nullptr;
            std::string bias_name = getAnnotation(node, "bias");
            if (!bias_name.empty()) {
                try {
                    const auto& bias = context->getParameter(bias_name);
                    if (bias.size() != out_rows) {
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
            const auto& raw = context->session().tensorData(w_it->second);
            if (w_it->second.dtype == frontend::GGML_TYPE_F32) {
                if (transpose_weight) {
                    ok = false;
                } else {
                    std::vector<float> weights(raw.size() / sizeof(float));
                    std::memcpy(weights.data(), raw.data(), raw.size());
                    ok = MetalExecutor::Instance().runMatMul(weight,
                                                             weights,
                                                             *input,
                                                             rows,
                                                             cols,
                                                             false,
                                                             output,
                                                             bias_tensor);
                }
            } else {
                size_t row_stride = raw.size() / rows;
                switch (w_it->second.dtype) {
                    case frontend::GGML_TYPE_Q4_0:
                        if (transpose_weight) {
                            ok = MetalExecutor::Instance().runMatMulQ4_0Transposed(
                                weight,
                                raw,
                                *input,
                                rows,
                                cols,
                                row_stride,
                                context->session().loader().quantizationVersion(),
                                output,
                                bias_tensor);
                        } else {
                            ok = MetalExecutor::Instance().runMatMulQ4_0(
                                weight,
                                raw,
                                *input,
                                rows,
                                cols,
                                row_stride,
                                context->session().loader().quantizationVersion(),
                                output,
                                bias_tensor);
                        }
                        break;
                    case frontend::GGML_TYPE_Q4_1:
                        if (transpose_weight) {
                            ok = false;
                            break;
                        }
                        ok = MetalExecutor::Instance().runMatMulQ4_1(weight,
                                                                     raw,
                                                                     *input,
                                                                     rows,
                                                                     cols,
                                                                     row_stride,
                                                                     output,
                                                                     bias_tensor);
                        break;
                    case frontend::GGML_TYPE_Q5_0:
                        if (transpose_weight) {
                            ok = false;
                            break;
                        }
                        ok = MetalExecutor::Instance().runMatMulQ5_0(weight,
                                                                     raw,
                                                                     *input,
                                                                     rows,
                                                                     cols,
                                                                     row_stride,
                                                                     output,
                                                                     bias_tensor);
                        break;
                    case frontend::GGML_TYPE_Q5_1:
                        if (transpose_weight) {
                            ok = false;
                            break;
                        }
                        ok = MetalExecutor::Instance().runMatMulQ5_1(weight,
                                                                     raw,
                                                                     *input,
                                                                     rows,
                                                                     cols,
                                                                     row_stride,
                                                                     output,
                                                                     bias_tensor);
                        break;
                    case frontend::GGML_TYPE_Q2_K:
                        // Q2_K Metal path is currently inaccurate; fall back to CPU for correctness.
                        ok = false;
                        break;
                    case frontend::GGML_TYPE_Q3_K:
                        if (transpose_weight) {
                            ok = false;
                            break;
                        }
                        ok = MetalExecutor::Instance().runMatMulQ3K(weight,
                                                                    raw,
                                                                    *input,
                                                                    rows,
                                                                    cols,
                                                                    row_stride,
                                                                    output,
                                                                    bias_tensor);
                        break;
                    case frontend::GGML_TYPE_Q4_K:
                        if (transpose_weight) {
                            ok = false;
                            break;
                        }
                        ok = MetalExecutor::Instance().runMatMulQ4K(weight,
                                                                    raw,
                                                                    *input,
                                                                    rows,
                                                                    cols,
                                                                    row_stride,
                                                                    output,
                                                                    bias_tensor);
                        break;
                    case frontend::GGML_TYPE_Q5_K:
                        if (transpose_weight) {
                            ok = false;
                            break;
                        }
                        ok = MetalExecutor::Instance().runMatMulQ5K(weight,
                                                                    raw,
                                                                    *input,
                                                                    rows,
                                                                    cols,
                                                                    row_stride,
                                                                    output,
                                                                    bias_tensor);
                        break;
                    case frontend::GGML_TYPE_Q6_K: {
                        static bool enable_q6k_transposed =
                            (std::getenv("MLC_ENABLE_Q6K_TRANSPOSED_METAL") != nullptr);
                        if (transpose_weight) {
                            if (enable_q6k_transposed) {
                                ok = MetalExecutor::Instance().runMatMulQ6KTransposed(
                                    weight,
                                    raw,
                                    *input,
                                    rows,
                                    cols,
                                    row_stride,
                                    output,
                                    bias_tensor);
                            } else {
                                ok = false;
                            }
                            break;
                        }
                        ok = MetalExecutor::Instance().runMatMulQ6K(weight,
                                                                    raw,
                                                                    *input,
                                                                    rows,
                                                                    cols,
                                                                    row_stride,
                                                                    output,
                                                                    bias_tensor);
                        break;
                    }
                    case frontend::GGML_TYPE_Q8_K:
                        if (transpose_weight) {
                            ok = false;
                            break;
                        }
                        ok = MetalExecutor::Instance().runMatMulQ8K(weight,
                                                                    raw,
                                                                    *input,
                                                                    rows,
                                                                    cols,
                                                                    row_stride,
                                                                    output,
                                                                    bias_tensor);
                        break;
                    case frontend::GGML_TYPE_Q8_0:
                        if (transpose_weight) {
                            ok = false;
                            break;
                        }
                        ok = MetalExecutor::Instance().runMatMulQ8_0(weight,
                                                                     raw,
                                                                     *input,
                                                                     rows,
                                                                     cols,
                                                                     row_stride,
                                                                     output,
                                                                     bias_tensor);
                        break;
                    case frontend::GGML_TYPE_Q8_1:
                        if (transpose_weight) {
                            ok = false;
                            break;
                        }
                        ok = MetalExecutor::Instance().runMatMulQ8_1(weight,
                                                                     raw,
                                                                     *input,
                                                                     rows,
                                                                     cols,
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
            if (context->seqLen() > 1) {
                return LoopExecutePerToken(*this, node, context, descriptor, {0});
            }
            const std::vector<float>* input = fetchInput(*context, node, 0, result);
            if (!result.success) return result;
            std::string weight_name = getAnnotation(node, "weight");
            std::string bias_name = getAnnotation(node, "bias");
            std::string norm_kind = getAnnotation(node, "norm_kind");
            if (weight_name.empty()) {
                CpuExecutionBackend cpu;
                return cpu.execute(node, context, descriptor);
            }
            try {
                const auto& weight = context->getParameter(weight_name);
                const std::vector<float>* bias_ptr = nullptr;
                if (!bias_name.empty()) {
                    bias_ptr = &context->getParameter(bias_name);
                    if (bias_ptr->size() != input->size()) {
                        CpuExecutionBackend cpu;
                        return cpu.execute(node, context, descriptor);
                    }
                }
                std::vector<float> output;
                bool ok = false;
                if (norm_kind == "layer") {
                    ok = MetalExecutor::Instance().runLayerNorm(*input, weight, bias_ptr, 1e-5f, output);
                } else {
                    ok = MetalExecutor::Instance().runRmsNorm(*input, weight, 1e-5f, output);
                    if (ok && bias_ptr) {
                        for (size_t i = 0; i < output.size(); ++i) output[i] += (*bias_ptr)[i];
                    }
                }
                if (!ok) {
                    CpuExecutionBackend cpu;
                    return cpu.execute(node, context, descriptor);
                }
                if (!node.outputs.empty()) {
                    context->setTensor(node.outputs[0], std::move(output));
                }
                result.message = (norm_kind == "layer") ? "metal-layer-norm" : "metal-norm";
                return result;
            } catch (const std::exception&) {
                CpuExecutionBackend cpu;
                return cpu.execute(node, context, descriptor);
            }
        }
        case ExecOpType::Add: {
            if (context->seqLen() > 1) {
                return LoopExecutePerToken(*this, node, context, descriptor, {0, 1});
            }
            const std::vector<float>* a = fetchInput(*context, node, 0, result);
            if (!result.success) return result;
            const std::vector<float>* b = fetchInput(*context, node, 1, result);
            if (!result.success) return result;
            if (a->size() != b->size()) {
                CpuExecutionBackend cpu;
                return cpu.execute(node, context, descriptor);
            }
            std::string bias_name = getAnnotation(node, "bias");
            const std::vector<float>* bias = nullptr;
            if (!bias_name.empty()) {
                try {
                    bias = &context->getParameter(bias_name);
                    if (bias->size() != a->size()) {
                        CpuExecutionBackend cpu;
                        return cpu.execute(node, context, descriptor);
                    }
                } catch (const std::exception&) {
                    CpuExecutionBackend cpu;
                    return cpu.execute(node, context, descriptor);
                }
            }
            std::vector<float> output;
            if (!MetalExecutor::Instance().runAdd(*a, *b, output, bias)) {
                CpuExecutionBackend cpu;
                return cpu.execute(node, context, descriptor);
            }
            if (!node.outputs.empty()) {
                context->setTensor(node.outputs[0], std::move(output));
            }
            result.message = bias ? "metal-add-bias" : "metal-add";
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

BackendExecutionResult MetalExecutionBackend::encode(const ExecutionNode& node,
                                                     ExecutionContext* context,
                                                     const KernelDescriptor* descriptor,
                                                     const FusionInputs& fusion) const {
    // Deferred-commit dispatch onto MetalExecutor's open fusion command
    // buffer. The executor pre-allocates the output buffer from the
    // intermediate buffer pool and passes it via fusion.output_buffer.
    // When fusion.primary_input_buffer is non-null, the previous op's
    // output is GPU-resident — dispatch to the FromBuffer variant and
    // skip the host upload.
    BackendExecutionResult result;
    result.actual_backend = BackendKind::Metal;
    if (descriptor) result.kernel_id = descriptor->id;
    else if (!node.kernel_id.empty()) result.kernel_id = node.kernel_id;
    if (!context) {
        result.success = false; result.message = "encode: no context"; return result;
    }
    if (!MetalExecutor::Instance().hasForwardPassCB()) {
        result.success = false; result.message = "encode: no fusion window open"; return result;
    }
    if (!fusion.output_buffer || !fusion.host_dst) {
        result.success = false; result.message = "encode: missing fusion output_buffer/host_dst"; return result;
    }

    switch (node.op) {
        case ExecOpType::MatMul:
        case ExecOpType::Linear: {
            std::string weight = getAnnotation(node, "weight");
            if (weight.empty()) {
                result.success = false; result.message = "encode: no weight name"; return result;
            }
            const auto& tensors = context->session().loader().tensors();
            auto w_it = tensors.find(weight);
            if (w_it == tensors.end() || w_it->second.shape.size() != 2) {
                result.success = false; result.message = "encode: weight not found"; return result;
            }
            if (w_it->second.dtype != frontend::GGML_TYPE_Q4_0) {
                result.success = false; result.message = "encode: dtype not supported"; return result;
            }
            size_t cols = static_cast<size_t>(w_it->second.shape[0]);
            size_t rows = static_cast<size_t>(w_it->second.shape[1]);
            // Bias on matmul: TinyLlama doesn't carry it; reject so executor
            // can fall back to execute() if some future model does.
            if (!getAnnotation(node, "bias").empty()) {
                result.success = false; result.message = "encode: matmul with bias not supported"; return result;
            }
            const auto& raw = context->session().tensorData(w_it->second);
            size_t row_stride = raw.size() / rows;
            uint32_t qv = context->session().loader().quantizationVersion();
            bool ok = false;
            if (fusion.primary_input_buffer) {
                if (fusion.primary_input_count != cols) {
                    result.success = false; result.message = "encode: gpu input count != cols"; return result;
                }
                ok = MetalExecutor::Instance().encodeMatMulQ4_0FromBuffer(
                    weight, raw, fusion.primary_input_buffer, fusion.primary_input_count,
                    rows, cols, row_stride, qv,
                    fusion.output_buffer, fusion.host_dst, fusion.needs_host_output);
            } else {
                const std::vector<float>* input = fetchInput(*context, node, 0, result);
                if (!result.success) return result;
                if (input->size() != cols) {
                    result.success = false; result.message = "encode: shape mismatch (transposed not supported)"; return result;
                }
                ok = MetalExecutor::Instance().encodeMatMulQ4_0FromHost(
                    weight, raw, *input, rows, cols, row_stride, qv,
                    fusion.output_buffer, fusion.host_dst, fusion.needs_host_output);
            }
            if (!ok) {
                result.success = false; result.message = "encode: encodeMatMulQ4_0 failed"; return result;
            }
            result.message = "metal-matmul-encoded";
            return result;
        }
        case ExecOpType::Norm: {
            std::string weight_name = getAnnotation(node, "weight");
            std::string bias_name = getAnnotation(node, "bias");
            std::string norm_kind = getAnnotation(node, "norm_kind");
            if (weight_name.empty() || norm_kind == "layer") {
                result.success = false; result.message = "encode: norm kind/weight unsupported"; return result;
            }
            if (!bias_name.empty()) {
                result.success = false; result.message = "encode: norm with bias not supported"; return result;
            }
            try {
                const auto& weight = context->getParameter(weight_name);
                bool ok = false;
                if (fusion.primary_input_buffer) {
                    if (fusion.primary_input_count != weight.size()) {
                        result.success = false; result.message = "encode: norm gpu input count != weight size"; return result;
                    }
                    ok = MetalExecutor::Instance().encodeRmsNormFromBuffer(
                        fusion.primary_input_buffer, fusion.primary_input_count,
                        weight, 1e-5f,
                        fusion.output_buffer, fusion.host_dst, fusion.needs_host_output);
                } else {
                    const std::vector<float>* input = fetchInput(*context, node, 0, result);
                    if (!result.success) return result;
                    ok = MetalExecutor::Instance().encodeRmsNormFromHost(
                        *input, weight, 1e-5f,
                        fusion.output_buffer, fusion.host_dst, fusion.needs_host_output);
                }
                if (!ok) {
                    result.success = false; result.message = "encode: encodeRmsNorm failed"; return result;
                }
                result.message = "metal-norm-encoded";
                return result;
            } catch (const std::exception&) {
                result.success = false; result.message = "encode: norm param missing"; return result;
            }
        }
        case ExecOpType::Add: {
            if (node.inputs.size() < 2) {
                result.success = false; result.message = "encode: add needs 2 inputs"; return result;
            }
            // Bias on Add: same story, TinyLlama doesn't carry it.
            if (!getAnnotation(node, "bias").empty()) {
                result.success = false; result.message = "encode: add with bias not supported"; return result;
            }
            // Determine element count from whichever side we have a host vec
            // for, OR from the GPU buffer counts if both are GPU-resident.
            const std::vector<float>* host_a = nullptr;
            const std::vector<float>* host_b = nullptr;
            size_t element_count = 0;
            if (fusion.primary_input_buffer) {
                element_count = fusion.primary_input_count;
            } else {
                host_a = fetchInput(*context, node, 0, result);
                if (!result.success) return result;
                element_count = host_a->size();
            }
            if (fusion.secondary_input_buffer) {
                if (fusion.secondary_input_count != element_count) {
                    result.success = false; result.message = "encode: add gpu input count mismatch"; return result;
                }
            } else {
                host_b = fetchInput(*context, node, 1, result);
                if (!result.success) return result;
                if (host_b->size() != element_count) {
                    result.success = false; result.message = "encode: add size mismatch"; return result;
                }
            }
            bool ok = MetalExecutor::Instance().encodeAddMixed(
                host_a, fusion.primary_input_buffer,
                host_b, fusion.secondary_input_buffer,
                element_count,
                fusion.output_buffer, fusion.host_dst, fusion.needs_host_output);
            if (!ok) {
                result.success = false; result.message = "encode: encodeAdd failed"; return result;
            }
            result.message = "metal-add-encoded";
            return result;
        }
        default:
            result.success = false;
            result.message = "encode: op type not supported";
            return result;
    }
}

BackendRegistry::BackendRegistry() = default;

const ExecutionBackend& BackendRegistry::backendFor(BackendKind kind) const {
    if (KernelDescriptorRegistry::forceCpu()) {
        return cpu_backend_;
    }

    switch (kind) {
        case BackendKind::CPU:
            return cpu_backend_;
        case BackendKind::Metal:
            return metal_backend_;
        case BackendKind::Auto:
        default:
            // Prefer Metal when available; fall back to CPU otherwise.
            if (MetalExecutor::Instance().isAvailable()) {
                return metal_backend_;
            }
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
