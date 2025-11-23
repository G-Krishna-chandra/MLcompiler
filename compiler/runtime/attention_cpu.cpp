#include "runtime/attention_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#include "runtime/operator_backend.hpp"
#include "frontends/ggml_types.hpp"
#include "runtime/quant_utils.hpp"
#include "runtime/quantization.hpp"

namespace mlc {
namespace runtime {
namespace {

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

void applyRotaryToBuffer(float* data,
                         size_t head_dim,
                         size_t rotary_dim,
                         const std::vector<float>& cos,
                         const std::vector<float>& sin) {
    if (!data || head_dim == 0 || rotary_dim == 0) return;
    rotary_dim = std::min(rotary_dim, head_dim);
    size_t pairs = rotary_dim / 2;
    for (size_t i = 0; i < pairs; ++i) {
        float c = (i < cos.size()) ? cos[i] : 1.0f;
        float s = (i < sin.size()) ? sin[i] : 0.0f;
        float x0 = data[2 * i];
        float x1 = data[2 * i + 1];
        data[2 * i] = x0 * c - x1 * s;
        data[2 * i + 1] = x0 * s + x1 * c;
    }
}

#if defined(__AVX512F__)
float dotProductSimd(const float* lhs, const float* rhs, size_t dim) {
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
}
#elif defined(__AVX2__)
float dotProductSimd(const float* lhs, const float* rhs, size_t dim) {
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
}
#endif

float dotProductScalar(const float* lhs, const float* rhs, size_t dim) {
    float acc = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        acc += lhs[i] * rhs[i];
    }
    return acc;
}

float dotProduct(const float* lhs, const float* rhs, size_t dim) {
#if defined(__AVX2__) || defined(__AVX512F__)
    return dotProductSimd(lhs, rhs, dim);
#else
    return dotProductScalar(lhs, rhs, dim);
#endif
}

struct CacheAccessor {
    TensorStorage* storage;
    size_t head_dim;
    size_t context_length;

    CacheAccessor(TensorStorage* s, size_t hd, size_t ctx)
        : storage(s), head_dim(hd), context_length(ctx) {}

    size_t rowOffset(size_t row) const {
        if (!storage) return 0;
        return row * storage->row_stride_bytes;
    }

    void readRow(size_t kv_head, size_t position, std::vector<float>& buffer) const {
        size_t row = kv_head * context_length + position;
        buffer.resize(head_dim);
        if (storage->dtype == frontend::GGML_TYPE_F32) {
            const float* src = storage->float_data.data() + row * head_dim;
            std::copy(src, src + head_dim, buffer.begin());
            return;
        }
        const uint8_t* ptr = storage->raw_data.data() + rowOffset(row);
        dequantizeRowTo(ptr, storage->dtype, head_dim, storage->quant_version, buffer.data());
    }

    void writeRow(size_t kv_head, size_t position, const float* values) {
        size_t row = kv_head * context_length + position;
        if (storage->dtype == frontend::GGML_TYPE_F32) {
            float* dst = storage->float_data.data() + row * head_dim;
            std::memcpy(dst, values, head_dim * sizeof(float));
            return;
        }
        uint8_t* dst = storage->raw_data.data() + rowOffset(row);
        quantizeRowFrom(values, storage->dtype, head_dim, storage->quant_version, dst);
    }
};

struct AttentionContext {
    ExecutionContext* runtime_context;
    const ExecutionNode& node;
    const std::vector<float>& q;
    const std::vector<float>& k_new;
    const std::vector<float>& v_new;
    size_t num_heads;
    size_t kv_heads;
    size_t head_dim;
    size_t context_length;
    size_t base_position;
    bool use_rotary;
    size_t rotary_dim;
    float rope_freq_base;
    float rope_freq_scale;
    std::vector<float> mask;
    bool apply_mask = false;
};

bool loadMask(const ExecutionNode& node,
              ExecutionContext* context,
              std::vector<float>& mask) {
    auto it = node.annotations.find("attention_mask");
    if (it == node.annotations.end()) return false;
    const auto* tensor = context->getTensor(it->second);
    if (!tensor) return false;
    mask = *tensor;
    return true;
}

std::vector<float> rotateVector(const float* data,
                                size_t length,
                                size_t head_dim,
                                size_t rotary_dim,
                                size_t position,
                                float freq_base,
                                float freq_scale) {
    std::vector<float> rotated(length);
    std::memcpy(rotated.data(), data, length * sizeof(float));
    std::vector<float> cos;
    std::vector<float> sin;
    computeRotaryCoefficients(position,
                               rotary_dim,
                               freq_base,
                               freq_scale,
                               cos,
                               sin);
    size_t num_heads = length / head_dim;
    for (size_t head = 0; head < num_heads; ++head) {
        float* ptr = rotated.data() + head * head_dim;
        applyRotaryToBuffer(ptr, head_dim, rotary_dim, cos, sin);
    }
    return rotated;
}

} // namespace

BackendExecutionResult RunAttentionCPU(const ExecutionNode& node,
                                       ExecutionContext* context,
                                       const std::vector<float>& q,
                                       const std::vector<float>& k_new,
                                       const std::vector<float>& v_new) {
    BackendExecutionResult result;
    if (!context) {
        result.success = false;
        result.message = "CPU attention requires execution context";
        return result;
    }

    size_t num_heads = node.attributes.count("heads")
                           ? static_cast<size_t>(node.attributes.at("heads"))
                           : 0;
    size_t head_dim = node.attributes.count("head_dim")
                          ? static_cast<size_t>(node.attributes.at("head_dim"))
                          : 0;
    if (num_heads == 0 && head_dim > 0) {
        num_heads = q.size() / head_dim;
    }
    if (head_dim == 0 && num_heads > 0) {
        head_dim = q.size() / std::max<size_t>(size_t(1), num_heads);
    }
    if (num_heads == 0 || head_dim == 0 || q.size() % (num_heads * head_dim) != 0) {
        result.success = false;
        result.message = "Invalid attention head configuration";
        return result;
    }
    size_t kv_heads = node.attributes.count("kv_heads")
                          ? static_cast<size_t>(node.attributes.at("kv_heads"))
                          : num_heads;
    if (kv_heads == 0) kv_heads = num_heads;
    size_t kv_span = kv_heads * head_dim;
    if (k_new.size() % kv_span != 0 || v_new.size() % kv_span != 0) {
        result.success = false;
        result.message = "KV input size mismatch for attention";
        return result;
    }

    size_t tokens_q = q.size() / (num_heads * head_dim);
    size_t tokens_k = k_new.size() / kv_span;
    size_t tokens_v = v_new.size() / kv_span;
    size_t tokens_available = std::min({tokens_q, tokens_k, tokens_v});
    if (tokens_available == 0) tokens_available = 1;

    const ExecutionTensor* cache_info = nullptr;
    std::string cache_k_name = node.annotations.count("kv_cache_k") ? node.annotations.at("kv_cache_k") : "";
    std::string cache_v_name = node.annotations.count("kv_cache_v") ? node.annotations.at("kv_cache_v") : "";
    if (cache_k_name.empty() || cache_v_name.empty()) {
        result.success = false;
        result.message = "KV cache annotations missing";
        return result;
    }
    cache_info = context->tensorInfo(cache_k_name);
    size_t context_length = 1;
    if (cache_info && cache_info->shape.size() >= 2) {
        context_length = static_cast<size_t>(std::max<int64_t>(1, cache_info->shape[1]));
    }
    size_t base_position = context->sequencePosition();
    if (context_length == 0) context_length = 1;
    if (base_position >= context_length) {
        base_position = context_length - 1;
    }
    tokens_available = std::min(tokens_available, context_length - base_position);
    if (tokens_available == 0) tokens_available = 1;

    TensorStorage* cache_k_storage = context->tensorStorage(cache_k_name);
    TensorStorage* cache_v_storage = context->tensorStorage(cache_v_name);
    if (!cache_k_storage || !cache_v_storage) {
        result.success = false;
        result.message = "KV cache tensors unavailable";
        return result;
    }

    CacheAccessor cache_k(cache_k_storage, head_dim, context_length);
    CacheAccessor cache_v(cache_v_storage, head_dim, context_length);

    size_t rotary_dim = 0;
    float rope_freq_base = 10000.0f;
    float rope_freq_scale = 1.0f;
    if (context->graph()) {
        const auto& cfg = context->graph()->modelConfig();
        if (cfg.rotary_dim > 0) {
            rotary_dim = std::min(cfg.rotary_dim, head_dim);
            if (cfg.rope_freq_base > 0.0f) rope_freq_base = cfg.rope_freq_base;
            if (cfg.rope_freq_scale > 0.0f) rope_freq_scale = cfg.rope_freq_scale;
        }
    }
    bool use_rotary = (rotary_dim > 0 && head_dim >= 2);

    std::vector<float> mask;
    bool apply_mask = loadMask(node, context, mask);

    std::vector<float> output(tokens_available * num_heads * head_dim, 0.0f);
    std::vector<float> decoded_row(head_dim, 0.0f);
    std::vector<float> logits;
    std::vector<float> attention;
    std::vector<float> accum(head_dim, 0.0f);

    for (size_t token_idx = 0; token_idx < tokens_available; ++token_idx) {
        const float* q_ptr = q.data() + token_idx * num_heads * head_dim;
        const float* k_ptr = k_new.data() + token_idx * kv_span;
        const float* v_ptr = v_new.data() + token_idx * kv_span;

        std::vector<float> q_rotated;
        std::vector<float> k_rotated;
        const float* q_data = q_ptr;
        const float* k_data = k_ptr;
        size_t position = std::min(context_length - 1, base_position + token_idx);
        if (use_rotary) {
            q_rotated = rotateVector(q_ptr, num_heads * head_dim, head_dim, rotary_dim, position, rope_freq_base, rope_freq_scale);
            k_rotated = rotateVector(k_ptr, kv_heads * head_dim, head_dim, rotary_dim, position, rope_freq_base, rope_freq_scale);
            q_data = q_rotated.data();
            k_data = k_rotated.data();
        }

        for (size_t kv = 0; kv < kv_heads; ++kv) {
            cache_k.writeRow(kv, position, k_data + kv * head_dim);
            cache_v.writeRow(kv, position, v_ptr + kv * head_dim);
        }

        size_t tokens_in_cache = std::min(context_length, position + 1);
        logits.assign(tokens_in_cache, 0.0f);
        attention.assign(tokens_in_cache, 0.0f);
        float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(head_dim));

        for (size_t head = 0; head < num_heads; ++head) {
            size_t kv_index = std::min(kv_heads - 1, head * kv_heads / std::max<size_t>(size_t(1), num_heads));
            const float* q_head = q_data + head * head_dim;
            for (size_t t = 0; t < tokens_in_cache; ++t) {
                cache_k.readRow(kv_index, t, decoded_row);
                logits[t] = dotProduct(q_head, decoded_row.data(), head_dim) * inv_sqrt;
                if (apply_mask && t < mask.size()) {
                    logits[t] += mask[t];
                }
            }
            softmax(logits);
            attention.assign(logits.begin(), logits.end());
            std::fill(accum.begin(), accum.end(), 0.0f);
            for (size_t t = 0; t < tokens_in_cache; ++t) {
                cache_v.readRow(kv_index, t, decoded_row);
                for (size_t d = 0; d < head_dim; ++d) {
                    accum[d] += attention[t] * decoded_row[d];
                }
            }
            size_t out_index = token_idx * num_heads * head_dim + head * head_dim;
            std::copy(accum.begin(), accum.end(), output.begin() + out_index);
        }
    }

    if (!node.outputs.empty()) {
        context->setTensor(node.outputs[0], std::move(output));
    }
    result.message = "attention-cpu";
    return result;
}

} // namespace runtime
} // namespace mlc
