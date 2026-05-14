#include "runtime/metal_runtime.hpp"
#include "runtime/execution_graph.hpp"
#include "runtime/float_convert.hpp"
#include "runtime/kernel_registry.hpp"
#include "runtime/quantization.hpp"
#include "runtime/quant_utils.hpp"
#include "frontends/ggml_types.hpp"

#if defined(__APPLE__)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>

#if defined(__APPLE__)
namespace {
static const char* kMetalKernelsSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void feedforward_silu_mul(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= length) return;
    float g = gate[index];
    float silu = g / (1.0f + exp(-g));
    output[index] = silu * up[index];
}

kernel void add_vectors(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= length) return;
    out[index] = a[index] + b[index];
}

kernel void add_residual_bias(
    device const float* a [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& length [[buffer(4)]],
    constant uint& has_bias [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= length) return;
    float v = a[index] + residual[index];
    if (has_bias) {
        v += bias[index];
    }
    out[index] = v;
}

kernel void add_bias_strided(
    device float* data [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    constant uint& stride_elems [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= length || stride_elems == 0) return;
    uint offset = index * stride_elems;
    data[offset] += bias[index];
}

kernel void rms_norm_kernel(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index > 0) return;
    if (length == 0) return;
    float sum = 0.0f;
    for (uint i = 0; i < length; ++i) {
        float v = input[i];
        sum += v * v;
    }
    float denom = static_cast<float>(length);
    float inv = rsqrt((sum / fmax(1.0f, denom)) + epsilon);
    for (uint i = 0; i < length; ++i) {
        float gamma = weight ? weight[i] : 1.0f;
        output[i] = input[i] * inv * gamma;
    }
}

kernel void layer_norm_kernel(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& length [[buffer(4)]],
    constant float& epsilon [[buffer(5)]],
    constant uint& has_bias [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    if (index > 0 || length == 0) return;
    float sum = 0.0f;
    for (uint i = 0; i < length; ++i) {
        sum += input[i];
    }
    float mean = sum / fmax(1.0f, static_cast<float>(length));
    float var = 0.0f;
    for (uint i = 0; i < length; ++i) {
        float d = input[i] - mean;
        var += d * d;
    }
    float inv_std = rsqrt((var / fmax(1.0f, static_cast<float>(length))) + epsilon);
    for (uint i = 0; i < length; ++i) {
        float gamma = weight ? weight[i] : 1.0f;
        float beta = (has_bias && bias) ? bias[i] : 0.0f;
        output[i] = (input[i] - mean) * inv_std * gamma + beta;
    }
}

kernel void vector_softmax(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index > 0 || length == 0) return;
    float max_val = input[0];
    for (uint i = 1; i < length; ++i) {
        max_val = fmax(max_val, input[i]);
    }
    float sum = 0.0f;
    for (uint i = 0; i < length; ++i) {
        float val = exp(input[i] - max_val);
        output[i] = val;
        sum += val;
    }
    float inv = sum > 0.0f ? (1.0f / sum) : 0.0f;
    for (uint i = 0; i < length; ++i) {
        output[i] *= inv;
    }
}

struct Q4_0Params {
    uint rows;
    uint cols;
    uint row_stride;
    uint quant_version;
};

struct KVWriteParams {
    uint head_dim;
    uint kv_heads;
    uint kv_stride;
    uint tokens;
    uint base_pos;
};

kernel void scatter_kv(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant KVWriteParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    uint total = params.head_dim * params.kv_heads * params.tokens;
    if (gid >= total) return;
    uint dim = gid % params.head_dim;
    uint tmp = gid / params.head_dim;
    uint token = tmp % params.tokens;
    uint kv = tmp / params.tokens;
    uint dst_index = kv * params.kv_stride +
                     (params.base_pos + token) * params.head_dim +
                     dim;
    dst[dst_index] = src[gid];
}

kernel void q4_0_matmul(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant Q4_0Params& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.rows) return;
    const device uchar* row = weights + gid * params.row_stride;
    const uint block_size = 18;
    uint cols = params.cols;
    uint blocks = (cols + 31u) / 32u;
    float acc = 0.0f;
    for (uint b = 0; b < blocks; ++b) {
        const device half* d_half = reinterpret_cast<const device half*>(row + b * block_size);
        float d = static_cast<float>(d_half[0]);
        const device uchar* qs = row + b * block_size + 2;
        uint base = b * 32u;
        for (uint j = 0; j < 16; ++j) {
            uchar byte = qs[j];
            int lo = static_cast<int>(byte & 0x0F) - 8;
            int hi = static_cast<int>((byte >> 4) & 0x0F) - 8;
            uint i0 = base + j;
            uint i1 = base + j + 16u;
            if (i0 < cols) acc += d * static_cast<float>(lo) * input[i0];
            if (i1 < cols) acc += d * static_cast<float>(hi) * input[i1];
        }
    }
    output[gid] = acc;
}

kernel void q4_0_matmul_transposed(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant Q4_0Params& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.cols) return;
    uint col = gid;
    const uint block_size = 18;
    uint block = col / 32u;
    uint offset = col % 32u;
    float acc = 0.0f;
    for (uint r = 0; r < params.rows; ++r) {
        const device uchar* row = weights + r * params.row_stride;
        const device half* d_half = reinterpret_cast<const device half*>(row + block * block_size);
        float d = static_cast<float>(d_half[0]);
        const device uchar* qs = row + block * block_size + 2;
        uchar byte;
        int q;
        if (offset < 16u) {
            byte = qs[offset];
            q = static_cast<int>(byte & 0x0F);
        } else {
            byte = qs[offset - 16u];
            q = static_cast<int>((byte >> 4) & 0x0F);
        }
        q -= 8;
        acc += d * static_cast<float>(q) * input[r];
    }
    output[col] = acc;
}

kernel void q4_1_matmul(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant Q4_0Params& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.rows) return;
    const device uchar* row = weights + gid * params.row_stride;
    const uint block_size = 20;
    uint cols = params.cols;
    uint blocks = (cols + 31u) / 32u;
    float acc = 0.0f;
    uint col_index = 0;
    for (uint b = 0; b < blocks && col_index < cols; ++b) {
        const device half* meta = reinterpret_cast<const device half*>(row + b * block_size);
        float d = static_cast<float>(meta[0]);
        float m = static_cast<float>(meta[1]);
        const device uchar* qs = row + b * block_size + 4;
        for (uint j = 0; j < 16 && col_index < cols; ++j) {
            uchar byte = qs[j];
            int x0 = byte & 0x0F;
            acc += (x0 * d + m) * input[col_index++];
            if (col_index < cols) {
                int x1 = (byte >> 4) & 0x0F;
                acc += (x1 * d + m) * input[col_index++];
            }
        }
    }
    output[gid] = acc;
}

kernel void q5_0_matmul(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant Q4_0Params& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.rows) return;
    const device uchar* row = weights + gid * params.row_stride;
    const uint block_size = 22;
    uint cols = params.cols;
    uint blocks = (cols + 31u) / 32u;
    float acc = 0.0f;
    uint col_index = 0;
    for (uint b = 0; b < blocks && col_index < cols; ++b) {
        const device half* meta = reinterpret_cast<const device half*>(row + b * block_size);
        float d = static_cast<float>(meta[0]);
        const device uchar* qh_ptr = row + b * block_size + 2;
        uint32_t qh = 0;
        // Assemble manually to avoid alignment pitfalls.
        qh |= static_cast<uint32_t>(qh_ptr[0]);
        qh |= static_cast<uint32_t>(qh_ptr[1]) << 8;
        qh |= static_cast<uint32_t>(qh_ptr[2]) << 16;
        qh |= static_cast<uint32_t>(qh_ptr[3]) << 24;
        const device uchar* qs = row + b * block_size + 6;
        for (uint j = 0; j < 16 && col_index < cols; ++j) {
            uchar byte = qs[j];
            uint8_t xh0 = static_cast<uint8_t>(((qh >> (j + 0)) << 4) & 0x10);
            uint8_t xh1 = static_cast<uint8_t>(((qh >> (j + 12))      ) & 0x10);
            int x0 = ((byte & 0x0F) | xh0) - 16;
            acc += static_cast<float>(x0) * d * input[col_index++];
            if (col_index < cols) {
                int x1 = ((byte >> 4) | xh1) - 16;
                acc += static_cast<float>(x1) * d * input[col_index++];
            }
        }
    }
    output[gid] = acc;
}

kernel void fused_bias_relu_matmul(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant Q4_0Params& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.rows) return;
    float acc = 0.0f;
    const device float* w = reinterpret_cast<const device float*>(weights + gid * params.row_stride);
    for (uint c = 0; c < params.cols; ++c) {
        acc += w[c] * input[c];
    }
    acc += bias ? bias[gid] : 0.0f;
    output[gid] = acc > 0.0f ? acc : 0.0f;
}

kernel void q5_1_matmul(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant Q4_0Params& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.rows) return;
    const device uchar* row = weights + gid * params.row_stride;
    const uint block_size = 24;
    uint cols = params.cols;
    uint blocks = (cols + 31u) / 32u;
    float acc = 0.0f;
    uint col_index = 0;
    for (uint b = 0; b < blocks && col_index < cols; ++b) {
        const device half* meta = reinterpret_cast<const device half*>(row + b * block_size);
        float d = static_cast<float>(meta[0]);
        float m = static_cast<float>(meta[1]);
        const device uchar* qh_ptr = row + b * block_size + 4;
        uint32_t qh = *(const device uint32_t*)(qh_ptr);
        const device uchar* qs = row + b * block_size + 8;
        for (uint j = 0; j < 16 && col_index < cols; ++j) {
            uchar byte = qs[j];
            uint8_t xh0 = static_cast<uint8_t>(((qh >> (2 * j)) & 0x1) << 4);
            uint8_t xh1 = static_cast<uint8_t>(((qh >> (2 * j + 1)) & 0x1) << 4);
            int x0 = (byte & 0x0F) | xh0;
            acc += (x0 * d + m) * input[col_index++];
            if (col_index < cols) {
                int x1 = ((byte >> 4) & 0x0F) | xh1;
                acc += (x1 * d + m) * input[col_index++];
            }
        }
    }
    output[gid] = acc;
}

struct QKParams {
    uint rows;
    uint cols;
    uint row_stride;
};

inline void getScaleMinK4(uint index,
                          const device uchar* data,
                          thread uchar& scale,
                          thread uchar& minv) {
    if (index < 4) {
        scale = data[index] & 63;
        minv = data[index + 4] & 63;
    } else {
        scale = static_cast<uchar>((data[index + 4] & 0xF) | ((data[index - 4] >> 6) << 4));
        minv = static_cast<uchar>((data[index + 4] >> 4) | ((data[index] >> 6) << 4));
    }
}

kernel void q2_k_matmul(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant QKParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.rows) return;
    const device uchar* row = weights + gid * params.row_stride;
    const uint block_size = 84;
    uint cols = params.cols;
    uint blocks = (cols + 255u) / 256u;
    float acc = 0.0f;
    uint col_index = 0;
    for (uint b = 0; b < blocks && col_index < cols; ++b) {
        const device uchar* block = row + b * block_size;
        const device uchar* scales = block;          // 16 bytes
        const device uchar* qs = block + 16;         // 64 bytes
        const device half* d_ptr = reinterpret_cast<const device half*>(block + 80);
        float d = static_cast<float>(d_ptr[0]);
        float m = static_cast<float>(d_ptr[1]);
        int is = 0;
        const uint total = min(256u, cols - col_index);
        for (uint n = 0; n < total; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4 && (col_index + n + j * 32) < cols; ++j) {
                uint8_t sc = scales[is++];
                float dl = d * static_cast<float>(sc & 0xF);
                float ml = m * static_cast<float>(sc >> 4);
                for (int l = 0; l < 16 && (col_index + n + j * 32 + l) < cols; ++l) {
                    int8_t val = static_cast<int8_t>((qs[l] >> shift) & 0x3);
                    float vf = dl * static_cast<float>(val) - ml;
                    acc += vf * input[col_index + n + j * 32 + l];
                }
                sc = scales[is++];
                dl = d * static_cast<float>(sc & 0xF);
                ml = m * static_cast<float>(sc >> 4);
                for (int l = 0; l < 16 && (col_index + n + j * 32 + 16 + l) < cols; ++l) {
                    int8_t val = static_cast<int8_t>((qs[l + 16] >> shift) & 0x3);
                    float vf = dl * static_cast<float>(val) - ml;
                    acc += vf * input[col_index + n + j * 32 + 16 + l];
                }
                shift += 2;
            }
            qs += 32;
        }
        col_index += 256;
    }
    output[gid] = acc;
}

// Dequantize QK (generic) row into float buffer.
kernel void dequant_qk_row(
    device const uchar* weights [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant QKParams& params [[buffer(2)]],
    constant uint& dtype [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.rows) return;
    const device uchar* row = weights + gid * params.row_stride;
    uint cols = params.cols;
    // Only one block per row for head_dim == cols; process sequentially.
    // Handles Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K.
    switch (dtype) {
    case 10: { // Q2_K
        const uint block_size = 84;
        const device uchar* block = row;
        const device uchar* scales = block;          // 16 bytes
        const device uchar* qs = block + 16;         // 64 bytes
        const device half* d_ptr = reinterpret_cast<const device half*>(block + 80);
        float d = static_cast<float>(d_ptr[0]);
        float m = static_cast<float>(d_ptr[1]);
        uint out_idx = gid * cols;
        int is = 0;
        for (uint n = 0; n < cols; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4 && n + j * 32 < cols; ++j) {
                uint8_t sc = scales[is++];
                float dl = d * static_cast<float>(sc & 0xF);
                float ml = m * static_cast<float>(sc >> 4);
                for (int l = 0; l < 16 && n + j * 32 + l < cols; ++l) {
                    int8_t val = static_cast<int8_t>((qs[l] >> shift) & 0x3);
                    output[out_idx + n + j * 32 + l] = dl * static_cast<float>(val) - ml;
                }
                sc = scales[is++];
                dl = d * static_cast<float>(sc & 0xF);
                ml = m * static_cast<float>(sc >> 4);
                for (int l = 0; l < 16 && n + j * 32 + 16 + l < cols; ++l) {
                    int8_t val = static_cast<int8_t>((qs[l + 16] >> shift) & 0x3);
                    output[out_idx + n + j * 32 + 16 + l] = dl * static_cast<float>(val) - ml;
                }
                shift += 2;
            }
            qs += 32;
        }
    } break;
    case 11: { // Q3_K
        const uint block_size = 110;
        const device uchar* block = row;
        const device uchar* hmask = block;
        const device uchar* qs = block + 32;
        const device uchar* scale_ptr = block + 96;
        uint32_t aux[4];
        aux[0] = *(const device uint32_t*)(scale_ptr + 0);
        aux[1] = *(const device uint32_t*)(scale_ptr + 4);
        aux[2] = *(const device uint32_t*)(scale_ptr + 8);
        aux[3] = *(const device uint32_t*)(scale_ptr + 12);
        uint32_t tmp = aux[2];
        const uint32_t kmask1 = 0x03030303;
        const uint32_t kmask2 = 0x0f0f0f0f;
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        thread int8_t scales[16];
        const thread uchar* aux_bytes = reinterpret_cast<const thread uchar*>(aux);
        for (int i = 0; i < 16; ++i) {
            scales[i] = static_cast<int8_t>(aux_bytes[i]);
        }
        const device half* d_ptr = reinterpret_cast<const device half*>(block + 108);
        float d = static_cast<float>(d_ptr[0]);
        uint out_idx = gid * cols;
        int is = 0;
        for (uint n = 0; n < cols; n += 128) {
            int shift = 0;
            uint8_t mask_bit = 1;
            for (int j = 0; j < 4 && n + j * 32 < cols; ++j) {
                float dl = d * static_cast<float>(scales[is++] - 32);
                for (int l = 0; l < 16 && n + j * 32 + l < cols; ++l) {
                    int8_t val = static_cast<int8_t>((qs[l] >> shift) & 0x3);
                    int8_t delta = (hmask[l] & mask_bit) ? 0 : 4;
                    output[out_idx + n + j * 32 + l] = dl * static_cast<float>(val - delta);
                }
                dl = d * static_cast<float>(scales[is++] - 32);
                for (int l = 0; l < 16 && n + j * 32 + 16 + l < cols; ++l) {
                    int8_t val = static_cast<int8_t>((qs[l + 16] >> shift) & 0x3);
                    int8_t delta = (hmask[l + 16] & mask_bit) ? 0 : 4;
                    output[out_idx + n + j * 32 + 16 + l] = dl * static_cast<float>(val - delta);
                }
                shift += 2;
                mask_bit <<= 1;
            }
            qs += 32;
            hmask += 32;
        }
    } break;
    case 12: { // Q4_K
        const uint block_size = 144;
        const device uchar* block = row;
        const device uchar* scales = block + 4;
        const device uchar* qs = block + 16;
        const device half* d_ptr = reinterpret_cast<const device half*>(block);
        float d = static_cast<float>(d_ptr[0]);
        float min = static_cast<float>(d_ptr[1]);
        uint out_idx = gid * cols;
        thread uchar sc_val = 0;
        thread uchar m_val = 0;
        for (uint n = 0, is = 0; n < cols; n += 64, is += 2) {
            getScaleMinK4(is + 0, scales, sc_val, m_val);
            float dl = d * static_cast<float>(sc_val);
            float ml = min * static_cast<float>(m_val);
            for (int l = 0; l < 32 && n + l < cols; ++l) {
                uint8_t val = (qs[l] & 0x0F);
                output[out_idx + n + l] = dl * static_cast<float>(val) - ml;
            }
            getScaleMinK4(is + 1, scales, sc_val, m_val);
            dl = d * static_cast<float>(sc_val);
            ml = min * static_cast<float>(m_val);
            for (int l = 0; l < 32 && n + 32 + l < cols; ++l) {
                uint8_t val = (qs[l] >> 4) & 0x0F;
                output[out_idx + n + 32 + l] = dl * static_cast<float>(val) - ml;
            }
            qs += 32;
        }
    } break;
    case 13: { // Q5_K
        const uint block_size = 176;
        const device uchar* block = row;
        const device uchar* scales = block + 4;
        const device uchar* qh = block + 16;
        const device uchar* qs = block + 48;
        const device half* d_ptr = reinterpret_cast<const device half*>(block);
        float d = static_cast<float>(d_ptr[0]);
        float minv = static_cast<float>(d_ptr[1]);
        uint out_idx = gid * cols;
        thread uchar sc_val = 0;
        thread uchar m_val = 0;
        int is = 0;
        uint8_t u1 = 1;
        uint8_t u2 = 2;
        for (uint n = 0; n < cols; n += 64) {
            getScaleMinK4(is + 0, scales, sc_val, m_val);
            float dl = d * static_cast<float>(sc_val);
            float ml = minv * static_cast<float>(m_val);
            for (int l = 0; l < 32 && n + l < cols; ++l) {
                uint8_t hi = (qh[l] & u1) ? 16 : 0;
                uint8_t val = (qs[l] & 0x0F) + hi;
                output[out_idx + n + l] = dl * static_cast<float>(val) - ml;
            }
            getScaleMinK4(is + 1, scales, sc_val, m_val);
            dl = d * static_cast<float>(sc_val);
            ml = minv * static_cast<float>(m_val);
            for (int l = 0; l < 32 && n + 32 + l < cols; ++l) {
                uint8_t hi = (qh[l] & u2) ? 16 : 0;
                uint8_t val = (qs[l] >> 4) + hi;
                output[out_idx + n + 32 + l] = dl * static_cast<float>(val) - ml;
            }
            qs += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    } break;
    case 14: { // Q6_K
        const uint block_size = 210;
        const device uchar* block = row;
        const device uchar* ql = block;
        const device uchar* qh = block + 128;
        thread int8_t scales[16];
        const device uchar* scale_bytes = block + 192;
        for (int i = 0; i < 16; ++i) {
            scales[i] = static_cast<int8_t>(scale_bytes[i]);
        }
        const device half* d_ptr = reinterpret_cast<const device half*>(block + 208);
        float d = static_cast<float>(d_ptr[0]);
        uint out_idx = gid * cols;
        for (uint i = 0; i < cols / 16; ++i) {
            int8_t sc = scales[i];
            float dl = d * static_cast<float>(sc);
            for (int j = 0; j < 16 && i * 16 + j < cols; ++j) {
                int idx = i * 16 + j;
                int bit = (idx & 1) ? (ql[idx / 2] >> 4) : (ql[idx / 2] & 0xF);
                int high = (qh[idx / 4] >> (2 * (idx % 4))) & 0x3;
                int val = bit | (high << 4);
                if (val > 31) val -= 64;
                output[out_idx + idx] = dl * static_cast<float>(val);
            }
        }
    } break;
    case 15: { // Q8_K
        const uint block_size = 292;
        const device uchar* row8k = row;
        const device float* d_ptr = reinterpret_cast<const device float*>(row8k);
        float d = d_ptr[0];
        const device int8_t* qs = reinterpret_cast<const device int8_t*>(row8k + 4);
        uint out_idx = gid * cols;
        for (uint i = 0; i < cols; ++i) {
            output[out_idx + i] = d * static_cast<float>(qs[i]);
        }
    } break;
    default:
        break;
    }
}
kernel void q3_k_matmul(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant QKParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.rows) return;
    const device uchar* row = weights + gid * params.row_stride;
    const uint block_size = 110;
    uint cols = params.cols;
    uint blocks = (cols + 255u) / 256u;
    float acc = 0.0f;
    uint col_index = 0;
    for (uint b = 0; b < blocks && col_index < cols; ++b) {
        const device uchar* block = row + b * block_size;
        const device uchar* hmask = block;
        const device uchar* qs = block + 32;
        const device uchar* scale_ptr = block + 64;
        uint32_t aux[4];
        aux[0] = *(const device uint32_t*)(scale_ptr + 0);
        aux[1] = *(const device uint32_t*)(scale_ptr + 4);
        aux[2] = *(const device uint32_t*)(scale_ptr + 8);
        aux[3] = *(const device uint32_t*)(scale_ptr + 12);
        uint32_t tmp = aux[2];
        const uint32_t kmask1 = 0x03030303;
        const uint32_t kmask2 = 0x0f0f0f0f;
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        thread int8_t scales[16];
        const thread uchar* aux_bytes = reinterpret_cast<const thread uchar*>(aux);
        for (int i = 0; i < 16; ++i) {
            scales[i] = static_cast<int8_t>(aux_bytes[i]);
        }
        const device half* d_ptr = reinterpret_cast<const device half*>(block + 108);
        float d = static_cast<float>(d_ptr[0]);
        int is = 0;
        const uint total = metal::min(256u, cols - col_index);
        for (uint n = 0; n < total; n += 128) {
            int shift = 0;
            uint8_t mask_bit = 1;
            for (int j = 0; j < 4 && (col_index + n + j * 32) < cols; ++j) {
                float dl = d * static_cast<float>(scales[is++] - 32);
                for (int l = 0; l < 16 && (col_index + n + j * 32 + l) < cols; ++l) {
                    int8_t val = static_cast<int8_t>((qs[l] >> shift) & 0x3);
                    int8_t delta = (hmask[l] & mask_bit) ? 0 : 4;
                    float vf = dl * static_cast<float>(val - delta);
                    acc += vf * input[col_index + n + j * 32 + l];
                }
                dl = d * static_cast<float>(scales[is++] - 32);
                for (int l = 0; l < 16 && (col_index + n + j * 32 + 16 + l) < cols; ++l) {
                    int8_t val = static_cast<int8_t>((qs[l + 16] >> shift) & 0x3);
                    int8_t delta = (hmask[l + 16] & mask_bit) ? 0 : 4;
                    float vf = dl * static_cast<float>(val - delta);
                    acc += vf * input[col_index + n + j * 32 + 16 + l];
                }
                shift += 2;
                mask_bit <<= 1;
            }
            qs += 32;
            hmask += 32;
        }
        col_index += 256;
    }
    output[gid] = acc;
}

kernel void q4_k_matmul(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant QKParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.rows) return;
    const device uchar* row = weights + gid * params.row_stride;
    const uint block_size = 144;
    uint cols = params.cols;
    uint blocks = (cols + 255u) / 256u;
    float acc = 0.0f;
    uint col_index = 0;
    for (uint b = 0; b < blocks && col_index < cols; ++b) {
        const device uchar* block = row + b * block_size;
        const device uchar* scales = block + 4;
        const device uchar* qs = block + 16;
        const device half* d_ptr = reinterpret_cast<const device half*>(block);
        float d = static_cast<float>(d_ptr[0]);
        float minv = static_cast<float>(d_ptr[1]);
        uint total = metal::min(256u, cols - col_index);
        thread uchar sc_val = 0;
        thread uchar m_val = 0;
        int is = 0;
    for (uint n = 0; n < total; n += 64) {
        getScaleMinK4(is + 0, scales, sc_val, m_val);
        float dl = d * static_cast<float>(sc_val);
        float ml = minv * static_cast<float>(m_val);
        for (int l = 0; l < 32 && (col_index + n + l) < cols; ++l) {
            uint8_t val = (qs[l] & 0x0F);
            float vf = dl * static_cast<float>(val) - ml;
            acc += vf * input[col_index + n + l];
        }
        getScaleMinK4(is + 1, scales, sc_val, m_val);
        dl = d * static_cast<float>(sc_val);
        ml = minv * static_cast<float>(m_val);
        for (int l = 0; l < 32 && (col_index + n + 32 + l) < cols; ++l) {
                uint8_t val = (qs[l] >> 4) & 0x0F;
                float vf = dl * static_cast<float>(val) - ml;
                acc += vf * input[col_index + n + 32 + l];
            }
        qs += 32;
        is += 2;
    }
    col_index += 256;
    }
    output[gid] = acc;
}

kernel void q5_k_matmul(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant QKParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.rows) return;
    const device uchar* row = weights + gid * params.row_stride;
    const uint block_size = 176;
    uint cols = params.cols;
    uint blocks = (cols + 255u) / 256u;
    float acc = 0.0f;
    uint col_index = 0;
    for (uint b = 0; b < blocks && col_index < cols; ++b) {
        const device uchar* block = row + b * block_size;
        const device uchar* scales = block + 4;
        const device uchar* qh = block + 16;
        const device uchar* qs = block + 48;
        const device half* d_ptr = reinterpret_cast<const device half*>(block);
        float d = static_cast<float>(d_ptr[0]);
        float minv = static_cast<float>(d_ptr[1]);
        uint total = metal::min(256u, cols - col_index);
        thread uchar sc_val = 0;
        thread uchar m_val = 0;
        int is = 0;
        uint8_t u1 = 1;
        uint8_t u2 = 2;
        for (uint n = 0; n < total; n += 64) {
            getScaleMinK4(is + 0, scales, sc_val, m_val);
            float dl = d * static_cast<float>(sc_val);
            float ml = minv * static_cast<float>(m_val);
            for (int l = 0; l < 32 && (col_index + n + l) < cols; ++l) {
                uint8_t hi = (qh[l] & u1) ? 16 : 0;
                uint8_t val = (qs[l] & 0x0F) + hi;
                float vf = dl * static_cast<float>(val) - ml;
                acc += vf * input[col_index + n + l];
            }
            getScaleMinK4(is + 1, scales, sc_val, m_val);
            dl = d * static_cast<float>(sc_val);
            ml = minv * static_cast<float>(m_val);
            for (int l = 0; l < 32 && (col_index + n + 32 + l) < cols; ++l) {
                uint8_t hi = (qh[l] & u2) ? 16 : 0;
                uint8_t val = ((qs[l] >> 4) & 0x0F) + hi;
                float vf = dl * static_cast<float>(val) - ml;
                acc += vf * input[col_index + n + 32 + l];
            }
            qs += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
        col_index += 256;
    }
    output[gid] = acc;
}

kernel void q6_k_matmul(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant QKParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.rows) return;
    const device uchar* row = weights + gid * params.row_stride;
    const uint block_size = 210;
    uint cols = params.cols;
    uint blocks = (cols + 255u) / 256u;
    float acc = 0.0f;
    uint col_index = 0;
    for (uint b = 0; b < blocks && col_index < cols; ++b) {
        const device uchar* block = row + b * block_size;
        const device uchar* ql = block;
        const device uchar* qh = block + 128;
        const device int8_t* scales = reinterpret_cast<const device int8_t*>(block + 192);
        const device half* d_ptr = reinterpret_cast<const device half*>(block + 208);
        float d = static_cast<float>(d_ptr[0]);
        uint total = metal::min(256u, cols - col_index);
        const device uchar* ql_ptr = ql;
        const device uchar* qh_ptr = qh;
        const device int8_t* sc_ptr = scales;
        for (uint base = 0; base < total; base += 128) {
            for (int l = 0; l < 32 && (col_index + base + l) < cols; ++l) {
                int is = l / 16;
                int8_t q1 = static_cast<int8_t>((ql_ptr[l + 0] & 0xF) | (((qh_ptr[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = static_cast<int8_t>((ql_ptr[l + 32] & 0xF) | (((qh_ptr[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = static_cast<int8_t>((ql_ptr[l + 0] >> 4) | (((qh_ptr[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = static_cast<int8_t>((ql_ptr[l + 32] >> 4) | (((qh_ptr[l] >> 6) & 3) << 4)) - 32;
                float s0 = d * static_cast<float>(sc_ptr[is + 0]);
                float s1 = d * static_cast<float>(sc_ptr[is + 2]);
                float s2 = d * static_cast<float>(sc_ptr[is + 4]);
                float s3 = d * static_cast<float>(sc_ptr[is + 6]);
                if (col_index + base + l < cols) acc += s0 * static_cast<float>(q1) * input[col_index + base + l];
                if (col_index + base + 32 + l < cols) acc += s1 * static_cast<float>(q2) * input[col_index + base + 32 + l];
                if (col_index + base + 64 + l < cols) acc += s2 * static_cast<float>(q3) * input[col_index + base + 64 + l];
                if (col_index + base + 96 + l < cols) acc += s3 * static_cast<float>(q4) * input[col_index + base + 96 + l];
            }
            ql_ptr += 64;
            qh_ptr += 32;
            sc_ptr += 8;
        }
        col_index += 256;
    }
    output[gid] = acc;
}

kernel void q6_k_matmul_transposed(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant QKParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.cols) return;
    uint col = gid;
    const uint block_size = 210;
    uint block = col / 256u;
    uint offset = col % 256u;
    uint base = offset / 128u;
    uint local = offset - base * 128u;
    uint segment = local / 32u;
    uint l = local % 32u;
    uint ql_offset = base * 64u;
    uint qh_offset = base * 32u;
    uint sc_offset = base * 8u;
    float acc = 0.0f;
    for (uint r = 0; r < params.rows; ++r) {
        const device uchar* row = weights + r * params.row_stride;
        const device uchar* block_ptr = row + block * block_size;
        const device uchar* ql = block_ptr + ql_offset;
        const device uchar* qh = block_ptr + 128 + qh_offset;
        const device int8_t* sc = reinterpret_cast<const device int8_t*>(block_ptr + 192) + sc_offset;
        const device half* d_ptr = reinterpret_cast<const device half*>(block_ptr + 208);
        float d = static_cast<float>(d_ptr[0]);
        int is = static_cast<int>(l / 16u);
        int8_t q = 0;
        int sc_index = 0;
        if (segment == 0u) {
            q = static_cast<int8_t>((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            sc_index = is + 0;
        } else if (segment == 1u) {
            q = static_cast<int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            sc_index = is + 2;
        } else if (segment == 2u) {
            q = static_cast<int8_t>((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            sc_index = is + 4;
        } else {
            q = static_cast<int8_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            sc_index = is + 6;
        }
        float s = d * static_cast<float>(sc[sc_index]);
        acc += s * static_cast<float>(q) * input[r];
    }
    output[col] = acc;
}

kernel void q8_k_matmul(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant QKParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.rows) return;
    const device uchar* row = weights + gid * params.row_stride;
    const uint block_size = 292;
    uint cols = params.cols;
    uint blocks = (cols + 255u) / 256u;
    float acc = 0.0f;
    uint col_index = 0;
    for (uint b = 0; b < blocks && col_index < cols; ++b) {
        const device float* d_ptr = reinterpret_cast<const device float*>(row + b * block_size);
        float d = d_ptr[0];
        const device int8_t* qs = reinterpret_cast<const device int8_t*>(row + b * block_size + 4);
        uint total = min(256u, cols - col_index);
        for (uint i = 0; i < total && (col_index + i) < cols; ++i) {
            float vf = d * static_cast<float>(qs[i]);
            acc += vf * input[col_index + i];
        }
        col_index += 256;
    }
    output[gid] = acc;
}

kernel void q8_0_matmul(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant Q4_0Params& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.rows) return;
    const device uchar* row = weights + gid * params.row_stride;
    const uint block_size = 34;
    uint cols = params.cols;
    uint blocks = (cols + 31u) / 32u;
    float acc = 0.0f;
    uint col_index = 0;
    for (uint b = 0; b < blocks && col_index < cols; ++b) {
        const device half* d_ptr = reinterpret_cast<const device half*>(row + b * block_size);
        float d = static_cast<float>(d_ptr[0]);
        const device int8_t* qs = reinterpret_cast<const device int8_t*>(row + b * block_size + 2);
        for (uint j = 0; j < 32 && col_index < cols; ++j) {
            float vf = d * static_cast<float>(qs[j]);
            acc += vf * input[col_index++];
        }
    }
    output[gid] = acc;
}

kernel void q8_1_matmul(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant Q4_0Params& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.rows) return;
    const device uchar* row = weights + gid * params.row_stride;
    const uint block_size = 36;
    uint cols = params.cols;
    uint blocks = (cols + 31u) / 32u;
    float acc = 0.0f;
    uint col_index = 0;
    for (uint b = 0; b < blocks && col_index < cols; ++b) {
        const device half* meta = reinterpret_cast<const device half*>(row + b * block_size);
        float d = static_cast<float>(meta[0]);
        float sum = static_cast<float>(meta[1]);
        float bias = sum / 32.0f;
        const device int8_t* qs = reinterpret_cast<const device int8_t*>(row + b * block_size + 4);
        for (uint j = 0; j < 32 && col_index < cols; ++j) {
            float vf = d * static_cast<float>(qs[j]) + bias;
            acc += vf * input[col_index++];
        }
    }
    output[gid] = acc;
}

struct RotaryParams {
    uint head_dim;
    uint rotary_dim;
    uint row_stride;
    uint pair_stride;
    uint count;
    uint offset_elems;
    uint heads;
    uint tokens;
};

kernel void apply_rotary_batch(
    device float* data [[buffer(0)]],
    device const float* cos_table [[buffer(1)]],
    device const float* sin_table [[buffer(2)]],
    constant RotaryParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.count) return;
    uint rotary_dim = min(params.rotary_dim, params.head_dim);
    if (rotary_dim < 2) return;
    uint pairs = rotary_dim / 2;
    uint offset_tokens = gid + params.offset_elems;
    device float* vec = reinterpret_cast<device float*>(
        reinterpret_cast<device uchar*>(data) + gid * params.row_stride);
    vec += params.offset_elems * params.head_dim;
    const device float* cos_row = cos_table + offset_tokens * params.pair_stride;
    const device float* sin_row = sin_table + offset_tokens * params.pair_stride;
    for (uint i = 0; i < pairs; ++i) {
        float c = (i < params.pair_stride) ? cos_row[i] : 1.0f;
        float s = (i < params.pair_stride) ? sin_row[i] : 0.0f;
        float x0 = vec[2 * i];
        float x1 = vec[2 * i + 1];
        vec[2 * i] = x0 * c - x1 * s;
        vec[2 * i + 1] = x0 * s + x1 * c;
    }
}

// Dequantize a single Q4_0 block into a float vector (32 elems)
kernel void dequant_q4_0_block(
    device const uchar* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= cols) return;
    const uint block_size = 18;
    uint block = gid / 32;
    uint local = gid % 32;
    const device uchar* row = src + block * block_size;
    const device half* d_half = reinterpret_cast<const device half*>(row);
    float d = static_cast<float>(d_half[0]);
    const device uchar* qs = row + 2;
    uint idx = local / 2;
    uchar byte = qs[idx];
    int8_t q = (local & 1) ? static_cast<int8_t>((byte >> 4) & 0x0F) : static_cast<int8_t>(byte & 0x0F);
    q = static_cast<int8_t>(q - 8);
    dst[gid] = d * static_cast<float>(q);
}

// Dequantize a single Q8_0 block into a float vector (32 elems)
kernel void dequant_q8_0_block(
    device const uchar* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= cols) return;
    const uint block_size = 34;
    uint block = gid / 32;
    uint local = gid % 32;
    const device uchar* row = src + block * block_size;
    const device half* d_half = reinterpret_cast<const device half*>(row);
    float d = static_cast<float>(d_half[0]);
    const device int8_t* qs = reinterpret_cast<const device int8_t*>(row + 2);
    dst[gid] = d * static_cast<float>(qs[local]);
}

// Dequantize a single Q4_1 block into a float vector (32 elems)
kernel void dequant_q4_1_block(
    device const uchar* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= cols) return;
    const uint block_size = 20;
    uint block = gid / 32;
    uint local = gid % 32;
    const device uchar* row = src + block * block_size;
    const device half* meta = reinterpret_cast<const device half*>(row);
    float d = static_cast<float>(meta[0]);
    float m = static_cast<float>(meta[1]);
    const device uchar* qs = row + 4;
    uint idx = local / 2;
    uchar byte = qs[idx];
    int x = (local & 1) ? static_cast<int>((byte >> 4) & 0x0F) : static_cast<int>(byte & 0x0F);
    dst[gid] = d * static_cast<float>(x) + m;
}

// Dequantize a single Q5_0 block into a float vector (32 elems)
kernel void dequant_q5_0_block(
    device const uchar* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= cols) return;
    const uint block_size = 22;
    uint block = gid / 32;
    uint local = gid % 32;
    const device uchar* row = src + block * block_size;
    const device half* meta = reinterpret_cast<const device half*>(row);
    float d = static_cast<float>(meta[0]);
    const device uint32_t* qh_ptr = reinterpret_cast<const device uint32_t*>(row + 2);
    uint32_t qh = *qh_ptr;
    const device uchar* qs = row + 6;
    uint idx = local / 2;
    uchar byte = qs[idx];
    uint8_t xh = (local & 1)
                     ? static_cast<uint8_t>(((qh >> (idx + 12)) & 0x1) << 4)
                     : static_cast<uint8_t>(((qh >> (idx + 0)) & 0x1) << 4);
    int x = (local & 1) ? ((byte >> 4) | xh) : ((byte & 0x0F) | xh);
    dst[gid] = (static_cast<int>(x) - 16) * d;
}

// Dequantize a single Q5_1 block into a float vector (32 elems)
kernel void dequant_q5_1_block(
    device const uchar* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= cols) return;
    const uint block_size = 24;
    uint block = gid / 32;
    uint local = gid % 32;
    const device uchar* row = src + block * block_size;
    const device half* meta = reinterpret_cast<const device half*>(row);
    float d = static_cast<float>(meta[0]);
    float m = static_cast<float>(meta[1]);
    const device uint32_t* qh_ptr = reinterpret_cast<const device uint32_t*>(row + 4);
    uint32_t qh = *qh_ptr;
    const device uchar* qs = row + 8;
    uint idx = local / 2;
    uchar byte = qs[idx];
    uint8_t xh = (local & 1)
                     ? static_cast<uint8_t>(((qh >> (idx + 12)) & 0x1) << 4)
                     : static_cast<uint8_t>(((qh >> (idx + 0)) & 0x1) << 4);
    int x = (local & 1) ? ((byte >> 4) | xh) : ((byte & 0x0F) | xh);
    dst[gid] = d * static_cast<float>(x) + m;
}

// Dequantize a single Q8_1 block into a float vector (32 elems)
kernel void dequant_q8_1_block(
    device const uchar* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= cols) return;
    const uint block_size = 36;
    uint block = gid / 32;
    uint local = gid % 32;
    const device uchar* row = src + block * block_size;
    const device half* meta = reinterpret_cast<const device half*>(row);
    float d = static_cast<float>(meta[0]);
    float sum = static_cast<float>(meta[1]);
    float bias = sum / 32.0f;
    const device int8_t* qs = reinterpret_cast<const device int8_t*>(row + 4);
    dst[gid] = d * static_cast<float>(qs[local]) + bias;
}

// ============================================================================
// flash_attention_v1: multi-tile online-softmax flash attention.
//
// Dispatch: one threadgroup per Q head, head_dim threads per threadgroup.
// Thread t owns feature dim t (loads, output accumulator); inside each tile
// it computes scores at strided kv positions and contributes one feature dim
// to the output. Tiles iterate over kv_seq in chunks of T tokens.
//
// Online softmax recurrence per tile:
//   m_new = max(m, tile_max)
//   alpha = exp(m - m_new)  (== 0 on first tile, where m == -INFINITY)
//   l = l * alpha + tile_sum                  (running denominator)
//   o = o * alpha + sum_s exp(s - m_new) * V  (running output)
//   m = m_new
// After all tiles, output = o / l.
//
// Threadgroup memory layout (T = TILE_SIZE):
//   [0, D)                              shared_q       (Q for this head)
//   [D, D + T*D)                        tile_K         (K for current tile)
//   [D + T*D, D + 2*T*D)                tile_V         (V for current tile)
//   [D + 2*T*D, D + 2*T*D + T)          tile_scores
//   [D + 2*T*D + T, 2*D + 2*T*D + T)    partial_buf    (max/sum reduction scratch)
//
// Determinism: reductions use fixed-stride tree (NOT simd_max/simd_sum) so the
// reduction order is bit-stable across runs.
//
// Causal mask: TODO when q_seq > 1 cases are exercised. For q_seq=1 (this
// session) causal is a no-op (the single Q sees all kv positions).
//
// Requires head_dim to be a power of 2. Asserted host-side.
// ============================================================================
struct FlashAttnParams {
    uint num_heads;
    uint kv_heads;
    uint head_dim;
    uint kv_seq;
    float inv_sqrt_d;
    uint debug_dump;        // 0 or 1; when 1, qk_out/sm_out are written
    uint apply_causal;      // 0 or 1; when 1, mask scores for kv_pos > q_position
    uint q_position;        // for q_seq=1, the kv position of the active query
    // K and V stride parameters. Address of (token=s, kv_head=kv, dim=d) is:
    //   base + s * stride_token + kv * stride_kv_head + d * stride_feature
    // Harness layout [kv_seq, kv_heads, head_dim]:
    //   stride_token = kv_heads * head_dim, stride_kv_head = head_dim, stride_feature = 1
    // Production cache layout [kv_heads, context_length, head_dim]:
    //   stride_token = head_dim, stride_kv_head = context_length * head_dim, stride_feature = 1
    uint k_stride_token;
    uint k_stride_kv_head;
    uint k_stride_feature;
    uint v_stride_token;
    uint v_stride_kv_head;
    uint v_stride_feature;
};

constant uint FLASH_TILE_SIZE = 32u;

kernel void flash_attention_v1(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    device float* qk_out [[buffer(4)]],
    device float* sm_out [[buffer(5)]],
    constant FlashAttnParams& params [[buffer(6)]],
    threadgroup float* tg_buf [[threadgroup(0)]],
    uint head [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]]
) {
    const uint T = FLASH_TILE_SIZE;
    const uint D = params.head_dim;
    const uint S = params.kv_seq;
    const uint kv = (head * params.kv_heads) / params.num_heads;

    threadgroup float* shared_q    = tg_buf;                          // [D]
    threadgroup float* tile_K      = tg_buf + D;                      // [T*D]
    threadgroup float* tile_V      = tg_buf + D + T * D;              // [T*D]
    threadgroup float* tile_scores = tg_buf + D + 2u * T * D;         // [T]
    threadgroup float* partial_buf = tg_buf + D + 2u * T * D + T;     // [D]

    // 1. Load Q for this head into threadgroup memory (one element per thread).
    shared_q[t] = Q[head * D + t];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-thread online-softmax running state.
    float m = -INFINITY;   // running max over all tiles seen so far
    float l = 0.0f;        // running sum (denominator)
    float o = 0.0f;        // running output for this thread's feature dim

    uint num_tiles = (S + T - 1u) / T;
    for (uint tile = 0u; tile < num_tiles; ++tile) {
        uint tile_start = tile * T;
        uint tile_n = (S - tile_start < T) ? (S - tile_start) : T;

        // Cooperative load: each thread loads its column (feature dim t)
        // across all positions in the tile. K and V address calc uses the
        // explicit per-axis strides so the same kernel works for the
        // harness's [kv_seq, kv_heads, head_dim] layout and the production
        // KV cache's [kv_heads, context_length, head_dim] layout.
        for (uint i = 0u; i < tile_n; ++i) {
            uint kv_pos = tile_start + i;
            uint k_idx = kv_pos * params.k_stride_token
                       + kv * params.k_stride_kv_head
                       + t * params.k_stride_feature;
            uint v_idx = kv_pos * params.v_stride_token
                       + kv * params.v_stride_kv_head
                       + t * params.v_stride_feature;
            tile_K[i * D + t] = K[k_idx];
            tile_V[i * D + t] = V[v_idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Tile scores: each thread computes whole-dot-products for its
        // strided positions within the tile. qk_out (debug) is written
        // inline per-position. Causal mask sets the score to -INFINITY when
        // kv_pos > q_position; downstream max-reduce and exp pass treat
        // masked positions as zero-weighted (exp(-INF - finite_max) = 0).
        for (uint s_local = t; s_local < tile_n; s_local += D) {
            float score = 0.0f;
            threadgroup const float* k_row = tile_K + s_local * D;
            for (uint d = 0u; d < D; ++d) {
                score += shared_q[d] * k_row[d];
            }
            float scaled = score * params.inv_sqrt_d;
            uint kv_pos = tile_start + s_local;
            if (params.apply_causal != 0u && kv_pos > params.q_position) {
                scaled = -INFINITY;
            }
            tile_scores[s_local] = scaled;
            if (params.debug_dump != 0u) {
                qk_out[head * S + kv_pos] = scaled;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. Tile max via fixed-stride tree reduction.
        float my_max = -INFINITY;
        for (uint s = t; s < tile_n; s += D) {
            my_max = fmax(my_max, tile_scores[s]);
        }
        partial_buf[t] = my_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = D / 2u; stride >= 1u; stride /= 2u) {
            if (t < stride) {
                partial_buf[t] = fmax(partial_buf[t], partial_buf[t + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (stride == 1u) break;
        }
        float tile_max = partial_buf[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 4. Online update — rescale factor for previous (l, o). The first
        // tile guard avoids the (-INFINITY - finite) subtraction and protects
        // against the (-INFINITY - -INFINITY) NaN that could surface once
        // causal masking lands and a row sees no positions.
        float m_new = fmax(m, tile_max);
        float alpha = (m == -INFINITY) ? 0.0f : exp(m - m_new);

        // 5. exp(score - m_new) into tile_scores; tile sum via tree reduce.
        // Guard against -INFINITY scores (from causal mask or an all-masked
        // tile with m_new == -INFINITY): exp(NaN) would propagate. Treating
        // masked scores as exp = 0 keeps the recurrence clean.
        float my_sum = 0.0f;
        for (uint s = t; s < tile_n; s += D) {
            float ts = tile_scores[s];
            float e = (ts == -INFINITY) ? 0.0f : exp(ts - m_new);
            tile_scores[s] = e;
            my_sum += e;
        }
        partial_buf[t] = my_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = D / 2u; stride >= 1u; stride /= 2u) {
            if (t < stride) {
                partial_buf[t] += partial_buf[t + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (stride == 1u) break;
        }
        float tile_sum = partial_buf[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 6. Tile output contribution for this thread's feature dim.
        float my_v_acc = 0.0f;
        for (uint s = 0u; s < tile_n; ++s) {
            my_v_acc += tile_scores[s] * tile_V[s * D + t];
        }

        // 7. Compose running state with this tile.
        o = o * alpha + my_v_acc;
        l = l * alpha + tile_sum;
        m = m_new;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 8. Final normalize.
    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    O[head * D + t] = o * inv_l;

    // 9. sm_out: softmax weights RECOMPUTED from raw scores in qk_out and the
    // final (m, l). Mathematically equivalent to running softmax weights at
    // the final tile (same m, l), but NOT a tap of intermediate-tile running
    // state. Validates that final m, l are correct; does not validate
    // per-tile updates. For per-tile-state validation, add an option C
    // harness (partial-tile reference) — TODO when intermediate-state
    // debugging is needed.
    if (params.debug_dump != 0u) {
        for (uint s = t; s < S; s += D) {
            float raw = qk_out[head * S + s];
            float e = (raw == -INFINITY) ? 0.0f : exp(raw - m);
            sm_out[head * S + s] = e * inv_l;
        }
    }
}
)";
} // namespace
#endif

namespace mlc {
namespace runtime {

struct Q4_0ParamsNative {
    uint32_t rows;
    uint32_t cols;
    uint32_t row_stride;
    uint32_t quant_version;
};

struct QKParamsNative {
    uint32_t rows;
    uint32_t cols;
    uint32_t row_stride;
};

struct RotaryParamsNative {
    uint32_t head_dim;
    uint32_t rotary_dim;
    uint32_t row_stride;
    uint32_t pair_stride;
    uint32_t count;
    uint32_t offset_elems;
};

struct KVWriteParamsNative {
    uint32_t head_dim;
    uint32_t kv_heads;
    uint32_t kv_stride;
    uint32_t tokens;
    uint32_t base_pos;
};

struct MetalExecutor::Impl {
#if defined(__APPLE__)
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> ffnPipeline = nil;
    id<MTLComputePipelineState> addPipeline = nil;
    id<MTLComputePipelineState> addResidualPipeline = nil;
    id<MTLComputePipelineState> normPipeline = nil;
    id<MTLComputePipelineState> layerNormPipeline = nil;
    id<MTLComputePipelineState> softmaxPipeline = nil;
    id<MTLComputePipelineState> q4MatmulPipeline = nil;
    id<MTLComputePipelineState> q4MatmulTransposedPipeline = nil;
    id<MTLComputePipelineState> q4_1MatmulPipeline = nil;
    id<MTLComputePipelineState> q5_0MatmulPipeline = nil;
    id<MTLComputePipelineState> q5_1MatmulPipeline = nil;
    id<MTLComputePipelineState> q2KMatmulPipeline = nil;
    id<MTLComputePipelineState> q3KMatmulPipeline = nil;
    id<MTLComputePipelineState> q4KMatmulPipeline = nil;
    id<MTLComputePipelineState> q5KMatmulPipeline = nil;
    id<MTLComputePipelineState> q6KMatmulPipeline = nil;
    id<MTLComputePipelineState> q6KMatmulTransposedPipeline = nil;
    id<MTLComputePipelineState> q8KMatmulPipeline = nil;
    id<MTLComputePipelineState> q8_0MatmulPipeline = nil;
    id<MTLComputePipelineState> q8_1MatmulPipeline = nil;
    id<MTLComputePipelineState> rotaryPipeline = nil;
    id<MTLComputePipelineState> dequantQ4Pipeline = nil;
    id<MTLComputePipelineState> dequantQ8Pipeline = nil;
    id<MTLComputePipelineState> dequantQ4_1Pipeline = nil;
    id<MTLComputePipelineState> dequantQ5_0Pipeline = nil;
    id<MTLComputePipelineState> dequantQ5_1Pipeline = nil;
    id<MTLComputePipelineState> dequantQ8_1Pipeline = nil;
    id<MTLComputePipelineState> dequantQKPipeline = nil;
    id<MTLComputePipelineState> biasAddPipeline = nil;
    id<MTLComputePipelineState> kvWritePipeline = nil;
    id<MTLComputePipelineState> flashAttentionPipeline = nil;
    bool debug_log = false;

    // Persistent device-side weight cache. Weights are immutable for the
    // lifetime of the runtime, so we upload each tensor once and reuse the
    // Metal buffer across every matmul that consumes it. Keyed by full
    // tensor name (e.g. "blk.5.attn_q.weight") to avoid cross-block collisions.
    mutable std::unordered_map<std::string, id<MTLBuffer>> weight_cache_;
    mutable std::mutex weight_cache_mutex_;
    mutable size_t cache_hits_ = 0;
    mutable size_t cache_misses_ = 0;
    mutable size_t cache_bytes_ = 0;

    // === Deferred-commit fusion window state ===
    // Active iff open_forward_pass_cb_ != nil. Single-threaded by construction —
    // the executor opens at most one window at a time on its run() thread.
    //
    // The "pass_retained_" vector holds strong refs to transient
    // MTLBuffers (input uploads via newBufferWithBytes for the FromHost
    // path) that must outlive the encode call but only until commit. The
    // pool (intermediate buffer pool below) owns longer-lived output
    // buffers that carry between ops within a window. pass_checked_out_
    // tracks pool buffers in use by the current window so flush can return
    // them all at once.
    mutable id<MTLCommandBuffer> open_forward_pass_cb_ = nil;
    mutable std::vector<id> pass_retained_;
    mutable std::vector<id<MTLBuffer>> pass_checked_out_;

    struct PendingReadback {
        id<MTLBuffer> source;
        std::vector<float>* dst;
        size_t byte_count;
    };
    mutable std::vector<PendingReadback> pass_readbacks_;

    // === Intermediate buffer pool ===
    // Four size classes cover TinyLlama Q4_0 cleanly:
    //   1 KB   — attn_k/v outputs (256 floats)
    //   8 KB   — most hidden states (2048 floats: norms, residuals, attn_q/output)
    //  32 KB   — FFN gate/up outputs (5632 floats)
    // 128 KB   — logits (32000 floats)
    // Power-of-2 rounding wastes 10-30% per buffer; total footprint stays
    // under ~4 MB on TinyLlama. Anything larger than the top class bypasses
    // the pool and allocates directly (released on return via ARC).
    static constexpr size_t kPoolClasses = 4;
    static constexpr size_t kPoolClassBytes[kPoolClasses] = {
        1024, 8192, 32768, 131072
    };
    mutable std::array<std::vector<id<MTLBuffer>>, kPoolClasses> pool_free_lists_;

    static size_t poolClassFor(size_t bytes) {
        for (size_t i = 0; i < kPoolClasses; ++i) {
            if (bytes <= kPoolClassBytes[i]) return i;
        }
        return kPoolClasses;  // oversized
    }

    id<MTLBuffer> poolCheckout(size_t bytes) const {
        size_t cls = poolClassFor(bytes);
        size_t alloc_bytes = (cls < kPoolClasses) ? kPoolClassBytes[cls] : bytes;
        if (cls < kPoolClasses && !pool_free_lists_[cls].empty()) {
            id<MTLBuffer> buf = pool_free_lists_[cls].back();
            pool_free_lists_[cls].pop_back();
            return buf;
        }
        return [device newBufferWithLength:alloc_bytes
                                   options:MTLResourceStorageModeShared];
    }

    void poolReturn(id<MTLBuffer> buf) const {
        if (!buf) return;
        size_t bytes = [buf length];
        for (size_t i = 0; i < kPoolClasses; ++i) {
            if (bytes == kPoolClassBytes[i]) {
                pool_free_lists_[i].push_back(buf);
                return;
            }
        }
        // Oversized direct allocation — release via ARC by letting it go.
    }

    void recordReadback(id<MTLBuffer> out_buf,
                        std::vector<float>* host_dst,
                        size_t byte_count,
                        bool needs_host) const {
        if (!needs_host) return;
        PendingReadback rb;
        rb.source = out_buf;
        rb.dst = host_dst;
        rb.byte_count = byte_count;
        pass_readbacks_.push_back(rb);
    }

    id<MTLBuffer> getOrCacheWeight(const std::string& name,
                                   const std::vector<uint8_t>& bytes) const {
        if (!name.empty()) {
            std::lock_guard<std::mutex> lock(weight_cache_mutex_);
            auto it = weight_cache_.find(name);
            if (it != weight_cache_.end()) {
                ++cache_hits_;
                return it->second;
            }
        }
        id<MTLBuffer> buffer = [device newBufferWithBytes:bytes.data()
                                                   length:bytes.size()
                                                  options:MTLResourceStorageModeShared];
        if (!buffer) return nil;
        if (!name.empty()) {
            std::lock_guard<std::mutex> lock(weight_cache_mutex_);
            // Re-check in case another thread inserted while we allocated.
            auto it = weight_cache_.find(name);
            if (it != weight_cache_.end()) {
                ++cache_hits_;
                return it->second;
            }
            weight_cache_.emplace(name, buffer);
            ++cache_misses_;
            cache_bytes_ += bytes.size();
        }
        return buffer;
    }
#endif

    Impl() {
#if defined(__APPLE__)
        @autoreleasepool {
            device = MTLCreateSystemDefaultDevice();
            if (!device) {
                fprintf(stderr, "[MetalRuntime] MTLCreateSystemDefaultDevice returned nil\n");
            }
            if (!device) {
                NSArray<id<MTLDevice>> *all = MTLCopyAllDevices();
                fprintf(stderr, "[MetalRuntime] MTLCopyAllDevices count = %lu\n", (unsigned long)all.count);
                if (all.count > 0) {
                    device = [all objectAtIndex:0];
                    fprintf(stderr, "[MetalRuntime] Selected device from MTLCopyAllDevices: %s\n",
                            device.name.UTF8String);
                }
            }
            if (!device) {
                fprintf(stderr,
                        "[MetalRuntime] Metal unavailable (no device); CPU backend will be used.\n");
                return;
            }
            queue = [device newCommandQueue];
            if (!queue) {
                fprintf(stderr,
                        "[MetalRuntime] Failed to create Metal command queue; CPU backend will be "
                        "used.\n");
                fprintf(stderr, "[MetalRuntime] Device name: %s\n", device.name.UTF8String);
                device = nil;
                return;
            }
            NSError* error = nil;
            NSString* source = [[NSString alloc] initWithUTF8String:kMetalKernelsSource];
            if (source) {
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:nil
                                                                error:&error];
                if (!library) {
                    fprintf(stderr, "[MetalRuntime] Failed to compile Metal library: %s\n",
                            error ? error.localizedDescription.UTF8String : "unknown");
                    fprintf(stderr, "[MetalRuntime] Device: %s\n", device.name.UTF8String);
                    device = nil;
                    queue = nil;
                    return;
                }
                auto buildPipeline = ^id<MTLComputePipelineState>(NSString* name) {
                    NSError* localError = nil;
                    id<MTLFunction> function = [library newFunctionWithName:name];
                    if (!function) return (id<MTLComputePipelineState>)nil;
                    id<MTLComputePipelineState> pipe =
                        [device newComputePipelineStateWithFunction:function error:&localError];
                    return pipe;
                };
                const char* dbg = getenv("MLC_METAL_DEBUG");
                debug_log = dbg && std::strlen(dbg) > 0;
                ffnPipeline = buildPipeline(@"feedforward_silu_mul");
                addPipeline = buildPipeline(@"add_vectors");
                addResidualPipeline = buildPipeline(@"add_residual_bias");
                normPipeline = buildPipeline(@"rms_norm_kernel");
                layerNormPipeline = buildPipeline(@"layer_norm_kernel");
                softmaxPipeline = buildPipeline(@"vector_softmax");
                q4MatmulPipeline = buildPipeline(@"q4_0_matmul");
                q4MatmulTransposedPipeline = buildPipeline(@"q4_0_matmul_transposed");
                q4_1MatmulPipeline = buildPipeline(@"q4_1_matmul");
                q5_0MatmulPipeline = buildPipeline(@"q5_0_matmul");
                q5_1MatmulPipeline = buildPipeline(@"q5_1_matmul");
                q2KMatmulPipeline = buildPipeline(@"q2_k_matmul");
                q3KMatmulPipeline = buildPipeline(@"q3_k_matmul");
                q4KMatmulPipeline = buildPipeline(@"q4_k_matmul");
                q5KMatmulPipeline = buildPipeline(@"q5_k_matmul");
                q6KMatmulPipeline = buildPipeline(@"q6_k_matmul");
                q6KMatmulTransposedPipeline = buildPipeline(@"q6_k_matmul_transposed");
                q8KMatmulPipeline = buildPipeline(@"q8_k_matmul");
                q8_0MatmulPipeline = buildPipeline(@"q8_0_matmul");
                q8_1MatmulPipeline = buildPipeline(@"q8_1_matmul");
                rotaryPipeline = buildPipeline(@"apply_rotary_batch");
                dequantQ4Pipeline = buildPipeline(@"dequant_q4_0_block");
                dequantQ8Pipeline = buildPipeline(@"dequant_q8_0_block");
                dequantQ4_1Pipeline = buildPipeline(@"dequant_q4_1_block");
                dequantQ5_0Pipeline = buildPipeline(@"dequant_q5_0_block");
                dequantQ5_1Pipeline = buildPipeline(@"dequant_q5_1_block");
                dequantQ8_1Pipeline = buildPipeline(@"dequant_q8_1_block");
                dequantQKPipeline = buildPipeline(@"dequant_qk_row");
                biasAddPipeline = buildPipeline(@"add_bias_strided");
                kvWritePipeline = buildPipeline(@"scatter_kv");
                flashAttentionPipeline = buildPipeline(@"flash_attention_v1");
            }
        }
#endif
    }

    ~Impl() = default;

    bool available() const {
#if defined(__APPLE__)
        return device != nil && queue != nil;
#else
        return false;
#endif
    }

    bool hasFeedForwardKernel() const {
#if defined(__APPLE__)
        return ffnPipeline != nil;
#else
        return false;
#endif
    }

    bool hasAddKernel() const {
#if defined(__APPLE__)
        return addPipeline != nil;
#else
        return false;
#endif
    }

    bool hasNormKernel() const {
#if defined(__APPLE__)
        return normPipeline != nil;
#else
        return false;
#endif
    }

    bool hasLayerNormKernel() const {
#if defined(__APPLE__)
        return layerNormPipeline != nil;
#else
        return false;
#endif
    }

    bool hasSoftmaxKernel() const {
#if defined(__APPLE__)
        return softmaxPipeline != nil;
#else
        return false;
#endif
    }

    static size_t alignedRowBytes(size_t elements) {
        size_t bytes = elements * sizeof(float);
        const size_t alignment = 16;
        return (bytes + (alignment - 1)) & ~(alignment - 1);
    }

    // CPU fallback dequant for K-series formats to keep attention working until GPU kernels exist.
    static void dequantizeKRowCPU(uint32_t dtype,
                                  const uint8_t* src,
                                  size_t head_dim,
                                  uint32_t quant_version,
                                  float* dst) {
        switch (dtype) {
        case frontend::GGML_TYPE_Q2_K:
            dequantizeRowQ2_K(src, head_dim, dst);
            break;
        case frontend::GGML_TYPE_Q3_K:
            dequantizeRowQ3_K(src, head_dim, dst);
            break;
        case frontend::GGML_TYPE_Q4_K:
            dequantizeRowQ4_K(src, head_dim, dst);
            break;
        case frontend::GGML_TYPE_Q5_K:
            dequantizeRowQ5_K(src, head_dim, dst);
            break;
        case frontend::GGML_TYPE_Q6_K:
            dequantizeRowQ6_K(src, head_dim, dst);
            break;
        case frontend::GGML_TYPE_Q8_K:
            dequantizeRowQ8_K(src, head_dim, dst);
            break;
        default:
            (void)quant_version;
            break;
        }
    }

    bool dequantCacheToGPU(const MetalExecutor::CacheDescriptor& desc,
                           size_t rows,
                           size_t cols,
                           id<MTLBuffer> dst) const {
#if defined(__APPLE__)
        if (!available() || !dst) return false;
        if (rows == 0 || cols == 0) return false;
        if (!desc.raw_quant) return false;
        size_t row_stride = desc.row_stride_bytes;
        if (row_stride == 0) {
            row_stride = ggmlRowSizeBytes(desc.dtype, cols, desc.quant_version);
        }
        size_t expected = rows * row_stride;
        if (desc.raw_quant->size() < expected) return false;
        if (@available(macOS 11.0, *)) {
            @autoreleasepool {
                // Choose pipeline up-front so we can bail without creating encoders.
                id<MTLComputePipelineState> pipe = nil;
                bool is_k = false;
                switch (desc.dtype) {
                case frontend::GGML_TYPE_Q4_0: pipe = dequantQ4Pipeline; break;
                case frontend::GGML_TYPE_Q4_1: pipe = dequantQ4_1Pipeline; break;
                case frontend::GGML_TYPE_Q5_0: pipe = dequantQ5_0Pipeline; break;
                case frontend::GGML_TYPE_Q5_1: pipe = dequantQ5_1Pipeline; break;
                case frontend::GGML_TYPE_Q8_0: pipe = dequantQ8Pipeline; break;
                case frontend::GGML_TYPE_Q8_1: pipe = dequantQ8_1Pipeline; break;
                case frontend::GGML_TYPE_Q2_K:
                case frontend::GGML_TYPE_Q3_K:
                case frontend::GGML_TYPE_Q4_K:
                case frontend::GGML_TYPE_Q5_K:
                case frontend::GGML_TYPE_Q6_K:
                case frontend::GGML_TYPE_Q8_K:
                    pipe = dequantQKPipeline;
                    is_k = true;
                    break;
                default:
                    pipe = nil;
                }
            if (!pipe) return false;

            id<MTLBuffer> srcBuf = [device newBufferWithBytes:desc.raw_quant->data()
                                                      length:expected
                                                     options:MTLResourceStorageModeShared];
            if (!srcBuf) return false;
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                if (!enc) return false;

                [enc setComputePipelineState:pipe];
                [enc setBuffer:srcBuf offset:0 atIndex:0];
                [enc setBuffer:dst offset:0 atIndex:1];
                if (is_k) {
                    QKParamsNative params{static_cast<uint32_t>(rows),
                                          static_cast<uint32_t>(cols),
                                          static_cast<uint32_t>(row_stride)};
                    uint32_t dt = desc.dtype;
                    [enc setBytes:&params length:sizeof(params) atIndex:2];
                    [enc setBytes:&dt length:sizeof(uint32_t) atIndex:3];
                    NSUInteger threadsPerGroup = pipe.threadExecutionWidth;
                    if (threadsPerGroup == 0) threadsPerGroup = 64;
                    NSUInteger groups = (rows + threadsPerGroup - 1) / threadsPerGroup;
                    if (groups == 0) groups = 1;
                    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
                } else {
                    uint32_t c = static_cast<uint32_t>(cols);
                    [enc setBytes:&c length:sizeof(uint32_t) atIndex:2];
                    NSUInteger threadsPerGroup = pipe.threadExecutionWidth;
                    if (threadsPerGroup == 0) threadsPerGroup = 64;
                    NSUInteger total = rows * cols;
                    NSUInteger groups = (total + threadsPerGroup - 1) / threadsPerGroup;
                    if (groups == 0) groups = 1;
                    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
                }
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                return cb.status == MTLCommandBufferStatusCompleted;
            }
        }
#else
        (void)desc;
        (void)rows;
        (void)cols;
        (void)dst;
#endif
        return false;
    }

    bool addBiasInPlace(id<MTLBuffer> buffer,
                        size_t elements,
                        size_t stride_elems,
                        const std::vector<float>* bias) const {
#if defined(__APPLE__)
        if (!bias || bias->empty()) return true;
        if (!buffer || bias->size() != elements || stride_elems == 0) return false;
        if (!available() || !biasAddPipeline) return false;
        if (@available(macOS 11.0, *)) {
            @autoreleasepool {
                id<MTLBuffer> biasBuffer = [device newBufferWithBytes:bias->data()
                                                             length:bias->size() * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
                if (!biasBuffer) return false;
                id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                if (!encoder) return false;
                [encoder setComputePipelineState:biasAddPipeline];
                [encoder setBuffer:buffer offset:0 atIndex:0];
                [encoder setBuffer:biasBuffer offset:0 atIndex:1];
                uint32_t len = static_cast<uint32_t>(elements);
                uint32_t stride = static_cast<uint32_t>(stride_elems);
                [encoder setBytes:&len length:sizeof(uint32_t) atIndex:2];
                [encoder setBytes:&stride length:sizeof(uint32_t) atIndex:3];
                NSUInteger threadsPerGroup = 64;
                NSUInteger groups = (elements + threadsPerGroup - 1) / threadsPerGroup;
                if (groups == 0) groups = 1;
                MTLSize tg = MTLSizeMake(groups, 1, 1);
                MTLSize th = MTLSizeMake(threadsPerGroup, 1, 1);
                [encoder dispatchThreadgroups:tg threadsPerThreadgroup:th];
                [encoder endEncoding];
                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];
                return commandBuffer.status == MTLCommandBufferStatusCompleted;
            }
        }
#else
        (void)buffer;
        (void)elements;
        (void)stride_elems;
        (void)bias;
#endif
        return false;
    }

    bool scatterTokens(id<MTLBuffer> src,
                       id<MTLBuffer> dst,
                       size_t kv_heads,
                       size_t tokens,
                       size_t head_dim,
                       size_t context_length,
                       size_t base_position) const {
#if defined(__APPLE__)
        if (!src || !dst || tokens == 0 || head_dim == 0 || kv_heads == 0) return false;
        size_t kv_stride = context_length * head_dim;
        // CPU fallback when pipeline unavailable.
        if (!available() || !kvWritePipeline) {
            float* s_ptr = reinterpret_cast<float*>([src contents]);
            float* d_ptr = reinterpret_cast<float*>([dst contents]);
            if (!s_ptr || !d_ptr) return false;
            for (size_t kvh = 0; kvh < kv_heads; ++kvh) {
                for (size_t t = 0; t < tokens; ++t) {
                    size_t pos = std::min(context_length - 1, base_position + t);
                    float* dst_row = d_ptr + kvh * kv_stride + pos * head_dim;
                    const float* src_row = s_ptr + (kvh * tokens + t) * head_dim;
                    std::memcpy(dst_row, src_row, head_dim * sizeof(float));
                }
            }
            return true;
        }
        if (@available(macOS 11.0, *)) {
            @autoreleasepool {
                id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                if (!encoder) return false;
                [encoder setComputePipelineState:kvWritePipeline];
                [encoder setBuffer:src offset:0 atIndex:0];
                [encoder setBuffer:dst offset:0 atIndex:1];
                KVWriteParamsNative params{static_cast<uint32_t>(head_dim),
                                           static_cast<uint32_t>(kv_heads),
                                           static_cast<uint32_t>(kv_stride),
                                           static_cast<uint32_t>(tokens),
                                           static_cast<uint32_t>(base_position)};
                [encoder setBytes:&params length:sizeof(params) atIndex:2];
                size_t total = kv_heads * tokens * head_dim;
                NSUInteger threadsPerGroup = kvWritePipeline.threadExecutionWidth;
                if (threadsPerGroup == 0) threadsPerGroup = 64;
                NSUInteger groups = (total + threadsPerGroup - 1) / threadsPerGroup;
                if (groups == 0) groups = 1;
                MTLSize tg = MTLSizeMake(groups, 1, 1);
                MTLSize th = MTLSizeMake(threadsPerGroup, 1, 1);
                [encoder dispatchThreadgroups:tg threadsPerThreadgroup:th];
                [encoder endEncoding];
                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];
                return commandBuffer.status == MTLCommandBufferStatusCompleted;
            }
        }
#else
        (void)src;
        (void)dst;
        (void)kv_heads;
        (void)tokens;
        (void)head_dim;
        (void)context_length;
        (void)base_position;
#endif
        return false;
    }

    bool dequantQ4Block(const std::vector<uint8_t>& src,
                        size_t cols,
                        std::vector<float>& dst) const {
#if defined(__APPLE__)
        if (!available() || !dequantQ4Pipeline) return false;
        if (cols == 0 || src.empty()) return false;
        size_t blocks = (cols + 31) / 32;
        size_t block_bytes = 18;
        if (src.size() < blocks * block_bytes) return false;
        dst.resize(blocks * 32);
        if (@available(macOS 11.0, *)) {
            @autoreleasepool {
                id<MTLBuffer> srcBuf = [device newBufferWithBytes:src.data()
                                                          length:src.size()
                                                         options:MTLResourceStorageModeShared];
                id<MTLBuffer> dstBuf = [device newBufferWithBytes:dst.data()
                                                          length:dst.size() * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
                if (!srcBuf || !dstBuf) return false;
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:dequantQ4Pipeline];
                [enc setBuffer:srcBuf offset:0 atIndex:0];
                [enc setBuffer:dstBuf offset:0 atIndex:1];
                uint32_t c = static_cast<uint32_t>(cols);
                [enc setBytes:&c length:sizeof(uint32_t) atIndex:2];
                NSUInteger threadsPerGroup = dequantQ4Pipeline.threadExecutionWidth;
                if (threadsPerGroup == 0) threadsPerGroup = 64;
                NSUInteger groups = (cols + threadsPerGroup - 1) / threadsPerGroup;
                if (groups == 0) groups = 1;
                [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                std::memcpy(dst.data(), [dstBuf contents], dst.size() * sizeof(float));
                return cb.status == MTLCommandBufferStatusCompleted;
            }
        }
#else
        (void)src;
        (void)cols;
        (void)dst;
#endif
        return false;
    }

    bool dequantQ8Block(const std::vector<uint8_t>& src,
                        size_t cols,
                        std::vector<float>& dst) const {
#if defined(__APPLE__)
        if (!available() || !dequantQ8Pipeline) return false;
        if (cols == 0 || src.empty()) return false;
        size_t blocks = (cols + 31) / 32;
        size_t block_bytes = 34;
        if (src.size() < blocks * block_bytes) return false;
        dst.resize(blocks * 32);
        if (@available(macOS 11.0, *)) {
            @autoreleasepool {
                id<MTLBuffer> srcBuf = [device newBufferWithBytes:src.data()
                                                          length:src.size()
                                                         options:MTLResourceStorageModeShared];
                id<MTLBuffer> dstBuf = [device newBufferWithBytes:dst.data()
                                                          length:dst.size() * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
                if (!srcBuf || !dstBuf) return false;
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:dequantQ8Pipeline];
                [enc setBuffer:srcBuf offset:0 atIndex:0];
                [enc setBuffer:dstBuf offset:0 atIndex:1];
                uint32_t c = static_cast<uint32_t>(cols);
                [enc setBytes:&c length:sizeof(uint32_t) atIndex:2];
                NSUInteger threadsPerGroup = dequantQ8Pipeline.threadExecutionWidth;
                if (threadsPerGroup == 0) threadsPerGroup = 64;
                NSUInteger groups = (cols + threadsPerGroup - 1) / threadsPerGroup;
                if (groups == 0) groups = 1;
                [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                std::memcpy(dst.data(), [dstBuf contents], dst.size() * sizeof(float));
                return cb.status == MTLCommandBufferStatusCompleted;
            }
        }
#else
        (void)src;
        (void)cols;
        (void)dst;
#endif
        return false;
    }

    bool dequantQ4_1Block(const std::vector<uint8_t>& src,
                          size_t cols,
                          std::vector<float>& dst) const {
#if defined(__APPLE__)
        if (!available() || !dequantQ4_1Pipeline) return false;
        if (cols == 0 || src.empty()) return false;
        size_t blocks = (cols + 31) / 32;
        size_t block_bytes = 20;
        if (src.size() < blocks * block_bytes) return false;
        dst.resize(blocks * 32);
        if (@available(macOS 11.0, *)) {
            @autoreleasepool {
                id<MTLBuffer> srcBuf = [device newBufferWithBytes:src.data()
                                                          length:src.size()
                                                         options:MTLResourceStorageModeShared];
                id<MTLBuffer> dstBuf = [device newBufferWithBytes:dst.data()
                                                          length:dst.size() * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
                if (!srcBuf || !dstBuf) return false;
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:dequantQ4_1Pipeline];
                [enc setBuffer:srcBuf offset:0 atIndex:0];
                [enc setBuffer:dstBuf offset:0 atIndex:1];
                uint32_t c = static_cast<uint32_t>(cols);
                [enc setBytes:&c length:sizeof(uint32_t) atIndex:2];
                NSUInteger threadsPerGroup = dequantQ4_1Pipeline.threadExecutionWidth;
                if (threadsPerGroup == 0) threadsPerGroup = 64;
                NSUInteger groups = (cols + threadsPerGroup - 1) / threadsPerGroup;
                if (groups == 0) groups = 1;
                [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                std::memcpy(dst.data(), [dstBuf contents], dst.size() * sizeof(float));
                return cb.status == MTLCommandBufferStatusCompleted;
            }
        }
#else
        (void)src;
        (void)cols;
        (void)dst;
#endif
        return false;
    }

    bool dequantQ5_0Block(const std::vector<uint8_t>& src,
                          size_t cols,
                          std::vector<float>& dst) const {
#if defined(__APPLE__)
        if (!available() || !dequantQ5_0Pipeline) return false;
        if (cols == 0 || src.empty()) return false;
        size_t blocks = (cols + 31) / 32;
        size_t block_bytes = 22;
        if (src.size() < blocks * block_bytes) return false;
        dst.resize(blocks * 32);
        if (@available(macOS 11.0, *)) {
            @autoreleasepool {
                id<MTLBuffer> srcBuf = [device newBufferWithBytes:src.data()
                                                          length:src.size()
                                                         options:MTLResourceStorageModeShared];
                id<MTLBuffer> dstBuf = [device newBufferWithBytes:dst.data()
                                                          length:dst.size() * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
                if (!srcBuf || !dstBuf) return false;
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:dequantQ5_0Pipeline];
                [enc setBuffer:srcBuf offset:0 atIndex:0];
                [enc setBuffer:dstBuf offset:0 atIndex:1];
                uint32_t c = static_cast<uint32_t>(cols);
                [enc setBytes:&c length:sizeof(uint32_t) atIndex:2];
                NSUInteger threadsPerGroup = dequantQ5_0Pipeline.threadExecutionWidth;
                if (threadsPerGroup == 0) threadsPerGroup = 64;
                NSUInteger groups = (cols + threadsPerGroup - 1) / threadsPerGroup;
                if (groups == 0) groups = 1;
                [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                std::memcpy(dst.data(), [dstBuf contents], dst.size() * sizeof(float));
                return cb.status == MTLCommandBufferStatusCompleted;
            }
        }
#else
        (void)src;
        (void)cols;
        (void)dst;
#endif
        return false;
    }

    bool dequantQ5_1Block(const std::vector<uint8_t>& src,
                          size_t cols,
                          std::vector<float>& dst) const {
#if defined(__APPLE__)
        if (!available() || !dequantQ5_1Pipeline) return false;
        if (cols == 0 || src.empty()) return false;
        size_t blocks = (cols + 31) / 32;
        size_t block_bytes = 24;
        if (src.size() < blocks * block_bytes) return false;
        dst.resize(blocks * 32);
        if (@available(macOS 11.0, *)) {
            @autoreleasepool {
                id<MTLBuffer> srcBuf = [device newBufferWithBytes:src.data()
                                                          length:src.size()
                                                         options:MTLResourceStorageModeShared];
                id<MTLBuffer> dstBuf = [device newBufferWithBytes:dst.data()
                                                          length:dst.size() * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
                if (!srcBuf || !dstBuf) return false;
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:dequantQ5_1Pipeline];
                [enc setBuffer:srcBuf offset:0 atIndex:0];
                [enc setBuffer:dstBuf offset:0 atIndex:1];
                uint32_t c = static_cast<uint32_t>(cols);
                [enc setBytes:&c length:sizeof(uint32_t) atIndex:2];
                NSUInteger threadsPerGroup = dequantQ5_1Pipeline.threadExecutionWidth;
                if (threadsPerGroup == 0) threadsPerGroup = 64;
                NSUInteger groups = (cols + threadsPerGroup - 1) / threadsPerGroup;
                if (groups == 0) groups = 1;
                [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                std::memcpy(dst.data(), [dstBuf contents], dst.size() * sizeof(float));
                return cb.status == MTLCommandBufferStatusCompleted;
            }
        }
#else
        (void)src;
        (void)cols;
        (void)dst;
#endif
        return false;
    }

    bool dequantQ8_1Block(const std::vector<uint8_t>& src,
                          size_t cols,
                          std::vector<float>& dst) const {
#if defined(__APPLE__)
        if (!available() || !dequantQ8_1Pipeline) return false;
        if (cols == 0 || src.empty()) return false;
        size_t blocks = (cols + 31) / 32;
        size_t block_bytes = 36;
        if (src.size() < blocks * block_bytes) return false;
        dst.resize(blocks * 32);
        if (@available(macOS 11.0, *)) {
            @autoreleasepool {
                id<MTLBuffer> srcBuf = [device newBufferWithBytes:src.data()
                                                          length:src.size()
                                                         options:MTLResourceStorageModeShared];
                id<MTLBuffer> dstBuf = [device newBufferWithBytes:dst.data()
                                                          length:dst.size() * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
                if (!srcBuf || !dstBuf) return false;
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:dequantQ8_1Pipeline];
                [enc setBuffer:srcBuf offset:0 atIndex:0];
                [enc setBuffer:dstBuf offset:0 atIndex:1];
                uint32_t c = static_cast<uint32_t>(cols);
                [enc setBytes:&c length:sizeof(uint32_t) atIndex:2];
                NSUInteger threadsPerGroup = dequantQ8_1Pipeline.threadExecutionWidth;
                if (threadsPerGroup == 0) threadsPerGroup = 64;
                NSUInteger groups = (cols + threadsPerGroup - 1) / threadsPerGroup;
                if (groups == 0) groups = 1;
                [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                std::memcpy(dst.data(), [dstBuf contents], dst.size() * sizeof(float));
                return cb.status == MTLCommandBufferStatusCompleted;
            }
        }
#else
        (void)src;
        (void)cols;
        (void)dst;
#endif
        return false;
    }

    bool dequantQKRow(const std::vector<uint8_t>& src,
                      uint32_t dtype,
                      size_t cols,
                      size_t row_stride,
                      std::vector<float>& dst) const {
#if defined(__APPLE__)
        if (!available() || !dequantQKPipeline) return false;
        if (cols == 0 || src.empty()) return false;
        dst.resize(cols);
        if (@available(macOS 11.0, *)) {
            @autoreleasepool {
                id<MTLBuffer> srcBuf = [device newBufferWithBytes:src.data()
                                                          length:src.size()
                                                         options:MTLResourceStorageModeShared];
                id<MTLBuffer> dstBuf = [device newBufferWithBytes:dst.data()
                                                          length:dst.size() * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
                if (!srcBuf || !dstBuf) return false;
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:dequantQKPipeline];
                [enc setBuffer:srcBuf offset:0 atIndex:0];
                [enc setBuffer:dstBuf offset:0 atIndex:1];
                QKParamsNative params{1u,
                                      static_cast<uint32_t>(cols),
                                      static_cast<uint32_t>(row_stride)};
                [enc setBytes:&params length:sizeof(params) atIndex:2];
                uint32_t dt = dtype;
                [enc setBytes:&dt length:sizeof(uint32_t) atIndex:3];
                NSUInteger threadsPerGroup = dequantQKPipeline.threadExecutionWidth;
                if (threadsPerGroup == 0) threadsPerGroup = 64;
                NSUInteger groups = (1 + threadsPerGroup - 1) / threadsPerGroup;
                if (groups == 0) groups = 1;
                [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                std::memcpy(dst.data(), [dstBuf contents], dst.size() * sizeof(float));
                return cb.status == MTLCommandBufferStatusCompleted;
            }
        }
#else
        (void)src;
        (void)dtype;
        (void)cols;
        (void)row_stride;
        (void)dst;
#endif
        return false;
    }

    bool applyRotaryGPU(id<MTLBuffer> buffer,
                        size_t count,
                        size_t row_stride,
                        size_t head_dim,
                        size_t rotary_dim,
                        const float* cos_ptr,
                        const float* sin_ptr,
                        size_t pair_stride,
                        size_t offset_tokens = 0) const {
#if defined(__APPLE__)
        if (!available() || !rotaryPipeline) return false;
        if (rotary_dim < 2 || count == 0) return true;
        if (pair_stride == 0 || !cos_ptr || !sin_ptr) return false;
        size_t pairs = rotary_dim / 2;
        if (pairs == 0) return true;
        size_t expected = pair_stride * (count + offset_tokens);
        if (@available(macOS 11.0, *)) {
            @autoreleasepool {
                id<MTLBuffer> cosBuffer = [device newBufferWithBytes:cos_ptr
                                                             length:expected * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
                id<MTLBuffer> sinBuffer = [device newBufferWithBytes:sin_ptr
                                                             length:expected * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
                if (!cosBuffer || !sinBuffer) return false;
                id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                if (!encoder) return false;
                [encoder setComputePipelineState:rotaryPipeline];
                [encoder setBuffer:buffer offset:0 atIndex:0];
                [encoder setBuffer:cosBuffer offset:0 atIndex:1];
                [encoder setBuffer:sinBuffer offset:0 atIndex:2];
                RotaryParamsNative params{
                    static_cast<uint32_t>(head_dim),
                    static_cast<uint32_t>(rotary_dim),
                    static_cast<uint32_t>(row_stride),
                    static_cast<uint32_t>(pair_stride),
                    static_cast<uint32_t>(count),
                    static_cast<uint32_t>(offset_tokens)};
                [encoder setBytes:&params length:sizeof(params) atIndex:3];
                MTLSize threads = MTLSizeMake(count, 1, 1);
                MTLSize threadgroup = MTLSizeMake(1, 1, 1);
                [encoder dispatchThreadgroups:threads threadsPerThreadgroup:threadgroup];
                [encoder endEncoding];
                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];
                return true;
            }
        }
#else
        (void)buffer;
        (void)count;
        (void)row_stride;
        (void)head_dim;
        (void)rotary_dim;
        (void)cos_ptr;
        (void)sin_ptr;
        (void)pair_stride;
#endif
        return false;
    }

    bool matmul(const std::string& weight_name,
                const std::vector<float>& weights,
                const std::vector<float>& input,
                size_t rows,
                size_t cols,
                bool transpose_weight,
                std::vector<float>& output,
                const std::vector<float>* bias = nullptr) const {
#if defined(__APPLE__)
        (void)weight_name;  // F32 path stages into a row-aligned buffer below;
                            // caching the raw float bytes wouldn't fit that layout.
        if (!available() || weights.size() != rows * cols) {
            return false;
        }
        if (transpose_weight) {
            return false;
        }
        if (input.size() != cols) {
            return false;
        }
        if (@available(macOS 11.0, *)) {
            @autoreleasepool {
            size_t leftRowBytes = alignedRowBytes(cols);
            size_t rightRowBytes = alignedRowBytes(1);
            size_t resultRowBytes = alignedRowBytes(1);

            id<MTLBuffer> leftBuffer = [device newBufferWithLength:leftRowBytes * rows
                                                             options:MTLResourceStorageModeShared];
            id<MTLBuffer> rightBuffer = [device newBufferWithLength:rightRowBytes * cols
                                                              options:MTLResourceStorageModeShared];
            id<MTLBuffer> resultBuffer = [device newBufferWithLength:resultRowBytes * rows
                                                               options:MTLResourceStorageModeShared];
            if (!leftBuffer || !rightBuffer || !resultBuffer) {
                return false;
            }

            uint8_t* leftPtr = reinterpret_cast<uint8_t*>([leftBuffer contents]);
            for (size_t r = 0; r < rows; ++r) {
                std::memcpy(leftPtr + r * leftRowBytes,
                            weights.data() + r * cols,
                            cols * sizeof(float));
                if (leftRowBytes > cols * sizeof(float)) {
                    std::memset(leftPtr + r * leftRowBytes + cols * sizeof(float),
                                0,
                                leftRowBytes - cols * sizeof(float));
                }
            }

            uint8_t* rightPtr = reinterpret_cast<uint8_t*>([rightBuffer contents]);
            for (size_t r = 0; r < cols; ++r) {
                std::memcpy(rightPtr + r * rightRowBytes,
                            input.data() + r,
                            sizeof(float));
                if (rightRowBytes > sizeof(float)) {
                    std::memset(rightPtr + r * rightRowBytes + sizeof(float),
                                0,
                                rightRowBytes - sizeof(float));
                }
            }

            MPSMatrixDescriptor* leftDesc =
                [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                       columns:cols
                                                      rowBytes:leftRowBytes
                                                      dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor* rightDesc =
                [MPSMatrixDescriptor matrixDescriptorWithRows:cols
                                                       columns:1
                                                      rowBytes:rightRowBytes
                                                      dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor* resultDesc =
                [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                       columns:1
                                                      rowBytes:resultRowBytes
                                                      dataType:MPSDataTypeFloat32];

            MPSMatrix* leftMatrix = [[MPSMatrix alloc] initWithBuffer:leftBuffer descriptor:leftDesc];
            MPSMatrix* rightMatrix = [[MPSMatrix alloc] initWithBuffer:rightBuffer descriptor:rightDesc];
            MPSMatrix* resultMatrix = [[MPSMatrix alloc] initWithBuffer:resultBuffer descriptor:resultDesc];

            MPSMatrixMultiplication* mm =
                [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                transposeLeft:false
                                               transposeRight:false
                                                  resultRows:rows
                                               resultColumns:1
                                            interiorColumns:cols
                                                       alpha:1.0f
                                                        beta:0.0f];

            if (!mm) {
                return false;
            }

            id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
            [mm encodeToCommandBuffer:commandBuffer
                           leftMatrix:leftMatrix
                          rightMatrix:rightMatrix
                         resultMatrix:resultMatrix];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];

            size_t stride_elems = resultRowBytes / sizeof(float);
            if (stride_elems == 0) stride_elems = 1;
            output.resize(rows);
            const uint8_t* resultPtr = reinterpret_cast<const uint8_t*>([resultBuffer contents]);
            for (size_t r = 0; r < rows; ++r) {
                std::memcpy(output.data() + r,
                            resultPtr + r * resultRowBytes,
                            sizeof(float));
            }
            if (bias && !bias->empty()) {
                for (size_t i = 0; i < rows && i < bias->size(); ++i) {
                    output[i] += (*bias)[i];
                }
            }
            }
            return true;
        }
#endif
        (void)weights;
        (void)input;
        (void)rows;
        (void)cols;
        (void)transpose_weight;
        (void)output;
        return false;
    }

bool matmulQuant(const std::string& weight_name,
                 const std::vector<uint8_t>& weights,
                 const std::vector<float>& input,
                 size_t rows,
                 size_t cols,
                 size_t row_stride,
                 uint32_t quant_version,
                 id<MTLComputePipelineState> pipeline,
                 const std::vector<float>* bias,
                 std::vector<float>& output) const {
#if defined(__APPLE__)
        (void)quant_version;
        if (!available() || !pipeline) return false;
        if (input.size() != cols) return false;
        if (weights.size() < row_stride * rows) return false;
        if (@available(macOS 11.0, *)) {
            @autoreleasepool {
                id<MTLBuffer> weightBuffer = getOrCacheWeight(weight_name, weights);
                id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input.data()
                                                                length:input.size() * sizeof(float)
                                                               options:MTLResourceStorageModeShared];
                id<MTLBuffer> outputBuffer = [device newBufferWithLength:rows * sizeof(float)
                                                                  options:MTLResourceStorageModeShared];
                if (!weightBuffer || !inputBuffer || !outputBuffer) return false;

                id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                if (!encoder) return false;
                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:weightBuffer offset:0 atIndex:0];
                [encoder setBuffer:inputBuffer offset:0 atIndex:1];
                [encoder setBuffer:outputBuffer offset:0 atIndex:2];
                Q4_0ParamsNative params = {static_cast<uint32_t>(rows),
                                           static_cast<uint32_t>(cols),
                                           static_cast<uint32_t>(row_stride),
                                           quant_version};
                [encoder setBytes:&params length:sizeof(Q4_0ParamsNative) atIndex:3];
                NSUInteger threadsPerGroup = pipeline.threadExecutionWidth;
                if (threadsPerGroup == 0) threadsPerGroup = 32;
                NSUInteger groups = (rows + threadsPerGroup - 1) / threadsPerGroup;
                if (groups == 0) groups = 1;
                MTLSize tg = MTLSizeMake(groups, 1, 1);
                MTLSize th = MTLSizeMake(threadsPerGroup, 1, 1);
                [encoder dispatchThreadgroups:tg threadsPerThreadgroup:th];
                [encoder endEncoding];
                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];

                bool ok = commandBuffer.status == MTLCommandBufferStatusCompleted;
                output.resize(rows);
                std::memcpy(output.data(), [outputBuffer contents], rows * sizeof(float));
                if (bias && !bias->empty()) {
                    for (size_t i = 0; i < rows && i < bias->size(); ++i) {
                        output[i] += (*bias)[i];
                    }
                }
                return ok;
            }
        }
#endif
        (void)weights;
        (void)input;
        (void)rows;
        (void)cols;
        (void)row_stride;
        (void)output;
        return false;
    }

bool matmulQuantTransposed(const std::string& weight_name,
                           const std::vector<uint8_t>& weights,
                           const std::vector<float>& input,
                           size_t rows,
                           size_t cols,
                           size_t row_stride,
                           uint32_t quant_version,
                           id<MTLComputePipelineState> pipeline,
                           const std::vector<float>* bias,
                           std::vector<float>& output) const {
#if defined(__APPLE__)
        (void)quant_version;
        if (!available() || !pipeline) return false;
        if (input.size() != rows) return false;
        if (weights.size() < row_stride * rows) return false;
        if (@available(macOS 11.0, *)) {
            @autoreleasepool {
                id<MTLBuffer> weightBuffer = getOrCacheWeight(weight_name, weights);
                id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input.data()
                                                                length:input.size() * sizeof(float)
                                                               options:MTLResourceStorageModeShared];
                id<MTLBuffer> outputBuffer = [device newBufferWithLength:cols * sizeof(float)
                                                                  options:MTLResourceStorageModeShared];
                if (!weightBuffer || !inputBuffer || !outputBuffer) return false;

                id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                if (!encoder) return false;
                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:weightBuffer offset:0 atIndex:0];
                [encoder setBuffer:inputBuffer offset:0 atIndex:1];
                [encoder setBuffer:outputBuffer offset:0 atIndex:2];
                Q4_0ParamsNative params = {static_cast<uint32_t>(rows),
                                           static_cast<uint32_t>(cols),
                                           static_cast<uint32_t>(row_stride),
                                           quant_version};
                [encoder setBytes:&params length:sizeof(Q4_0ParamsNative) atIndex:3];
                NSUInteger threadsPerGroup = pipeline.threadExecutionWidth;
                if (threadsPerGroup == 0) threadsPerGroup = 32;
                NSUInteger groups = (cols + threadsPerGroup - 1) / threadsPerGroup;
                if (groups == 0) groups = 1;
                MTLSize tg = MTLSizeMake(groups, 1, 1);
                MTLSize th = MTLSizeMake(threadsPerGroup, 1, 1);
                [encoder dispatchThreadgroups:tg threadsPerThreadgroup:th];
                [encoder endEncoding];
                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];

                bool ok = commandBuffer.status == MTLCommandBufferStatusCompleted;
                output.resize(cols);
                std::memcpy(output.data(), [outputBuffer contents], cols * sizeof(float));
                if (bias && !bias->empty()) {
                    for (size_t i = 0; i < cols && i < bias->size(); ++i) {
                        output[i] += (*bias)[i];
                    }
                }
                return ok;
            }
        }
#endif
        (void)weights;
        (void)input;
        (void)rows;
        (void)cols;
        (void)row_stride;
        (void)output;
        return false;
    }

bool matmulQuantKTransposed(const std::string& weight_name,
                            const std::vector<uint8_t>& weights,
                            const std::vector<float>& input,
                            size_t rows,
                            size_t cols,
                            size_t row_stride,
                            id<MTLComputePipelineState> pipeline,
                            const std::vector<float>* bias,
                            std::vector<float>& output) const {
#if defined(__APPLE__)
        if (!available() || !pipeline) return false;
        if (input.size() != rows) return false;
        if (weights.size() < row_stride * rows) return false;
        if (@available(macOS 11.0, *)) {
            @autoreleasepool {
                id<MTLBuffer> weightBuffer = getOrCacheWeight(weight_name, weights);
                id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input.data()
                                                                length:input.size() * sizeof(float)
                                                               options:MTLResourceStorageModeShared];
                id<MTLBuffer> outputBuffer = [device newBufferWithLength:cols * sizeof(float)
                                                                  options:MTLResourceStorageModeShared];
                if (!weightBuffer || !inputBuffer || !outputBuffer) return false;

                id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                if (!encoder) return false;
                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:weightBuffer offset:0 atIndex:0];
                [encoder setBuffer:inputBuffer offset:0 atIndex:1];
                [encoder setBuffer:outputBuffer offset:0 atIndex:2];
                QKParamsNative params = {static_cast<uint32_t>(rows),
                                         static_cast<uint32_t>(cols),
                                         static_cast<uint32_t>(row_stride)};
                [encoder setBytes:&params length:sizeof(QKParamsNative) atIndex:3];
                NSUInteger threadsPerGroup = pipeline.threadExecutionWidth;
                if (threadsPerGroup == 0) threadsPerGroup = 32;
                NSUInteger groups = (cols + threadsPerGroup - 1) / threadsPerGroup;
                if (groups == 0) groups = 1;
                MTLSize tg = MTLSizeMake(groups, 1, 1);
                MTLSize th = MTLSizeMake(threadsPerGroup, 1, 1);
                [encoder dispatchThreadgroups:tg threadsPerThreadgroup:th];
                [encoder endEncoding];
                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];

                bool ok = commandBuffer.status == MTLCommandBufferStatusCompleted;
                output.resize(cols);
                std::memcpy(output.data(), [outputBuffer contents], cols * sizeof(float));
                if (bias && !bias->empty()) {
                    for (size_t i = 0; i < cols && i < bias->size(); ++i) {
                        output[i] += (*bias)[i];
                    }
                }
                return ok;
            }
        }
#endif
        (void)weights;
        (void)input;
        (void)rows;
        (void)cols;
        (void)row_stride;
        (void)output;
        return false;
    }

bool matmulQuantK(const std::string& weight_name,
                  const std::vector<uint8_t>& weights,
                  const std::vector<float>& input,
                  size_t rows,
                  size_t cols,
                  size_t row_stride,
                  id<MTLComputePipelineState> pipeline,
                  const std::vector<float>* bias,
                  std::vector<float>& output) const {
#if defined(__APPLE__)
        if (!available() || !pipeline) return false;
        if (input.size() != cols) return false;
        if (weights.size() < row_stride * rows) return false;
        if (@available(macOS 11.0, *)) {
            @autoreleasepool {
                id<MTLBuffer> weightBuffer = getOrCacheWeight(weight_name, weights);
                id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input.data()
                                                                length:input.size() * sizeof(float)
                                                               options:MTLResourceStorageModeShared];
                id<MTLBuffer> outputBuffer = [device newBufferWithLength:rows * sizeof(float)
                                                                  options:MTLResourceStorageModeShared];
                if (!weightBuffer || !inputBuffer || !outputBuffer) return false;

                id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                if (!encoder) return false;
                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:weightBuffer offset:0 atIndex:0];
                [encoder setBuffer:inputBuffer offset:0 atIndex:1];
                [encoder setBuffer:outputBuffer offset:0 atIndex:2];
                QKParamsNative params = {static_cast<uint32_t>(rows),
                                         static_cast<uint32_t>(cols),
                                         static_cast<uint32_t>(row_stride)};
                [encoder setBytes:&params length:sizeof(QKParamsNative) atIndex:3];
                NSUInteger threadsPerGroup = pipeline.threadExecutionWidth;
                if (threadsPerGroup == 0) threadsPerGroup = 32;
                NSUInteger groups = (rows + threadsPerGroup - 1) / threadsPerGroup;
                if (groups == 0) groups = 1;
                MTLSize tg = MTLSizeMake(groups, 1, 1);
                MTLSize th = MTLSizeMake(threadsPerGroup, 1, 1);
                [encoder dispatchThreadgroups:tg threadsPerThreadgroup:th];
                [encoder endEncoding];
                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];

                bool ok = commandBuffer.status == MTLCommandBufferStatusCompleted;
                output.resize(rows);
                std::memcpy(output.data(), [outputBuffer contents], rows * sizeof(float));
                if (bias && !bias->empty()) {
                    for (size_t i = 0; i < rows && i < bias->size(); ++i) {
                        output[i] += (*bias)[i];
                    }
                }
                return ok;
            }
        }
#endif
        (void)weights;
        (void)input;
        (void)rows;
        (void)cols;
        (void)row_stride;
        (void)output;
        return false;
    }

bool matmulQ4_0(const std::string& weight_name,
                const std::vector<uint8_t>& weights,
                const std::vector<float>& input,
                size_t rows,
                size_t cols,
                size_t row_stride,
                uint32_t quant_version,
                const std::vector<float>* bias,
                std::vector<float>& output) const {
    return matmulQuant(weight_name, weights, input, rows, cols, row_stride, quant_version,
                       q4MatmulPipeline, bias, output);
}

bool matmulQ4_0Transposed(const std::string& weight_name,
                          const std::vector<uint8_t>& weights,
                          const std::vector<float>& input,
                          size_t rows,
                          size_t cols,
                          size_t row_stride,
                          uint32_t quant_version,
                          const std::vector<float>* bias,
                          std::vector<float>& output) const {
    return matmulQuantTransposed(weight_name, weights, input, rows, cols, row_stride, quant_version,
                                 q4MatmulTransposedPipeline, bias, output);
}

bool matmulQ4_1(const std::string& weight_name,
                const std::vector<uint8_t>& weights,
                const std::vector<float>& input,
                size_t rows,
                size_t cols,
                size_t row_stride,
                const std::vector<float>* bias,
                std::vector<float>& output) const {
    return matmulQuant(weight_name, weights, input, rows, cols, row_stride, 0,
                       q4_1MatmulPipeline, bias, output);
}

bool matmulQ5_0(const std::string& weight_name,
                const std::vector<uint8_t>& weights,
                const std::vector<float>& input,
                size_t rows,
                size_t cols,
                size_t row_stride,
                const std::vector<float>* bias,
                std::vector<float>& output) const {
    return matmulQuant(weight_name, weights, input, rows, cols, row_stride, 0,
                       q5_0MatmulPipeline, bias, output);
}

bool matmulQ5_1(const std::string& weight_name,
                const std::vector<uint8_t>& weights,
                const std::vector<float>& input,
                size_t rows,
                size_t cols,
                size_t row_stride,
                const std::vector<float>* bias,
                std::vector<float>& output) const {
    return matmulQuant(weight_name, weights, input, rows, cols, row_stride, 0,
                       q5_1MatmulPipeline, bias, output);
}

bool matmulQ2_K(const std::string& weight_name,
                const std::vector<uint8_t>& weights,
                const std::vector<float>& input,
                size_t rows,
                size_t cols,
                size_t row_stride,
                const std::vector<float>* bias,
                std::vector<float>& output) const {
#if defined(__APPLE__)
    if (!available() || !q2KMatmulPipeline) return false;
    auto result = matmulQuantK(weight_name, weights, input, rows, cols, row_stride, q2KMatmulPipeline, bias, output);
    return result;
#else
    (void)weights;
    (void)input;
        (void)rows;
        (void)cols;
        (void)row_stride;
        (void)output;
        return false;
#endif
    }

bool matmulQ3_K(const std::string& weight_name,
                const std::vector<uint8_t>& weights,
                const std::vector<float>& input,
                size_t rows,
                size_t cols,
                size_t row_stride,
                const std::vector<float>* bias,
                std::vector<float>& output) const {
#if defined(__APPLE__)
    if (!available() || !q3KMatmulPipeline) return false;
    return matmulQuantK(weight_name, weights, input, rows, cols, row_stride, q3KMatmulPipeline, bias, output);
#else
    (void)weights;
        (void)input;
        (void)rows;
        (void)cols;
        (void)row_stride;
        (void)output;
        return false;
#endif
    }

bool matmulQ4_K(const std::string& weight_name,
                const std::vector<uint8_t>& weights,
                const std::vector<float>& input,
                size_t rows,
                size_t cols,
                size_t row_stride,
                const std::vector<float>* bias,
                std::vector<float>& output) const {
    (void)weight_name;  // CPU reference path; no Metal weight upload.
#if defined(__APPLE__)
    // Q4_K matches the K-quant kernel template that matmulQ6_K uses. That
    // path was validated and shipped in 748ff30 with full logits-cosine
    // parity on TinyLlama Q4_0 (the model has one Q6_K tensor: lm_head). The
    // identical one-line wiring works here too:
    //     return matmulQuantK(weight_name, weights, input, rows, cols,
    //                         row_stride, q4KMatmulPipeline, bias, output);
    // Flip it when a Q4_K-quantized model is on the bench AND you have run
    // `mlc compare --metal-vs-cpu` against it to confirm cosine 1.0 at the
    // logits boundary. Until then we keep the host scalar loop so callers
    // of an untested Metal path can't get silently-wrong tokens.
    if (input.size() != cols || weights.size() < row_stride * rows) return false;
    output.resize(rows, 0.0f);
    for (size_t r = 0; r < rows; ++r) {
        const uint8_t* row_ptr = weights.data() + r * row_stride;
        output[r] = dotProductRowQ4_K(row_ptr, cols, input.data());
        if (bias && r < bias->size()) output[r] += (*bias)[r];
    }
    return true;
#else
    (void)weights;
    (void)input;
    (void)rows;
    (void)cols;
    (void)row_stride;
    (void)bias;
    (void)output;
    return false;
#endif
}

    bool matmulQ5_K(const std::string& weight_name,
                    const std::vector<uint8_t>& weights,
                    const std::vector<float>& input,
                    size_t rows,
                    size_t cols,
                    size_t row_stride,
                    const std::vector<float>* bias,
                    std::vector<float>& output) const {
#if defined(__APPLE__)
        if (!available() || !q5KMatmulPipeline) return false;
        return matmulQuantK(weight_name, weights, input, rows, cols, row_stride, q5KMatmulPipeline, bias, output);
#else
        (void)weights;
        (void)input;
        (void)rows;
        (void)cols;
        (void)row_stride;
        (void)output;
        return false;
#endif
    }

    bool matmulQ6_K(const std::string& weight_name,
                    const std::vector<uint8_t>& weights,
                    const std::vector<float>& input,
                    size_t rows,
                    size_t cols,
                    size_t row_stride,
                    const std::vector<float>* bias,
                    std::vector<float>& output) const {
#if defined(__APPLE__)
        if (!available() || !q6KMatmulPipeline) return false;
        return matmulQuantK(weight_name, weights, input, rows, cols, row_stride,
                            q6KMatmulPipeline, bias, output);
#else
        (void)weights;
        (void)input;
        (void)rows;
        (void)cols;
        (void)row_stride;
        (void)output;
        return false;
#endif
    }

    bool matmulQ6_KTransposed(const std::string& weight_name,
                              const std::vector<uint8_t>& weights,
                              const std::vector<float>& input,
                              size_t rows,
                              size_t cols,
                              size_t row_stride,
                              const std::vector<float>* bias,
                              std::vector<float>& output) const {
#if defined(__APPLE__)
        if (!available() || !q6KMatmulTransposedPipeline) return false;
        return matmulQuantKTransposed(weight_name, weights, input, rows, cols, row_stride,
                                      q6KMatmulTransposedPipeline, bias, output);
#else
        (void)weights;
        (void)input;
        (void)rows;
        (void)cols;
        (void)row_stride;
        (void)output;
        return false;
#endif
    }

    bool matmulQ8_K(const std::string& weight_name,
                    const std::vector<uint8_t>& weights,
                    const std::vector<float>& input,
                    size_t rows,
                    size_t cols,
                    size_t row_stride,
                    const std::vector<float>* bias,
                    std::vector<float>& output) const {
#if defined(__APPLE__)
        if (!available() || !q8KMatmulPipeline) return false;
        return matmulQuantK(weight_name, weights, input, rows, cols, row_stride, q8KMatmulPipeline, bias, output);
#else
        (void)weights;
        (void)input;
        (void)rows;
        (void)cols;
        (void)row_stride;
        (void)output;
        return false;
#endif
    }

    bool matmulQ8_0(const std::string& weight_name,
                    const std::vector<uint8_t>& weights,
                    const std::vector<float>& input,
                    size_t rows,
                    size_t cols,
                    size_t row_stride,
                    const std::vector<float>* bias,
                    std::vector<float>& output) const {
#if defined(__APPLE__)
        if (!available() || !q8_0MatmulPipeline) return false;
        return matmulQuant(weight_name, weights, input, rows, cols, row_stride, 0, q8_0MatmulPipeline, bias, output);
#else
        (void)weights;
        (void)input;
        (void)rows;
        (void)cols;
        (void)row_stride;
        (void)output;
        return false;
#endif
    }

    bool matmulQ8_1(const std::string& weight_name,
                    const std::vector<uint8_t>& weights,
                    const std::vector<float>& input,
                    size_t rows,
                    size_t cols,
                    size_t row_stride,
                    const std::vector<float>* bias,
                    std::vector<float>& output) const {
#if defined(__APPLE__)
        if (!available() || !q8_1MatmulPipeline) return false;
        return matmulQuant(weight_name, weights, input, rows, cols, row_stride, 0, q8_1MatmulPipeline, bias, output);
#else
        (void)weights;
        (void)input;
        (void)rows;
        (void)cols;
        (void)row_stride;
        (void)output;
        return false;
#endif
    }

    bool feedForward(const std::vector<float>& gate,
                     const std::vector<float>& up,
                     std::vector<float>& output) const {
#if defined(__APPLE__)
        if (!available() || !ffnPipeline) return false;
        if (gate.size() != up.size() || gate.empty()) return false;
        if (gate.size() > std::numeric_limits<uint32_t>::max()) return false;

        size_t bytes = gate.size() * sizeof(float);
        output.resize(gate.size());
        if (@available(macOS 11.0, *)) {
            @autoreleasepool {
                id<MTLBuffer> gateBuffer = [device newBufferWithBytes:gate.data()
                                                               length:bytes
                                                              options:MTLResourceStorageModeShared];
                id<MTLBuffer> upBuffer = [device newBufferWithBytes:up.data()
                                                             length:bytes
                                                            options:MTLResourceStorageModeShared];
                id<MTLBuffer> outBuffer = [device newBufferWithLength:bytes
                                                               options:MTLResourceStorageModeShared];
                if (!gateBuffer || !upBuffer || !outBuffer) {
                    return false;
                }

                id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                if (!encoder) return false;
                [encoder setComputePipelineState:ffnPipeline];
                [encoder setBuffer:gateBuffer offset:0 atIndex:0];
                [encoder setBuffer:upBuffer offset:0 atIndex:1];
                [encoder setBuffer:outBuffer offset:0 atIndex:2];
                uint32_t length = static_cast<uint32_t>(gate.size());
                [encoder setBytes:&length length:sizeof(uint32_t) atIndex:3];

                NSUInteger threadWidth = ffnPipeline.threadExecutionWidth;
                if (threadWidth == 0) threadWidth = 32;
                NSUInteger threadsPerGroup = threadWidth;
                NSUInteger threadgroups = (gate.size() + threadsPerGroup - 1) / threadsPerGroup;

                MTLSize tgSize = MTLSizeMake(threadgroups, 1, 1);
                MTLSize thSize = MTLSizeMake(threadsPerGroup, 1, 1);
                [encoder dispatchThreadgroups:tgSize threadsPerThreadgroup:thSize];
                [encoder endEncoding];
                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];

                std::memcpy(output.data(), [outBuffer contents], bytes);
                return true;
            }
        }
#endif
        (void)gate;
        (void)up;
        (void)output;
        return false;
    }

    bool addVectors(const std::vector<float>& a,
                    const std::vector<float>& b,
                    const std::vector<float>* bias,
                    std::vector<float>& output) const {
#if defined(__APPLE__)
        if (!available()) return false;
        if (a.size() != b.size() || a.empty()) return false;
        bool use_bias = bias && !bias->empty();
        if (use_bias && bias->size() != a.size()) return false;
        id<MTLComputePipelineState> pipe = use_bias ? addResidualPipeline : addPipeline;
        if (!pipe) return false;
        if (a.size() > std::numeric_limits<uint32_t>::max()) return false;
        size_t bytes = a.size() * sizeof(float);
        output.resize(a.size());
        if (@available(macOS 11.0, *)) {
            @autoreleasepool {
                id<MTLBuffer> aBuffer = [device newBufferWithBytes:a.data()
                                                             length:bytes
                                                            options:MTLResourceStorageModeShared];
                id<MTLBuffer> bBuffer = [device newBufferWithBytes:b.data()
                                                             length:bytes
                                                            options:MTLResourceStorageModeShared];
                id<MTLBuffer> outBuffer = [device newBufferWithLength:bytes
                                                               options:MTLResourceStorageModeShared];
                if (!aBuffer || !bBuffer || !outBuffer) return false;

                id<MTLBuffer> biasBuffer = nil;
                if (use_bias) {
                    biasBuffer = [device newBufferWithBytes:bias->data()
                                                     length:bytes
                                                    options:MTLResourceStorageModeShared];
                    if (!biasBuffer) return false;
                }

                id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                if (!encoder) return false;
                [encoder setComputePipelineState:pipe];
                [encoder setBuffer:aBuffer offset:0 atIndex:0];
                [encoder setBuffer:bBuffer offset:0 atIndex:1];
                if (use_bias) {
                    [encoder setBuffer:biasBuffer offset:0 atIndex:2];
                    [encoder setBuffer:outBuffer offset:0 atIndex:3];
                    uint32_t length = static_cast<uint32_t>(a.size());
                    [encoder setBytes:&length length:sizeof(uint32_t) atIndex:4];
                    uint32_t has_bias = 1;
                    [encoder setBytes:&has_bias length:sizeof(uint32_t) atIndex:5];
                } else {
                    [encoder setBuffer:outBuffer offset:0 atIndex:2];
                    uint32_t length = static_cast<uint32_t>(a.size());
                    [encoder setBytes:&length length:sizeof(uint32_t) atIndex:3];
                }

                NSUInteger threadWidth = pipe.threadExecutionWidth;
                if (threadWidth == 0) threadWidth = 32;
                NSUInteger threadsPerGroup = threadWidth;
                NSUInteger threadgroups = (a.size() + threadsPerGroup - 1) / threadsPerGroup;
                if (threadgroups == 0) threadgroups = 1;
                MTLSize tgSize = MTLSizeMake(threadgroups, 1, 1);
                MTLSize thSize = MTLSizeMake(threadsPerGroup, 1, 1);
                [encoder dispatchThreadgroups:tgSize threadsPerThreadgroup:thSize];
                [encoder endEncoding];
                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];

                std::memcpy(output.data(), [outBuffer contents], bytes);
                return true;
            }
        }
#endif
        (void)a;
        (void)b;
        (void)output;
        return false;
    }

    bool rmsNorm(const std::vector<float>& input,
                 const std::vector<float>& weight,
                 float epsilon,
                 std::vector<float>& output) const {
#if defined(__APPLE__)
        if (!available() || !normPipeline) return false;
        if (input.size() != weight.size() || input.empty()) return false;
        if (input.size() > std::numeric_limits<uint32_t>::max()) return false;
        size_t bytes = input.size() * sizeof(float);
        output.resize(input.size());
        if (@available(macOS 11.0, *)) {
            @autoreleasepool {
                id<MTLBuffer> inBuffer = [device newBufferWithBytes:input.data()
                                                             length:bytes
                                                            options:MTLResourceStorageModeShared];
                id<MTLBuffer> wBuffer = [device newBufferWithBytes:weight.data()
                                                            length:bytes
                                                           options:MTLResourceStorageModeShared];
                id<MTLBuffer> outBuffer = [device newBufferWithLength:bytes
                                                               options:MTLResourceStorageModeShared];
                if (!inBuffer || !wBuffer || !outBuffer) return false;
                id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                if (!encoder) return false;
                [encoder setComputePipelineState:normPipeline];
                [encoder setBuffer:inBuffer offset:0 atIndex:0];
                [encoder setBuffer:wBuffer offset:0 atIndex:1];
                [encoder setBuffer:outBuffer offset:0 atIndex:2];
                uint32_t length = static_cast<uint32_t>(input.size());
                [encoder setBytes:&length length:sizeof(uint32_t) atIndex:3];
                float eps = epsilon;
                [encoder setBytes:&eps length:sizeof(float) atIndex:4];

                MTLSize tgSize = MTLSizeMake(1, 1, 1);
                MTLSize thSize = MTLSizeMake(1, 1, 1);
                [encoder dispatchThreadgroups:tgSize threadsPerThreadgroup:thSize];
                [encoder endEncoding];
                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];

                std::memcpy(output.data(), [outBuffer contents], bytes);
                return true;
            }
        }
#endif
        (void)input;
        (void)weight;
        (void)epsilon;
        (void)output;
        return false;
    }

    bool layerNorm(const std::vector<float>& input,
                   const std::vector<float>& weight,
                   const std::vector<float>* bias,
                   float epsilon,
                   std::vector<float>& output) const {
#if defined(__APPLE__)
        if (!available() || !layerNormPipeline) return false;
        if (input.size() != weight.size() || input.empty()) return false;
        if (input.size() > std::numeric_limits<uint32_t>::max()) return false;
        bool use_bias = bias && !bias->empty();
        if (use_bias && bias->size() != input.size()) return false;
        size_t bytes = input.size() * sizeof(float);
        output.resize(input.size());
        if (@available(macOS 11.0, *)) {
            @autoreleasepool {
                id<MTLBuffer> inBuffer = [device newBufferWithBytes:input.data()
                                                             length:bytes
                                                            options:MTLResourceStorageModeShared];
                id<MTLBuffer> wBuffer = [device newBufferWithBytes:weight.data()
                                                            length:bytes
                                                           options:MTLResourceStorageModeShared];
                id<MTLBuffer> bBuffer = nil;
                if (use_bias) {
                    bBuffer = [device newBufferWithBytes:bias->data()
                                                 length:bytes
                                                options:MTLResourceStorageModeShared];
                }
                id<MTLBuffer> outBuffer = [device newBufferWithLength:bytes
                                                               options:MTLResourceStorageModeShared];
                if (!inBuffer || !wBuffer || !outBuffer) return false;
                id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                if (!encoder) return false;
                [encoder setComputePipelineState:layerNormPipeline];
                [encoder setBuffer:inBuffer offset:0 atIndex:0];
                [encoder setBuffer:wBuffer offset:0 atIndex:1];
                if (use_bias && bBuffer) {
                    [encoder setBuffer:bBuffer offset:0 atIndex:2];
                } else {
                    id<MTLBuffer> zeroBuffer = [device newBufferWithLength:sizeof(float)
                                                                     options:MTLResourceStorageModeShared];
                    float zero = 0.0f;
                    std::memcpy([zeroBuffer contents], &zero, sizeof(float));
                    [encoder setBuffer:zeroBuffer offset:0 atIndex:2];
                }
                [encoder setBuffer:outBuffer offset:0 atIndex:3];
                uint32_t length = static_cast<uint32_t>(input.size());
                [encoder setBytes:&length length:sizeof(uint32_t) atIndex:4];
                float eps = epsilon;
                [encoder setBytes:&eps length:sizeof(float) atIndex:5];
                uint32_t has_bias = use_bias ? 1 : 0;
                [encoder setBytes:&has_bias length:sizeof(uint32_t) atIndex:6];

                MTLSize tgSize = MTLSizeMake(1, 1, 1);
                MTLSize thSize = MTLSizeMake(1, 1, 1);
                [encoder dispatchThreadgroups:tgSize threadsPerThreadgroup:thSize];
                [encoder endEncoding];
                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];

                std::memcpy(output.data(), [outBuffer contents], bytes);
                return true;
            }
        }
#endif
        (void)input;
        (void)weight;
        (void)bias;
        (void)epsilon;
        (void)output;
        return false;
    }

    bool softmax(const std::vector<float>& input,
                 std::vector<float>& output) const {
#if defined(__APPLE__)
        if (!available() || !softmaxPipeline) return false;
        if (input.empty()) return false;
        if (input.size() > std::numeric_limits<uint32_t>::max()) return false;
        size_t bytes = input.size() * sizeof(float);
        output.resize(input.size());
        if (@available(macOS 11.0, *)) {
            @autoreleasepool {
                id<MTLBuffer> inBuffer = [device newBufferWithBytes:input.data()
                                                            length:bytes
                                                           options:MTLResourceStorageModeShared];
                id<MTLBuffer> outBuffer = [device newBufferWithLength:bytes
                                                               options:MTLResourceStorageModeShared];
                if (!inBuffer || !outBuffer) return false;
                id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                if (!encoder) return false;
                [encoder setComputePipelineState:softmaxPipeline];
                [encoder setBuffer:inBuffer offset:0 atIndex:0];
                [encoder setBuffer:outBuffer offset:0 atIndex:1];
                uint32_t length = static_cast<uint32_t>(input.size());
                [encoder setBytes:&length length:sizeof(uint32_t) atIndex:2];
                MTLSize tgSize = MTLSizeMake(1, 1, 1);
                MTLSize thSize = MTLSizeMake(1, 1, 1);
                [encoder dispatchThreadgroups:tgSize threadsPerThreadgroup:thSize];
                [encoder endEncoding];
                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];
                std::memcpy(output.data(), [outBuffer contents], bytes);
                return true;
            }
        }
#endif
        (void)input;
        (void)output;
        return false;
    }

    bool attention(const std::vector<float>& q,
                   const std::vector<float>& k,
                   const std::vector<float>& v,
                   size_t num_heads,
                   size_t kv_heads,
                   size_t head_dim,
                   size_t context_length,
                   const std::vector<float>& mask,
                   const std::vector<float>* alibi_slopes,
                   size_t position,
                   size_t rotary_dim,
                   float rope_freq_base,
                   float rope_freq_scale,
                   const MetalExecutor::CacheDescriptor& cache_k_desc,
                   const MetalExecutor::CacheDescriptor& cache_v_desc,
                   std::vector<float>& output) const {
#if defined(__APPLE__)
        if (!available()) return false;
        if (head_dim == 0 || num_heads == 0) return false;

        size_t effective_kv_heads = kv_heads > 0 ? kv_heads : num_heads;
        if (effective_kv_heads == 0) effective_kv_heads = 1;
        size_t expected_q = num_heads * head_dim;
        if (expected_q == 0 || q.size() != expected_q) {
            return false;
        }

        if (context_length == 0) context_length = 1;
        size_t kv_stride = context_length * head_dim;

        size_t kv_span = effective_kv_heads * head_dim;
        if (kv_span == 0) return false;
        if (k.size() < kv_span || v.size() < kv_span) {
            return false;
        }
        if (k.size() % kv_span != 0 || v.size() % kv_span != 0) {
            return false;
        }
        size_t tokens_from_k = k.size() / kv_span;
        size_t tokens_from_v = v.size() / kv_span;
        if (tokens_from_k == 0 || tokens_from_k != tokens_from_v) {
            return false;
        }

        // Prepare cache buffers (float) from descriptors.
        auto prepareCache = [&](const MetalExecutor::CacheDescriptor& desc,
                                size_t total_elems,
                                std::vector<float>& tmp,
                                std::vector<float>*& out_vec) -> bool {
            out_vec = nullptr;
            if (desc.float_data) {
                if (desc.float_data->size() != total_elems) {
                    desc.float_data->resize(total_elems, 0.0f);
                }
            out_vec = desc.float_data;
            }
            if (out_vec && desc.dtype == frontend::GGML_TYPE_F32) {
                return true;
            }
            if (!desc.raw_quant) return false;
            // Need dequant into out_vec (or tmp).
            std::vector<float>* target = out_vec ? out_vec : &tmp;
            target->resize(total_elems);
            bool ok = false;
        switch (desc.dtype) {
        case frontend::GGML_TYPE_Q4_0:
            ok = desc.raw_quant &&
                 MetalExecutor::Instance().dequantQ4Block(*desc.raw_quant, total_elems, *target);
            break;
            case frontend::GGML_TYPE_Q4_1:
                ok = desc.raw_quant &&
                     MetalExecutor::Instance().dequantQ4_1Block(*desc.raw_quant, total_elems, *target);
                break;
            case frontend::GGML_TYPE_Q5_0:
                ok = desc.raw_quant &&
                     MetalExecutor::Instance().dequantQ5_0Block(*desc.raw_quant, total_elems, *target);
                break;
            case frontend::GGML_TYPE_Q5_1:
                ok = desc.raw_quant &&
                     MetalExecutor::Instance().dequantQ5_1Block(*desc.raw_quant, total_elems, *target);
                break;
            case frontend::GGML_TYPE_Q8_0:
                ok = desc.raw_quant &&
                     MetalExecutor::Instance().dequantQ8Block(*desc.raw_quant, total_elems, *target);
                break;
            case frontend::GGML_TYPE_Q8_1:
                ok = desc.raw_quant &&
                     MetalExecutor::Instance().dequantQ8_1Block(*desc.raw_quant, total_elems, *target);
                break;
            case frontend::GGML_TYPE_Q2_K:
            case frontend::GGML_TYPE_Q3_K:
            case frontend::GGML_TYPE_Q4_K:
            case frontend::GGML_TYPE_Q5_K:
        case frontend::GGML_TYPE_Q6_K:
        case frontend::GGML_TYPE_Q8_K: {
            if (!desc.raw_quant || desc.row_stride_bytes == 0) break;
#if defined(__APPLE__)
            if (dequantQKPipeline) {
                ok = MetalExecutor::Instance().dequantQKRow(*desc.raw_quant,
                                                            desc.dtype,
                                                            head_dim,
                                                            desc.row_stride_bytes,
                                                            *target);
            }
#endif
            if (!ok) {
                size_t rows = total_elems / head_dim;
                for (size_t r = 0; r < rows; ++r) {
                    const uint8_t* src = desc.raw_quant->data() + r * desc.row_stride_bytes;
                    dequantizeKRowCPU(desc.dtype,
                                      src,
                                      head_dim,
                                      desc.quant_version,
                                      target->data() + r * head_dim);
                }
                ok = true;
            }
            break;
        }
        case frontend::GGML_TYPE_F32:
            ok = true;
            break;
        default:
            ok = false;
            }
            if (!ok && desc.raw_quant) {
                // CPU fallback dequant.
                size_t rows = head_dim ? total_elems / head_dim : 0;
                size_t stride = desc.row_stride_bytes;
                if (stride == 0) {
                    stride = ggmlRowSizeBytes(desc.dtype, head_dim, desc.quant_version);
                }
                for (size_t r = 0; r < rows; ++r) {
                    const uint8_t* src = desc.raw_quant->data() + r * stride;
                    dequantizeRowTo(src,
                                    desc.dtype,
                                    head_dim,
                                    desc.quant_version,
                                    target->data() + r * head_dim);
                }
                ok = true;
            }
            if (ok && out_vec == nullptr) {
                out_vec = target;
            }
            return ok;
        };

        size_t cache_elems = effective_kv_heads * kv_stride;
        std::vector<float> tmp_k;
        std::vector<float> tmp_v;
        std::vector<float>* kv_cache_k = nullptr;
        std::vector<float>* kv_cache_v = nullptr;
        if (!prepareCache(cache_k_desc, cache_elems, tmp_k, kv_cache_k)) return false;
        if (!prepareCache(cache_v_desc, cache_elems, tmp_v, kv_cache_v)) return false;

        bool sharedCacheValid = false;
        bool gpu_cache_k_ready = false;
        bool gpu_cache_v_ready = false;
#if defined(__APPLE__)
        id<MTLBuffer> sharedKCache = nil;
        id<MTLBuffer> sharedVCache = nil;
        if (cache_k_desc.handle && cache_v_desc.handle &&
            cache_k_desc.handle->buffer && cache_v_desc.handle->buffer &&
            cache_k_desc.handle->bytes >= cache_elems * sizeof(float) &&
            cache_v_desc.handle->bytes >= cache_elems * sizeof(float)) {
            sharedKCache = (__bridge id<MTLBuffer>)cache_k_desc.handle->buffer;
            sharedVCache = (__bridge id<MTLBuffer>)cache_v_desc.handle->buffer;
            sharedCacheValid = (sharedKCache != nil) && (sharedVCache != nil);
        }
        // TODO: re-enable GPU cache dequant + residency once kernels are fully validated.
        // For now keep cache copies on host to avoid intermittent Metal crashes with tiny tensors.
        sharedCacheValid = false;
        gpu_cache_k_ready = false;
        gpu_cache_v_ready = false;
#endif
        size_t effective_rotary = 0;
        float freq_base = 10000.0f;
        float freq_scale = 1.0f;
        if (rotary_dim > 0 && head_dim >= 2) {
            effective_rotary = std::min(rotary_dim, head_dim);
            if (rope_freq_base > 0.0f) {
                freq_base = rope_freq_base;
            }
            if (rope_freq_scale != 0.0f) {
                freq_scale = rope_freq_scale;
            }
        }
        bool use_rotary = effective_rotary > 0;
        std::vector<float> rotary_cos;
        std::vector<float> rotary_sin;
        std::vector<float> q_single_cos;
        std::vector<float> q_single_sin;
        std::vector<float> q_cos_table;
        std::vector<float> q_sin_table;

        size_t base_position = std::min(position, context_length - 1);
        size_t writable = context_length - base_position;
        size_t tokens_to_write = std::min(tokens_from_k, writable);
        if (tokens_to_write == 0) {
            tokens_to_write = std::min(tokens_from_k, context_length);
        }
        if (tokens_to_write == 0 || kv_cache_k == nullptr || kv_cache_v == nullptr) return false;

        const std::vector<float>* k_ptr = &k;
        std::vector<float> rotated_k;
        if (use_rotary) {
            rotated_k = k;
            bool rotated = false;
#if defined(__APPLE__)
            // DEAD CODE — gated off by sharedCacheValid being force-pinned to
            // false a few lines above (search 'sharedCacheValid = false'). Kept
            // for the future GPU-cache-residency work, but it has TWO known
            // sizing bugs that would fire the moment sharedCacheValid is
            // re-enabled. Don't silently re-enable that flag without fixing
            // them. The assert below makes that impossible.
            //
            // Bug 1 — cos/sin table is too small.
            //   The host fills cos_table / sin_table of size pairs * tokens_to_write,
            //   then calls applyRotaryGPU with count = tokens_to_write *
            //   effective_kv_heads. Inside applyRotaryGPU,
            //     expected = pair_stride * (count + offset_tokens)
            //              = pairs * tokens_to_write * effective_kv_heads
            //   which is effective_kv_heads-x larger than what the host
            //   actually allocated. newBufferWithBytes therefore reads past
            //   the std::vectors. For TinyLlama (kv_heads=4) that is a 4-x
            //   over-read of host memory.
            //
            // Bug 2 — per-thread vec stride is wrong.
            //   row_stride = kv_span * sizeof(float) means each gid jumps by a
            //   whole [kv_heads, head_dim] row instead of by one head_dim
            //   vector, so threads gid >= tokens_to_write address bytes past
            //   the end of tmpK (which is only tokens_to_write * kv_span
            //   floats long). Symmetric to bug 1 but on the data side.
            //
            // A correct rewrite probably wants:
            //   row_stride = head_dim * sizeof(float)
            //   count      = tokens_to_write * effective_kv_heads
            //   cos/sin    = sized [count, pairs], with the same cos values
            //                replicated across kv_heads for each token (or
            //                use offset_elems to encode the kv_head dim).
            // But the cleanest place to do that is inside a new shader path
            // that knows the [token, kv_head, head_dim] layout explicitly.
            //
            // For now: trip an assert so this path can never silently run
            // without a sizing review. The CPU fallback path below produces
            // bit-equal results and is the production code.
            if (sharedCacheValid && effective_rotary > 0) {
                assert(false && "K-RoPE GPU path has unfixed sizing bugs; "
                                "see comment block at this line before enabling sharedCacheValid");
                (void)tokens_to_write; (void)base_position; (void)effective_kv_heads;
            }
#endif
            if (!rotated) {
                for (size_t t = 0; t < tokens_to_write; ++t) {
                    size_t pos = std::min(context_length - 1, base_position + t);
                    computeRotaryCoefficients(pos,
                                              effective_rotary,
                                              freq_base,
                                              freq_scale,
                                              rotary_cos,
                                              rotary_sin);
                    for (size_t kvh = 0; kvh < effective_kv_heads; ++kvh) {
                        float* vec = rotated_k.data() + t * kv_span + kvh * head_dim;
                        applyRotaryEmbedding(vec,
                                             rotary_cos,
                                             rotary_sin,
                                             head_dim,
                                             effective_rotary);
                    }
                }
                k_ptr = &rotated_k;
            }
        }

        bool gpu_scatter = false;
        // Temporarily prefer CPU scatter for correctness while we finalize GPU cache writes.
        gpu_scatter = false;

        if (!gpu_scatter) {
            for (size_t kvh = 0; kvh < effective_kv_heads; ++kvh) {
                for (size_t t = 0; t < tokens_to_write; ++t) {
                    size_t pos = std::min(context_length - 1, base_position + t);
                    float* dest_k = kv_cache_k->data() + kvh * kv_stride + pos * head_dim;
                    float* dest_v = kv_cache_v->data() + kvh * kv_stride + pos * head_dim;
                    const float* src_k = k_ptr->data() + t * kv_span + kvh * head_dim;
                    const float* src_v = v.data() + t * kv_span + kvh * head_dim;
                    std::memcpy(dest_k, src_k, head_dim * sizeof(float));
                    std::memcpy(dest_v, src_v, head_dim * sizeof(float));
                }
            }
#if defined(__APPLE__)
            if (sharedCacheValid) {
                if (@available(macOS 11.0, *)) {
                    size_t update_bytes = tokens_to_write * head_dim * sizeof(float);
                    size_t start_token = std::min(context_length - 1, base_position);
                    for (size_t kvh = 0; kvh < effective_kv_heads; ++kvh) {
                        size_t base_bytes = (kvh * kv_stride + start_token) * head_dim * sizeof(float);
                        NSRange range = NSMakeRange(base_bytes, update_bytes);
                        [sharedKCache didModifyRange:range];
                        [sharedVCache didModifyRange:range];
                    }
                }
            }
#endif
        }

        size_t tokens = std::min(context_length, base_position + tokens_to_write);
        if (tokens == 0) tokens = 1;

        if (@available(macOS 11.0, *)) {
            @autoreleasepool {
                output.assign(num_heads * head_dim, 0.0f);
                float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(head_dim));
                size_t kv_divisor = std::max<size_t>(size_t(1), num_heads);
                size_t headRowBytes = alignedRowBytes(head_dim);
                size_t tokensRowBytes = alignedRowBytes(tokens);
                size_t alignedVecRowBytes = alignedRowBytes(head_dim);
                size_t sharedRowBytes = head_dim * sizeof(float);
                if (use_rotary) {
                    computeRotaryCoefficients(base_position,
                                              effective_rotary,
                                              freq_base,
                                              freq_scale,
                                              rotary_cos,
                                              rotary_sin);
                    q_single_cos = rotary_cos;
                    q_single_sin = rotary_sin;
                    size_t pairs = rotary_cos.size();
                    q_cos_table.resize(pairs * num_heads);
                    q_sin_table.resize(pairs * num_heads);
                    for (size_t head = 0; head < num_heads; ++head) {
                        std::copy(rotary_cos.begin(),
                                  rotary_cos.end(),
                                  q_cos_table.begin() + head * pairs);
                        std::copy(rotary_sin.begin(),
                                  rotary_sin.end(),
                                  q_sin_table.begin() + head * pairs);
                    }
                }

                // === Flash-attention v1 dispatch (Session C, gated default-off) ===
                // MLC_FLASH_ATTN=1 routes through the custom Metal kernel that
                // does Q·K + softmax + ·V in one launch. Reads K/V directly
                // from the KV cache via strided addressing; no GQA broadcast
                // needed (the kernel maps q_head → kv_head internally).
                // For q_seq=1 (decode and item-3 batched prefill, which loops
                // per-token externally), the Q sees all cached positions
                // [0, tokens) and causal masking is a no-op.
                static const bool use_flash_attn = (std::getenv("MLC_FLASH_ATTN") != nullptr);
                if (use_flash_attn && flashAttentionPipeline && kv_cache_k && kv_cache_v) {
                    // Q rotation (CPU; cheap for num_heads × head_dim floats).
                    std::vector<float> q_for_flash(q.begin(),
                                                    q.begin() + num_heads * head_dim);
                    if (use_rotary && !q_single_cos.empty()) {
                        for (size_t h = 0; h < num_heads; ++h) {
                            applyRotaryEmbedding(q_for_flash.data() + h * head_dim,
                                                 q_single_cos, q_single_sin,
                                                 head_dim, effective_rotary);
                        }
                    }
                    bool ok = MetalExecutor::Instance().runFlashAttentionStrided(
                        q_for_flash,
                        kv_cache_k->data(), kv_cache_k->size(),
                        kv_cache_v->data(), kv_cache_v->size(),
                        num_heads, effective_kv_heads, head_dim,
                        /*kv_seq=*/tokens,
                        /*apply_causal=*/false,
                        /*q_position=*/tokens > 0 ? tokens - 1 : 0,
                        /*k_stride_token=*/head_dim,
                        /*k_stride_kv_head=*/kv_stride,
                        /*v_stride_token=*/head_dim,
                        /*v_stride_kv_head=*/kv_stride,
                        output);
                    if (ok) return true;
                    // Fall through to MPS path on kernel failure (shape outside
                    // the kernel's accepted range, etc.).
                }

                // === Attention dispatch path selection ===
                // Default: batched-heads path (Tier 2) — one command buffer per
                // attention call, all num_heads matmuls fused via MPS batching,
                // CPU-side GQA broadcast of K/V. Per-token attention cost drops
                // from ~15 ms to ~1-2 ms on TinyLlama (M3 Pro).
                // Fallback: legacy per-head loop, kept here behind
                // MLC_ATTN_LEGACY=1 for A/B comparison during validation.
                static const bool use_legacy_attn = (std::getenv("MLC_ATTN_LEGACY") != nullptr);
                if (use_legacy_attn) {
                for (size_t head = 0; head < num_heads; ++head) {
                    size_t kv_index = std::min(effective_kv_heads - 1,
                                               head * effective_kv_heads / kv_divisor);
        const float* q_ptr = q.data() + head * head_dim;

        id<MTLBuffer> qBuffer = [device newBufferWithLength:headRowBytes
                                                      options:MTLResourceStorageModeShared];
        if (!qBuffer) return false;
        std::memcpy([qBuffer contents], q_ptr, head_dim * sizeof(float));
        if (headRowBytes > head_dim * sizeof(float)) {
            std::memset(((uint8_t*)[qBuffer contents]) + head_dim * sizeof(float), 0,
                        headRowBytes - head_dim * sizeof(float));
        }

        if (use_rotary && !q_cos_table.empty() && !q_single_cos.empty()) {
            size_t stride = q_single_cos.size();
            const float* cos_ptr = q_cos_table.data() + head * stride;
            const float* sin_ptr = q_sin_table.data() + head * stride;
            // offset_tokens=0 is intentional: this is the single-token Q-rotation
            // path, and `cos_ptr`/`sin_ptr` already point at this head's slice of
            // the cos/sin table (filled by computeRotaryCoefficients for the
            // current base_position above). The shader interprets offset_elems as
            // a buffer index — see apply_rotary_batch at metal_runtime.mm:996,
            // specifically `vec += offset_elems * head_dim` — so passing
            // base_position here would push the write pointer base_position *
            // head_dim floats past the start of qBuffer, which is only head_dim
            // floats long. Passing 0 keeps the rotation in-bounds.
            bool rotated = applyRotaryGPU(qBuffer,
                                          1,
                                          headRowBytes,
                                          head_dim,
                                          effective_rotary,
                                          cos_ptr,
                                          sin_ptr,
                                          stride,
                                          0);
            if (!rotated) {
                float* vec = reinterpret_cast<float*>([qBuffer contents]);
                applyRotaryEmbedding(vec,
                                     q_single_cos,
                                     q_single_sin,
                                     head_dim,
                                     effective_rotary);
            }
        }

        bool useSharedMatrix = sharedCacheValid && gpu_cache_k_ready && gpu_cache_v_ready;
        id<MTLBuffer> kBuffer = nil;
        NSUInteger kRowBytes = useSharedMatrix ? sharedRowBytes : alignedVecRowBytes;
        if (useSharedMatrix) {
            kBuffer = sharedKCache;
        } else {
                        kBuffer = [device newBufferWithLength:alignedVecRowBytes * tokens
                                                      options:MTLResourceStorageModeShared];
                        if (!kBuffer) return false;
                        uint8_t* kDst = reinterpret_cast<uint8_t*>([kBuffer contents]);
                        for (size_t t = 0; t < tokens; ++t) {
                            const float* src = kv_cache_k->data() + kv_index * kv_stride + t * head_dim;
                            std::memcpy(kDst + t * alignedVecRowBytes, src, head_dim * sizeof(float));
                            if (alignedVecRowBytes > head_dim * sizeof(float)) {
                                std::memset(kDst + t * alignedVecRowBytes + head_dim * sizeof(float), 0,
                                            alignedVecRowBytes - head_dim * sizeof(float));
                            }
                        }
                    }

                    id<MTLBuffer> logitsBuffer = [device newBufferWithLength:tokensRowBytes
                                                                       options:MTLResourceStorageModeShared];
                    if (!logitsBuffer) return false;

                    MPSMatrixDescriptor* qDesc =
                        [MPSMatrixDescriptor matrixDescriptorWithRows:1
                                                             columns:head_dim
                                                            rowBytes:headRowBytes
                                                            dataType:MPSDataTypeFloat32];
                    MPSMatrixDescriptor* kDesc =
                        [MPSMatrixDescriptor matrixDescriptorWithRows:tokens
                                                             columns:head_dim
                                                            rowBytes:kRowBytes
                                                            dataType:MPSDataTypeFloat32];
                    MPSMatrixDescriptor* logitsDesc =
                        [MPSMatrixDescriptor matrixDescriptorWithRows:1
                                                             columns:tokens
                                                            rowBytes:tokensRowBytes
                                                            dataType:MPSDataTypeFloat32];

                    MPSMatrix* qMatrix = [[MPSMatrix alloc] initWithBuffer:qBuffer descriptor:qDesc];
                    MPSMatrix* kMatrix = nil;
                    if (useSharedMatrix) {
                        NSUInteger offsetBytes = static_cast<NSUInteger>(kv_index * kv_stride * head_dim * sizeof(float));
                        kMatrix = [[MPSMatrix alloc] initWithBuffer:kBuffer offset:offsetBytes descriptor:kDesc];
                    } else {
                        kMatrix = [[MPSMatrix alloc] initWithBuffer:kBuffer descriptor:kDesc];
                    }
                    MPSMatrix* logitsMatrix = [[MPSMatrix alloc] initWithBuffer:logitsBuffer descriptor:logitsDesc];

                    MPSMatrixMultiplication* qk =
                        [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                        transposeLeft:false
                                                       transposeRight:true
                                                          resultRows:1
                                                       resultColumns:tokens
                                                    interiorColumns:head_dim
                                                               alpha:inv_sqrt
                                                                beta:0.0f];

                    if (!qk) return false;

                    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
                    [qk encodeToCommandBuffer:commandBuffer
                                    leftMatrix:qMatrix
                                   rightMatrix:kMatrix
                                  resultMatrix:logitsMatrix];
                    [commandBuffer commit];
                    [commandBuffer waitUntilCompleted];

                    float* logitsPtr = reinterpret_cast<float*>([logitsBuffer contents]);
                    // Apply ALiBi slopes if provided (per-head, tokens along time).
                    if (alibi_slopes && head < alibi_slopes->size()) {
                        float slope = (*alibi_slopes)[head];
                        for (size_t t = 0; t < tokens; ++t) {
                            logitsPtr[t] += slope * static_cast<float>(t);
                        }
                    }
                    if (!mask.empty()) {
                        // Support head-specific masks packed as [head, token].
                        size_t mask_tokens = std::min(tokens, mask.size());
                        size_t head_stride = 0;
                        if (mask.size() >= tokens && mask.size() % tokens == 0) {
                            head_stride = tokens;
                        }
                        for (size_t t = 0; t < tokens && t < mask_tokens; ++t) {
                            size_t idx = t;
                            if (head_stride > 0) {
                                size_t mask_head_count = mask.size() / head_stride;
                                if (head < mask_head_count) {
                                    idx = head * head_stride + t;
                                }
                            }
                            if (idx < mask.size()) {
                                logitsPtr[t] += mask[idx];
                            }
                        }
                    }

                    MPSMatrixSoftMax* softmaxKernel = [[MPSMatrixSoftMax alloc] initWithDevice:device];
                    if (!softmaxKernel) return false;
                    id<MTLCommandBuffer> softmaxBuffer = [queue commandBuffer];
                    [softmaxKernel encodeToCommandBuffer:softmaxBuffer
                                             inputMatrix:logitsMatrix
                                            resultMatrix:logitsMatrix];
                    [softmaxBuffer commit];
                    [softmaxBuffer waitUntilCompleted];

                    id<MTLBuffer> attBuffer = logitsBuffer;

                    bool useSharedV = sharedCacheValid && gpu_cache_v_ready;
        id<MTLBuffer> vBuffer = nil;
        NSUInteger vRowBytes = useSharedV ? sharedRowBytes : alignedVecRowBytes;
        if (useSharedV) {
            vBuffer = sharedVCache;
        } else {
                        vBuffer = [device newBufferWithLength:alignedVecRowBytes * tokens
                                                       options:MTLResourceStorageModeShared];
                        if (!vBuffer) return false;
                        uint8_t* vDst = reinterpret_cast<uint8_t*>([vBuffer contents]);
                        for (size_t t = 0; t < tokens; ++t) {
                            const float* src = kv_cache_v->data() + kv_index * kv_stride + t * head_dim;
                            std::memcpy(vDst + t * alignedVecRowBytes, src, head_dim * sizeof(float));
                            if (alignedVecRowBytes > head_dim * sizeof(float)) {
                                std::memset(vDst + t * alignedVecRowBytes + head_dim * sizeof(float), 0,
                                            alignedVecRowBytes - head_dim * sizeof(float));
                            }
                        }
                    }

                    id<MTLBuffer> resultBuffer = [device newBufferWithLength:headRowBytes
                                                                      options:MTLResourceStorageModeShared];
                    if (!resultBuffer) return false;

                    MPSMatrixDescriptor* attDesc =
                        [MPSMatrixDescriptor matrixDescriptorWithRows:1
                                                            columns:tokens
                                                           rowBytes:tokensRowBytes
                                                           dataType:MPSDataTypeFloat32];
                    MPSMatrixDescriptor* vDesc =
                        [MPSMatrixDescriptor matrixDescriptorWithRows:tokens
                                                             columns:head_dim
                                                            rowBytes:vRowBytes
                                                            dataType:MPSDataTypeFloat32];
                    MPSMatrixDescriptor* outDesc =
                        [MPSMatrixDescriptor matrixDescriptorWithRows:1
                                                             columns:head_dim
                                                            rowBytes:headRowBytes
                                                            dataType:MPSDataTypeFloat32];

                    MPSMatrix* attMatrix = [[MPSMatrix alloc] initWithBuffer:attBuffer descriptor:attDesc];
                    MPSMatrix* vMatrix = nil;
                    if (useSharedV) {
                        NSUInteger offsetBytes = static_cast<NSUInteger>(kv_index * kv_stride * head_dim * sizeof(float));
                        vMatrix = [[MPSMatrix alloc] initWithBuffer:vBuffer offset:offsetBytes descriptor:vDesc];
                    } else {
                        vMatrix = [[MPSMatrix alloc] initWithBuffer:vBuffer descriptor:vDesc];
                    }
                    MPSMatrix* outMatrix = [[MPSMatrix alloc] initWithBuffer:resultBuffer descriptor:outDesc];

                    MPSMatrixMultiplication* av =
                        [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                        transposeLeft:false
                                                       transposeRight:false
                                                          resultRows:1
                                                       resultColumns:head_dim
                                                    interiorColumns:tokens
                                                               alpha:1.0f
                                                                beta:0.0f];
                    if (!av) return false;
                    id<MTLCommandBuffer> commandBuffer2 = [queue commandBuffer];
                    [av encodeToCommandBuffer:commandBuffer2
                                   leftMatrix:attMatrix
                                  rightMatrix:vMatrix
                                 resultMatrix:outMatrix];
                    [commandBuffer2 commit];
                    [commandBuffer2 waitUntilCompleted];

                    std::memcpy(output.data() + head * head_dim,
                                [resultBuffer contents],
                                head_dim * sizeof(float));
                }
                } else {
                // === BATCHED-HEADS PATH (Tier 2) ===
                // One MTLCommandBuffer per attention call. All num_heads matmuls
                // run as batched MPSMatrixMultiplication (matrices=num_heads),
                // softmax runs once over [num_heads, tokens] as per-row softmax,
                // and GQA broadcast is done host-side by gathering each Q head's
                // kv_head slice into a num_heads-batched K/V buffer.
                //
                // Q-RoPE is applied on the CPU here (it's 32*64 floats of work,
                // a few microseconds) so we don't need a separate GPU dispatch
                // for it. Mask + ALiBi are folded into the logits buffer before
                // the Q.K matmul, which runs with beta=1.0 to add the scores
                // on top, eliminating any mid-call sync.
                //
                // fp16 attention path. Default ON per item-4 fp16 arc:
                // - Q, K, V, logits, result allocated as half-precision.
                // - All five MPSMatrixDescriptors use MPSDataTypeFloat16.
                // - Q is NEON-cast to fp16 on upload; K, V on copy from cache;
                //   mask + ALiBi pre-fill in fp16; result NEON-cast back to
                //   fp32 on output. MPS' simdgroup_half8x8 matmul runs at 2×
                //   throughput vs fp32. At decode shape the gain is small
                //   (~5% per attention call) because dispatch overhead
                //   dominates compute, but combined with fp16 KV cache the
                //   path is wedge-compatible with paged-KV (item 5). Set
                //   MLC_FP16_ATTN=0 to opt out.
                static const bool fp16_attn = []() {
                    const char* env = std::getenv("MLC_FP16_ATTN");
                    if (!env) return true;  // default on
                    return !(env[0] == '0' && env[1] == 0);
                }();
                const size_t kElemBytes      = fp16_attn ? sizeof(uint16_t) : sizeof(float);
                const size_t kHeadDimBytes   = head_dim * kElemBytes;
                const size_t kKVMatrixBytes  = tokens * kHeadDimBytes;
                const size_t kLogitsRowBytes = tokens * kElemBytes;
                const MPSDataType mpsDtype   = fp16_attn ? MPSDataTypeFloat16 : MPSDataTypeFloat32;

                id<MTLBuffer> qBuffer = [device newBufferWithLength:num_heads * kHeadDimBytes
                                                            options:MTLResourceStorageModeShared];
                id<MTLBuffer> kBatchBuffer = [device newBufferWithLength:num_heads * kKVMatrixBytes
                                                                options:MTLResourceStorageModeShared];
                id<MTLBuffer> vBatchBuffer = [device newBufferWithLength:num_heads * kKVMatrixBytes
                                                                options:MTLResourceStorageModeShared];
                id<MTLBuffer> logitsBuffer = [device newBufferWithLength:num_heads * kLogitsRowBytes
                                                                options:MTLResourceStorageModeShared];
                id<MTLBuffer> resultBuffer = [device newBufferWithLength:num_heads * kHeadDimBytes
                                                                options:MTLResourceStorageModeShared];
                if (!qBuffer || !kBatchBuffer || !vBatchBuffer ||
                    !logitsBuffer || !resultBuffer) {
                    return false;
                }

                // 1. CPU-rotate Q (fp32 working buffer), then cast to fp16 (or
                //    plain memcpy in fp32 mode).
                std::vector<float> q_staging;
                const float* q_src;
                if (use_rotary && !q_single_cos.empty()) {
                    q_staging.assign(q.begin(), q.begin() + num_heads * head_dim);
                    for (size_t h = 0; h < num_heads; ++h) {
                        applyRotaryEmbedding(q_staging.data() + h * head_dim,
                                             q_single_cos,
                                             q_single_sin,
                                             head_dim,
                                             effective_rotary);
                    }
                    q_src = q_staging.data();
                } else {
                    q_src = q.data();
                }
                if (fp16_attn) {
                    castF32toF16(q_src,
                                 reinterpret_cast<uint16_t*>([qBuffer contents]),
                                 num_heads * head_dim);
                } else {
                    std::memcpy([qBuffer contents], q_src, num_heads * kHeadDimBytes);
                }

                // 2. GQA broadcast: gather K[kv_index] and V[kv_index] into
                //    per-Q-head batch slots. kv_cache_{k,v} is laid out as
                //    [kv_heads, context_length, head_dim]; we take the first
                //    `tokens` rows of each Q head's owning kv_head. Cast to
                //    fp16 inline when fp16_attn is on.
                uint8_t* kDst = reinterpret_cast<uint8_t*>([kBatchBuffer contents]);
                uint8_t* vDst = reinterpret_cast<uint8_t*>([vBatchBuffer contents]);
                size_t per_head_elems = tokens * head_dim;
                for (size_t h = 0; h < num_heads; ++h) {
                    size_t kv_idx = std::min(effective_kv_heads - 1,
                                             h * effective_kv_heads / kv_divisor);
                    const float* kSrc = kv_cache_k->data() + kv_idx * kv_stride;
                    const float* vSrc = kv_cache_v->data() + kv_idx * kv_stride;
                    if (fp16_attn) {
                        castF32toF16(kSrc,
                                     reinterpret_cast<uint16_t*>(kDst + h * kKVMatrixBytes),
                                     per_head_elems);
                        castF32toF16(vSrc,
                                     reinterpret_cast<uint16_t*>(vDst + h * kKVMatrixBytes),
                                     per_head_elems);
                    } else {
                        std::memcpy(kDst + h * kKVMatrixBytes, kSrc, kKVMatrixBytes);
                        std::memcpy(vDst + h * kKVMatrixBytes, vSrc, kKVMatrixBytes);
                    }
                }

                // 3. Pre-fill logits with mask + ALiBi per head, so the Q.K
                //    matmul (with beta=1.0) adds the scores on top in one step.
                //    Compute mask values in fp32 (numerical headroom for
                //    -INF / large negative), then cast the whole logits buffer
                //    to fp16 in one bulk pass when fp16_attn is on.
                bool have_mask = !mask.empty();
                bool have_alibi = (alibi_slopes != nullptr);
                size_t mask_head_stride = 0;
                if (have_mask && mask.size() >= tokens && mask.size() % tokens == 0) {
                    mask_head_stride = tokens;
                }
                std::vector<float> logits_staging;
                float* logitsF32;
                if (fp16_attn) {
                    logits_staging.assign(num_heads * tokens, 0.0f);
                    logitsF32 = logits_staging.data();
                } else {
                    logitsF32 = reinterpret_cast<float*>([logitsBuffer contents]);
                    std::memset(logitsF32, 0, num_heads * kLogitsRowBytes);
                }
                for (size_t h = 0; h < num_heads; ++h) {
                    float* row = logitsF32 + h * tokens;
                    if (have_alibi && h < alibi_slopes->size()) {
                        float slope = (*alibi_slopes)[h];
                        for (size_t t = 0; t < tokens; ++t) {
                            row[t] += slope * static_cast<float>(t);
                        }
                    }
                    if (have_mask) {
                        size_t mask_tokens = std::min(tokens, mask.size());
                        for (size_t t = 0; t < tokens && t < mask_tokens; ++t) {
                            size_t idx = t;
                            if (mask_head_stride > 0) {
                                size_t mask_head_count = mask.size() / mask_head_stride;
                                if (h < mask_head_count) {
                                    idx = h * mask_head_stride + t;
                                }
                            }
                            if (idx < mask.size()) row[t] += mask[idx];
                        }
                    }
                }
                if (fp16_attn) {
                    castF32toF16(logits_staging.data(),
                                 reinterpret_cast<uint16_t*>([logitsBuffer contents]),
                                 num_heads * tokens);
                }

                // 4. Build batched MPS descriptors + matrices.
                MPSMatrixDescriptor* qDesc =
                    [MPSMatrixDescriptor matrixDescriptorWithRows:1
                                                          columns:head_dim
                                                         matrices:num_heads
                                                         rowBytes:kHeadDimBytes
                                                      matrixBytes:kHeadDimBytes
                                                         dataType:mpsDtype];
                MPSMatrixDescriptor* kDesc =
                    [MPSMatrixDescriptor matrixDescriptorWithRows:tokens
                                                          columns:head_dim
                                                         matrices:num_heads
                                                         rowBytes:kHeadDimBytes
                                                      matrixBytes:kKVMatrixBytes
                                                         dataType:mpsDtype];
                MPSMatrixDescriptor* lBatchDesc =
                    [MPSMatrixDescriptor matrixDescriptorWithRows:1
                                                          columns:tokens
                                                         matrices:num_heads
                                                         rowBytes:kLogitsRowBytes
                                                      matrixBytes:kLogitsRowBytes
                                                         dataType:mpsDtype];
                MPSMatrixDescriptor* lSoftmaxDesc =
                    [MPSMatrixDescriptor matrixDescriptorWithRows:num_heads
                                                          columns:tokens
                                                         matrices:1
                                                         rowBytes:kLogitsRowBytes
                                                      matrixBytes:num_heads * kLogitsRowBytes
                                                         dataType:mpsDtype];
                MPSMatrixDescriptor* rDesc =
                    [MPSMatrixDescriptor matrixDescriptorWithRows:1
                                                          columns:head_dim
                                                         matrices:num_heads
                                                         rowBytes:kHeadDimBytes
                                                      matrixBytes:kHeadDimBytes
                                                         dataType:mpsDtype];

                MPSMatrix* qMat = [[MPSMatrix alloc] initWithBuffer:qBuffer descriptor:qDesc];
                MPSMatrix* kMat = [[MPSMatrix alloc] initWithBuffer:kBatchBuffer descriptor:kDesc];
                MPSMatrix* lMatBatched = [[MPSMatrix alloc] initWithBuffer:logitsBuffer descriptor:lBatchDesc];
                MPSMatrix* lMatSoftmax = [[MPSMatrix alloc] initWithBuffer:logitsBuffer descriptor:lSoftmaxDesc];
                MPSMatrix* vMat = [[MPSMatrix alloc] initWithBuffer:vBatchBuffer descriptor:kDesc];
                MPSMatrix* rMat = [[MPSMatrix alloc] initWithBuffer:resultBuffer descriptor:rDesc];

                // 5. Single command buffer for all 3 MPS ops.
                id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

                // 5a. Batched Q.K with beta=1.0 (adds scaled scores onto the
                //     pre-populated mask/alibi values).
                MPSMatrixMultiplication* qk =
                    [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                      transposeLeft:false
                                                     transposeRight:true
                                                         resultRows:1
                                                      resultColumns:tokens
                                                    interiorColumns:head_dim
                                                              alpha:inv_sqrt
                                                               beta:1.0f];
                if (!qk) return false;
                [qk encodeToCommandBuffer:commandBuffer
                               leftMatrix:qMat
                              rightMatrix:kMat
                             resultMatrix:lMatBatched];

                // 5b. Softmax over [num_heads, tokens] — one call, per-row.
                MPSMatrixSoftMax* softmaxKernel =
                    [[MPSMatrixSoftMax alloc] initWithDevice:device];
                if (!softmaxKernel) return false;
                [softmaxKernel encodeToCommandBuffer:commandBuffer
                                         inputMatrix:lMatSoftmax
                                        resultMatrix:lMatSoftmax];

                // 5c. Batched att.V.
                MPSMatrixMultiplication* av =
                    [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                      transposeLeft:false
                                                     transposeRight:false
                                                         resultRows:1
                                                      resultColumns:head_dim
                                                    interiorColumns:tokens
                                                              alpha:1.0f
                                                               beta:0.0f];
                if (!av) return false;
                [av encodeToCommandBuffer:commandBuffer
                               leftMatrix:lMatBatched
                              rightMatrix:vMat
                             resultMatrix:rMat];

                // 6. Commit + wait once.
                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];

                // 7. Copy result back into output (head-major layout matches CPU).
                if (fp16_attn) {
                    castF16toF32(reinterpret_cast<const uint16_t*>([resultBuffer contents]),
                                 output.data(),
                                 num_heads * head_dim);
                } else {
                    std::memcpy(output.data(), [resultBuffer contents],
                                num_heads * kHeadDimBytes);
                }
                }
                return true;
            }
        }
#endif
        return false;
    }

    // Skeleton single-tile flash-attention. Math correctness only; not yet
    // wired into the production attention dispatch. See the kernel comment
    // for layout details. If qk_debug or sm_debug is non-null, the kernel
    // dumps post-Q·K scores and post-softmax weights into them for harness
    // sub-step parity reporting.
    bool flashAttention(const std::vector<float>& q,
                        const float* k_data, size_t k_size,
                        const float* v_data, size_t v_size,
                        size_t num_heads,
                        size_t kv_heads,
                        size_t head_dim,
                        size_t kv_seq,
                        bool apply_causal,
                        size_t q_position,
                        size_t k_stride_token, size_t k_stride_kv_head, size_t k_stride_feature,
                        size_t v_stride_token, size_t v_stride_kv_head, size_t v_stride_feature,
                        std::vector<float>& output,
                        std::vector<float>* qk_debug,
                        std::vector<float>* sm_debug) const {
#if defined(__APPLE__)
        if (!available() || !flashAttentionPipeline) return false;
        if (num_heads == 0 || head_dim == 0 || kv_seq == 0) return false;
        size_t effective_kv_heads = kv_heads > 0 ? kv_heads : num_heads;
        if (num_heads % effective_kv_heads != 0) return false;
        // head_dim must be a power of 2 (tree reduction assumption).
        if ((head_dim & (head_dim - 1)) != 0) return false;
        size_t q_per = num_heads * head_dim;
        if (q.size() != q_per) return false;
        // K/V sizes depend on the layout. Caller passes both pointer and
        // total size so the kernel can bind the right buffer length.
        if (k_data == nullptr || v_data == nullptr) return false;
        if (k_size == 0 || v_size == 0) return false;

        // TG memory layout must match the kernel's. Tile size hard-coded to 32
        // here and in the kernel (`FLASH_TILE_SIZE`); they're a paired constant.
        static constexpr size_t kFlashTileSize = 32;
        size_t tg_floats = 2 * head_dim + 2 * kFlashTileSize * head_dim + kFlashTileSize;
        size_t tg_bytes = tg_floats * sizeof(float);
        if (head_dim > flashAttentionPipeline.maxTotalThreadsPerThreadgroup) return false;
        if (tg_bytes > device.maxThreadgroupMemoryLength) return false;

        output.assign(q_per, 0.0f);
        bool dump = (qk_debug != nullptr) || (sm_debug != nullptr);
        if (qk_debug) qk_debug->assign(num_heads * kv_seq, 0.0f);
        if (sm_debug) sm_debug->assign(num_heads * kv_seq, 0.0f);
        // Always bind a valid buffer at slots 4/5 even when not dumping.
        size_t dump_bytes = num_heads * kv_seq * sizeof(float);
        if (dump_bytes == 0) dump_bytes = sizeof(float);

        if (@available(macOS 11.0, *)) {
            @autoreleasepool {
                id<MTLBuffer> qBuf = [device newBufferWithBytes:q.data()
                                                          length:q.size() * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
                id<MTLBuffer> kBuf = [device newBufferWithBytes:k_data
                                                          length:k_size * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
                id<MTLBuffer> vBuf = [device newBufferWithBytes:v_data
                                                          length:v_size * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
                id<MTLBuffer> oBuf = [device newBufferWithLength:output.size() * sizeof(float)
                                                          options:MTLResourceStorageModeShared];
                id<MTLBuffer> qkBuf = [device newBufferWithLength:dump_bytes
                                                           options:MTLResourceStorageModeShared];
                id<MTLBuffer> smBuf = [device newBufferWithLength:dump_bytes
                                                           options:MTLResourceStorageModeShared];
                if (!qBuf || !kBuf || !vBuf || !oBuf || !qkBuf || !smBuf) return false;

                struct FlashAttnParams {
                    uint32_t num_heads;
                    uint32_t kv_heads;
                    uint32_t head_dim;
                    uint32_t kv_seq;
                    float inv_sqrt_d;
                    uint32_t debug_dump;
                    uint32_t apply_causal;
                    uint32_t q_position;
                    uint32_t k_stride_token;
                    uint32_t k_stride_kv_head;
                    uint32_t k_stride_feature;
                    uint32_t v_stride_token;
                    uint32_t v_stride_kv_head;
                    uint32_t v_stride_feature;
                } params;
                params.num_heads = static_cast<uint32_t>(num_heads);
                params.kv_heads = static_cast<uint32_t>(effective_kv_heads);
                params.head_dim = static_cast<uint32_t>(head_dim);
                params.kv_seq = static_cast<uint32_t>(kv_seq);
                params.inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(head_dim));
                params.debug_dump = dump ? 1u : 0u;
                params.apply_causal = apply_causal ? 1u : 0u;
                params.q_position = static_cast<uint32_t>(q_position);
                params.k_stride_token = static_cast<uint32_t>(k_stride_token);
                params.k_stride_kv_head = static_cast<uint32_t>(k_stride_kv_head);
                params.k_stride_feature = static_cast<uint32_t>(k_stride_feature);
                params.v_stride_token = static_cast<uint32_t>(v_stride_token);
                params.v_stride_kv_head = static_cast<uint32_t>(v_stride_kv_head);
                params.v_stride_feature = static_cast<uint32_t>(v_stride_feature);

                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:flashAttentionPipeline];
                [enc setBuffer:qBuf offset:0 atIndex:0];
                [enc setBuffer:kBuf offset:0 atIndex:1];
                [enc setBuffer:vBuf offset:0 atIndex:2];
                [enc setBuffer:oBuf offset:0 atIndex:3];
                [enc setBuffer:qkBuf offset:0 atIndex:4];
                [enc setBuffer:smBuf offset:0 atIndex:5];
                [enc setBytes:&params length:sizeof(params) atIndex:6];
                [enc setThreadgroupMemoryLength:tg_bytes atIndex:0];
                MTLSize tg = MTLSizeMake(num_heads, 1, 1);
                MTLSize th = MTLSizeMake(head_dim, 1, 1);
                [enc dispatchThreadgroups:tg threadsPerThreadgroup:th];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (cb.status != MTLCommandBufferStatusCompleted) return false;

                std::memcpy(output.data(), [oBuf contents], output.size() * sizeof(float));
                if (dump) {
                    if (qk_debug) std::memcpy(qk_debug->data(), [qkBuf contents],
                                              qk_debug->size() * sizeof(float));
                    if (sm_debug) std::memcpy(sm_debug->data(), [smBuf contents],
                                              sm_debug->size() * sizeof(float));
                }
                return true;
            }
        }
#endif
        (void)q; (void)k_data; (void)k_size; (void)v_data; (void)v_size;
        (void)num_heads; (void)kv_heads; (void)head_dim; (void)kv_seq;
        (void)apply_causal; (void)q_position;
        (void)k_stride_token; (void)k_stride_kv_head; (void)k_stride_feature;
        (void)v_stride_token; (void)v_stride_kv_head; (void)v_stride_feature;
        (void)output; (void)qk_debug; (void)sm_debug;
        return false;
    }
};

MetalExecutor& MetalExecutor::Instance() {
    static MetalExecutor instance;
    return instance;
}

MetalExecutor::MetalExecutor() : impl_(new Impl()) {}
MetalExecutor::~MetalExecutor() = default;

bool MetalExecutor::isAvailable() const {
    return impl_ && impl_->available();
}

void MetalExecutor::requireAvailable() const {
    if (!isAvailable()) {
        throw std::runtime_error("Metal is required but no device is available");
    }
}

bool MetalExecutor::shouldUseFor(const ExecutionNode& node) const {
    return node.backend == BackendKind::Metal
        && isAvailable()
        && !KernelDescriptorRegistry::forceCpu();
}

bool MetalExecutor::runMatMul(const std::string& weight_name,
                              const std::vector<float>& weights,
                              const std::vector<float>& input,
                              size_t rows,
                              size_t cols,
                              bool transpose_weight,
                              std::vector<float>& output,
                              const std::vector<float>* bias) const {
    if (!impl_) return false;
    return impl_->matmul(weight_name, weights, input, rows, cols, transpose_weight, output, bias);
}

bool MetalExecutor::runMatMulQ4_0(const std::string& weight_name,
                                  const std::vector<uint8_t>& weights,
                                  const std::vector<float>& input,
                                  size_t rows,
                                  size_t cols,
                                  size_t row_stride,
                                  uint32_t quant_version,
                                  std::vector<float>& output,
                                  const std::vector<float>* bias) const {
    if (!impl_) return false;
    return impl_->matmulQ4_0(weight_name, weights, input, rows, cols, row_stride, quant_version, bias, output);
}

bool MetalExecutor::runMatMulQ4_0Transposed(const std::string& weight_name,
                                            const std::vector<uint8_t>& weights,
                                            const std::vector<float>& input,
                                            size_t rows,
                                            size_t cols,
                                            size_t row_stride,
                                            uint32_t quant_version,
                                            std::vector<float>& output,
                                            const std::vector<float>* bias) const {
    if (!impl_) return false;
    return impl_->matmulQ4_0Transposed(weight_name, weights, input, rows, cols, row_stride, quant_version, bias, output);
}

bool MetalExecutor::runMatMulQ4_1(const std::string& weight_name,
                                  const std::vector<uint8_t>& weights,
                                  const std::vector<float>& input,
                                  size_t rows,
                                  size_t cols,
                                  size_t row_stride,
                                  std::vector<float>& output,
                                  const std::vector<float>* bias) const {
    if (!impl_) return false;
    return impl_->matmulQ4_1(weight_name, weights, input, rows, cols, row_stride, bias, output);
}

bool MetalExecutor::runMatMulQ5_0(const std::string& weight_name,
                                  const std::vector<uint8_t>& weights,
                                  const std::vector<float>& input,
                                  size_t rows,
                                  size_t cols,
                                  size_t row_stride,
                                  std::vector<float>& output,
                                  const std::vector<float>* bias) const {
    if (!impl_) return false;
    return impl_->matmulQ5_0(weight_name, weights, input, rows, cols, row_stride, bias, output);
}

bool MetalExecutor::runMatMulQ5_1(const std::string& weight_name,
                                  const std::vector<uint8_t>& weights,
                                  const std::vector<float>& input,
                                  size_t rows,
                                  size_t cols,
                                  size_t row_stride,
                                  std::vector<float>& output,
                                  const std::vector<float>* bias) const {
    if (!impl_) return false;
    return impl_->matmulQ5_1(weight_name, weights, input, rows, cols, row_stride, bias, output);
}

bool MetalExecutor::runMatMulQ2K(const std::string& weight_name,
                                 const std::vector<uint8_t>& weights,
                                 const std::vector<float>& input,
                                 size_t rows,
                                 size_t cols,
                                 size_t row_stride,
                                 std::vector<float>& output,
                                 const std::vector<float>* bias) const {
    if (!impl_) return false;
    return impl_->matmulQ2_K(weight_name, weights, input, rows, cols, row_stride, bias, output);
}

bool MetalExecutor::runMatMulQ3K(const std::string& weight_name,
                                 const std::vector<uint8_t>& weights,
                                 const std::vector<float>& input,
                                 size_t rows,
                                 size_t cols,
                                 size_t row_stride,
                                 std::vector<float>& output,
                                 const std::vector<float>* bias) const {
    if (!impl_) return false;
    return impl_->matmulQ3_K(weight_name, weights, input, rows, cols, row_stride, bias, output);
}

bool MetalExecutor::runMatMulQ4K(const std::string& weight_name,
                                 const std::vector<uint8_t>& weights,
                                 const std::vector<float>& input,
                                 size_t rows,
                                 size_t cols,
                                 size_t row_stride,
                                 std::vector<float>& output,
                                 const std::vector<float>* bias) const {
    if (!impl_) return false;
    return impl_->matmulQ4_K(weight_name, weights, input, rows, cols, row_stride, bias, output);
}

bool MetalExecutor::runMatMulQ5K(const std::string& weight_name,
                                 const std::vector<uint8_t>& weights,
                                 const std::vector<float>& input,
                                 size_t rows,
                                 size_t cols,
                                 size_t row_stride,
                                 std::vector<float>& output,
                                 const std::vector<float>* bias) const {
    if (!impl_) return false;
    return impl_->matmulQ5_K(weight_name, weights, input, rows, cols, row_stride, bias, output);
}

bool MetalExecutor::runMatMulQ6K(const std::string& weight_name,
                                 const std::vector<uint8_t>& weights,
                                 const std::vector<float>& input,
                                 size_t rows,
                                 size_t cols,
                                 size_t row_stride,
                                 std::vector<float>& output,
                                 const std::vector<float>* bias) const {
    if (!impl_) return false;
    return impl_->matmulQ6_K(weight_name, weights, input, rows, cols, row_stride, bias, output);
}

bool MetalExecutor::runMatMulQ6KTransposed(const std::string& weight_name,
                                           const std::vector<uint8_t>& weights,
                                           const std::vector<float>& input,
                                           size_t rows,
                                           size_t cols,
                                           size_t row_stride,
                                           std::vector<float>& output,
                                           const std::vector<float>* bias) const {
    if (!impl_) return false;
    return impl_->matmulQ6_KTransposed(weight_name, weights, input, rows, cols, row_stride, bias, output);
}

bool MetalExecutor::runMatMulQ8K(const std::string& weight_name,
                                 const std::vector<uint8_t>& weights,
                                 const std::vector<float>& input,
                                 size_t rows,
                                 size_t cols,
                                 size_t row_stride,
                                 std::vector<float>& output,
                                 const std::vector<float>* bias) const {
    if (!impl_) return false;
    return impl_->matmulQ8_K(weight_name, weights, input, rows, cols, row_stride, bias, output);
}

bool MetalExecutor::runMatMulQ8_0(const std::string& weight_name,
                                  const std::vector<uint8_t>& weights,
                                  const std::vector<float>& input,
                                  size_t rows,
                                  size_t cols,
                                  size_t row_stride,
                                  std::vector<float>& output,
                                  const std::vector<float>* bias) const {
    if (!impl_) return false;
    return impl_->matmulQ8_0(weight_name, weights, input, rows, cols, row_stride, bias, output);
}

bool MetalExecutor::runMatMulQ8_1(const std::string& weight_name,
                                  const std::vector<uint8_t>& weights,
                                  const std::vector<float>& input,
                                  size_t rows,
                                  size_t cols,
                                  size_t row_stride,
                                  std::vector<float>& output,
                                  const std::vector<float>* bias) const {
    if (!impl_) return false;
    return impl_->matmulQ8_1(weight_name, weights, input, rows, cols, row_stride, bias, output);
}

std::string MetalExecutor::weightCacheSummary() const {
    if (!impl_) return std::string{};
    std::lock_guard<std::mutex> lock(impl_->weight_cache_mutex_);
    std::ostringstream oss;
    double mb = static_cast<double>(impl_->cache_bytes_) / (1024.0 * 1024.0);
    oss << "weight cache: " << impl_->cache_hits_ << " hits, "
        << impl_->cache_misses_ << " misses, "
        << std::fixed << std::setprecision(1) << mb
        << " MB resident across " << impl_->weight_cache_.size() << " tensors";
    return oss.str();
}

bool MetalExecutor::hasForwardPassCB() const {
#if defined(__APPLE__)
    return impl_ && impl_->open_forward_pass_cb_ != nil;
#else
    return false;
#endif
}

bool MetalExecutor::beginForwardPassCB() const {
#if defined(__APPLE__)
    if (!impl_ || !impl_->available()) return false;
    if (impl_->open_forward_pass_cb_ != nil) return true;  // already open, idempotent
    if (@available(macOS 11.0, *)) {
        impl_->open_forward_pass_cb_ = [impl_->queue commandBuffer];
        return impl_->open_forward_pass_cb_ != nil;
    }
#endif
    return false;
}

bool MetalExecutor::flushForwardPassCB() const {
#if defined(__APPLE__)
    if (!impl_ || impl_->open_forward_pass_cb_ == nil) {
        // No open CB but possibly still some pool buffers to return / pending
        // readbacks to drain — defensive cleanup.
        if (impl_) {
            for (id<MTLBuffer> buf : impl_->pass_checked_out_) impl_->poolReturn(buf);
            impl_->pass_checked_out_.clear();
            impl_->pass_readbacks_.clear();
            impl_->pass_retained_.clear();
        }
        return true;
    }
    [impl_->open_forward_pass_cb_ commit];
    [impl_->open_forward_pass_cb_ waitUntilCompleted];
    bool ok = impl_->open_forward_pass_cb_.status == MTLCommandBufferStatusCompleted;
    // Drain pending host memcpys (pool output buffer -> host vector) only
    // for outputs whose needs_host flag is true. Outputs flagged
    // needs_host=false stay GPU-resident and would have already been
    // consumed by a downstream encoded op via its FromBuffer variant.
    for (const auto& rb : impl_->pass_readbacks_) {
        if (rb.source && rb.dst) {
            std::memcpy(rb.dst->data(), [rb.source contents], rb.byte_count);
        }
    }
    impl_->pass_readbacks_.clear();
    // Return all pool buffers checked out by this window.
    for (id<MTLBuffer> buf : impl_->pass_checked_out_) impl_->poolReturn(buf);
    impl_->pass_checked_out_.clear();
    // Transient (non-pool) retained buffers — input uploads via
    // newBufferWithBytes, weight uploads — can be released now.
    impl_->pass_retained_.clear();
    impl_->open_forward_pass_cb_ = nil;
    return ok;
#else
    return true;
#endif
}

void MetalExecutor::discardForwardPassCB() const {
#if defined(__APPLE__)
    if (!impl_) return;
    // Safe to drop without commit: Metal's explicit-commit API keeps the
    // buffer in MTLCommandBufferStatusNotEnqueued state until commit() is
    // called. If we adopt async scheduling, revisit.
    for (id<MTLBuffer> buf : impl_->pass_checked_out_) impl_->poolReturn(buf);
    impl_->pass_checked_out_.clear();
    impl_->pass_readbacks_.clear();
    impl_->pass_retained_.clear();
    impl_->open_forward_pass_cb_ = nil;
#endif
}

void* MetalExecutor::checkoutPoolBuffer(size_t bytes) const {
#if defined(__APPLE__)
    if (!impl_) return nullptr;
    id<MTLBuffer> buf = impl_->poolCheckout(bytes);
    if (!buf) return nullptr;
    // Atomically transfer the strong reference into pass_checked_out_
    // so the buffer survives the C++ void* round-trip back to the caller.
    // The (__bridge void*) cast is unretained; pass_checked_out_'s
    // std::vector<id<MTLBuffer>> is the actual owner under ARC. Without
    // this, the local `buf` would drop on function return and the void*
    // would point at freed memory by the time the caller used it.
    impl_->pass_checked_out_.push_back(buf);
    return (__bridge void*)buf;
#else
    (void)bytes;
    return nullptr;
#endif
}

void MetalExecutor::trackWindowBuffer(void* buffer) const {
    // No-op: ownership is established at checkoutPoolBuffer time so the
    // void* the caller holds is never an unretained dangler. Method kept
    // on the API for callers that still invoke it; the retention happened
    // earlier.
    (void)buffer;
#if defined(__APPLE__)
    (void)impl_;
#endif
}

bool MetalExecutor::encodeMatMulQ4_0FromHost(const std::string& weight_name,
                                             const std::vector<uint8_t>& weights,
                                             const std::vector<float>& host_input,
                                             size_t rows,
                                             size_t cols,
                                             size_t row_stride,
                                             uint32_t quant_version,
                                             void* output_buffer,
                                             std::vector<float>* host_dst,
                                             bool needs_host) const {
#if defined(__APPLE__)
    if (!impl_ || !impl_->available() || !impl_->q4MatmulPipeline) return false;
    if (impl_->open_forward_pass_cb_ == nil) return false;
    if (host_input.size() != cols) return false;
    if (weights.size() < row_stride * rows) return false;
    if (!output_buffer) return false;
    if (@available(macOS 11.0, *)) {
        id<MTLBuffer> weightBuffer = impl_->getOrCacheWeight(weight_name, weights);
        id<MTLBuffer> inputBuffer = [impl_->device newBufferWithBytes:host_input.data()
                                                               length:host_input.size() * sizeof(float)
                                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> outBuf = (__bridge id<MTLBuffer>)output_buffer;
        if (!weightBuffer || !inputBuffer) return false;
        id<MTLComputeCommandEncoder> encoder = [impl_->open_forward_pass_cb_ computeCommandEncoder];
        if (!encoder) return false;
        [encoder setComputePipelineState:impl_->q4MatmulPipeline];
        [encoder setBuffer:weightBuffer offset:0 atIndex:0];
        [encoder setBuffer:inputBuffer offset:0 atIndex:1];
        [encoder setBuffer:outBuf offset:0 atIndex:2];
        Q4_0ParamsNative params = {static_cast<uint32_t>(rows),
                                   static_cast<uint32_t>(cols),
                                   static_cast<uint32_t>(row_stride),
                                   quant_version};
        [encoder setBytes:&params length:sizeof(Q4_0ParamsNative) atIndex:3];
        NSUInteger threadsPerGroup = impl_->q4MatmulPipeline.threadExecutionWidth;
        if (threadsPerGroup == 0) threadsPerGroup = 32;
        NSUInteger groups = (rows + threadsPerGroup - 1) / threadsPerGroup;
        if (groups == 0) groups = 1;
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [encoder endEncoding];
        impl_->pass_retained_.push_back(inputBuffer);
        impl_->recordReadback(outBuf, host_dst, rows * sizeof(float), needs_host);
        return true;
    }
#endif
    (void)weight_name; (void)weights; (void)host_input; (void)rows; (void)cols;
    (void)row_stride; (void)quant_version; (void)output_buffer; (void)host_dst; (void)needs_host;
    return false;
}

bool MetalExecutor::encodeMatMulQ4_0FromBuffer(const std::string& weight_name,
                                               const std::vector<uint8_t>& weights,
                                               void* input_buffer,
                                               size_t input_count,
                                               size_t rows,
                                               size_t cols,
                                               size_t row_stride,
                                               uint32_t quant_version,
                                               void* output_buffer,
                                               std::vector<float>* host_dst,
                                               bool needs_host) const {
#if defined(__APPLE__)
    if (!impl_ || !impl_->available() || !impl_->q4MatmulPipeline) return false;
    if (impl_->open_forward_pass_cb_ == nil) return false;
    if (input_count != cols) return false;
    if (weights.size() < row_stride * rows) return false;
    if (!input_buffer || !output_buffer) return false;
    if (@available(macOS 11.0, *)) {
        id<MTLBuffer> weightBuffer = impl_->getOrCacheWeight(weight_name, weights);
        id<MTLBuffer> inputBuf = (__bridge id<MTLBuffer>)input_buffer;
        id<MTLBuffer> outBuf = (__bridge id<MTLBuffer>)output_buffer;
        if (!weightBuffer) return false;
        id<MTLComputeCommandEncoder> encoder = [impl_->open_forward_pass_cb_ computeCommandEncoder];
        if (!encoder) return false;
        [encoder setComputePipelineState:impl_->q4MatmulPipeline];
        [encoder setBuffer:weightBuffer offset:0 atIndex:0];
        [encoder setBuffer:inputBuf offset:0 atIndex:1];
        [encoder setBuffer:outBuf offset:0 atIndex:2];
        Q4_0ParamsNative params = {static_cast<uint32_t>(rows),
                                   static_cast<uint32_t>(cols),
                                   static_cast<uint32_t>(row_stride),
                                   quant_version};
        [encoder setBytes:&params length:sizeof(Q4_0ParamsNative) atIndex:3];
        NSUInteger threadsPerGroup = impl_->q4MatmulPipeline.threadExecutionWidth;
        if (threadsPerGroup == 0) threadsPerGroup = 32;
        NSUInteger groups = (rows + threadsPerGroup - 1) / threadsPerGroup;
        if (groups == 0) groups = 1;
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [encoder endEncoding];
        impl_->recordReadback(outBuf, host_dst, rows * sizeof(float), needs_host);
        return true;
    }
#endif
    (void)weight_name; (void)weights; (void)input_buffer; (void)input_count;
    (void)rows; (void)cols; (void)row_stride; (void)quant_version;
    (void)output_buffer; (void)host_dst; (void)needs_host;
    return false;
}

bool MetalExecutor::encodeRmsNormFromHost(const std::vector<float>& host_input,
                                          const std::vector<float>& weight,
                                          float epsilon,
                                          void* output_buffer,
                                          std::vector<float>* host_dst,
                                          bool needs_host) const {
#if defined(__APPLE__)
    if (!impl_ || !impl_->available() || !impl_->normPipeline) return false;
    if (impl_->open_forward_pass_cb_ == nil) return false;
    if (host_input.size() != weight.size() || host_input.empty()) return false;
    if (host_input.size() > std::numeric_limits<uint32_t>::max()) return false;
    if (!output_buffer) return false;
    if (@available(macOS 11.0, *)) {
        size_t bytes = host_input.size() * sizeof(float);
        id<MTLBuffer> inBuffer = [impl_->device newBufferWithBytes:host_input.data()
                                                            length:bytes
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> wBuffer = [impl_->device newBufferWithBytes:weight.data()
                                                           length:bytes
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> outBuf = (__bridge id<MTLBuffer>)output_buffer;
        if (!inBuffer || !wBuffer) return false;
        id<MTLComputeCommandEncoder> encoder = [impl_->open_forward_pass_cb_ computeCommandEncoder];
        if (!encoder) return false;
        [encoder setComputePipelineState:impl_->normPipeline];
        [encoder setBuffer:inBuffer offset:0 atIndex:0];
        [encoder setBuffer:wBuffer offset:0 atIndex:1];
        [encoder setBuffer:outBuf offset:0 atIndex:2];
        uint32_t length = static_cast<uint32_t>(host_input.size());
        [encoder setBytes:&length length:sizeof(uint32_t) atIndex:3];
        float eps = epsilon;
        [encoder setBytes:&eps length:sizeof(float) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];
        impl_->pass_retained_.push_back(inBuffer);
        impl_->pass_retained_.push_back(wBuffer);
        impl_->recordReadback(outBuf, host_dst, bytes, needs_host);
        return true;
    }
#endif
    (void)host_input; (void)weight; (void)epsilon; (void)output_buffer;
    (void)host_dst; (void)needs_host;
    return false;
}

bool MetalExecutor::encodeRmsNormFromBuffer(void* input_buffer,
                                            size_t input_count,
                                            const std::vector<float>& weight,
                                            float epsilon,
                                            void* output_buffer,
                                            std::vector<float>* host_dst,
                                            bool needs_host) const {
#if defined(__APPLE__)
    if (!impl_ || !impl_->available() || !impl_->normPipeline) return false;
    if (impl_->open_forward_pass_cb_ == nil) return false;
    if (input_count != weight.size() || input_count == 0) return false;
    if (input_count > std::numeric_limits<uint32_t>::max()) return false;
    if (!input_buffer || !output_buffer) return false;
    if (@available(macOS 11.0, *)) {
        size_t bytes = input_count * sizeof(float);
        id<MTLBuffer> inBuf = (__bridge id<MTLBuffer>)input_buffer;
        id<MTLBuffer> wBuffer = [impl_->device newBufferWithBytes:weight.data()
                                                           length:bytes
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> outBuf = (__bridge id<MTLBuffer>)output_buffer;
        if (!wBuffer) return false;
        id<MTLComputeCommandEncoder> encoder = [impl_->open_forward_pass_cb_ computeCommandEncoder];
        if (!encoder) return false;
        [encoder setComputePipelineState:impl_->normPipeline];
        [encoder setBuffer:inBuf offset:0 atIndex:0];
        [encoder setBuffer:wBuffer offset:0 atIndex:1];
        [encoder setBuffer:outBuf offset:0 atIndex:2];
        uint32_t length = static_cast<uint32_t>(input_count);
        [encoder setBytes:&length length:sizeof(uint32_t) atIndex:3];
        float eps = epsilon;
        [encoder setBytes:&eps length:sizeof(float) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];
        impl_->pass_retained_.push_back(wBuffer);
        impl_->recordReadback(outBuf, host_dst, bytes, needs_host);
        return true;
    }
#endif
    (void)input_buffer; (void)input_count; (void)weight; (void)epsilon;
    (void)output_buffer; (void)host_dst; (void)needs_host;
    return false;
}

bool MetalExecutor::encodeAddFromHost(const std::vector<float>& host_a,
                                      const std::vector<float>& host_b,
                                      void* output_buffer,
                                      std::vector<float>* host_dst,
                                      bool needs_host) const {
    return encodeAddMixed(&host_a, nullptr, &host_b, nullptr,
                          host_a.size(), output_buffer, host_dst, needs_host);
}

bool MetalExecutor::encodeAddMixed(const std::vector<float>* host_a, void* buffer_a,
                                   const std::vector<float>* host_b, void* buffer_b,
                                   size_t element_count,
                                   void* output_buffer,
                                   std::vector<float>* host_dst,
                                   bool needs_host) const {
#if defined(__APPLE__)
    if (!impl_ || !impl_->available() || !impl_->addPipeline) return false;
    if (impl_->open_forward_pass_cb_ == nil) return false;
    if (element_count == 0) return false;
    if (element_count > std::numeric_limits<uint32_t>::max()) return false;
    if (!output_buffer) return false;
    if ((host_a == nullptr) == (buffer_a == nullptr)) return false;  // exactly one set
    if ((host_b == nullptr) == (buffer_b == nullptr)) return false;
    if (host_a && host_a->size() != element_count) return false;
    if (host_b && host_b->size() != element_count) return false;
    if (@available(macOS 11.0, *)) {
        size_t bytes = element_count * sizeof(float);
        id<MTLBuffer> aBuf;
        if (host_a) {
            id<MTLBuffer> upload = [impl_->device newBufferWithBytes:host_a->data()
                                                              length:bytes
                                                             options:MTLResourceStorageModeShared];
            if (!upload) return false;
            impl_->pass_retained_.push_back(upload);
            aBuf = upload;
        } else {
            aBuf = (__bridge id<MTLBuffer>)buffer_a;
        }
        id<MTLBuffer> bBuf;
        if (host_b) {
            id<MTLBuffer> upload = [impl_->device newBufferWithBytes:host_b->data()
                                                              length:bytes
                                                             options:MTLResourceStorageModeShared];
            if (!upload) return false;
            impl_->pass_retained_.push_back(upload);
            bBuf = upload;
        } else {
            bBuf = (__bridge id<MTLBuffer>)buffer_b;
        }
        id<MTLBuffer> outBuf = (__bridge id<MTLBuffer>)output_buffer;
        id<MTLComputeCommandEncoder> encoder = [impl_->open_forward_pass_cb_ computeCommandEncoder];
        if (!encoder) return false;
        [encoder setComputePipelineState:impl_->addPipeline];
        [encoder setBuffer:aBuf offset:0 atIndex:0];
        [encoder setBuffer:bBuf offset:0 atIndex:1];
        [encoder setBuffer:outBuf offset:0 atIndex:2];
        uint32_t length = static_cast<uint32_t>(element_count);
        [encoder setBytes:&length length:sizeof(uint32_t) atIndex:3];
        NSUInteger threadWidth = impl_->addPipeline.threadExecutionWidth;
        if (threadWidth == 0) threadWidth = 32;
        NSUInteger threadgroups = (element_count + threadWidth - 1) / threadWidth;
        if (threadgroups == 0) threadgroups = 1;
        [encoder dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)];
        [encoder endEncoding];
        impl_->recordReadback(outBuf, host_dst, bytes, needs_host);
        return true;
    }
#endif
    (void)host_a; (void)buffer_a; (void)host_b; (void)buffer_b;
    (void)element_count; (void)output_buffer; (void)host_dst; (void)needs_host;
    return false;
}

bool MetalExecutor::ensureSharedBuffer(std::vector<float>& data, MetalBufferHandle& handle) const {
#if defined(__APPLE__)
    if (!impl_ || !impl_->available()) return false;
    size_t bytes = data.size() * sizeof(float);
    if (handle.buffer && handle.bytes == bytes) {
        return true;
    }
    releaseBuffer(handle);
    if (@available(macOS 11.0, *)) {
        id<MTLBuffer> buffer = [impl_->device newBufferWithBytesNoCopy:data.data()
                                                               length:bytes
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];
        if (!buffer) return false;
        handle.buffer = (__bridge_retained void*)buffer;
        handle.bytes = bytes;
        handle.host_dirty = false;
        handle.device_dirty = false;
        return true;
    }
    return false;
#else
    (void)data;
    (void)handle;
    return false;
#endif
}

void MetalExecutor::releaseBuffer(MetalBufferHandle& handle) const {
#if defined(__APPLE__)
    if (handle.buffer) {
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)handle.buffer;
        buf = nil;
        (void)buf;
        handle.buffer = nullptr;
    }
#else
    (void)handle;
#endif
    handle.bytes = 0;
    handle.host_dirty = false;
    handle.device_dirty = false;
}

void MetalExecutor::markHostModified(MetalBufferHandle& handle,
                                     size_t offset_bytes,
                                     size_t length_bytes) const {
#if defined(__APPLE__)
    if (!handle.buffer || length_bytes == 0) return;
    if (@available(macOS 11.0, *)) {
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)handle.buffer;
        NSRange range = NSMakeRange(offset_bytes, length_bytes);
        [buf didModifyRange:range];
        handle.host_dirty = false;
    }
#else
    (void)handle;
    (void)offset_bytes;
    (void)length_bytes;
#endif
}

bool MetalExecutor::scatterKVCache(const std::vector<float>& src,
                                   MetalBufferHandle& dst_handle,
                                   size_t kv_heads,
                                   size_t tokens,
                                   size_t head_dim,
                                   size_t context_length,
                                   size_t base_position) const {
#if defined(__APPLE__)
    if (!impl_ || !impl_->available()) return false;
    if (!dst_handle.buffer || tokens == 0 || head_dim == 0 || kv_heads == 0) return false;
    size_t expected = kv_heads * tokens * head_dim;
    if (src.size() != expected) return false;
    id<MTLBuffer> dst = (__bridge id<MTLBuffer>)dst_handle.buffer;
    if (!dst) return false;
    if (@available(macOS 11.0, *)) {
        @autoreleasepool {
            id<MTLBuffer> stage = [impl_->device newBufferWithBytes:src.data()
                                                             length:expected * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
            if (!stage) return false;
            return impl_->scatterTokens(stage,
                                        dst,
                                        kv_heads,
                                        tokens,
                                        head_dim,
                                        context_length,
                                        base_position);
        }
    }
#else
    (void)src;
    (void)dst_handle;
    (void)kv_heads;
    (void)tokens;
    (void)head_dim;
    (void)context_length;
    (void)base_position;
#endif
    return false;
}

bool MetalExecutor::dequantQ4Block(const std::vector<uint8_t>& src,
                                   size_t cols,
                                   std::vector<float>& dst) const {
    if (!impl_) return false;
    return impl_->dequantQ4Block(src, cols, dst);
}

bool MetalExecutor::dequantQ8Block(const std::vector<uint8_t>& src,
                                   size_t cols,
                                   std::vector<float>& dst) const {
    if (!impl_) return false;
    return impl_->dequantQ8Block(src, cols, dst);
}

bool MetalExecutor::dequantQ4_1Block(const std::vector<uint8_t>& src,
                                     size_t cols,
                                     std::vector<float>& dst) const {
    if (!impl_) return false;
    return impl_->dequantQ4_1Block(src, cols, dst);
}

bool MetalExecutor::dequantQ5_0Block(const std::vector<uint8_t>& src,
                                     size_t cols,
                                     std::vector<float>& dst) const {
    if (!impl_) return false;
    return impl_->dequantQ5_0Block(src, cols, dst);
}

bool MetalExecutor::dequantQ5_1Block(const std::vector<uint8_t>& src,
                                     size_t cols,
                                     std::vector<float>& dst) const {
    if (!impl_) return false;
    return impl_->dequantQ5_1Block(src, cols, dst);
}

bool MetalExecutor::dequantQ8_1Block(const std::vector<uint8_t>& src,
                                     size_t cols,
                                     std::vector<float>& dst) const {
    if (!impl_) return false;
    return impl_->dequantQ8_1Block(src, cols, dst);
}

bool MetalExecutor::dequantQKRow(const std::vector<uint8_t>& src,
                                 uint32_t dtype,
                                 size_t cols,
                                 size_t row_stride,
                                 std::vector<float>& dst) const {
    if (!impl_) return false;
    return impl_->dequantQKRow(src, dtype, cols, row_stride, dst);
}

bool MetalExecutor::runFeedForward(const std::vector<float>& gate,
                                   const std::vector<float>& up,
                                   std::vector<float>& output) const {
    if (!impl_) return false;
    return impl_->feedForward(gate, up, output);
}

bool MetalExecutor::runAdd(const std::vector<float>& a,
                           const std::vector<float>& b,
                           std::vector<float>& output,
                           const std::vector<float>* bias) const {
    if (!impl_) return false;
    return impl_->addVectors(a, b, bias, output);
}

bool MetalExecutor::runRmsNorm(const std::vector<float>& input,
                               const std::vector<float>& weight,
                               float epsilon,
                               std::vector<float>& output) const {
    if (!impl_) return false;
    return impl_->rmsNorm(input, weight, epsilon, output);
}

bool MetalExecutor::runLayerNorm(const std::vector<float>& input,
                                 const std::vector<float>& weight,
                                 const std::vector<float>* bias,
                                 float epsilon,
                                 std::vector<float>& output) const {
    if (!impl_) return false;
    return impl_->layerNorm(input, weight, bias, epsilon, output);
}

bool MetalExecutor::runSoftmax(const std::vector<float>& input,
                               std::vector<float>& output) const {
    if (!impl_) return false;
    return impl_->softmax(input, output);
}

bool MetalExecutor::runAttention(const std::vector<float>& q,
                                 const std::vector<float>& k,
                                 const std::vector<float>& v,
                                 size_t num_heads,
                                 size_t kv_heads,
                                 size_t head_dim,
                                 size_t context_length,
                                 const std::vector<float>& mask,
                                 const std::vector<float>* alibi_slopes,
                                 size_t position,
                                 size_t rotary_dim,
                                 float rope_freq_base,
                                 float rope_freq_scale,
                                 const CacheDescriptor& cache_k,
                                 const CacheDescriptor& cache_v,
                                 std::vector<float>& output) const {
    if (!impl_) return false;
    return impl_->attention(q, k, v, num_heads, kv_heads, head_dim,
                            context_length, mask, alibi_slopes, position,
                            rotary_dim, rope_freq_base, rope_freq_scale,
                            cache_k, cache_v, output);
}

bool MetalExecutor::runFlashAttention(const std::vector<float>& q,
                                       const std::vector<float>& k,
                                       const std::vector<float>& v,
                                       size_t num_heads,
                                       size_t kv_heads,
                                       size_t head_dim,
                                       size_t kv_seq,
                                       bool apply_causal,
                                       size_t q_position,
                                       std::vector<float>& output,
                                       std::vector<float>* qk_debug,
                                       std::vector<float>* sm_debug) const {
    if (!impl_) return false;
    // Harness layout: K/V are [kv_seq, kv_heads, head_dim] token-major.
    size_t effective_kv_heads = kv_heads > 0 ? kv_heads : num_heads;
    size_t k_stride_token = effective_kv_heads * head_dim;
    size_t k_stride_kv_head = head_dim;
    size_t k_stride_feature = 1;
    return impl_->flashAttention(q, k.data(), k.size(), v.data(), v.size(),
                                 num_heads, kv_heads, head_dim, kv_seq,
                                 apply_causal, q_position,
                                 k_stride_token, k_stride_kv_head, k_stride_feature,
                                 k_stride_token, k_stride_kv_head, k_stride_feature,
                                 output, qk_debug, sm_debug);
}

bool MetalExecutor::runFlashAttentionStrided(const std::vector<float>& q,
                                              const float* k_data, size_t k_size,
                                              const float* v_data, size_t v_size,
                                              size_t num_heads,
                                              size_t kv_heads,
                                              size_t head_dim,
                                              size_t kv_seq,
                                              bool apply_causal,
                                              size_t q_position,
                                              size_t k_stride_token,
                                              size_t k_stride_kv_head,
                                              size_t v_stride_token,
                                              size_t v_stride_kv_head,
                                              std::vector<float>& output) const {
    if (!impl_) return false;
    return impl_->flashAttention(q, k_data, k_size, v_data, v_size,
                                 num_heads, kv_heads, head_dim, kv_seq,
                                 apply_causal, q_position,
                                 k_stride_token, k_stride_kv_head, 1,
                                 v_stride_token, v_stride_kv_head, 1,
                                 output, nullptr, nullptr);
}

bool MetalExecutor::hasFeedForwardKernel() const {
    return impl_ && impl_->hasFeedForwardKernel();
}

bool MetalExecutor::hasAddKernel() const {
    return impl_ && impl_->hasAddKernel();
}

bool MetalExecutor::hasRmsNormKernel() const {
    return impl_ && impl_->hasNormKernel();
}

bool MetalExecutor::hasLayerNormKernel() const {
    return impl_ && impl_->hasLayerNormKernel();
}

bool MetalExecutor::hasSoftmaxKernel() const {
    return impl_ && impl_->hasSoftmaxKernel();
}

bool MetalExecutor::hasBiasAddKernel() const {
    return impl_ && impl_->available() && impl_->biasAddPipeline != nil;
}

bool MetalExecutor::hasKVWriteKernel() const {
    return impl_ && impl_->available() && impl_->kvWritePipeline != nil;
}

bool MetalExecutor::hasDequantQKKernel() const {
    return impl_ && impl_->available() && impl_->dequantQKPipeline != nil;
}

} // namespace runtime
} // namespace mlc
