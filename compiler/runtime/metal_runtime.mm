#include "runtime/metal_runtime.hpp"
#include "runtime/quantization.hpp"

#if defined(__APPLE__)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>

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
    uint col_index = 0;
    for (uint b = 0; b < blocks && col_index < cols; ++b) {
        const device half* d_half = reinterpret_cast<const device half*>(row + b * block_size);
        float d = static_cast<float>(d_half[0]);
        const device uchar* qs = row + b * block_size + 2;
        for (uint j = 0; j < 16 && col_index < cols; ++j) {
            uchar byte = qs[j];
            int lo = static_cast<int>(byte & 0x0F) - 8;
            acc += d * static_cast<float>(lo) * input[col_index++];
            if (col_index < cols) {
                int hi = static_cast<int>((byte >> 4) & 0x0F) - 8;
                acc += d * static_cast<float>(hi) * input[col_index++];
            }
        }
    }
    output[gid] = acc;
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
        uint32_t qh = *(const device uint32_t*)(qh_ptr);
        const device uchar* qs = row + b * block_size + 6;
        for (uint j = 0; j < 16 && col_index < cols; ++j) {
            uchar byte = qs[j];
            uint8_t xh0 = ((qh >> (j + 0)) << 4) & 0x10;
            uint8_t xh1 = ((qh >> (j + 12))     ) & 0x10;
            int x0 = ((byte & 0x0F) | xh0) - 16;
            acc += (static_cast<float>(x0) * d) * input[col_index++];
            if (col_index < cols) {
                int x1 = (((byte >> 4) & 0x0F) | xh1) - 16;
                acc += (static_cast<float>(x1) * d) * input[col_index++];
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
            uint8_t xh0 = ((qh >> (j + 0)) << 4) & 0x10;
            uint8_t xh1 = ((qh >> (j + 12))     ) & 0x10;
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
        const device uchar* scales = block;
        const device uchar* qs = block + 16;
        const device half* d_ptr = reinterpret_cast<const device half*>(block + 16 + 64);
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
        const device int8_t* scales = reinterpret_cast<const device int8_t*>(aux);
        const device half* d_ptr = reinterpret_cast<const device half*>(block + 64 + 16);
        float d = static_cast<float>(d_ptr[0]);
        int is = 0;
        const uint total = min(256u, cols - col_index);
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
        const device uchar* scales = block;
        const device uchar* qs = block + 12;
        const device half* d_ptr = reinterpret_cast<const device half*>(block + 12 + 128);
        float d = static_cast<float>(d_ptr[0]);
        float min = static_cast<float>(d_ptr[1]);
        uint total = min(256u, cols - col_index);
        for (uint n = 0; n < total; n += 32) {
            uint8_t sc = scales[n / 32];
            uint8_t sc2 = scales[n / 32 + 8];
            float dl = d * static_cast<float>(sc & 0xF);
            float ml = min * static_cast<float>(sc >> 4);
            for (int l = 0; l < 16 && (col_index + n + l) < cols; ++l) {
                uint8_t val = (qs[l] & 0x0F);
                float vf = dl * static_cast<float>(val) - ml;
                acc += vf * input[col_index + n + l];
            }
            dl = d * static_cast<float>(sc2 & 0xF);
            ml = min * static_cast<float>(sc2 >> 4);
            for (int l = 0; l < 16 && (col_index + n + 16 + l) < cols; ++l) {
                uint8_t val = ((qs[l] >> 4) & 0x0F);
                float vf = dl * static_cast<float>(val) - ml;
                acc += vf * input[col_index + n + 16 + l];
            }
            qs += 16;
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
        const device uchar* scales = block;
        const device uchar* qh = block + 12;
        const device uchar* qs = block + 12 + 32;
        const device half* d_ptr = reinterpret_cast<const device half*>(block + 12 + 32 + 128);
        float d = static_cast<float>(d_ptr[0]);
        float min = static_cast<float>(d_ptr[1]);
        uint total = min(256u, cols - col_index);
        for (uint n = 0; n < total; n += 32) {
            uint8_t sc = scales[n / 32];
            uint8_t sc2 = scales[n / 32 + 8];
            float dl = d * static_cast<float>(sc & 0xF);
            float ml = min * static_cast<float>(sc >> 4);
            for (int l = 0; l < 16 && (col_index + n + l) < cols; ++l) {
                uint8_t vh = ((qh[(n / 32) * 4 + l / 4] >> (2 * (l % 4))) & 0x3) << 4;
                uint8_t val = (qs[l] & 0x0F) | vh;
                float vf = dl * static_cast<float>(static_cast<int>(val) - 16) - ml;
                acc += vf * input[col_index + n + l];
            }
            dl = d * static_cast<float>(sc2 & 0xF);
            ml = min * static_cast<float>(sc2 >> 4);
            for (int l = 0; l < 16 && (col_index + n + 16 + l) < cols; ++l) {
                uint8_t vh = ((qh[(n / 32) * 4 + l / 4 + 16/4] >> (2 * (l % 4))) & 0x3) << 4;
                uint8_t val = ((qs[l] >> 4) & 0x0F) | vh;
                float vf = dl * static_cast<float>(static_cast<int>(val) - 16) - ml;
                acc += vf * input[col_index + n + 16 + l];
            }
            qs += 16;
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
        const device int8_t* scales = reinterpret_cast<const device int8_t*>(block + 128 + 64);
        const device half* d_ptr = reinterpret_cast<const device half*>(block + 128 + 64 + 32);
        float d = static_cast<float>(d_ptr[0]);
        uint total = min(256u, cols - col_index);
        for (uint i = 0; i < total / 16 && (col_index + i * 16) < cols; ++i) {
            int8_t sc = scales[i];
            float dl = d * static_cast<float>(sc);
            for (int j = 0; j < 16 && (col_index + i * 16 + j) < cols; ++j) {
                int idx = i * 16 + j;
                int bit = (idx & 1) ? (ql[idx / 2] >> 4) : (ql[idx / 2] & 0xF);
                int high = (qh[idx / 4] >> (2 * (idx % 4))) & 0x3;
                int val = bit | (high << 4);
                if (val > 31) val -= 64;
                float vf = dl * static_cast<float>(val);
                acc += vf * input[col_index + idx];
            }
        }
        col_index += 256;
    }
    output[gid] = acc;
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
    device float* vec = reinterpret_cast<device float*>(
        reinterpret_cast<device uchar*>(data) + gid * params.row_stride);
    const device float* cos_row = cos_table + gid * params.pair_stride;
    const device float* sin_row = sin_table + gid * params.pair_stride;
    for (uint i = 0; i < pairs; ++i) {
        float c = (i < params.pair_stride) ? cos_row[i] : 1.0f;
        float s = (i < params.pair_stride) ? sin_row[i] : 0.0f;
        float x0 = vec[2 * i];
        float x1 = vec[2 * i + 1];
        vec[2 * i] = x0 * c - x1 * s;
        vec[2 * i + 1] = x0 * s + x1 * c;
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

struct RotaryParamsNative {
    uint32_t head_dim;
    uint32_t rotary_dim;
    uint32_t row_stride;
    uint32_t pair_stride;
    uint32_t count;
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
    id<MTLComputePipelineState> normPipeline = nil;
    id<MTLComputePipelineState> softmaxPipeline = nil;
    id<MTLComputePipelineState> q4MatmulPipeline = nil;
    id<MTLComputePipelineState> q4_1MatmulPipeline = nil;
    id<MTLComputePipelineState> q5_0MatmulPipeline = nil;
    id<MTLComputePipelineState> q5_1MatmulPipeline = nil;
    id<MTLComputePipelineState> q2KMatmulPipeline = nil;
    id<MTLComputePipelineState> q3KMatmulPipeline = nil;
    id<MTLComputePipelineState> q4KMatmulPipeline = nil;
    id<MTLComputePipelineState> q5KMatmulPipeline = nil;
    id<MTLComputePipelineState> q6KMatmulPipeline = nil;
    id<MTLComputePipelineState> q8KMatmulPipeline = nil;
    id<MTLComputePipelineState> q8_0MatmulPipeline = nil;
    id<MTLComputePipelineState> q8_1MatmulPipeline = nil;
    id<MTLComputePipelineState> rotaryPipeline = nil;
    id<MTLComputePipelineState> biasAddPipeline = nil;
    id<MTLComputePipelineState> kvWritePipeline = nil;
#endif

    Impl() {
#if defined(__APPLE__)
        @autoreleasepool {
            device = MTLCreateSystemDefaultDevice();
            if (device) {
                queue = [device newCommandQueue];
                if (queue) {
                    NSError* error = nil;
                    NSString* source = [[NSString alloc] initWithUTF8String:kMetalKernelsSource];
                    if (source) {
                        id<MTLLibrary> library = [device newLibraryWithSource:source
                                                                      options:nil
                                                                        error:&error];
                        if (library) {
                            auto buildPipeline = ^id<MTLComputePipelineState>(NSString* name) {
                                NSError* localError = nil;
                                id<MTLFunction> function = [library newFunctionWithName:name];
                                if (!function) return (id<MTLComputePipelineState>)nil;
                                id<MTLComputePipelineState> pipe =
                                    [device newComputePipelineStateWithFunction:function error:&localError];
                                return pipe;
                            };
                            ffnPipeline = buildPipeline(@"feedforward_silu_mul");
                            addPipeline = buildPipeline(@"add_vectors");
                            normPipeline = buildPipeline(@"rms_norm_kernel");
                            softmaxPipeline = buildPipeline(@"vector_softmax");
                            q4MatmulPipeline = buildPipeline(@"q4_0_matmul");
                            q4_1MatmulPipeline = buildPipeline(@"q4_1_matmul");
                            q5_0MatmulPipeline = buildPipeline(@"q5_0_matmul");
                            q5_1MatmulPipeline = buildPipeline(@"q5_1_matmul");
                            q2KMatmulPipeline = buildPipeline(@"q2_k_matmul");
                            q3KMatmulPipeline = buildPipeline(@"q3_k_matmul");
                            q4KMatmulPipeline = buildPipeline(@"q4_k_matmul");
                            q5KMatmulPipeline = buildPipeline(@"q5_k_matmul");
                            q6KMatmulPipeline = buildPipeline(@"q6_k_matmul");
                            q8KMatmulPipeline = buildPipeline(@"q8_k_matmul");
                            q8_0MatmulPipeline = buildPipeline(@"q8_0_matmul");
                            q8_1MatmulPipeline = buildPipeline(@"q8_1_matmul");
                            rotaryPipeline = buildPipeline(@"apply_rotary_batch");
                            biasAddPipeline = buildPipeline(@"add_bias_strided");
                            kvWritePipeline = buildPipeline(@"scatter_kv");
                        }
                    }
                }
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

    bool addBiasInPlace(id<MTLBuffer> buffer,
                        size_t elements,
                        size_t stride_elems,
                        const std::vector<float>* bias) const {
#if defined(__APPLE__)
        if (!bias || bias->empty()) return true;
        if (!available() || !biasAddPipeline) return false;
        if (!buffer || bias->size() != elements || stride_elems == 0) return false;
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
        if (!available() || !kvWritePipeline) return false;
        if (!src || !dst || tokens == 0 || head_dim == 0 || kv_heads == 0) return false;
        size_t kv_stride = context_length * head_dim;
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

    bool applyRotaryGPU(id<MTLBuffer> buffer,
                        size_t count,
                        size_t row_stride,
                        size_t head_dim,
                        size_t rotary_dim,
                        const float* cos_ptr,
                        const float* sin_ptr,
                        size_t pair_stride) const {
#if defined(__APPLE__)
        if (!available() || !rotaryPipeline) return false;
        if (rotary_dim < 2 || count == 0) return true;
        if (pair_stride == 0 || !cos_ptr || !sin_ptr) return false;
        size_t pairs = rotary_dim / 2;
        if (pairs == 0) return true;
        size_t expected = pair_stride * count;
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
                    static_cast<uint32_t>(count)};
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

    bool matmul(const std::vector<float>& weights,
                const std::vector<float>& input,
                size_t rows,
                size_t cols,
                bool transpose_weight,
                std::vector<float>& output,
                const std::vector<float>* bias = nullptr) const {
#if defined(__APPLE__)
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
            if (bias && !addBiasInPlace(resultBuffer, rows, stride_elems, bias)) {
                return false;
            }
            output.resize(rows);
            const uint8_t* resultPtr = reinterpret_cast<const uint8_t*>([resultBuffer contents]);
            for (size_t r = 0; r < rows; ++r) {
                std::memcpy(output.data() + r,
                            resultPtr + r * resultRowBytes,
                            sizeof(float));
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

bool matmulQuant(const std::vector<uint8_t>& weights,
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
                id<MTLBuffer> weightBuffer = [device newBufferWithBytes:weights.data()
                                                                 length:weights.size()
                                                                options:MTLResourceStorageModeShared];
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

                if (bias && !addBiasInPlace(outputBuffer, rows, 1, bias)) {
                    return false;
                }
                output.resize(rows);
                std::memcpy(output.data(), [outputBuffer contents], rows * sizeof(float));
                return commandBuffer.status == MTLCommandBufferStatusCompleted;
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

bool matmulQ4_0(const std::vector<uint8_t>& weights,
                const std::vector<float>& input,
                size_t rows,
                size_t cols,
                size_t row_stride,
                uint32_t quant_version,
                const std::vector<float>* bias,
                std::vector<float>& output) const {
    return matmulQuant(weights, input, rows, cols, row_stride, quant_version,
                       q4MatmulPipeline, bias, output);
}

bool matmulQ4_1(const std::vector<uint8_t>& weights,
                const std::vector<float>& input,
                size_t rows,
                size_t cols,
                size_t row_stride,
                const std::vector<float>* bias,
                std::vector<float>& output) const {
    return matmulQuant(weights, input, rows, cols, row_stride, 0,
                       q4_1MatmulPipeline, bias, output);
}

bool matmulQ5_0(const std::vector<uint8_t>& weights,
                const std::vector<float>& input,
                size_t rows,
                size_t cols,
                size_t row_stride,
                const std::vector<float>* bias,
                std::vector<float>& output) const {
    return matmulQuant(weights, input, rows, cols, row_stride, 0,
                       q5_0MatmulPipeline, bias, output);
}

bool matmulQ5_1(const std::vector<uint8_t>& weights,
                const std::vector<float>& input,
                size_t rows,
                size_t cols,
                size_t row_stride,
                const std::vector<float>* bias,
                std::vector<float>& output) const {
    return matmulQuant(weights, input, rows, cols, row_stride, 0,
                       q5_1MatmulPipeline, bias, output);
}

bool matmulQ2_K(const std::vector<uint8_t>& weights,
                const std::vector<float>& input,
                size_t rows,
                size_t cols,
                size_t row_stride,
                const std::vector<float>* bias,
                std::vector<float>& output) const {
#if defined(__APPLE__)
    if (!available() || !q2KMatmulPipeline) return false;
    auto result = matmulQuant(weights, input, rows, cols, row_stride, 0, q2KMatmulPipeline, bias, output);
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

bool matmulQ3_K(const std::vector<uint8_t>& weights,
                const std::vector<float>& input,
                size_t rows,
                size_t cols,
                size_t row_stride,
                const std::vector<float>* bias,
                std::vector<float>& output) const {
#if defined(__APPLE__)
    if (!available() || !q3KMatmulPipeline) return false;
    return matmulQuant(weights, input, rows, cols, row_stride, 0, q3KMatmulPipeline, bias, output);
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

bool matmulQ4_K(const std::vector<uint8_t>& weights,
                const std::vector<float>& input,
                size_t rows,
                size_t cols,
                size_t row_stride,
                const std::vector<float>* bias,
                std::vector<float>& output) const {
#if defined(__APPLE__)
    if (!available() || !q4KMatmulPipeline) return false;
    return matmulQuant(weights, input, rows, cols, row_stride, 0, q4KMatmulPipeline, bias, output);
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

    bool matmulQ5_K(const std::vector<uint8_t>& weights,
                    const std::vector<float>& input,
                    size_t rows,
                    size_t cols,
                    size_t row_stride,
                    const std::vector<float>* bias,
                    std::vector<float>& output) const {
#if defined(__APPLE__)
        if (!available() || !q5KMatmulPipeline) return false;
        return matmulQuant(weights, input, rows, cols, row_stride, 0, q5KMatmulPipeline, bias, output);
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

    bool matmulQ6_K(const std::vector<uint8_t>& weights,
                    const std::vector<float>& input,
                    size_t rows,
                    size_t cols,
                    size_t row_stride,
                    const std::vector<float>* bias,
                    std::vector<float>& output) const {
#if defined(__APPLE__)
        if (!available() || !q6KMatmulPipeline) return false;
        return matmulQuant(weights, input, rows, cols, row_stride, 0, q6KMatmulPipeline, bias, output);
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

    bool matmulQ8_K(const std::vector<uint8_t>& weights,
                    const std::vector<float>& input,
                    size_t rows,
                    size_t cols,
                    size_t row_stride,
                    const std::vector<float>* bias,
                    std::vector<float>& output) const {
#if defined(__APPLE__)
        if (!available() || !q8KMatmulPipeline) return false;
        return matmulQuant(weights, input, rows, cols, row_stride, 0, q8KMatmulPipeline, bias, output);
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

    bool matmulQ8_0(const std::vector<uint8_t>& weights,
                    const std::vector<float>& input,
                    size_t rows,
                    size_t cols,
                    size_t row_stride,
                    const std::vector<float>* bias,
                    std::vector<float>& output) const {
#if defined(__APPLE__)
        if (!available() || !q8_0MatmulPipeline) return false;
        return matmulQuant(weights, input, rows, cols, row_stride, 0, q8_0MatmulPipeline, bias, output);
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

    bool matmulQ8_1(const std::vector<uint8_t>& weights,
                    const std::vector<float>& input,
                    size_t rows,
                    size_t cols,
                    size_t row_stride,
                    const std::vector<float>* bias,
                    std::vector<float>& output) const {
#if defined(__APPLE__)
        if (!available() || !q8_1MatmulPipeline) return false;
        return matmulQuant(weights, input, rows, cols, row_stride, 0, q8_1MatmulPipeline, bias, output);
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
                    std::vector<float>& output) const {
#if defined(__APPLE__)
        if (!available() || !addPipeline) return false;
        if (a.size() != b.size() || a.empty()) return false;
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

                id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                if (!encoder) return false;
                [encoder setComputePipelineState:addPipeline];
                [encoder setBuffer:aBuffer offset:0 atIndex:0];
                [encoder setBuffer:bBuffer offset:0 atIndex:1];
                [encoder setBuffer:outBuffer offset:0 atIndex:2];
                uint32_t length = static_cast<uint32_t>(a.size());
                [encoder setBytes:&length length:sizeof(uint32_t) atIndex:3];

                NSUInteger threadWidth = addPipeline.threadExecutionWidth;
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
                   size_t position,
                   size_t rotary_dim,
                   float rope_freq_base,
                   float rope_freq_scale,
                   std::vector<float>& kv_cache_k,
                   std::vector<float>& kv_cache_v,
                   MetalBufferHandle* cache_k_handle,
                   MetalBufferHandle* cache_v_handle,
                   std::vector<float>& output) const {
        (void)cache_k_handle;
        (void)cache_v_handle;
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
        if (kv_cache_k.size() < effective_kv_heads * kv_stride ||
            kv_cache_v.size() < effective_kv_heads * kv_stride) {
            return false;
        }

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
        bool sharedCacheValid = false;
#if defined(__APPLE__)
        id<MTLBuffer> sharedKCache = nil;
        id<MTLBuffer> sharedVCache = nil;
        if (cache_k_handle && cache_v_handle &&
            cache_k_handle->buffer && cache_v_handle->buffer &&
            cache_k_handle->bytes >= kv_cache_k.size() * sizeof(float) &&
            cache_v_handle->bytes >= kv_cache_v.size() * sizeof(float)) {
            sharedKCache = (__bridge id<MTLBuffer>)cache_k_handle->buffer;
            sharedVCache = (__bridge id<MTLBuffer>)cache_v_handle->buffer;
            sharedCacheValid = (sharedKCache != nil) && (sharedVCache != nil);
        }
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
        if (tokens_to_write == 0) return false;

        const std::vector<float>* k_ptr = &k;
        std::vector<float> rotated_k;
        if (use_rotary) {
            rotated_k = k;
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

        bool gpu_scatter = false;
#if defined(__APPLE__)
        if (sharedCacheValid && kvWritePipeline) {
            size_t write_elements = tokens_to_write * kv_span;
            id<MTLBuffer> kStage = [device newBufferWithBytes:k_ptr->data()
                                                       length:write_elements * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
            id<MTLBuffer> vStage = [device newBufferWithBytes:v.data()
                                                       length:write_elements * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
            if (kStage && vStage &&
                scatterTokens(kStage, sharedKCache, effective_kv_heads, tokens_to_write,
                              head_dim, context_length, base_position) &&
                scatterTokens(vStage, sharedVCache, effective_kv_heads, tokens_to_write,
                              head_dim, context_length, base_position)) {
                gpu_scatter = true;
            }
        }
#endif

        if (!gpu_scatter) {
            for (size_t kvh = 0; kvh < effective_kv_heads; ++kvh) {
                for (size_t t = 0; t < tokens_to_write; ++t) {
                    size_t pos = std::min(context_length - 1, base_position + t);
                    float* dest_k = kv_cache_k.data() + kvh * kv_stride + pos * head_dim;
                    float* dest_v = kv_cache_v.data() + kvh * kv_stride + pos * head_dim;
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
                        bool rotated = applyRotaryGPU(qBuffer,
                                                      1,
                                                      headRowBytes,
                                                      head_dim,
                                                      effective_rotary,
                                                      cos_ptr,
                                                      sin_ptr,
                                                      stride);
                        if (!rotated) {
                            float* vec = reinterpret_cast<float*>([qBuffer contents]);
                            applyRotaryEmbedding(vec,
                                                 q_single_cos,
                                                 q_single_sin,
                                                 head_dim,
                                                 effective_rotary);
                        }
                    }

                    bool useSharedMatrix = sharedCacheValid;
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
                            const float* src = kv_cache_k.data() + kv_index * kv_stride + t * head_dim;
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

                    if (!mask.empty()) {
                        float* logitsPtr = reinterpret_cast<float*>([logitsBuffer contents]);
                        for (size_t t = 0; t < tokens && t < mask.size(); ++t) {
                            logitsPtr[t] += mask[t];
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

                    bool useSharedV = sharedCacheValid;
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
                            const float* src = kv_cache_v.data() + kv_index * kv_stride + t * head_dim;
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
                return true;
            }
        }
#endif
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

bool MetalExecutor::runMatMul(const std::vector<float>& weights,
                              const std::vector<float>& input,
                              size_t rows,
                              size_t cols,
                              bool transpose_weight,
                              std::vector<float>& output,
                              const std::vector<float>* bias) const {
    if (!impl_) return false;
    return impl_->matmul(weights, input, rows, cols, transpose_weight, output, bias);
}

bool MetalExecutor::runMatMulQ4_0(const std::vector<uint8_t>& weights,
                                  const std::vector<float>& input,
                                  size_t rows,
                                  size_t cols,
                                  size_t row_stride,
                                  uint32_t quant_version,
                                  std::vector<float>& output,
                                  const std::vector<float>* bias) const {
    if (!impl_) return false;
    return impl_->matmulQ4_0(weights, input, rows, cols, row_stride, quant_version, bias, output);
}

bool MetalExecutor::runMatMulQ4_1(const std::vector<uint8_t>& weights,
                                  const std::vector<float>& input,
                                  size_t rows,
                                  size_t cols,
                                  size_t row_stride,
                                  std::vector<float>& output,
                                  const std::vector<float>* bias) const {
    if (!impl_) return false;
    return impl_->matmulQ4_1(weights, input, rows, cols, row_stride, bias, output);
}

bool MetalExecutor::runMatMulQ5_0(const std::vector<uint8_t>& weights,
                                  const std::vector<float>& input,
                                  size_t rows,
                                  size_t cols,
                                  size_t row_stride,
                                  std::vector<float>& output,
                                  const std::vector<float>* bias) const {
    if (!impl_) return false;
    return impl_->matmulQ5_0(weights, input, rows, cols, row_stride, bias, output);
}

bool MetalExecutor::runMatMulQ5_1(const std::vector<uint8_t>& weights,
                                  const std::vector<float>& input,
                                  size_t rows,
                                  size_t cols,
                                  size_t row_stride,
                                  std::vector<float>& output,
                                  const std::vector<float>* bias) const {
    if (!impl_) return false;
    return impl_->matmulQ5_1(weights, input, rows, cols, row_stride, bias, output);
}

bool MetalExecutor::runMatMulQ2K(const std::vector<uint8_t>& weights,
                                 const std::vector<float>& input,
                                 size_t rows,
                                 size_t cols,
                                 size_t row_stride,
                                 std::vector<float>& output,
                                 const std::vector<float>* bias) const {
    if (!impl_) return false;
    return impl_->matmulQ2_K(weights, input, rows, cols, row_stride, bias, output);
}

bool MetalExecutor::runMatMulQ3K(const std::vector<uint8_t>& weights,
                                 const std::vector<float>& input,
                                 size_t rows,
                                 size_t cols,
                                 size_t row_stride,
                                 std::vector<float>& output,
                                 const std::vector<float>* bias) const {
    if (!impl_) return false;
    return impl_->matmulQ3_K(weights, input, rows, cols, row_stride, bias, output);
}

bool MetalExecutor::runMatMulQ4K(const std::vector<uint8_t>& weights,
                                 const std::vector<float>& input,
                                 size_t rows,
                                 size_t cols,
                                 size_t row_stride,
                                 std::vector<float>& output,
                                 const std::vector<float>* bias) const {
    if (!impl_) return false;
    return impl_->matmulQ4_K(weights, input, rows, cols, row_stride, bias, output);
}

bool MetalExecutor::runMatMulQ5K(const std::vector<uint8_t>& weights,
                                 const std::vector<float>& input,
                                 size_t rows,
                                 size_t cols,
                                 size_t row_stride,
                                 std::vector<float>& output,
                                 const std::vector<float>* bias) const {
    if (!impl_) return false;
    return impl_->matmulQ5_K(weights, input, rows, cols, row_stride, bias, output);
}

bool MetalExecutor::runMatMulQ6K(const std::vector<uint8_t>& weights,
                                 const std::vector<float>& input,
                                 size_t rows,
                                 size_t cols,
                                 size_t row_stride,
                                 std::vector<float>& output,
                                 const std::vector<float>* bias) const {
    if (!impl_) return false;
    return impl_->matmulQ6_K(weights, input, rows, cols, row_stride, bias, output);
}

bool MetalExecutor::runMatMulQ8K(const std::vector<uint8_t>& weights,
                                 const std::vector<float>& input,
                                 size_t rows,
                                 size_t cols,
                                 size_t row_stride,
                                 std::vector<float>& output,
                                 const std::vector<float>* bias) const {
    if (!impl_) return false;
    return impl_->matmulQ8_K(weights, input, rows, cols, row_stride, bias, output);
}

bool MetalExecutor::runMatMulQ8_0(const std::vector<uint8_t>& weights,
                                  const std::vector<float>& input,
                                  size_t rows,
                                  size_t cols,
                                  size_t row_stride,
                                  std::vector<float>& output,
                                  const std::vector<float>* bias) const {
    if (!impl_) return false;
    return impl_->matmulQ8_0(weights, input, rows, cols, row_stride, bias, output);
}

bool MetalExecutor::runMatMulQ8_1(const std::vector<uint8_t>& weights,
                                  const std::vector<float>& input,
                                  size_t rows,
                                  size_t cols,
                                  size_t row_stride,
                                  std::vector<float>& output,
                                  const std::vector<float>* bias) const {
    if (!impl_) return false;
    return impl_->matmulQ8_1(weights, input, rows, cols, row_stride, bias, output);
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
    if (!impl_->kvWritePipeline) return false;
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

bool MetalExecutor::runFeedForward(const std::vector<float>& gate,
                                   const std::vector<float>& up,
                                   std::vector<float>& output) const {
    if (!impl_) return false;
    return impl_->feedForward(gate, up, output);
}

bool MetalExecutor::runAdd(const std::vector<float>& a,
                           const std::vector<float>& b,
                           std::vector<float>& output) const {
    if (!impl_) return false;
    return impl_->addVectors(a, b, output);
}

bool MetalExecutor::runRmsNorm(const std::vector<float>& input,
                               const std::vector<float>& weight,
                               float epsilon,
                               std::vector<float>& output) const {
    if (!impl_) return false;
    return impl_->rmsNorm(input, weight, epsilon, output);
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
                                 size_t position,
                                 size_t rotary_dim,
                                 float rope_freq_base,
                                 float rope_freq_scale,
                                 std::vector<float>& kv_cache_k,
                                 std::vector<float>& kv_cache_v,
                                 MetalBufferHandle* cache_k_handle,
                                 MetalBufferHandle* cache_v_handle,
                                 std::vector<float>& output) const {
    if (!impl_) return false;
    return impl_->attention(q, k, v, num_heads, kv_heads, head_dim,
                            context_length, mask, position,
                            rotary_dim, rope_freq_base, rope_freq_scale,
                            kv_cache_k, kv_cache_v,
                            cache_k_handle, cache_v_handle,
                            output);
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

bool MetalExecutor::hasSoftmaxKernel() const {
    return impl_ && impl_->hasSoftmaxKernel();
}

} // namespace runtime
} // namespace mlc
