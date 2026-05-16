// Phase N1 — minimal Metal hazard tests.
//
// Goal: determine why M1's compound encoder pattern (multi-encoder on one
// CB + memoryBarrier inside each encoder) gave stale reads, and whether
// a single-encoder-per-pass rewrite is safe.
//
// Each test runs a known-correct computation and verifies the output:
//   pass 0: write [0,1,2,...,N-1] into a buffer
//   step 1: multiply by 2 → [0,2,4,...]   (writes the buffer)
//   step 2: multiply by 3 → [0,6,12,...]  (reads + writes the buffer)
//
// We expect dst[i] == 6*i for every i.
//
// Five test variants, each toggling one variable:
//   A. one encoder, two dispatches, NO barrier between them
//   B. one encoder, two dispatches, memoryBarrierWithScope:Buffers between
//   C. two encoders on one CB, no explicit sync (M1's compound pattern)
//   D. two encoders on one CB, fence updated/waited between them
//   E. two encoders on one CB, useResource called explicitly
//
// Cached buffer is recreated with MTLResourceHazardTrackingModeTracked.

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cstring>

static const char* kSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void scale(
    device float* buf [[buffer(0)]],
    constant float& factor [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    buf[gid] = buf[gid] * factor;
}
)";

static int gFailures = 0;

static void check(const char* label, const std::vector<float>& got,
                  float expected_factor, size_t n) {
    int bad = 0;
    float maxerr = 0;
    for (size_t i = 0; i < n; ++i) {
        float want = static_cast<float>(i) * expected_factor;
        float err = std::abs(got[i] - want);
        if (err > 1e-3f) ++bad;
        if (err > maxerr) maxerr = err;
    }
    if (bad == 0) {
        printf("  PASS  %s: maxerr=%g\n", label, maxerr);
    } else {
        ++gFailures;
        printf("  FAIL  %s: bad=%d/%zu maxerr=%g  got[0..5]=%g,%g,%g,%g,%g,%g\n",
               label, bad, n, maxerr,
               got[0], got[1], got[2], got[3], got[4], got[5]);
    }
}

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { fprintf(stderr, "no Metal device\n"); return 1; }
        id<MTLCommandQueue> queue = [device newCommandQueue];

        NSError* err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:[NSString stringWithUTF8String:kSource]
                                                  options:nil
                                                    error:&err];
        if (!lib) { fprintf(stderr, "compile err: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"scale"];
        id<MTLComputePipelineState> ps =
            [device newComputePipelineStateWithFunction:fn error:&err];
        if (!ps) { fprintf(stderr, "pipeline err: %s\n", err.localizedDescription.UTF8String); return 1; }

        const size_t N = 4096;
        const size_t bytes = N * sizeof(float);

        auto fillSrc = [&](id<MTLBuffer> buf) {
            float* p = (float*)buf.contents;
            for (size_t i = 0; i < N; ++i) p[i] = static_cast<float>(i);
        };
        auto dispatchScale = [&](id<MTLComputeCommandEncoder> enc, id<MTLBuffer> buf, float factor) {
            [enc setComputePipelineState:ps];
            [enc setBuffer:buf offset:0 atIndex:0];
            [enc setBytes:&factor length:sizeof(float) atIndex:1];
            NSUInteger tw = ps.threadExecutionWidth * 4;
            if (tw == 0) tw = 64;
            NSUInteger tg = (N + tw - 1) / tw;
            [enc dispatchThreadgroups:MTLSizeMake(tg, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        };
        auto readback = [&](id<MTLBuffer> buf, std::vector<float>& out) {
            out.resize(N);
            std::memcpy(out.data(), buf.contents, bytes);
        };

        MTLResourceOptions tracked = MTLResourceStorageModeShared |
                                      MTLResourceHazardTrackingModeTracked;

        printf("Metal hazard test on device: %s\n", device.name.UTF8String);
        printf("Each test: scale buffer by 2 then by 3; expect dst[i] = 6*i.\n\n");

        // -------- A. one encoder, two dispatches, NO barrier --------
        {
            id<MTLBuffer> buf = [device newBufferWithLength:bytes options:tracked];
            fillSrc(buf);
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            dispatchScale(enc, buf, 2.0f);
            dispatchScale(enc, buf, 3.0f);
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            std::vector<float> out; readback(buf, out);
            check("A: one encoder, NO barrier", out, 6.0f, N);
        }

        // -------- B. one encoder, two dispatches, memoryBarrier ----
        {
            id<MTLBuffer> buf = [device newBufferWithLength:bytes options:tracked];
            fillSrc(buf);
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            dispatchScale(enc, buf, 2.0f);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            dispatchScale(enc, buf, 3.0f);
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            std::vector<float> out; readback(buf, out);
            check("B: one encoder, memoryBarrier", out, 6.0f, N);
        }

        // -------- C. two encoders on one CB, no explicit sync ------
        {
            id<MTLBuffer> buf = [device newBufferWithLength:bytes options:tracked];
            fillSrc(buf);
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> e1 = [cb computeCommandEncoder];
            dispatchScale(e1, buf, 2.0f);
            [e1 endEncoding];
            id<MTLComputeCommandEncoder> e2 = [cb computeCommandEncoder];
            dispatchScale(e2, buf, 3.0f);
            [e2 endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            std::vector<float> out; readback(buf, out);
            check("C: two encoders, no sync (M1 pattern)", out, 6.0f, N);
        }

        // -------- D. two encoders, MTLFence between --------------
        if (@available(macOS 10.13, *)) {
            id<MTLFence> fence = [device newFence];
            id<MTLBuffer> buf = [device newBufferWithLength:bytes options:tracked];
            fillSrc(buf);
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> e1 = [cb computeCommandEncoder];
            dispatchScale(e1, buf, 2.0f);
            [e1 updateFence:fence];
            [e1 endEncoding];
            id<MTLComputeCommandEncoder> e2 = [cb computeCommandEncoder];
            [e2 waitForFence:fence];
            dispatchScale(e2, buf, 3.0f);
            [e2 endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            std::vector<float> out; readback(buf, out);
            check("D: two encoders, MTLFence", out, 6.0f, N);
        }

        // -------- E. two encoders, useResource explicit ----------
        {
            id<MTLBuffer> buf = [device newBufferWithLength:bytes options:tracked];
            fillSrc(buf);
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> e1 = [cb computeCommandEncoder];
            [e1 useResource:buf usage:MTLResourceUsageWrite];
            dispatchScale(e1, buf, 2.0f);
            [e1 endEncoding];
            id<MTLComputeCommandEncoder> e2 = [cb computeCommandEncoder];
            [e2 useResource:buf usage:MTLResourceUsageRead | MTLResourceUsageWrite];
            dispatchScale(e2, buf, 3.0f);
            [e2 endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            std::vector<float> out; readback(buf, out);
            check("E: two encoders, useResource", out, 6.0f, N);
        }

        // -------- F. two encoders, separate src/dst buffers ------
        // This mirrors compound's actual case better: dispatch 1 writes
        // buf1, dispatch 2 reads buf1 and writes buf2. If the issue is
        // cross-encoder RAW on the SAME buffer, F should pass when C
        // fails (different buffers shouldn't alias).
        {
            id<MTLBuffer> buf1 = [device newBufferWithLength:bytes options:tracked];
            id<MTLBuffer> buf2 = [device newBufferWithLength:bytes options:tracked];
            fillSrc(buf1);
            // Initialize buf2 with NaN-ish value to detect stale writes.
            float* p2 = (float*)buf2.contents;
            for (size_t i = 0; i < N; ++i) p2[i] = -999.0f;

            // Need a kernel that copies buf1*factor → buf2 for this case.
            const char* copySrc = R"(
                #include <metal_stdlib>
                using namespace metal;
                kernel void copy_scale(
                    device const float* src [[buffer(0)]],
                    device float* dst [[buffer(1)]],
                    constant float& factor [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
                    dst[gid] = src[gid] * factor;
                }
            )";
            NSError* e = nil;
            id<MTLLibrary> lib2 = [device newLibraryWithSource:[NSString stringWithUTF8String:copySrc]
                                                       options:nil error:&e];
            id<MTLComputePipelineState> ps2 =
                [device newComputePipelineStateWithFunction:[lib2 newFunctionWithName:@"copy_scale"]
                                                       error:&e];

            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> e1 = [cb computeCommandEncoder];
            // dispatch 1: scale buf1 by 2 in place.
            dispatchScale(e1, buf1, 2.0f);
            [e1 endEncoding];
            id<MTLComputeCommandEncoder> e2 = [cb computeCommandEncoder];
            // dispatch 2: dst[i] = buf1[i] * 3 → expect 6*i in buf2.
            [e2 setComputePipelineState:ps2];
            [e2 setBuffer:buf1 offset:0 atIndex:0];
            [e2 setBuffer:buf2 offset:0 atIndex:1];
            float factor = 3.0f;
            [e2 setBytes:&factor length:sizeof(float) atIndex:2];
            NSUInteger tw = ps2.threadExecutionWidth * 4;
            if (tw == 0) tw = 64;
            NSUInteger tg = (N + tw - 1) / tw;
            [e2 dispatchThreadgroups:MTLSizeMake(tg, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
            [e2 endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            std::vector<float> out; readback(buf2, out);
            check("F: two encoders, distinct src/dst", out, 6.0f, N);
        }

        // -------- G. exact M1 chain: split + rope + cast --------
        // Reproduces the actual chain that broke in M1:
        //   buf_qkv [N, q+k+v]  →  strided_copy_f32  →  buf_q [N, q]
        //                                              →  buf_k [N, k]
        //                          batched_rope(buf_q, cos, sin)  in-place
        //                          batched_rope(buf_k, cos, sin)  in-place
        //                          cast_f32_to_f16(buf_k → buf_k16)
        // Then check buf_q, buf_k, buf_k16 against CPU expected.
        //
        // Two variants:
        //   G1. all 7 dispatches in ONE compute encoder, barriers between
        //       split→rope and rope→cast.
        //   G2. M1's compound pattern: one encoder per "compound" with
        //       intra-encoder barriers; verify post-encoder values WITHOUT
        //       extra flush.
        {
            const char* src = R"(
                #include <metal_stdlib>
                using namespace metal;

                struct StridedParams { uint dim; uint src_stride; uint offset; uint batch; };
                kernel void strided_copy_f32(
                    device const float* src [[buffer(0)]],
                    device       float* dst [[buffer(1)]],
                    constant StridedParams& p [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
                    uint total = p.batch * p.dim;
                    if (gid >= total) return;
                    uint r = gid / p.dim;
                    uint i = gid - r * p.dim;
                    dst[gid] = src[r * p.src_stride + p.offset + i];
                }

                struct RopeParams { uint batch; uint n_heads; uint head_dim; uint rotary_dim; };
                kernel void batched_rope(
                    device       float* data [[buffer(0)]],
                    device const float* cos_t [[buffer(1)]],
                    device const float* sin_t [[buffer(2)]],
                    constant RopeParams& p [[buffer(3)]],
                    uint gid [[thread_position_in_grid]]) {
                    uint pairs = p.rotary_dim / 2u;
                    uint total = p.batch * p.n_heads * pairs;
                    if (gid >= total) return;
                    uint pair_idx = gid % pairs;
                    uint tmp = gid / pairs;
                    uint head_idx = tmp % p.n_heads;
                    uint req_idx = tmp / p.n_heads;
                    uint head_off = (req_idx * p.n_heads + head_idx) * p.head_dim;
                    uint c_off = req_idx * pairs + pair_idx;
                    float c = cos_t[c_off], s = sin_t[c_off];
                    uint i0 = head_off + 2u * pair_idx;
                    uint i1 = i0 + 1u;
                    float x0 = data[i0], x1 = data[i1];
                    data[i0] = x0 * c - x1 * s;
                    data[i1] = x0 * s + x1 * c;
                }

                kernel void cast_f32_to_f16(
                    device const float* src [[buffer(0)]],
                    device       half*  dst [[buffer(1)]],
                    constant uint& length [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
                    if (gid >= length) return;
                    dst[gid] = half(src[gid]);
                }
            )";
            NSError* e = nil;
            id<MTLLibrary> lib2 = [device newLibraryWithSource:[NSString stringWithUTF8String:src]
                                                       options:nil error:&e];
            if (!lib2) { fprintf(stderr, "G compile: %s\n", e.localizedDescription.UTF8String); return 1; }
            id<MTLComputePipelineState> ps_copy =
                [device newComputePipelineStateWithFunction:[lib2 newFunctionWithName:@"strided_copy_f32"] error:&e];
            id<MTLComputePipelineState> ps_rope =
                [device newComputePipelineStateWithFunction:[lib2 newFunctionWithName:@"batched_rope"] error:&e];
            id<MTLComputePipelineState> ps_cast =
                [device newComputePipelineStateWithFunction:[lib2 newFunctionWithName:@"cast_f32_to_f16"] error:&e];

            // TinyLlama shapes: hidden=2048, heads=32, kv_heads=4, head_dim=64.
            // qkv layout: [batch=1, q_rows=2048 + k_rows=256 + v_rows=256] = 2560.
            const uint32_t batch = 1, num_heads = 32, kv_heads = 4, head_dim = 64;
            const uint32_t rotary_dim = 64;
            const uint32_t q_rows = num_heads * head_dim;       // 2048
            const uint32_t k_rows = kv_heads * head_dim;        // 256
            const uint32_t v_rows = kv_heads * head_dim;        // 256
            const uint32_t qkv_rows = q_rows + k_rows + v_rows; // 2560

            // Buffers (all hazard-tracked shared).
            id<MTLBuffer> buf_qkv  = [device newBufferWithLength:batch * qkv_rows * sizeof(float) options:tracked];
            id<MTLBuffer> buf_q    = [device newBufferWithLength:batch * q_rows  * sizeof(float) options:tracked];
            id<MTLBuffer> buf_k    = [device newBufferWithLength:batch * k_rows  * sizeof(float) options:tracked];
            id<MTLBuffer> buf_k16  = [device newBufferWithLength:batch * k_rows  * sizeof(uint16_t) options:tracked];
            uint32_t pairs = rotary_dim / 2;
            id<MTLBuffer> buf_cos  = [device newBufferWithLength:batch * pairs * sizeof(float) options:tracked];
            id<MTLBuffer> buf_sin  = [device newBufferWithLength:batch * pairs * sizeof(float) options:tracked];

            // Fill buf_qkv with values: qkv[r,i] = (r*100 + i) * 0.001  (small for fp16 safety).
            // Fill cos/sin with sample rope coefficients for position p=5.
            float* qkv = (float*)buf_qkv.contents;
            for (uint32_t r = 0; r < batch; ++r) {
                for (uint32_t i = 0; i < qkv_rows; ++i) {
                    qkv[r * qkv_rows + i] = static_cast<float>((r * 100 + i)) * 0.001f;
                }
            }
            float* cos_p = (float*)buf_cos.contents;
            float* sin_p = (float*)buf_sin.contents;
            for (uint32_t r = 0; r < batch; ++r) {
                float pos = 5.0f + r;
                for (uint32_t i = 0; i < pairs; ++i) {
                    float exp = (2.0f * i) / rotary_dim;
                    float invf = std::pow(10000.0f, -exp);
                    cos_p[r * pairs + i] = std::cos(pos * invf);
                    sin_p[r * pairs + i] = std::sin(pos * invf);
                }
            }

            auto cpu_expected = [&](std::vector<float>& q_exp,
                                    std::vector<float>& k_exp,
                                    std::vector<uint16_t>& k16_exp) {
                q_exp.assign(batch * q_rows, 0);
                k_exp.assign(batch * k_rows, 0);
                for (uint32_t r = 0; r < batch; ++r) {
                    const float* src = qkv + r * qkv_rows;
                    std::copy(src, src + q_rows, q_exp.begin() + r * q_rows);
                    std::copy(src + q_rows, src + q_rows + k_rows, k_exp.begin() + r * k_rows);
                }
                // Apply rope.
                for (uint32_t r = 0; r < batch; ++r) {
                    for (uint32_t h = 0; h < num_heads; ++h) {
                        for (uint32_t i = 0; i < pairs; ++i) {
                            float c = cos_p[r * pairs + i];
                            float s = sin_p[r * pairs + i];
                            float* v = q_exp.data() + (r * num_heads + h) * head_dim + 2 * i;
                            float x0 = v[0], x1 = v[1];
                            v[0] = x0 * c - x1 * s;
                            v[1] = x0 * s + x1 * c;
                        }
                    }
                    for (uint32_t h = 0; h < kv_heads; ++h) {
                        for (uint32_t i = 0; i < pairs; ++i) {
                            float c = cos_p[r * pairs + i];
                            float s = sin_p[r * pairs + i];
                            float* v = k_exp.data() + (r * kv_heads + h) * head_dim + 2 * i;
                            float x0 = v[0], x1 = v[1];
                            v[0] = x0 * c - x1 * s;
                            v[1] = x0 * s + x1 * c;
                        }
                    }
                }
                // Cast K to fp16.
                k16_exp.assign(batch * k_rows, 0);
                for (size_t i = 0; i < k_exp.size(); ++i) {
                    __fp16 h = static_cast<__fp16>(k_exp[i]);
                    std::memcpy(&k16_exp[i], &h, sizeof(uint16_t));
                }
            };

            auto run_compound = [&](id<MTLCommandBuffer> cb, bool barriers, bool single_encoder) {
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

                auto copyDispatch = [&](id<MTLBuffer> dst, uint32_t dim, uint32_t offset) {
                    [enc setComputePipelineState:ps_copy];
                    [enc setBuffer:buf_qkv offset:0 atIndex:0];
                    [enc setBuffer:dst     offset:0 atIndex:1];
                    struct { uint32_t dim; uint32_t src_stride; uint32_t offset; uint32_t batch; } p;
                    p.dim = dim; p.src_stride = qkv_rows; p.offset = offset; p.batch = batch;
                    [enc setBytes:&p length:sizeof(p) atIndex:2];
                    uint32_t total = batch * dim;
                    NSUInteger tw = ps_copy.threadExecutionWidth * 4; if (tw == 0) tw = 64;
                    NSUInteger tg = (total + tw - 1) / tw;
                    [enc dispatchThreadgroups:MTLSizeMake(tg, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
                };
                copyDispatch(buf_q, q_rows, 0);
                copyDispatch(buf_k, k_rows, q_rows);
                if (barriers) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                if (!single_encoder) {
                    [enc endEncoding];
                    enc = [cb computeCommandEncoder];
                }
                auto ropeDispatch = [&](id<MTLBuffer> data, uint32_t nh) {
                    [enc setComputePipelineState:ps_rope];
                    [enc setBuffer:data    offset:0 atIndex:0];
                    [enc setBuffer:buf_cos offset:0 atIndex:1];
                    [enc setBuffer:buf_sin offset:0 atIndex:2];
                    struct { uint32_t batch; uint32_t n_heads; uint32_t head_dim; uint32_t rotary_dim; } p;
                    p.batch = batch; p.n_heads = nh; p.head_dim = head_dim; p.rotary_dim = rotary_dim;
                    [enc setBytes:&p length:sizeof(p) atIndex:3];
                    uint32_t total = batch * nh * pairs;
                    NSUInteger tw = ps_rope.threadExecutionWidth; if (tw == 0) tw = 32;
                    NSUInteger tg = (total + tw - 1) / tw;
                    [enc dispatchThreadgroups:MTLSizeMake(tg, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
                };
                ropeDispatch(buf_q, num_heads);
                ropeDispatch(buf_k, kv_heads);
                if (barriers) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                if (!single_encoder) {
                    [enc endEncoding];
                    enc = [cb computeCommandEncoder];
                }
                [enc setComputePipelineState:ps_cast];
                [enc setBuffer:buf_k    offset:0 atIndex:0];
                [enc setBuffer:buf_k16  offset:0 atIndex:1];
                uint32_t len = batch * k_rows;
                [enc setBytes:&len length:sizeof(len) atIndex:2];
                NSUInteger tw = ps_cast.threadExecutionWidth * 4; if (tw == 0) tw = 64;
                NSUInteger tg = (len + tw - 1) / tw;
                [enc dispatchThreadgroups:MTLSizeMake(tg, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
                [enc endEncoding];
            };

            auto verify_g = [&](const char* label) {
                std::vector<float> q_exp, k_exp; std::vector<uint16_t> k16_exp;
                cpu_expected(q_exp, k_exp, k16_exp);
                std::vector<float> q_got(batch * q_rows), k_got(batch * k_rows);
                std::vector<uint16_t> k16_got(batch * k_rows);
                std::memcpy(q_got.data(),   buf_q.contents,  q_got.size() * sizeof(float));
                std::memcpy(k_got.data(),   buf_k.contents,  k_got.size() * sizeof(float));
                std::memcpy(k16_got.data(), buf_k16.contents, k16_got.size() * sizeof(uint16_t));
                double qe = 0, ke = 0; int k16_bad = 0;
                for (size_t i = 0; i < q_got.size(); ++i) qe += std::abs(q_got[i] - q_exp[i]);
                for (size_t i = 0; i < k_got.size(); ++i) ke += std::abs(k_got[i] - k_exp[i]);
                for (size_t i = 0; i < k16_got.size(); ++i)
                    if (k16_got[i] != k16_exp[i]) ++k16_bad;
                bool ok = (qe < 1e-3) && (ke < 1e-3) && (k16_bad == 0);
                if (ok) printf("  PASS  %s: q_err=%g k_err=%g k16_bad=%d\n", label, qe, ke, k16_bad);
                else {
                    ++gFailures;
                    printf("  FAIL  %s: q_err=%g k_err=%g k16_bad=%d\n", label, qe, ke, k16_bad);
                }
            };

            // G1: all 7 dispatches in ONE encoder, with barriers.
            {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                run_compound(cb, /*barriers=*/true, /*single_encoder=*/true);
                [cb commit]; [cb waitUntilCompleted];
                verify_g("G1: M1 chain in ONE encoder with barriers");
            }
            // G2: three encoders (M1's "compound" pattern), barriers in each.
            {
                // Reset buf_q / buf_k by re-running the pipeline; OK because
                // the buffers get overwritten by strided_copy.
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                run_compound(cb, /*barriers=*/true, /*single_encoder=*/false);
                [cb commit]; [cb waitUntilCompleted];
                verify_g("G2: M1 chain in 3 encoders (M1 actual pattern)");
            }
            // G3: three encoders, NO barriers.
            {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                run_compound(cb, /*barriers=*/false, /*single_encoder=*/false);
                [cb commit]; [cb waitUntilCompleted];
                verify_g("G3: M1 chain in 3 encoders, no barriers");
            }
            // G4: ONE encoder, NO barriers.
            {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                run_compound(cb, /*barriers=*/false, /*single_encoder=*/true);
                [cb commit]; [cb waitUntilCompleted];
                verify_g("G4: M1 chain in ONE encoder, no barriers");
            }
        }

        printf("\n");
        if (gFailures == 0) printf("ALL TESTS PASSED\n");
        else printf("FAILED tests: %d\n", gFailures);
        return gFailures > 0 ? 1 : 0;
    }
}
