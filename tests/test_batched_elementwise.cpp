#include <gtest/gtest.h>

#include "runtime/metal_runtime.hpp"

#include <cmath>
#include <random>
#include <vector>

namespace {
std::vector<float> randomVector(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

// CPU reference RMSNorm (matches the kernel formula: scale = rsqrt(mean+eps),
// out = in * scale * gain).
std::vector<float> rmsNormCpuRef(const std::vector<float>& input,
                                 const std::vector<float>& weight,
                                 float epsilon) {
    double sumsq = 0.0;
    for (float v : input) sumsq += static_cast<double>(v) * v;
    float mean = static_cast<float>(sumsq / input.size());
    float scale = 1.0f / std::sqrt(mean + epsilon);
    std::vector<float> out(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        out[i] = input[i] * scale * weight[i];
    }
    return out;
}
} // namespace

TEST(BatchedElementwiseB2a, RmsNormBatchedMatchesPerRow) {
    auto& exec = mlc::runtime::MetalExecutor::Instance();
    if (!exec.isAvailable()) {
        GTEST_SKIP() << "Metal unavailable";
    }

    constexpr size_t batch = 4;
    constexpr size_t length = 2048;
    constexpr float epsilon = 1e-5f;

    auto weight = randomVector(length, /*seed=*/123);

    // Build the batched input as N independent random rows.
    std::vector<float> batched_input(batch * length);
    std::vector<std::vector<float>> per_row_inputs;
    for (size_t n = 0; n < batch; ++n) {
        auto row = randomVector(length, /*seed=*/1000 + static_cast<uint32_t>(n));
        std::copy(row.begin(), row.end(), batched_input.begin() + n * length);
        per_row_inputs.push_back(std::move(row));
    }

    // Run the batched kernel.
    std::vector<float> batched_output;
    ASSERT_TRUE(exec.runRmsNormBatched(batched_input, weight, epsilon,
                                       batch, length, batched_output));
    ASSERT_EQ(batched_output.size(), batch * length);

    // Compare each row to the existing single-row runRmsNorm AND to a CPU
    // reference (the latter catches kernel correctness; the former catches
    // any per-row vs batched divergence).
    for (size_t n = 0; n < batch; ++n) {
        std::vector<float> single_out;
        ASSERT_TRUE(exec.runRmsNorm(per_row_inputs[n], weight, epsilon, single_out));
        auto cpu_ref = rmsNormCpuRef(per_row_inputs[n], weight, epsilon);

        for (size_t i = 0; i < length; ++i) {
            float batched_val = batched_output[n * length + i];
            // Batched kernel must match the single-row kernel bit-for-bit
            // (same arithmetic, same reduction order).
            EXPECT_EQ(batched_val, single_out[i])
                << "row=" << n << " i=" << i;
            // And both must match the CPU reference within fp32 tolerance.
            EXPECT_NEAR(batched_val, cpu_ref[i], 1e-4f) << "row=" << n << " i=" << i;
        }
    }
}

// Add and SiLU+mul existing kernels are 1-D parallel — the same dispatch
// over batch*length elements produces the correct batched result. Explicitly
// verify this by feeding a [batch*length] buffer and checking each row
// independently matches the per-row call.
TEST(BatchedElementwiseB2a, AddOverBatchTreatsAsFlat) {
    auto& exec = mlc::runtime::MetalExecutor::Instance();
    if (!exec.isAvailable()) {
        GTEST_SKIP() << "Metal unavailable";
    }

    constexpr size_t batch = 3;
    constexpr size_t length = 512;
    auto a_flat = randomVector(batch * length, 7);
    auto b_flat = randomVector(batch * length, 11);

    std::vector<float> batched_out;
    ASSERT_TRUE(exec.runAdd(a_flat, b_flat, batched_out, /*bias=*/nullptr));
    ASSERT_EQ(batched_out.size(), batch * length);

    for (size_t n = 0; n < batch; ++n) {
        std::vector<float> a_row(a_flat.begin() + n * length,
                                 a_flat.begin() + (n + 1) * length);
        std::vector<float> b_row(b_flat.begin() + n * length,
                                 b_flat.begin() + (n + 1) * length);
        std::vector<float> single_out;
        ASSERT_TRUE(exec.runAdd(a_row, b_row, single_out, /*bias=*/nullptr));
        for (size_t i = 0; i < length; ++i) {
            EXPECT_EQ(batched_out[n * length + i], single_out[i])
                << "row=" << n << " i=" << i;
        }
    }
}

TEST(BatchedElementwiseB2a, FeedForwardOverBatchTreatsAsFlat) {
    auto& exec = mlc::runtime::MetalExecutor::Instance();
    if (!exec.isAvailable()) {
        GTEST_SKIP() << "Metal unavailable";
    }
    if (!exec.hasFeedForwardKernel()) {
        GTEST_SKIP() << "FeedForward kernel unavailable";
    }

    constexpr size_t batch = 2;
    constexpr size_t length = 1024;
    auto gate = randomVector(batch * length, 31);
    auto up   = randomVector(batch * length, 37);

    std::vector<float> batched_out;
    ASSERT_TRUE(exec.runFeedForward(gate, up, batched_out));
    ASSERT_EQ(batched_out.size(), batch * length);

    for (size_t n = 0; n < batch; ++n) {
        std::vector<float> g_row(gate.begin() + n * length,
                                 gate.begin() + (n + 1) * length);
        std::vector<float> u_row(up.begin() + n * length,
                                 up.begin() + (n + 1) * length);
        std::vector<float> single_out;
        ASSERT_TRUE(exec.runFeedForward(g_row, u_row, single_out));
        for (size_t i = 0; i < length; ++i) {
            EXPECT_EQ(batched_out[n * length + i], single_out[i])
                << "row=" << n << " i=" << i;
        }
    }
}
