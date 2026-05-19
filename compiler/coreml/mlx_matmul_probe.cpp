// Companion to `coreml_matmul_probe`: times MLX-on-GPU matmul for the
// same (M, K, N) shape so we can compare ANE vs GPU latency for the
// single-request decode path.
//
// Usage: mlx_matmul_probe <M> <K> <N> [iters]

#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace mx = mlx::core;

int main(int argc, char **argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <M> <K> <N> [iters]\n", argv[0]);
        return 2;
    }
    int M = std::atoi(argv[1]);
    int K = std::atoi(argv[2]);
    int N = std::atoi(argv[3]);
    int iters = (argc >= 5) ? std::atoi(argv[4]) : 100;

    auto make_fp16_arr = [&](const std::vector<int> &shape) {
        size_t n = 1;
        for (int d : shape) n *= d;
        std::vector<uint16_t> data(n, 0x3C00);  // fp16 1.0
        uint16_t *buf = static_cast<uint16_t *>(std::malloc(n * sizeof(uint16_t)));
        std::memcpy(buf, data.data(), n * sizeof(uint16_t));
        mx::Shape s(shape.begin(), shape.end());
        return mx::array(static_cast<void *>(buf), std::move(s), mx::float16,
                         [](void *p) { std::free(p); });
    };
    auto x = make_fp16_arr({M, K});
    auto w = make_fp16_arr({K, N});

    // Warm-up.
    auto warm = mx::matmul(x, w);
    mx::eval(warm);

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        auto y = mx::matmul(x, w);
        mx::eval(y);
    }
    auto t1 = std::chrono::steady_clock::now();
    double avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    std::printf("[mlx-gpu] M=%d K=%d N=%d iters=%d avg=%.1f us/call\n",
                M, K, N, iters, avg_us);
    return 0;
}
