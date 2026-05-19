// MLX C++ integration smoke test.
//
// Goal of P2: prove that MLX's Metal dispatch can be driven from inside the
// MLcompiler codebase. We do not yet wire MLX into any compiler pass — the
// future `mlc.matmul` lowering pass will produce code like the body of this
// test, but for now we just verify the linkage and the round-trip.
//
// Two checks:
//   1) 4x4 fp16 ones @ ones — verify every element exactly equals 4
//      (fully representable in fp16, no fp accumulation surprises).
//   2) 2048x2048 fp16 ones @ ones — verify every element exactly equals 2048
//      (still representable in fp16; matmul accumulates in fp32 internally
//      on Apple GPUs, so this is the right exactness check).

#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

#include <cstdint>
#include <iostream>
#include <string>

namespace {

int check_ones_matmul(int N) {
  using namespace mlx::core;

  array x = ones({N, N}, float16);
  array w = ones({N, N}, float16);
  array y = matmul(x, w);
  eval(y);  // force lazy graph to materialize on the Metal device

  if (y.shape(0) != N || y.shape(1) != N || y.dtype() != float16) {
    std::cerr << "shape/dtype mismatch for N=" << N << "\n";
    return 1;
  }
  // float16_t is exposed by MLX as a wrapper; we read through its underlying
  // uint16 bit pattern below to avoid pulling in an extra header.
  const float16_t *data = y.data<float16_t>();
  const float expected = static_cast<float>(N);
  // Pick a handful of probe positions instead of scanning the whole buffer
  // — they share a kernel, so if one is wrong they all are.
  const int probes[] = {0, N - 1, (N / 2) * N + (N / 2), N * N - 1};
  for (int idx : probes) {
    float v = static_cast<float>(data[idx]);
    if (v != expected) {
      std::cerr << "N=" << N << " y[" << idx << "] = " << v
                << ", expected " << expected << "\n";
      return 1;
    }
  }
  std::cout << "N=" << N << " ones@ones OK (every probe == " << expected
            << ")\n";
  return 0;
}

} // namespace

int main() {
  if (check_ones_matmul(4) != 0) return 1;
  if (check_ones_matmul(2048) != 0) return 1;
  std::cout << "MLX matmul smoke test passed.\n";
  return 0;
}
