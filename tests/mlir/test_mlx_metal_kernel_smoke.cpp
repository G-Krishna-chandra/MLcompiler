// Smoke test for MLX's metal_kernel custom-op API. Builds a trivial
// "multiply by 2" kernel, dispatches it on a small array, and asserts the
// host-readback matches. Establishes the pattern that S1's Q4_0 kernel
// will follow.

#include "compiler/mlir/exec/MLXBuilder.h"

#include <mlx/array.h>
#include <mlx/fast.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

#include <gtest/gtest.h>

#include <string>

namespace mx = mlx::core;

TEST(MLXMetalKernelSmoke, MultiplyByTwo) {
  const char *src = R"(
    uint idx = thread_position_in_grid.x;
    if (idx >= n) return;
    out[idx] = inp[idx] * (T)2;
  )";
  auto fn = mx::fast::metal_kernel(
      /*name=*/"mul_by_two",
      /*input_names=*/{"inp"},
      /*output_names=*/{"out"},
      /*source=*/src,
      /*header=*/"",
      /*ensure_row_contiguous=*/true,
      /*atomic_outputs=*/false);

  std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8};
  auto inp = mlir::mlc::exec::fp32ToMLX(data, {8});
  auto results = fn(
      /*inputs=*/{inp},
      /*output_shapes=*/{mx::Shape{8}},
      /*output_dtypes=*/{mx::float32},
      /*grid=*/{8, 1, 1},
      /*threadgroup=*/{8, 1, 1},
      /*template_args=*/{{"T", mx::float32}, {"n", 8}},
      /*init_value=*/std::nullopt,
      /*verbose=*/false,
      /*stream=*/{});

  ASSERT_EQ(results.size(), 1u);
  auto host = mlir::mlc::exec::mlxToF32(results[0]);
  ASSERT_EQ(host.size(), 8u);
  for (size_t i = 0; i < host.size(); ++i)
    EXPECT_EQ(host[i], static_cast<float>((i + 1) * 2)) << "at " << i;
}
