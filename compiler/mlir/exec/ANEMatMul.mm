#include "compiler/mlir/exec/ANEMatMul.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx = mlx::core;

namespace mlir::mlc::exec {

struct ANEMatMul::Impl {
  MLModel *model = nullptr;       // ARC: owned, retained until destructor
  MLMultiArray *input_buf = nullptr;
  MLDictionaryFeatureProvider *inputs = nullptr;
  NSString *input_key = nullptr;
  NSString *output_key = nullptr;
  int M = 0, K = 0, N = 0;
};

namespace {

NSString *compileModelOrThrow(NSURL *mlpackage_url) {
  NSError *err = nil;
  NSURL *compiled = [MLModel compileModelAtURL:mlpackage_url error:&err];
  if (err)
    throw std::runtime_error(std::string("MLModel compile failed: ") +
                             err.localizedDescription.UTF8String);
  return compiled.path;
}

// Read input/output feature names from the model description so we can
// build the right key in the feature provider without hardcoding "x" /
// "y" — MIL programs converted to mlprogram occasionally rename outputs.
void grabIONames(MLModel *model, NSString **inKey, NSString **outKey) {
  auto desc = model.modelDescription;
  for (NSString *name in desc.inputDescriptionsByName) { *inKey = name; break; }
  for (NSString *name in desc.outputDescriptionsByName) { *outKey = name; break; }
  if (!*inKey || !*outKey)
    throw std::runtime_error("ANE matmul model missing input/output");
}

} // namespace

ANEMatMul::ANEMatMul(const std::string &path, int M, int K, int N)
    : p_(std::make_unique<Impl>()) {
  p_->M = M; p_->K = K; p_->N = N;
  @autoreleasepool {
    NSError *err = nil;
    NSURL *pkg = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
    NSString *compiledPath = compileModelOrThrow(pkg);
    NSURL *compiled = [NSURL fileURLWithPath:compiledPath];

    MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
    cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    MLModel *m = [MLModel modelWithContentsOfURL:compiled
                                   configuration:cfg
                                           error:&err];
    if (!m || err)
      throw std::runtime_error(std::string("MLModel load failed: ") +
                               err.localizedDescription.UTF8String);
    p_->model = m;
    NSString *in_local = nil, *out_local = nil;
    grabIONames(m, &in_local, &out_local);
    p_->input_key = in_local;
    p_->output_key = out_local;

    // Preallocate the input MLMultiArray once. Predictions reuse the
    // same buffer to avoid per-call allocation cost.
    p_->input_buf = [[MLMultiArray alloc]
        initWithShape:@[ @(M), @(K) ]
             dataType:MLMultiArrayDataTypeFloat16
                error:&err];
    if (err)
      throw std::runtime_error("alloc input MLMultiArray failed");
    p_->inputs = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{ p_->input_key : p_->input_buf }
                     error:&err];
    if (err)
      throw std::runtime_error("init MLDictionaryFeatureProvider failed");
  }
}

ANEMatMul::~ANEMatMul() = default;
ANEMatMul::ANEMatMul(ANEMatMul &&) noexcept = default;
ANEMatMul &ANEMatMul::operator=(ANEMatMul &&) noexcept = default;

int ANEMatMul::M() const { return p_->M; }
int ANEMatMul::K() const { return p_->K; }
int ANEMatMul::N() const { return p_->N; }

mx::array ANEMatMul::predict(const mx::array &x) {
  if (x.ndim() != 2 || x.shape(0) != p_->M || x.shape(1) != p_->K)
    throw std::runtime_error("ANEMatMul: input shape mismatch");

  // mx::array → MLMultiArray. eval first to materialize the bytes,
  // then copy them into the preallocated MLMultiArray buffer.
  mx::array x16 = x.dtype() == mx::float16 ? x : mx::astype(x, mx::float16);
  mx::eval(x16);
  const void *src = x16.data<uint16_t>();
  std::memcpy(p_->input_buf.dataPointer, src,
              static_cast<size_t>(p_->M) * p_->K * sizeof(uint16_t));

  // Predict.
  NSError *err = nil;
  id<MLFeatureProvider> outFeatures =
      [p_->model predictionFromFeatures:p_->inputs error:&err];
  if (!outFeatures || err)
    throw std::runtime_error(std::string("CoreML predict failed: ") +
                             err.localizedDescription.UTF8String);
  MLMultiArray *y = [outFeatures featureValueForName:p_->output_key].multiArrayValue;
  if (!y)
    throw std::runtime_error("CoreML predict produced no output");

  // MLMultiArray → mx::array. Return CoreML's native output dtype (fp32
  // even when compute_precision is fp16 — see model spec); caller can
  // astype lazily so back-to-back predict() calls don't sync MLX each
  // round-trip.
  size_t out_count = static_cast<size_t>(p_->M) * p_->N;
  mx::Shape shape{p_->M, p_->N};
  if (y.dataType == MLMultiArrayDataTypeFloat16) {
    uint16_t *buf = static_cast<uint16_t *>(std::malloc(out_count * sizeof(uint16_t)));
    if (!buf) throw std::bad_alloc();
    std::memcpy(buf, y.dataPointer, out_count * sizeof(uint16_t));
    return mx::array(static_cast<void *>(buf), std::move(shape), mx::float16,
                     [](void *p) { std::free(p); });
  }
  if (y.dataType == MLMultiArrayDataTypeFloat32) {
    float *fbuf =
        static_cast<float *>(std::malloc(out_count * sizeof(float)));
    if (!fbuf) throw std::bad_alloc();
    std::memcpy(fbuf, y.dataPointer, out_count * sizeof(float));
    return mx::array(static_cast<void *>(fbuf), std::move(shape),
                     mx::float32,
                     [](void *p) { std::free(p); });
  }
  throw std::runtime_error("unexpected CoreML output dtype");
}

// --- offline package generation ---

namespace {

const char *kPythonVenv = "/tmp/coreml-venv312/bin/python";
const char *kGenScript =
    "/Users/kc/MLcompiler/compiler/coreml/gen_matmul_model.py";

void writeFP16Binary(const std::string &path, const mx::array &w) {
  mx::array w16 = w.dtype() == mx::float16 ? w : mx::astype(w, mx::float16);
  mx::eval(w16);
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("can't write weight binary: " + path);
  out.write(reinterpret_cast<const char *>(w16.data<uint16_t>()),
            static_cast<std::streamsize>(w16.size() * sizeof(uint16_t)));
}

} // namespace

std::string buildANEMatMulPackage(const std::string &out_dir,
                                  const std::string &cache_key,
                                  int M, int K, int N,
                                  const mx::array &weight_fp16) {
  namespace fs = std::filesystem;
  fs::create_directories(out_dir);
  std::string pkg_path = out_dir + "/" + cache_key + ".mlpackage";
  if (fs::exists(pkg_path))
    return pkg_path;  // Already generated; cached on disk between runs.

  std::string bin_path = out_dir + "/" + cache_key + ".fp16";
  writeFP16Binary(bin_path, weight_fp16);

  // Build the shell command. Single-quote-escape the paths to be safe.
  std::string cmd = std::string(kPythonVenv) + " " + kGenScript + " '" +
                    pkg_path + "' " + std::to_string(M) + " " +
                    std::to_string(K) + " " + std::to_string(N) + " '" +
                    bin_path + "'";
  int rc = std::system(cmd.c_str());
  if (rc != 0)
    throw std::runtime_error("gen_matmul_model.py failed (rc=" +
                              std::to_string(rc) + ") for " + cache_key);
  fs::remove(bin_path);
  return pkg_path;
}

} // namespace mlir::mlc::exec
