#include "compiler/mlir/exec/ANELayer.h"

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

struct ANELayer::Impl {
  MLModel *model = nullptr;
  MLMultiArray *in_qkv_buf = nullptr;
  MLMultiArray *in_attn_buf = nullptr;
  MLDictionaryFeatureProvider *inputs = nullptr;
  NSString *in_qkv_key = nullptr;
  NSString *in_attn_key = nullptr;
  NSString *out_qkv_key = nullptr;
  NSString *out_attn_key = nullptr;
  int M = 0;
  int K_qkv = 0, N_qkv = 0;
  int K_attn = 0, N_attn = 0;
};

namespace {

NSString *compileOrThrow(NSURL *url) {
  NSError *err = nil;
  NSURL *compiled = [MLModel compileModelAtURL:url error:&err];
  if (err)
    throw std::runtime_error(std::string("MLModel compile failed: ") +
                             err.localizedDescription.UTF8String);
  return compiled.path;
}

// Pull the (input name, second input name) and (output name, second
// output name) out of the model's description. We sort by name so the
// mapping `x_qkv → in_qkv_key`, `x_attn → in_attn_key` is deterministic
// regardless of which order CoreML stored them.
void grabIONames(MLModel *m, NSString **in1, NSString **in2,
                 NSString **out1, NSString **out2) {
  auto desc = m.modelDescription;
  NSMutableArray *ins = [NSMutableArray array];
  for (NSString *name in desc.inputDescriptionsByName)
    [ins addObject:name];
  NSMutableArray *outs = [NSMutableArray array];
  for (NSString *name in desc.outputDescriptionsByName)
    [outs addObject:name];
  [ins sortUsingSelector:@selector(compare:)];
  [outs sortUsingSelector:@selector(compare:)];
  if (ins.count != 2 || outs.count != 2)
    throw std::runtime_error("ANELayer model must have 2 inputs / 2 outputs");
  // Names: x_attn, x_qkv (alpha sort) and out_attn, out_qkv.
  // x_qkv → in_qkv_key requires picking the one that ends with "qkv".
  for (NSString *name in ins) {
    if ([name containsString:@"qkv"]) *in1 = name; else *in2 = name;
  }
  for (NSString *name in outs) {
    if ([name containsString:@"qkv"]) *out1 = name; else *out2 = name;
  }
  if (!*in1 || !*in2 || !*out1 || !*out2)
    throw std::runtime_error("ANELayer I/O naming convention broken");
}

} // namespace

ANELayer::ANELayer(const std::string &path, int M,
                   int K_qkv, int N_qkv,
                   int K_attn, int N_attn)
    : p_(std::make_unique<Impl>()) {
  p_->M = M;
  p_->K_qkv = K_qkv; p_->N_qkv = N_qkv;
  p_->K_attn = K_attn; p_->N_attn = N_attn;

  @autoreleasepool {
    NSError *err = nil;
    NSURL *pkg = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
    NSString *compiledPath = compileOrThrow(pkg);
    NSURL *compiled = [NSURL fileURLWithPath:compiledPath];

    MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
    cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    MLModel *m = [MLModel modelWithContentsOfURL:compiled
                                   configuration:cfg
                                           error:&err];
    if (!m || err)
      throw std::runtime_error(std::string("ANELayer model load failed: ") +
                                err.localizedDescription.UTF8String);
    p_->model = m;

    NSString *in_qkv = nil, *in_attn = nil, *out_qkv = nil, *out_attn = nil;
    grabIONames(m, &in_qkv, &in_attn, &out_qkv, &out_attn);
    p_->in_qkv_key = in_qkv;
    p_->in_attn_key = in_attn;
    p_->out_qkv_key = out_qkv;
    p_->out_attn_key = out_attn;

    p_->in_qkv_buf = [[MLMultiArray alloc]
        initWithShape:@[ @(M), @(K_qkv) ]
             dataType:MLMultiArrayDataTypeFloat16 error:&err];
    if (err) throw std::runtime_error("alloc in_qkv MLMultiArray failed");
    p_->in_attn_buf = [[MLMultiArray alloc]
        initWithShape:@[ @(M), @(K_attn) ]
             dataType:MLMultiArrayDataTypeFloat16 error:&err];
    if (err) throw std::runtime_error("alloc in_attn MLMultiArray failed");

    p_->inputs = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{
          p_->in_qkv_key : p_->in_qkv_buf,
          p_->in_attn_key : p_->in_attn_buf,
        }
                     error:&err];
    if (err) throw std::runtime_error("init MLDictionaryFeatureProvider failed");
  }
}

ANELayer::~ANELayer() = default;
ANELayer::ANELayer(ANELayer &&) noexcept = default;
ANELayer &ANELayer::operator=(ANELayer &&) noexcept = default;

int ANELayer::M() const { return p_->M; }
int ANELayer::K_qkv() const { return p_->K_qkv; }
int ANELayer::N_qkv() const { return p_->N_qkv; }
int ANELayer::K_attn() const { return p_->K_attn; }
int ANELayer::N_attn() const { return p_->N_attn; }

namespace {

void copyFP16Input(MLMultiArray *buf, const mx::array &x, int M, int K) {
  if (x.ndim() != 2 || x.shape(0) != M || x.shape(1) != K)
    throw std::runtime_error("ANELayer input shape mismatch");
  mx::array x16 = x.dtype() == mx::float16 ? x : mx::astype(x, mx::float16);
  mx::eval(x16);
  std::memcpy(buf.dataPointer, x16.data<uint16_t>(),
              static_cast<size_t>(M) * K * sizeof(uint16_t));
}

mx::array readOutput(MLMultiArray *y, int M, int N) {
  size_t count = static_cast<size_t>(M) * N;
  mx::Shape shape{M, N};
  if (y.dataType == MLMultiArrayDataTypeFloat16) {
    uint16_t *buf = static_cast<uint16_t *>(std::malloc(count * sizeof(uint16_t)));
    if (!buf) throw std::bad_alloc();
    std::memcpy(buf, y.dataPointer, count * sizeof(uint16_t));
    return mx::array(static_cast<void *>(buf), std::move(shape), mx::float16,
                     [](void *p) { std::free(p); });
  }
  if (y.dataType == MLMultiArrayDataTypeFloat32) {
    float *fbuf = static_cast<float *>(std::malloc(count * sizeof(float)));
    if (!fbuf) throw std::bad_alloc();
    std::memcpy(fbuf, y.dataPointer, count * sizeof(float));
    return mx::array(static_cast<void *>(fbuf), std::move(shape), mx::float32,
                     [](void *p) { std::free(p); });
  }
  throw std::runtime_error("unexpected ANELayer output dtype");
}

} // namespace

std::pair<mx::array, mx::array>
ANELayer::predict(const mx::array &x_qkv, const mx::array &x_attn) {
  copyFP16Input(p_->in_qkv_buf, x_qkv, p_->M, p_->K_qkv);
  copyFP16Input(p_->in_attn_buf, x_attn, p_->M, p_->K_attn);

  NSError *err = nil;
  id<MLFeatureProvider> outFeatures =
      [p_->model predictionFromFeatures:p_->inputs error:&err];
  if (!outFeatures || err)
    throw std::runtime_error(std::string("ANELayer predict failed: ") +
                              err.localizedDescription.UTF8String);
  MLMultiArray *y_qkv =
      [outFeatures featureValueForName:p_->out_qkv_key].multiArrayValue;
  MLMultiArray *y_attn =
      [outFeatures featureValueForName:p_->out_attn_key].multiArrayValue;
  if (!y_qkv || !y_attn)
    throw std::runtime_error("ANELayer predict missing output");

  return {readOutput(y_qkv, p_->M, p_->N_qkv),
          readOutput(y_attn, p_->M, p_->N_attn)};
}

// --- offline build ---

namespace {

const char *kPythonVenv = "/tmp/coreml-venv312/bin/python";
const char *kGenScript =
    "/Users/kc/MLcompiler/compiler/coreml/gen_layer_model.py";

void writeFP16(const std::string &path, const mx::array &w) {
  mx::array w16 = w.dtype() == mx::float16 ? w : mx::astype(w, mx::float16);
  mx::eval(w16);
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("can't write weight binary: " + path);
  out.write(reinterpret_cast<const char *>(w16.data<uint16_t>()),
            static_cast<std::streamsize>(w16.size() * sizeof(uint16_t)));
}

} // namespace

std::string buildANELayerPackage(const std::string &out_dir,
                                 const std::string &cache_key,
                                 int M, int K_qkv, int N_qkv,
                                 int K_attn, int N_attn,
                                 const mx::array &w_qkv_fp16,
                                 const mx::array &w_o_fp16) {
  namespace fs = std::filesystem;
  fs::create_directories(out_dir);
  std::string pkg = out_dir + "/" + cache_key + ".mlpackage";
  if (fs::exists(pkg)) return pkg;

  std::string b1 = out_dir + "/" + cache_key + ".w_qkv.fp16";
  std::string b2 = out_dir + "/" + cache_key + ".w_o.fp16";
  writeFP16(b1, w_qkv_fp16);
  writeFP16(b2, w_o_fp16);

  std::string cmd = std::string(kPythonVenv) + " " + kGenScript + " '" + pkg +
                    "' " + std::to_string(M) + " " + std::to_string(K_qkv) +
                    " " + std::to_string(N_qkv) + " " + std::to_string(K_attn) +
                    " " + std::to_string(N_attn) + " '" + b1 + "' '" + b2 + "'";
  int rc = std::system(cmd.c_str());
  if (rc != 0)
    throw std::runtime_error("gen_layer_model.py failed (rc=" +
                              std::to_string(rc) + ") for " + cache_key);
  fs::remove(b1);
  fs::remove(b2);
  return pkg;
}

} // namespace mlir::mlc::exec
