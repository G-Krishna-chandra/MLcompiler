#include "compiler/mlir/emit/GGUFToMLIR.h"

#include "compiler/frontends/gguf_loader.hpp"
#include "compiler/frontends/ggml_types.hpp"
#include "compiler/mlir/dialect/MLCDialect.h"
#include "compiler/mlir/dialect/MLCOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include <stdexcept>
#include <string>

namespace mlir::mlc {

namespace {

using ::mlc::frontend::GGUFLoader;
using ::mlc::frontend::GGUFTensorInfo;
using ::mlc::frontend::GGUFValueType;
using namespace ::mlc::frontend;  // GGML_TYPE_* constants

// Pull a kv metadata value, narrowing through GGUF's set of integer/float
// types. Throws if the key is missing or the type is unexpected.
uint64_t getKvU(const GGUFLoader &loader, const std::string &key) {
  const auto &kv = loader.kvMetadata();
  auto it = kv.find(key);
  if (it == kv.end())
    throw std::runtime_error("missing GGUF kv: " + key);
  switch (it->second.type) {
  case GGUFValueType::UINT32: return std::get<uint32_t>(it->second.data);
  case GGUFValueType::UINT64: return std::get<uint64_t>(it->second.data);
  case GGUFValueType::INT32:  return static_cast<uint64_t>(std::get<int32_t>(it->second.data));
  case GGUFValueType::INT64:  return static_cast<uint64_t>(std::get<int64_t>(it->second.data));
  default:
    throw std::runtime_error("kv " + key + " is not an integer");
  }
}

float getKvF(const GGUFLoader &loader, const std::string &key) {
  const auto &kv = loader.kvMetadata();
  auto it = kv.find(key);
  if (it == kv.end())
    throw std::runtime_error("missing GGUF kv: " + key);
  if (it->second.type == GGUFValueType::FLOAT32)
    return std::get<float>(it->second.data);
  if (it->second.type == GGUFValueType::FLOAT64)
    return static_cast<float>(std::get<double>(it->second.data));
  throw std::runtime_error("kv " + key + " is not a float");
}

std::string getArch(const GGUFLoader &loader) {
  const auto &kv = loader.kvMetadata();
  auto it = kv.find("general.architecture");
  if (it == kv.end() || it->second.type != GGUFValueType::STRING)
    throw std::runtime_error("missing general.architecture");
  return std::get<std::string>(it->second.data);
}

Type ggufElementType(MLIRContext &ctx, uint32_t dtype) {
  switch (dtype) {
  case GGML_TYPE_F32:  return Float32Type::get(&ctx);
  case GGML_TYPE_F16:  return Float16Type::get(&ctx);
  case GGML_TYPE_BF16: return BFloat16Type::get(&ctx);
  // Quantized weights tag with the underlying element bit-width; the
  // quant_format string attr disambiguates the exact GGUF layout downstream.
  case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q4_K:
  case GGML_TYPE_Q2_K:
    return IntegerType::get(&ctx, 4);
  case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1: case GGML_TYPE_Q5_K:
    return IntegerType::get(&ctx, 5);
  case GGML_TYPE_Q6_K:
    return IntegerType::get(&ctx, 6);
  case GGML_TYPE_Q8_0: case GGML_TYPE_Q8_1: case GGML_TYPE_Q8_K:
    return IntegerType::get(&ctx, 8);
  case GGML_TYPE_Q3_K:
    return IntegerType::get(&ctx, 3);
  default:
    return Float32Type::get(&ctx);
  }
}

const char *ggufQuantFormat(uint32_t dtype) {
  switch (dtype) {
  case GGML_TYPE_F32:  return "f32";
  case GGML_TYPE_F16:  return "f16";
  case GGML_TYPE_BF16: return "bf16";
  case GGML_TYPE_Q4_0: return "q4_0";
  case GGML_TYPE_Q4_1: return "q4_1";
  case GGML_TYPE_Q4_K: return "q4_k";
  case GGML_TYPE_Q5_0: return "q5_0";
  case GGML_TYPE_Q5_1: return "q5_1";
  case GGML_TYPE_Q5_K: return "q5_k";
  case GGML_TYPE_Q6_K: return "q6_k";
  case GGML_TYPE_Q8_0: return "q8_0";
  case GGML_TYPE_Q8_1: return "q8_1";
  case GGML_TYPE_Q8_K: return "q8_k";
  case GGML_TYPE_Q2_K: return "q2_k";
  case GGML_TYPE_Q3_K: return "q3_k";
  default:             return "f32";
  }
}

RankedTensorType tensorTypeOf(MLIRContext &ctx, const GGUFTensorInfo &t) {
  ::llvm::SmallVector<int64_t> shape;
  shape.reserve(t.shape.size());
  for (uint64_t d : t.shape)
    shape.push_back(static_cast<int64_t>(d));
  return RankedTensorType::get(shape, ggufElementType(ctx, t.dtype));
}

const GGUFTensorInfo &mustGet(const GGUFLoader &loader,
                              const std::string &name) {
  const auto &tensors = loader.tensors();
  auto it = tensors.find(name);
  if (it == tensors.end())
    throw std::runtime_error("missing GGUF tensor: " + name);
  return it->second;
}

} // namespace

OwningOpRef<ModuleOp> emitMLIR(MLIRContext &ctx, const GGUFLoader &loader) {
  ctx.loadDialect<MLCDialect, func::FuncDialect>();

  const std::string arch = getArch(loader);
  if (arch != "llama")
    throw std::runtime_error("emitMLIR currently supports only llama-family GGUFs (got " + arch + ")");

  const int64_t nLayers  = static_cast<int64_t>(getKvU(loader, "llama.block_count"));
  const int64_t hidden   = static_cast<int64_t>(getKvU(loader, "llama.embedding_length"));
  const int64_t nHeads   = static_cast<int64_t>(getKvU(loader, "llama.attention.head_count"));
  const int64_t nKvHeads = static_cast<int64_t>(getKvU(loader, "llama.attention.head_count_kv"));
  const int64_t headDim  = hidden / nHeads;
  const float   epsilon  = getKvF(loader, "llama.attention.layer_norm_rms_epsilon");

  const int64_t kvDim = nKvHeads * headDim;
  // Vocab dim: whichever side of the 2D output.weight is NOT hidden. GGUF
  // models differ on whether output.weight is stored as [vocab, hidden]
  // (Llama 2, 3) or [hidden, vocab] (TinyLlama, some 1.1B variants).
  const auto &outW = mustGet(loader, "output.weight").shape;
  if (outW.size() != 2)
    throw std::runtime_error("output.weight must be rank-2");
  const int64_t vocab = (static_cast<int64_t>(outW[0]) == hidden)
                            ? static_cast<int64_t>(outW[1])
                            : static_cast<int64_t>(outW[0]);

  OpBuilder b(&ctx);
  auto loc   = b.getUnknownLoc();
  auto i32   = IntegerType::get(&ctx, 32);
  auto f16   = Float16Type::get(&ctx);

  auto module = ModuleOp::create(loc);

  // Assemble argument types + name annotations. The order is fixed so the
  // test can index by position without parsing names.
  ::llvm::SmallVector<Type> argTypes;
  ::llvm::SmallVector<std::string> argNames;
  auto pushArg = [&](Type t, std::string name) {
    argTypes.push_back(t);
    argNames.push_back(std::move(name));
  };
  auto pushTensorArg = [&](const std::string &name) {
    pushArg(tensorTypeOf(ctx, mustGet(loader, name)), name);
  };

  pushArg(RankedTensorType::get({ShapedType::kDynamic}, i32), "ids");
  pushArg(RankedTensorType::get({ShapedType::kDynamic}, i32), "positions");
  pushTensorArg("token_embd.weight");
  pushTensorArg("output_norm.weight");
  pushTensorArg("output.weight");
  for (int64_t i = 0; i < nLayers; ++i) {
    std::string p = "blk." + std::to_string(i) + ".";
    pushTensorArg(p + "attn_norm.weight");
    pushTensorArg(p + "attn_q.weight");
    pushTensorArg(p + "attn_k.weight");
    pushTensorArg(p + "attn_v.weight");
    pushTensorArg(p + "attn_output.weight");
    pushTensorArg(p + "ffn_norm.weight");
    pushTensorArg(p + "ffn_gate.weight");
    pushTensorArg(p + "ffn_up.weight");
    pushTensorArg(p + "ffn_down.weight");
  }

  auto hiddenTy = RankedTensorType::get({ShapedType::kDynamic, hidden}, f16);
  auto kvTy     = RankedTensorType::get({ShapedType::kDynamic, kvDim},  f16);
  auto logitsTy = RankedTensorType::get({ShapedType::kDynamic, vocab},  f16);

  auto fnTy = b.getFunctionType(argTypes, {logitsTy});
  b.setInsertionPointToEnd(module.getBody());
  auto fn = b.create<func::FuncOp>(loc, "inference", fnTy);
  for (size_t i = 0; i < argNames.size(); ++i)
    fn.setArgAttr(i, "mlc.name", b.getStringAttr(argNames[i]));

  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);

  auto autoDev = DeviceAttr::get(&ctx, Device::Auto);
  // TinyLlama (and similar 1.1B Llama variants) stores weights as [in, out]
  // — same convention the matmul wants — so no transpose. Models that store
  // [out, in] (e.g., Llama 2/3 output.weight) would need transpose_b=true on
  // the affected ops; handle per-tensor when we add support for them.
  auto falseAttr = b.getBoolAttr(false);
  auto emitMatMul = [&](Value x, Value w, RankedTensorType outTy,
                        const char *qf) -> Value {
    return b.create<MatMulOp>(loc, outTy, x, w, autoDev,
                              b.getStringAttr(qf), falseAttr);
  };

  // Convenience: format strings for each layer-arg slot.
  auto qfAt = [&](size_t argIdx) {
    return ggufQuantFormat(mustGet(loader, argNames[argIdx]).dtype);
  };

  Value ids            = entry->getArgument(0);
  Value positions      = entry->getArgument(1);
  Value tokenEmbd      = entry->getArgument(2);
  Value outputNorm     = entry->getArgument(3);
  Value outputWeight   = entry->getArgument(4);

  // mlc.embedding
  Value h = b.create<EmbeddingOp>(loc, hiddenTy, ids, tokenEmbd, autoDev,
                                  b.getStringAttr(qfAt(2)));

  // 22 (or however many) decoder layers
  constexpr size_t kPerLayer = 9;
  constexpr size_t kHeader = 5;  // ids, positions, token_embd, output_norm, output
  for (int64_t i = 0; i < nLayers; ++i) {
    const size_t base = kHeader + i * kPerLayer;
    Value attnNorm = entry->getArgument(base + 0);
    Value wQ       = entry->getArgument(base + 1);
    Value wK       = entry->getArgument(base + 2);
    Value wV       = entry->getArgument(base + 3);
    Value wO       = entry->getArgument(base + 4);
    Value ffnNorm  = entry->getArgument(base + 5);
    Value wGate    = entry->getArgument(base + 6);
    Value wUp      = entry->getArgument(base + 7);
    Value wDown    = entry->getArgument(base + 8);

    Value n1 = b.create<NormOp>(loc, hiddenTy, h, attnNorm, autoDev,
                                b.getF32FloatAttr(epsilon));
    Value q = emitMatMul(n1, wQ, hiddenTy, qfAt(base + 1));
    Value k = emitMatMul(n1, wK, kvTy,     qfAt(base + 2));
    Value v = emitMatMul(n1, wV, kvTy,     qfAt(base + 3));
    Value a = b.create<AttentionOp>(loc, hiddenTy, q, k, v, positions, autoDev,
                                    b.getI64IntegerAttr(nHeads),
                                    b.getI64IntegerAttr(nKvHeads),
                                    b.getI64IntegerAttr(headDim));
    Value o = emitMatMul(a, wO, hiddenTy, qfAt(base + 4));
    h = b.create<AddOp>(loc, hiddenTy, h, o, autoDev);

    Value n2 = b.create<NormOp>(loc, hiddenTy, h, ffnNorm, autoDev,
                                b.getF32FloatAttr(epsilon));
    Value f = b.create<FeedForwardOp>(loc, hiddenTy, n2, wGate, wUp, wDown,
                                      autoDev, b.getStringAttr(qfAt(base + 6)));
    h = b.create<AddOp>(loc, hiddenTy, h, f, autoDev);
  }

  // Final norm + LM head
  Value finalNorm = b.create<NormOp>(loc, hiddenTy, h, outputNorm, autoDev,
                                     b.getF32FloatAttr(epsilon));
  Value logits = b.create<LMHeadOp>(loc, logitsTy, finalNorm, outputWeight,
                                    autoDev, b.getStringAttr(qfAt(4)));

  b.create<func::ReturnOp>(loc, logits);

  return module;
}

} // namespace mlir::mlc
