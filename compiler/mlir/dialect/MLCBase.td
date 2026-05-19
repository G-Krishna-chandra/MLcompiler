#ifndef MLC_BASE_TD
#define MLC_BASE_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

def MLC_Dialect : Dialect {
  let name = "mlc";
  let cppNamespace = "::mlir::mlc";
  let summary = "MLcompiler dialect for heterogeneous Apple-silicon inference";
  let description = [{
    High-level ops executed during LLM inference. Each op carries a
    `target_device` attribute (`gpu`, `ane`, or `auto`) that the scheduling
    pass resolves before lowering to MLX (GPU), MIL (ANE), or custom Metal.
    Quantization format lives in a string attribute on ops that take
    quantized weight operands, not in the operand type, so the same op can
    carry any GGUF quant format without an explosion of MLIR types.
  }];
  let useDefaultAttributePrinterParser = 1;
}

class MLC_Op<string mnemonic, list<Trait> traits = []> :
    Op<MLC_Dialect, mnemonic, traits>;

def MLC_Device_GPU  : I32EnumAttrCase<"GPU",  0, "gpu">;
def MLC_Device_ANE  : I32EnumAttrCase<"ANE",  1, "ane">;
def MLC_Device_Auto : I32EnumAttrCase<"Auto", 2, "auto">;

def MLC_DeviceEnum : I32EnumAttr<"Device", "Target device for op execution",
    [MLC_Device_GPU, MLC_Device_ANE, MLC_Device_Auto]> {
  let cppNamespace = "::mlir::mlc";
  let genSpecializedAttr = 0;
}

def MLC_DeviceAttr : EnumAttr<MLC_Dialect, MLC_DeviceEnum, "device"> {
  let assemblyFormat = "`<` $value `>`";
}

def MLC_FloatTensor : TensorOf<[F16, F32, BF16]>;
def MLC_QuantTensor : TensorOf<[AnyType]>;
def MLC_IntTensor   : TensorOf<[I32, I64]>;

#endif // MLC_BASE_TD
