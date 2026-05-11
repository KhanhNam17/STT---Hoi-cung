from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "models\\eres2net_fixed.onnx",
    "models\\eres2net_int8.onnx",
    weight_type=QuantType.QInt8
)