from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "clip_text.onnx",
    "clip_text_int8.onnx",
    weight_type=QuantType.QInt8
)

print("INT8 model ready")
