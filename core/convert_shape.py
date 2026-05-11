import onnx
from onnxruntime.tools.make_dynamic_shape_fixed import make_dim_param_fixed

model = onnx.load("models\\eres2net_sim.onnx")

make_dim_param_fixed(model.graph, "frames", 345)

onnx.save(model, "eres2net_fixed.onnx")