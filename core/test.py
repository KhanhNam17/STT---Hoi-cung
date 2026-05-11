import onnxruntime as ort
import numpy as np

so = ort.SessionOptions()
so.enable_profiling = True

session = ort.InferenceSession(
    "models/eres2net_int8.onnx",
    sess_options=so,
    providers=["QNNExecutionProvider"]
)

print(session.get_providers())

x = np.random.randn(1, 345, 80).astype(np.float32)

input_name = session.get_inputs()[0].name

y = session.run(None, {input_name: x})

print(y[0].shape)

profile_file = session.end_profiling()

print(profile_file)

# import onnxruntime as ort
# import numpy as np

# so = ort.SessionOptions()
# so.enable_profiling = True
# model_path = "models\\3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"

# session = ort.InferenceSession(
#     model_path,
#     sess_options=so,
#     providers=["QNNExecutionProvider"]
# )

# x = np.random.randn(1, 345, 80).astype(np.float32)

# input_name = session.get_inputs()[0].name

# y = session.run(None, {input_name: x})

# profile_file = session.end_profiling()

# print(profile_file)