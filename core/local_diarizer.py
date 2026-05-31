import os
import torch
from pyannote.audio import Model, Inference



device = torch.device("cpu")

model_path = "models/wespeaker/pytorch_model.bin"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Không tìm thấy file tại:{model_path}")

model = Model.from_pretrained(model_path)

model.to(device)

inference = Inference(model, window='whole', device=device)

print("✅ Load model thành công")