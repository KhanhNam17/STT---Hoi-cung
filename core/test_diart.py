import time
import wave
import numpy as np
import torch
import os

# Tắt cảnh báo K2 của SpeechBrain (để tránh lỗi log rác)
os.environ["SPEECHBRAIN_DISABLE_K2"] = "1"

# Import trực tiếp Pipeline của Pyannote thay vì dùng qua Diart cho chế độ Offline
from pyannote.audio import Pipeline
from dotenv import load_dotenv

# Tự động nạp biến môi trường (ví dụ: HF_TOKEN) từ file .env cùng thư mục
load_dotenv()

# ────────────────────────────────────────────────────────────────────────────
# 1. KHỞI TẠO CẤU HÌNH PYANNOTE DIARIZATION
# ────────────────────────────────────────────────────────────────────────────
print("⏳ Đang nạp mô hình Pyannote Diarization lên RAM...")
try:
    # Nạp pipeline từ Hugging Face. Thư viện sẽ tự động dùng HF_TOKEN từ os.environ
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.environ.get("HF_TOKEN")
    )
    
    # Kiểm tra nếu máy có hỗ trợ GPU (CUDA), đẩy model sang GPU để xử lý nhanh hơn
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        print("✅ Đã load model thành công (Chế độ GPU)!\n")
    else:
        print("✅ Đã load model thành công (Chế độ CPU)!\n")

except Exception as e:
    print(f"❌ Lỗi khi nạp mô hình. (Bạn đã cấp quyền truy cập model trên web chưa?)")
    print(f"Chi tiết lỗi: {e}")
    exit(1)

# ────────────────────────────────────────────────────────────────────────────
# 2. ĐỌC FILE AUDIO TEST
# ────────────────────────────────────────────────────────────────────────────
audio_file = "data/podcast_HAT_16k.wav"

if not os.path.exists(audio_file):
    print(f"❌ Không tìm thấy file: {audio_file}")
    exit(1)

print(f"🎵 Đang đọc file: {audio_file}")
with wave.open(audio_file, 'rb') as f:
    sample_rate = f.getframerate()
    frames = f.readframes(f.getnframes())
    # Chuyển đổi sang float32 theo chuẩn đầu vào của Pytorch
    audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

# Tạo tensor Pytorch có shape (1, samples)
waveform = torch.from_numpy(audio_np).unsqueeze(0)
duration = len(audio_np) / sample_rate

# ────────────────────────────────────────────────────────────────────────────
# 3. CHẠY PHÂN TÁCH NGƯỜI NÓI (OFFLINE)
# ────────────────────────────────────────────────────────────────────────────
print(f"⏳ Bắt đầu phân tích Diarization ({duration:.2f} giây audio)...")
start_time = time.perf_counter()

# Pyannote Pipeline yêu cầu đầu vào dạng dictionary chứa 'waveform' và 'sample_rate'
annotation = pipeline({"waveform": waveform, "sample_rate": sample_rate})

end_time = time.perf_counter()
process_time = end_time - start_time
rtf = process_time / duration if duration > 0 else 0

# ────────────────────────────────────────────────────────────────────────────
# 4. IN KẾT QUẢ BÁO CÁO
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("🗣️ KẾT QUẢ PHÂN TÁCH NGƯỜI NÓI (DIARIZATION):")
print("-" * 50)

# Lặp qua các đoạn được gán nhãn
for segment, _, speaker in annotation.itertracks(yield_label=True):
    bar = "█" * min(40, int((segment.end - segment.start) * 2))
    print(f"[{segment.start:05.1f}s - {segment.end:05.1f}s] {speaker:12s} | {bar}")

print("="*50)
print(f"⏱  Độ dài Audio : {duration:.2f}s")
print(f"⚡ Thời gian xử lý: {process_time:.2f}s")
print(f"🚀 Hệ số RTF    : {rtf:.3f}x")
print("="*50)