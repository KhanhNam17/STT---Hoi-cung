import os
from dotenv import load_dotenv

# Import các module chuẩn của Diart như trong ảnh
from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.sources import MicrophoneAudioSource
from diart.inference import StreamingInference

# 1. Nạp token từ file .env
load_dotenv()
my_token = os.getenv("HF_TOKEN")

if not my_token:
    print("❌ Lỗi: Không tìm thấy HF_TOKEN trong file .env")
    exit(1)

# 2. Cấu hình Pipeline với Token
print("⏳ Đang khởi tạo Diart Pipeline...")
config = SpeakerDiarizationConfig(
    hf_token=my_token,
    # Bạn có thể ép model dùng CPU hoặc GPU ở đây nếu cần:
    # device=torch.device("cuda") 
)
pipeline = SpeakerDiarization(config)

# 3. Khởi tạo nguồn âm thanh từ Microphone
mic = MicrophoneAudioSource()

# 4. Khởi chạy bộ suy luận trực tiếp (có kèm vẽ biểu đồ)
print("\n" + "="*50)
print("🎙️ ĐÃ KẾT NỐI MICROPHONE. HÃY THỬ NÓI CHUYỆN!")
print("📊 Biểu đồ sóng âm sẽ hiện ra. Nhấn Ctrl+C ở Terminal để dừng.")
print("="*50 + "\n")

# do_plot=True sẽ bật cửa sổ đồ thị real-time như trong ảnh của bạn
inference = StreamingInference(pipeline, mic, do_plot=True)

# Bắt đầu vòng lặp nghe
prediction = inference()