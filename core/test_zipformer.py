import os
import sys
import queue
import numpy as np
import sounddevice as sd
import sherpa_onnx

# ────────────────────────────────────────────────────────────────────────────
# 1. CẤU HÌNH ĐƯỜNG DẪN MÔ HÌNH (Sử dụng cấu hình từ Batch Mode)
# ────────────────────────────────────────────────────────────────────────────
MODEL_DIR = "models/zipformer"

tokens_path  = f"{MODEL_DIR}/config.json"
encoder_path = f"{MODEL_DIR}/encoder-epoch-31-avg-11-chunk-32-left-128.fp16.onnx"
decoder_path = f"{MODEL_DIR}/decoder-epoch-31-avg-11-chunk-32-left-128.fp16.onnx"
joiner_path  = f"{MODEL_DIR}/joiner-epoch-31-avg-11-chunk-32-left-128.fp16.onnx"

SAMPLE_RATE = 16000

# ────────────────────────────────────────────────────────────────────────────
# 2. KHỞI TẠO MÔ HÌNH LÊN RAM
# ────────────────────────────────────────────────────────────────────────────
print("⏳ Đang nạp mô hình Zipformer 30M lên RAM...")
try:
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=tokens_path,
        encoder=encoder_path,
        decoder=decoder_path,
        joiner=joiner_path,
        num_threads=1,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=20 # Giảm xuống để phản hồi Live nhanh hơn
    )
    print("✅ Đã load model thành công!\n")
except Exception as e:
    print(f"❌ Lỗi khi tải mô hình: {e}")
    sys.exit(1)

# ────────────────────────────────────────────────────────────────────────────
# 3. THIẾT LẬP LUỒNG MICROPHONE & HÀNG ĐỢI
# ────────────────────────────────────────────────────────────────────────────
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    """Hàm callback ném âm thanh liên tục từ Micro vào Hàng đợi"""
    if status:
        print(f"\n⚠️ Lỗi Micro: {status}", file=sys.stderr)
    # indata có dạng float32 trong khoảng [-1.0, 1.0], nén thành mảng 1 chiều
    audio_queue.put(indata.copy().flatten())

# ────────────────────────────────────────────────────────────────────────────
# 4. VÒNG LẶP XỬ LÝ REAL-TIME (STT LOOP)
# ────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("🎙️ BẮT ĐẦU GHI ÂM (Ấn Ctrl+C để dừng)")
print("=" * 60)

stream = recognizer.create_stream()
last_text = ""

try:
    # Mở luồng thu âm liên tục với sounddevice
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=audio_callback):
        while True:
            # Rút khối âm thanh ra khỏi hàng đợi
            chunk = audio_queue.get()
            
            # Đẩy khối âm thanh vào luồng của Zipformer
            stream.accept_waveform(SAMPLE_RATE, chunk)
            
            # Chạy giải mã liên tục miễn là mô hình báo đã sẵn sàng
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            
            # Lấy chuỗi văn bản nhận dạng được (Loại bỏ .text theo chuẩn Sherpa cũ)
            text = recognizer.get_result(stream).strip()
            
            # Kiểm tra xem người dùng đã dứt câu chưa
            is_endpoint = recognizer.is_endpoint(stream)
            
            # Hiển thị trực tiếp lên Terminal (In đè lên dòng hiện tại)
            if text != last_text and text:
                # Xóa dòng cũ và in dòng đang nhận dạng
                sys.stdout.write(f"\r\033[KĐang nghe: {text}")
                sys.stdout.flush()
                last_text = text
                
            # Nếu phát hiện dứt câu (Khoảng lặng)
            if is_endpoint:
                if text:
                    sys.stdout.write(f"\r\033[K✅ Chốt câu: {text}\n")
                    sys.stdout.flush()
                # Reset luồng để bắt đầu nghe câu mới
                recognizer.reset(stream)
                last_text = ""

except KeyboardInterrupt:
    print("\n\n⏹️ Đã ngắt luồng ghi âm. Chương trình kết thúc.")
except Exception as e:
    print(f"\n❌ Lỗi hệ thống thu âm: {e}")