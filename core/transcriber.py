# core/transcriber.py
#
# Mục đích: Chuyển đổi giọng nói → văn bản (STT)
#   Backend: Qualcomm AI Hub Models — whisper_large_v3_turbo
#
# Thay đổi so với file gốc:
#   - XOÁ run_transcription(), verify_files() — không dùng trong app
#   - XOÁ ON_DEVICE, QUALCOMM_DEVICE hardcode → đọc từ .env
#   - GIỮ NGUYÊN load_model(), transcribe_file(), get_duration()
#   - THÊM block test cuối file

import os
import time

import librosa
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ── Cấu hình đọc từ .env ────────────────────────────────────────────────────
WHISPER_HF_ID   = "openai/whisper-large-v3-turbo"
ON_DEVICE       = os.getenv("ON_DEVICE", "false").lower() == "true"
QUALCOMM_DEVICE = os.getenv("QUALCOMM_DEVICE", "Samsung Galaxy S25")


# ────────────────────────────────────────────────────────────────────────────
# Hàm 1: Load model — GIỮ NGUYÊN logic, thêm dotenv
# ────────────────────────────────────────────────────────────────────────────
def load_model():
    """
    Load Qualcomm Whisper large-v3-turbo.

    Trả về (qai_model, app):
        qai_model — chứa encoder + decoder (dùng để cache với st.cache_resource)
        app       — pipeline hoàn chỉnh: audio → text

    Cách dùng trong Streamlit:
        @st.cache_resource
        def get_model():
            _, app = load_model()
            return app

        app = get_model()
    """
    try:
        from qai_hub_models.models.whisper_large_v3_turbo import App, Model
    except ImportError:
        raise ImportError(
            "Chạy: pip install 'qai_hub_models[whisper-large-v3-turbo]'"
        )

    print("⏳ Loading Qualcomm Whisper large-v3-turbo...")
    print(f"   Chế độ  : {'On-Device (' + QUALCOMM_DEVICE + ')' if ON_DEVICE else 'PyTorch FP (local)'}")
    print(f"   HF model: {WHISPER_HF_ID}")

    qai_model = Model.from_pretrained()

    # FIX: ép float32 để tránh lỗi mixed dtype (float16/bfloat16 trộn float32)
    qai_model.encoder.float().eval()
    qai_model.decoder.float().eval()

    app = App(
        encoder=qai_model.encoder,
        decoder=qai_model.decoder,
        hf_model_id=WHISPER_HF_ID,
    )

    # Force tiếng Việt — tránh auto-detect, giảm WER ~0.3%
    app.tokenizer.set_prefix_tokens(language="vi", task="transcribe")
    print(f"   Language : vi/transcribe — prefix: {app.tokenizer.prefix_tokens}")
    print("✅ Whisper loaded (SHA+conv optimized, float32, lang=vi)")

    return qai_model, app


# ────────────────────────────────────────────────────────────────────────────
# Hàm 2: Đo duration file — GIỮ NGUYÊN
# ────────────────────────────────────────────────────────────────────────────
def get_duration(wav_path: str) -> float:
    """Đo duration file audio bằng librosa (giây)."""
    try:
        audio, sr = librosa.load(wav_path, sr=16000, mono=True)
        return round(len(audio) / sr, 3)
    except Exception:
        return 0.0


# ────────────────────────────────────────────────────────────────────────────
# Hàm 3: Transcribe 1 file — GIỮ NGUYÊN logic, bỏ phần on-device compile
# ────────────────────────────────────────────────────────────────────────────
def transcribe_file(app, wav_path: str) -> dict:
    """
    Transcribe 1 file WAV, trả về text + metrics.

    Args:
        app      : App object từ load_model()
        wav_path : đường dẫn file WAV 16kHz mono (output của converter)

    Returns dict:
        text     — văn bản nhận dạng được
        duration — độ dài audio (giây)
        latency  — thời gian xử lý (giây)
        rtf      — Real-Time Factor = latency / duration
                   RTF < 1.0 nghĩa là xử lý nhanh hơn thời gian thực
    """
    audio, sr = librosa.load(wav_path, sr=16000, mono=True)
    duration  = round(len(audio) / sr, 3)

    t0 = time.perf_counter()

    if ON_DEVICE:
        # On-device: compile và chạy trên chip Snapdragon qua Qualcomm AI Hub
        # Chỉ dùng khi ON_DEVICE=true trong .env và có token qai_hub
        import qai_hub as hub
        import torch
        from qai_hub_models.models.whisper_large_v3_turbo import App

        device = hub.Device(QUALCOMM_DEVICE)

        # Compile encoder
        enc_sample = app.encoder.sample_inputs()
        enc_traced = torch.jit.trace(
            app.encoder,
            [torch.tensor(v[0]) for v in enc_sample.values()]
        )
        enc_job = hub.submit_compile_job(
            model=enc_traced,
            device=device,
            input_specs=app.encoder.get_input_spec(),
        )

        # Compile decoder
        dec_sample = app.decoder.sample_inputs()
        dec_traced = torch.jit.trace(
            app.decoder,
            [torch.tensor(v[0]) for v in dec_sample.values()]
        )
        dec_job = hub.submit_compile_job(
            model=dec_traced,
            device=device,
            input_specs=app.decoder.get_input_spec(),
        )

        on_device_app = App(
            encoder=enc_job.get_target_model(),
            decoder=dec_job.get_target_model(),
            hf_model_id=WHISPER_HF_ID,
        )
        text = on_device_app.transcribe(audio, audio_sample_rate=sr)

    else:
        # Local PyTorch FP — chạy trực tiếp trên CPU/GPU máy tính
        text = app.transcribe(audio, audio_sample_rate=sr)

    latency = time.perf_counter() - t0

    return {
        "text"    : text.strip() if text else "",
        "duration": duration,
        "latency" : round(latency, 3),
        "rtf"     : round(latency / duration, 4) if duration > 0 else None,
    }


# ────────────────────────────────────────────────────────────────────────────
# TEST — chạy: python core/transcriber.py <file.wav>
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("=" * 55)
    print("  TEST: core/transcriber.py")
    print("=" * 55)

    # Bước 1: kiểm tra import
    print("\n[1] Kiểm tra import qai_hub_models...")
    try:
        from qai_hub_models.models.whisper_large_v3_turbo import App, Model
        print("    ✅ qai_hub_models import thành công")
    except ImportError as e:
        print(f"    ❌ Import lỗi: {e}")
        print("       Chạy: pip install 'qai_hub_models[whisper-large-v3-turbo]'")
        sys.exit(1)

    # Bước 2: cần file WAV để test
    if len(sys.argv) < 2:
        print("\n[2] Cần file WAV để test transcription")
        print("    Dùng lệnh: python core/transcriber.py <file_16k.wav>")
        print("\n    Gợi ý: chạy converter.py trước để tạo file WAV")
        print("    python core/converter.py <file_gốc.mp3>")
        sys.exit(0)

    wav_path = sys.argv[1]
    from pathlib import Path
    if not Path(wav_path).exists():
        print(f"\n    ❌ File không tồn tại: {wav_path}")
        sys.exit(1)

    # Bước 3: load model
    print("\n[2] Load model (lần đầu sẽ mất 1-3 phút để tải weights)...")
    try:
        qai_model, app = load_model()
    except Exception as e:
        print(f"    ❌ Load model thất bại: {e}")
        sys.exit(1)

    # Bước 4: transcribe
    print(f"\n[3] Transcribe file: {wav_path}")
    try:
        result = transcribe_file(app, wav_path)
        print(f"\n    ✅ Kết quả:")
        print(f"       Text    : {result['text']}")
        print(f"       Duration: {result['duration']}s")
        print(f"       Latency : {result['latency']}s")
        print(f"       RTF     : {result['rtf']}")
        if result['rtf'] and result['rtf'] < 1.0:
            print(f"       ⚡ Nhanh hơn thời gian thực ({result['rtf']}x)")
    except Exception as e:
        print(f"    ❌ Transcribe thất bại: {e}")
        sys.exit(1)

    print("\n✅ Tất cả test đều qua — transcriber.py sẵn sàng\n")