# core/transcriber.py
#
# Mục đích: Chuyển đổi giọng nói → văn bản (STT)
#   Backend: Qualcomm AI Hub Models — whisper_large_v3_turbo

import os
import time

import librosa
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ── Cấu hình đọc từ .env ────────────────────────────────────────────────────
WHISPER_HF_ID   = os.getenv("WHISPER_HF_ID", r"models\whisper-tokenizer")
ON_DEVICE       = os.getenv("ON_DEVICE", "false").lower() == "true"
QUALCOMM_DEVICE = os.getenv("QUALCOMM_DEVICE", "Snapdragon X Elite CRD")
ENCODER_PATH    = os.getenv("WHISPER_ENCODER_PATH", r"models\\HfWhisperEncoder.onnx")
DECODER_PATH    = os.getenv("WHISPER_DECODER_PATH", r"models\\HfWhisperDecoder.onnx")
 

# ────────────────────────────────────────────────────────────────────────────
# Hàm 1: Load model — GIỮ NGUYÊN logic, thêm dotenv
# ────────────────────────────────────────────────────────────────────────────
def load_model():
    """
    Load Whisper large-v3-turbo.
 
    ON_DEVICE=false → PyTorch FP local (App + Model từ qai_hub_models)
    ON_DEVICE=true  → NPU Snapdragon (OnnxModelTorchWrapper.OnNPU + HfWhisperApp)
                      giống hệt demo.py đã chạy được
 
    Trả về (model_ref, app):
        model_ref — encoder/decoder object (dùng để cache với st.cache_resource)
        app       — pipeline hoàn chỉnh: audio → text
 
    Cách dùng trong Streamlit:
        @st.cache_resource
        def get_model():
            _, app = load_model()
            return app
 
        app = get_model()
    """
    print("⏳ Loading Qualcomm Whisper large-v3-turbo...")
    print(f"   Chế độ  : {'On-Device NPU (' + QUALCOMM_DEVICE + ')' if ON_DEVICE else 'PyTorch FP (local CPU)'}")
    print(f"   HF model: {WHISPER_HF_ID}")
 
    if ON_DEVICE:
        # ── NPU path: copy y chang demo.py ──────────────────────────────────
        try:
            from qai_hub_models.models._shared.hf_whisper.app import HfWhisperApp
            from qai_hub_models.utils.onnx.torch_wrapper import OnnxModelTorchWrapper
        except ImportError:
            raise ImportError(
                "Chạy: pip install 'qai_hub_models[whisper-large-v3-turbo]'"
            )
 
        print(f"   Encoder : {ENCODER_PATH}")
        print(f"   Decoder : {DECODER_PATH}")
 
        encoder = OnnxModelTorchWrapper.OnNPU(ENCODER_PATH)
        print("   ✅ Encoder loaded")
 
        decoder = OnnxModelTorchWrapper.OnNPU(DECODER_PATH)
        print("   ✅ Decoder loaded")
 
        app = HfWhisperApp(encoder, decoder, WHISPER_HF_ID)
        print("✅ Whisper loaded (NPU, ONNX QNN EP)")
 
        # model_ref trả về tuple để cache_resource có thể giữ tham chiếu
        return (encoder, decoder), app
 
    else:
        # ── CPU path: giữ nguyên logic cũ ───────────────────────────────────
        try:
            from qai_hub_models.models.whisper_large_v3_turbo import App, Model
        except ImportError:
            raise ImportError(
                "Chạy: pip install 'qai_hub_models[whisper-large-v3-turbo]'"
            )
 
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
 
    t0   = time.perf_counter()
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