# core/converter.py
#
# Mục đích: Tiền xử lý audio đầu vào cho app STT
#   - Nhận file upload từ Streamlit (bytes) hoặc đường dẫn file
#   - Convert sang WAV 16kHz mono — chuẩn cho Whisper + pyannote
#   - Đọc thông tin file sau khi convert để hiển thị UI
#
# Thay đổi so với file gốc:
#   - XOÁ convert_dataset() — không dùng trong app thực tế
#   - THÊM convert_from_bytes() — nhận bytes từ st.file_uploader
#   - THÊM get_audio_info()     — đọc duration/sample_rate cho UI
#   - FFMPEG_PATH đọc từ .env thay vì hardcode

import os
import subprocess
import tempfile
import wave
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Đọc từ .env, fallback "ffmpeg" nếu đã có trong PATH hệ thống
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")


# ────────────────────────────────────────────────────────────────────────────
# Hàm 1: convert file từ đường dẫn — hàm core, các hàm khác đều gọi vào đây
# ────────────────────────────────────────────────────────────────────────────
def convert_to_wav(
    input_path: str,
    output_path: str,
    sample_rate: int = 16000,
    normalize: bool = True,
) -> bool:
    """
    Convert 1 file bất kỳ sang WAV 16kHz mono bằng ffmpeg.

    Output format chuẩn cho Whisper + pyannote:
      - Sample rate : 16000 Hz
      - Channels    : 1 (mono)
      - Sample fmt  : s16 (signed 16-bit PCM)
      - Loudness: - 16 LUFS EBU R128 (normalize = True)

    Returns:
        True nếu thành công, False nếu ffmpeg báo lỗi.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        FFMPEG_PATH,
        "-y",                     # overwrite nếu file đã tồn tại
        "-i", input_path,
        "-ar", str(sample_rate),  # sample rate
        "-ac", "1",               # mono
        "-sample_fmt", "s16",     # 16-bit PCM
    ]

    if normalize:
        cmd += ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11:linear=true"]
    
    cmd.append(output_path)

    result = subprocess.run(cmd, capture_output=True)

    if result.returncode != 0:
        err = result.stderr.decode(errors="replace")
        print(f"[converter] ffmpeg error:\n{err}")

    return result.returncode == 0


# ────────────────────────────────────────────────────────────────────────────
# Hàm 2: nhận bytes từ Streamlit st.file_uploader — dùng trong Batch Mode
# ────────────────────────────────────────────────────────────────────────────
def convert_from_bytes(
    file_bytes: bytes,
    original_filename: str,
    output_dir: str = None,
    sample_rate: int = 16000,
    normalize: bool = True,
) -> str | None:
    """
    Nhận bytes từ Streamlit upload, convert sang WAV 16kHz mono.

    Cách dùng trong Streamlit:
        uploaded = st.file_uploader("Chọn file", type=["mp3","wav","mp4","mkv"])
        if uploaded:
            wav_path = convert_from_bytes(uploaded.read(), uploaded.name)
            if wav_path:
                st.success(f"Convert xong: {wav_path}")

    Args:
        file_bytes        : nội dung file (uploaded.read())
        original_filename : tên file gốc — dùng để giữ đúng extension
                            để ffmpeg nhận dạng codec (.mp3 / .mp4 / ...)
        output_dir        : thư mục lưu WAV output
                            None = lưu cùng thư mục tempfile hệ thống
        sample_rate       : mặc định 16000 Hz

    Returns:
        Đường dẫn file WAV nếu thành công.
        None nếu convert thất bại.
    """
    ext  = Path(original_filename).suffix.lower()  # .mp3 / .wav / .mp4 ...
    stem = Path(original_filename).stem            # tên file không có extension

    # Lưu bytes vào file tạm — giữ đúng extension để ffmpeg đọc đúng codec
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_input_path = tmp.name

    # Xác định đường dẫn file WAV output
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        wav_path = str(Path(output_dir) / f"{stem}_16k.wav")
    else:
        wav_path = tmp_input_path.replace(ext, "_16k.wav")

    success = convert_to_wav(tmp_input_path, wav_path, sample_rate, normalize)

    # Dọn file tạm đầu vào
    try:
        os.unlink(tmp_input_path)
    except OSError:
        pass

    if success:
        return wav_path

    # Dọn file output nếu convert thất bại
    try:
        os.unlink(wav_path)
    except OSError:
        pass

    return None


# ────────────────────────────────────────────────────────────────────────────
# Hàm 3: đọc thông tin WAV để hiển thị lên UI
# ────────────────────────────────────────────────────────────────────────────
def get_audio_info(wav_path: str) -> dict:
    """
    Đọc metadata của file WAV sau khi convert.
    Dùng để hiển thị thông tin trên Streamlit trước khi xử lý AI.

    Returns dict:
        duration_sec  — tổng giây (float)
        duration_str  — "MM:SS" để hiển thị trên UI
        sample_rate   — Hz (phải là 16000 sau convert)
        channels      — số kênh (phải là 1 sau convert)
        frames        — tổng số frame
        ok            — False nếu đọc file thất bại
    """
    try:
        with wave.open(wav_path, "rb") as wf:
            frames      = wf.getnframes()
            sample_rate = wf.getframerate()
            channels    = wf.getnchannels()
            duration    = frames / sample_rate if sample_rate > 0 else 0.0

        minutes = int(duration // 60)
        seconds = int(duration % 60)

        return {
            "duration_sec" : round(duration, 2),
            "duration_str" : f"{minutes:02d}:{seconds:02d}",
            "sample_rate"  : sample_rate,
            "channels"     : channels,
            "frames"       : frames,
            "ok"           : True,
        }

    except Exception as e:
        print(f"[converter] get_audio_info lỗi: {e}")
        return {
            "duration_sec" : 0.0,
            "duration_str" : "00:00",
            "sample_rate"  : 0,
            "channels"     : 0,
            "frames"       : 0,
            "ok"           : False,
        }


# ────────────────────────────────────────────────────────────────────────────
# TEST — chạy: python core/converter.py <input_file>
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("=" * 55)
    print("  TEST: core/converter.py")
    print("=" * 55)

    # Bước 1: kiểm tra ffmpeg có chạy được không
    print(f"\n[1] Kiểm tra ffmpeg...")
    print(f"    FFMPEG_PATH = {FFMPEG_PATH}")
    check = subprocess.run([FFMPEG_PATH, "-version"], capture_output=True)
    if check.returncode == 0:
        version_line = check.stdout.decode(errors="replace").splitlines()[0]
        print(f"    ✅ {version_line}")
    else:
        print("    ❌ ffmpeg không chạy được!")
        print("       Kiểm tra FFMPEG_PATH trong file .env")
        sys.exit(1)

    # Bước 2: convert file nếu có truyền tham số
    if len(sys.argv) < 2:
        print("\n[2] Không có file test — bỏ qua bước convert")
        print("    Dùng lệnh: python core/converter.py <đường_dẫn_file_audio>")
        print("\n✅ Converter sẵn sàng sử dụng\n")
        sys.exit(0)

    input_file = sys.argv[1]
    print(f"\n[2] Convert file: {input_file}")

    if not Path(input_file).exists():
        print(f"    ❌ File không tồn tại: {input_file}")
        sys.exit(1)

    output_file = str(Path(input_file).with_name(
        Path(input_file).stem + "_16k.wav"
    ))

    ok = convert_to_wav(input_file, output_file)
    if not ok:
        print("    ❌ Convert thất bại — kiểm tra log ffmpeg ở trên")
        sys.exit(1)

    print(f"    ✅ Convert thành công → {output_file}")

    # Bước 3: đọc và in thông tin file vừa tạo
    print(f"\n[3] Đọc thông tin file WAV...")
    info = get_audio_info(output_file)
    if info["ok"]:
        print(f"    ✅ Duration   : {info['duration_str']}  ({info['duration_sec']}s)")
        print(f"       Sample rate: {info['sample_rate']} Hz")
        print(f"       Channels   : {info['channels']}")
        print(f"       Frames     : {info['frames']}")
    else:
        print("    ❌ Không đọc được thông tin file")
        sys.exit(1)

    print("\n✅ Tất cả test đều qua — converter.py sẵn sàng\n")