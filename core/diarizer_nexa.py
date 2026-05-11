import os
import re
import subprocess
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Import bộ khung chuẩn từ file gốc của bạn
from core.diarizer import SpeakerSegment, postprocess_segments, validate_wav

NEXA_CLI_PATH = os.getenv("NEXA_CLI_PATH", "nexa")
NEXA_DIAR_MODEL = os.getenv("NEXA_DIAR_MODEL", "Pyannote-NPU")

def load_diarizer_nexa():
    """Kiểm tra Nexa CLI đã sẵn sàng chưa"""
    print(f"⏳ Kích hoạt bộ chuyển nối Nexa AI ({NEXA_DIAR_MODEL})...")
    try:
        r = subprocess.run([NEXA_CLI_PATH, "-h"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            print("✅ Nexa AI Engine sẵn sàng!")
            return {"backend": "nexa"}
        else:
            raise RuntimeError(f"Lỗi khởi động Nexa:\n{r.stderr}")
    except FileNotFoundError:
        raise RuntimeError("Không tìm thấy lệnh 'nexa'. Vui lòng kiểm tra lại môi trường!")

def diarize_file_nexa(pipeline, wav_path: str, num_speakers=None, min_duration=0.5, merge_gap=2.5) -> list:
    valid, reason = validate_wav(wav_path)
    if not valid:
        raise ValueError(f"File không hợp lệ: {reason}")

    # 1. Lấy đường dẫn tuyệt đối
    abs_wav_path = str(Path(wav_path).resolve())

    print(f"🎙️ Diarizing (Nexa NPU): {Path(abs_wav_path).name}")
    t0 = time.perf_counter()

    # 2. Gọi lệnh với cờ -i
    command = [NEXA_CLI_PATH, "infer", NEXA_DIAR_MODEL, "-i", abs_wav_path]
    print(f"   [Nexa] CMD: {' '.join(command)}")
    print("   [Nexa] Đang giao việc cho NPU xử lý...")
    
    process = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace')
    
    output_text = process.stdout + "\n" + process.stderr
    
    # 3. Biểu thức Regex đã được tối ưu để bỏ qua rác
    pattern = r"\[\d+\].*?([\d.]+)s.*?([\d.]+)s.*?(SPEAKER_\d+)"
    raw_segments = []

    # 4. Đọc log và xóa mã màu ANSI tàng hình
    for line in output_text.splitlines():
        line = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', line)
        line = line.strip()
        if not line:
            continue
            
        match = re.search(pattern, line)
        if match:
            start = float(match.group(1))
            end = float(match.group(2))
            speaker = match.group(3)
            raw_segments.append(SpeakerSegment(speaker=speaker, start=start, end=end))

    if not raw_segments:
        print("\n⚠️ [CẢNH BÁO] NPU chạy xong nhưng không bắt được kết quả! Đây là log thô:")
        print("-" * 60)
        print(output_text.strip()[:1000]) # In 1000 ký tự đầu để debug
        print("-" * 60)
        return []

    elapsed_raw = round(time.perf_counter() - t0, 2)
    print(f"   [Nexa] Hoàn thành phân tích {len(raw_segments)} đoạn thô | Thời gian: {elapsed_raw}s")

    # 5. Đưa qua máy lọc dọn rác của file gốc
    segments = postprocess_segments(raw_segments, min_duration=min_duration, merge_gap=merge_gap)
    print(f"   [Postprocess] Sau lọc: {len({s.speaker for s in segments})} người nói | {len(segments)} đoạn hội thoại chuẩn")
    
    return segments

# ─────────────────────────────────────────────────────────────────────────────
# ĐOẠN DƯỚI ĐÂY DÙNG ĐỂ TEST ĐỘC LẬP TỐC ĐỘ CỦA FILE NÀY
# Chạy lệnh: python core/diarizer_nexa.py podcast_HAT_16k.wav
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import wave
    
    print("=" * 60)
    print("  TEST: diarizer_nexa.py (Nexa AI - Qualcomm NPU)")
    print("=" * 60)
    
    try:
        engine = load_diarizer_nexa()
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)
        
    wav_file = sys.argv[1] if len(sys.argv) > 1 else "podcast_HAT_16k.wav"
    
    if not Path(wav_file).exists():
        print(f"❌ Không tìm thấy file: {wav_file}")
        sys.exit(1)
    
    # 1. Bắt đầu đo thời gian
    t_start = time.perf_counter()
    
    # 2. Chạy hàm xử lý
    segs = diarize_file_nexa(engine, wav_file)
    
    # 3. Kết thúc đo thời gian
    t_total = round(time.perf_counter() - t_start, 2)
    
    # 4. Lấy độ dài file audio để tính RTF
    try:
        with wave.open(wav_file, "rb") as wf:
            dur = wf.getnframes() / wf.getframerate()
    except Exception:
        dur = 0
        
    rtf = t_total / dur if dur > 0 else 0
    
    # 5. In báo cáo mượt mà
    print(f"\n[Kết quả] ({len({s.speaker for s in segs})} người nói | {len(segs)} đoạn):")
    for i, s in enumerate(segs[:15]):
        bar = "█" * min(40, int((s.end - s.start) * 2)) # Vẽ thanh bar độ dài
        print(f"  [{i+1:02d}] {s.speaker:12s} | {s.start:6.2f}s → {s.end:6.2f}s | {bar}")
        
    if len(segs) > 15:
        print(f"  ... và {len(segs)-15} đoạn nữa")
    
    print(f"\n[Hiệu năng NPU]:")
    print(f"  Audio : {dur:.1f}s")
    print(f"  Xử lý : {t_total}s")
    if rtf > 0:
        print(f"  RTF   : {rtf:.3f}x (nhanh gấp {1/rtf:.1f} lần realtime)")
    print("✅ Test xong!\n")