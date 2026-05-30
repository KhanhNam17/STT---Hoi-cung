# test_pipeline.py
#
# Test toàn bộ pipeline core end-to-end với 1 file audio thật.
# Chạy script này trước khi đưa vào Streamlit để xác nhận mọi thứ hoạt động.
#
# Cách chạy:
#   python test_pipeline.py <đường_dẫn_file_audio>
#
# Ví dụ:
#   python test_pipeline.py test_audio/interview_sample.mp3
#   python test_pipeline.py C:\Audio\buoi_hoicung.wav
#
# Script sẽ chạy tuần tự 4 bước:
#   Bước 1: convert_from_bytes  → WAV 16kHz mono
#   Bước 2: transcribe_file     → text
#   Bước 3: diarize_file        → speaker segments
#   Bước 4: align + merge       → AlignedTurn[]
#
# Nếu tất cả pass → app Streamlit sẽ chạy được với file này.

import sys
import time
from pathlib import Path

# ── Import toàn bộ core ──────────────────────────────────────────────────────
from converter   import convert_to_wav, get_audio_info
from transcriber import load_model, transcribe_file
from diarizer    import load_diarizer, diarize_file, validate_wav, get_speaker_stats
from aligner     import align, merge_consecutive, rename_turns, format_timestamp


def separator(title: str):
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


def run_pipeline(input_file: str, num_speakers: int = None):
    """
    Chạy toàn bộ pipeline với 1 file audio bất kỳ.
    In kết quả chi tiết ở từng bước.
    """
    print("=" * 55)
    print("  TEST PIPELINE: converter → transcriber → diarizer → aligner")
    print("=" * 55)
    print(f"  Input: {input_file}")
    print(f"  Speakers: {'tự phát hiện' if not num_speakers else num_speakers}")

    t_total = time.perf_counter()

    # ── Bước 1: Convert ───────────────────────────────────────────────────────
    separator("Bước 1/4: Convert → WAV 16kHz mono")

    input_path = Path(input_file)
    if not input_path.exists():
        print(f"  ❌ File không tồn tại: {input_file}")
        sys.exit(1)

    wav_path = str(input_path.with_name(input_path.stem + "_test_16k.wav"))

    print(f"  Đang convert: {input_path.name}")
    t0 = time.perf_counter()
    ok = convert_to_wav(input_file, wav_path)
    elapsed = round(time.perf_counter() - t0, 2)

    if not ok:
        print("  ❌ Convert thất bại — kiểm tra FFMPEG_PATH trong .env")
        sys.exit(1)

    info = get_audio_info(wav_path)
    print(f"  ✅ Convert xong ({elapsed}s)")
    print(f"     Duration   : {info['duration_str']} ({info['duration_sec']}s)")
    print(f"     Sample rate: {info['sample_rate']} Hz")
    print(f"     Channels   : {info['channels']}")

    # ── Bước 2: Transcribe ───────────────────────────────────────────────────
    separator("Bước 2/4: Transcribe → text")

    print("  Load Qualcomm Whisper (lần đầu ~1-3 phút)...")
    try:
        _, app = load_model()
    except Exception as e:
        print(f"  ❌ Load model thất bại: {e}")
        sys.exit(1)

    print(f"  Đang transcribe: {Path(wav_path).name}")
    t0 = time.perf_counter()
    try:
        result = transcribe_file(app, wav_path)
    except Exception as e:
        print(f"  ❌ Transcribe thất bại: {e}")
        sys.exit(1)

    print(f"  ✅ Transcribe xong")
    print(f"     Latency    : {result['latency']}s")
    print(f"     RTF        : {result['rtf']}  {'⚡ nhanh hơn real-time' if result['rtf'] and result['rtf'] < 1.0 else ''}")
    print(f"     Text ({len(result['text'].split())} từ):")
    # In text xuống dòng nếu dài
    words = result["text"].split()
    lines = [" ".join(words[i:i+12]) for i in range(0, len(words), 12)]
    for line in lines:
        print(f"       {line}")

    # ── Bước 3: Diarize ───────────────────────────────────────────────────────
    separator("Bước 3/4: Diarize → speaker segments")

    # Validate trước
    valid, reason = validate_wav(wav_path)
    if not valid:
        print(f"  ❌ File WAV không hợp lệ: {reason}")
        sys.exit(1)

    print("  Load pyannote diarizer (lần đầu ~vài phút)...")
    try:
        pipeline = load_diarizer()
    except Exception as e:
        print(f"  ❌ Load diarizer thất bại: {e}")
        sys.exit(1)

    print(f"  Đang diarize...")
    t0 = time.perf_counter()
    try:
        segments = diarize_file(
            pipeline, wav_path,
            num_speakers=num_speakers,
            min_speakers=1,
            max_speakers=6,
        )
    except Exception as e:
        print(f"  ❌ Diarize thất bại: {e}")
        sys.exit(1)

    stats = get_speaker_stats(segments)
    print(f"  ✅ Diarize xong — {len({s.speaker for s in segments})} người nói, {len(segments)} đoạn")
    for spk, info in stats.items():
        bar = "█" * int(info["percent"] / 5)
        print(f"     {spk:14s} {bar:20s} {info['percent']:5.1f}% | {info['turns']} lượt")

    # ── Bước 4: Align ─────────────────────────────────────────────────────────
    separator("Bước 4/4: Align → transcript hoàn chỉnh")

    turns  = align(segments, result["text"])
    merged = merge_consecutive(turns, gap_limit=1.5)

    print(f"  ✅ Align xong — {len(turns)} turns → sau merge: {len(merged)} turns")
    print()

    # In transcript mẫu
    print("  TRANSCRIPT:")
    print("  " + "─" * 51)
    for t in merged:
        ts      = format_timestamp(t.start)
        ts_end  = format_timestamp(t.end)
        preview = t.text[:60] + ("..." if len(t.text) > 60 else "")
        print(f"  [{ts} → {ts_end}]  {t.speaker}")
        print(f"  {preview}")
        print()

    # ── Tổng kết ─────────────────────────────────────────────────────────────
    t_elapsed = round(time.perf_counter() - t_total, 1)
    separator("KẾT QUẢ TỔNG KẾT")
    audio_info = get_audio_info(wav_path) if Path(wav_path).exists() else {"duration_sec": "?"}
    print(f"  ✅ Pipeline hoàn thành trong {t_elapsed}s")
    print(f"  Audio duration : {audio_info.get('duration_sec', '?')}s")
    print(f"  Số lượt nói    : {len(merged)}")
    print(f"  Số người nói   : {len(stats)}")
    print(f"  Tổng từ        : {len(result['text'].split())}")
    print()
    print("  🎉 Sẵn sàng đưa vào Streamlit!")
    print()

    # Dọn file WAV test
    try:
        Path(wav_path).unlink()
        print(f"  🗑  Đã xoá file test tạm: {Path(wav_path).name}")
    except OSError:
        pass

    return merged


# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cách dùng:")
        print("  python test_pipeline.py <file_audio>")
        print("  python test_pipeline.py <file_audio> <số_người_nói>")
        print()
        print("Ví dụ:")
        print("  python test_pipeline.py sample.mp3")
        print("  python test_pipeline.py interview.wav 2")
        sys.exit(0)

    input_file   = sys.argv[1]
    num_speakers = int(sys.argv[2]) if len(sys.argv) >= 3 else None

    run_pipeline(input_file, num_speakers)