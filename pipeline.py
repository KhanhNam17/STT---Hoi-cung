# pipeline.py
# Gộp Whisper transcription + pyannote diarization
# Output: transcript có tên speaker theo từng đoạn

import json
import numpy as np
import librosa
from pathlib import Path
from dataclasses import dataclass, asdict

from diarizer import SpeakerSegment, get_speaker_stats


@dataclass
class MeetingTurn:
    """Một lượt nói hoàn chỉnh trong meeting."""
    speaker:    str
    start:      float
    end:        float
    text:       str
    duration:   float


def assign_text_to_speakers(
    segments:    list[SpeakerSegment],
    whisper_result: dict,
) -> list[SpeakerSegment]:
    """
    Gán text từ Whisper vào từng speaker segment của pyannote.

    Chiến lược: dùng word-level timestamps của Whisper (nếu có),
    hoặc dùng segment-level overlap nếu không có word timestamps.

    Parameters
    ----------
    segments       : output của diarizer.diarize_file()
    whisper_result : output của model.transcribe() với word_timestamps=True
                     Cần có key "segments" chứa list các segment với "words"

    Returns
    -------
    Cập nhật trường text trong mỗi SpeakerSegment
    """
    # Lấy tất cả words với timestamp
    all_words = []
    for seg in whisper_result.get("segments", []):
        for w in seg.get("words", []):
            all_words.append({
                "word" : w["word"].strip(),
                "start": w["start"],
                "end"  : w["end"],
            })

    if not all_words:
        # Fallback: dùng segment-level overlap nếu không có word timestamps
        return _assign_by_segment_overlap(segments, whisper_result)

    # Với mỗi speaker segment, lấy các words nằm trong khoảng thời gian đó
    for spk_seg in segments:
        words_in_seg = [
            w["word"] for w in all_words
            if w["start"] >= spk_seg.start - 0.1  # tolerance 100ms
            and w["end"] <= spk_seg.end + 0.1
        ]
        spk_seg.text = " ".join(words_in_seg).strip()

    return segments


def _assign_by_segment_overlap(
    speaker_segs: list[SpeakerSegment],
    whisper_result: dict,
) -> list[SpeakerSegment]:
    """
    Fallback: gán text dựa trên overlap lớn nhất giữa
    whisper segment và speaker segment.
    """
    whisper_segs = whisper_result.get("segments", [])

    for spk_seg in speaker_segs:
        best_text = ""
        best_overlap = 0.0

        for w_seg in whisper_segs:
            # Tính overlap
            overlap_start = max(spk_seg.start, w_seg["start"])
            overlap_end   = min(spk_seg.end,   w_seg["end"])
            overlap       = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_text    = w_seg["text"].strip()

        spk_seg.text = best_text

    return speaker_segs


def merge_consecutive_turns(
    segments: list[SpeakerSegment],
    gap_threshold: float = 1.5,  # giây — ghép các turn cùng speaker nếu khoảng cách < threshold
) -> list[MeetingTurn]:
    """
    Gộp các segment liên tiếp của cùng 1 speaker thành 1 MeetingTurn.
    Tránh transcript bị chặt quá vụn.

    Parameters
    ----------
    gap_threshold : khoảng cách tối đa (giây) giữa 2 segment để gộp
    """
    if not segments:
        return []

    turns = []
    current = MeetingTurn(
        speaker  = segments[0].speaker,
        start    = segments[0].start,
        end      = segments[0].end,
        text     = segments[0].text,
        duration = round(segments[0].end - segments[0].start, 3),
    )

    for seg in segments[1:]:
        same_speaker = seg.speaker == current.speaker
        small_gap    = seg.start - current.end <= gap_threshold

        if same_speaker and small_gap:
            # Gộp vào turn hiện tại
            current.end  = seg.end
            current.text = (current.text + " " + seg.text).strip()
            current.duration = round(current.end - current.start, 3)
        else:
            turns.append(current)
            current = MeetingTurn(
                speaker  = seg.speaker,
                start    = seg.start,
                end      = seg.end,
                text     = seg.text,
                duration = round(seg.end - seg.start, 3),
            )

    turns.append(current)
    return turns


def run_meeting_pipeline(
    wav_path:      str,
    transcriber_app,           # Qualcomm HfWhisperApp hoặc openai whisper model
    diarizer_pipeline,         # pyannote Pipeline
    use_qualcomm:  bool = True,
    num_speakers:  int  = None,
    speaker_names: dict = None,  # {"SPEAKER_00": "An", "SPEAKER_01": "Bình"}
    output_dir:    str  = "results/meetings",
) -> dict:
    """
    Pipeline hoàn chỉnh cho 1 file meeting audio.

    Bước 1: Diarization (ai nói lúc nào)
    Bước 2: Transcription toàn bộ audio
    Bước 3: Gán text vào từng speaker
    Bước 4: Gộp turn liên tiếp
    Bước 5: Lưu kết quả

    Returns
    -------
    dict chứa turns, stats, và metadata
    """
    import time
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  MEETING PIPELINE: {Path(wav_path).name}")
    print(f"{'='*60}")

    total_start = time.perf_counter()

    # ── Step 1: Diarization ───────────────────────────────────────────────────
    from diarizer import diarize_file, rename_speakers
    speaker_segs = diarize_file(
        diarizer_pipeline, wav_path, num_speakers=num_speakers
    )

    if speaker_names:
        speaker_segs = rename_speakers(speaker_segs, speaker_names)

    # ── Step 2: Transcription ─────────────────────────────────────────────────
    print("📝 Transcribing full audio...")
    t_trans = time.perf_counter()

    audio, sr = librosa.load(wav_path, sr=16000, mono=True)

    if use_qualcomm:
        # Qualcomm HfWhisperApp
        full_text = transcriber_app.transcribe(audio, audio_sample_rate=sr)
        whisper_result = {"text": full_text, "segments": []}
        # Note: để có word-level timestamps với Qualcomm cần custom thêm
        # Hiện tại dùng full text và phân chia theo overlap
    else:
        # OpenAI whisper — hỗ trợ word_timestamps
        whisper_result = transcriber_app.transcribe(
            wav_path,
            language="vi",
            task="transcribe",
            word_timestamps=True,   # ← quan trọng để align chính xác
            verbose=False,
        )

    trans_time = round(time.perf_counter() - t_trans, 2)
    print(f"   ✅ Transcription xong ({trans_time}s)")

    # ── Step 3: Gán text vào speaker ─────────────────────────────────────────
    print("🔗 Aligning text to speakers...")
    speaker_segs = assign_text_to_speakers(speaker_segs, whisper_result)

    # ── Step 4: Gộp turns ────────────────────────────────────────────────────
    turns = merge_consecutive_turns(speaker_segs, gap_threshold=1.5)

    # ── Step 5: Stats và lưu kết quả ─────────────────────────────────────────
    stats   = get_speaker_stats(speaker_segs)
    elapsed = round(time.perf_counter() - total_start, 2)

    result = {
        "file"          : str(wav_path),
        "total_time_s"  : elapsed,
        "num_turns"     : len(turns),
        "speakers"      : stats,
        "transcript"    : [asdict(t) for t in turns],
        "full_text"     : whisper_result.get("text", ""),
    }

    # Lưu JSON
    stem    = Path(wav_path).stem
    out_json = Path(output_dir) / f"{stem}_meeting.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Lưu TXT dễ đọc
    out_txt = Path(output_dir) / f"{stem}_transcript.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"MEETING TRANSCRIPT: {Path(wav_path).name}\n")
        f.write("=" * 60 + "\n\n")
        for t in turns:
            timestamp = f"[{_fmt_time(t.start)} → {_fmt_time(t.end)}]"
            f.write(f"{t.speaker} {timestamp}\n")
            f.write(f"{t.text}\n\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("SPEAKER STATS:\n")
        for spk, s in stats.items():
            f.write(f"  {spk}: {s['duration']:.1f}s ({s['percent']}%) — {s['turns']} turns\n")

    print(f"\n✅ Pipeline xong ({elapsed}s)")
    print(f"   {len(turns)} turns | {len(stats)} speakers")
    print(f"   Transcript: {out_txt}")
    print(f"   JSON      : {out_json}")

    _print_transcript_preview(turns, max_turns=5)

    return result


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _print_transcript_preview(turns: list[MeetingTurn], max_turns: int = 5):
    print(f"\n📋 Preview (first {max_turns} turns):")
    print("-" * 50)
    for t in turns[:max_turns]:
        ts  = f"[{_fmt_time(t.start)}]"
        txt = t.text[:70] + "..." if len(t.text) > 70 else t.text
        print(f"  {t.speaker} {ts}: {txt}")
    print("-" * 50)
