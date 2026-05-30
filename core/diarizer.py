# core/diarizer.py
#
# Mục đích: Phân tách người nói (Speaker Diarization)
#
# Model được hỗ trợ (đọc từ .env DIARIZATION_MODEL):
#   - pyannote/speaker-diarization-community-1  ← mặc định, local, tốt hơn 3.1
#   - pyannote/speaker-diarization-3.1          ← legacy, fallback
#   - pyannote/speaker-diarization-precision-2  ← cloud, cần PYANNOTE_API_KEY
#
# Thay đổi so với phiên bản trước:
#   - Hỗ trợ community-1: output mới (output.speaker_diarization)
#   - Hỗ trợ exclusive_speaker_diarization của community-1
#     → timestamp chính xác hơn, ít overlap hơn, tốt cho alignment
#   - postprocess_segments: merge_gap 1.5 → 2.5s, thêm Bước 4 drop ghost speaker
#   - load_diarizer: tự detect model type để xử lý output đúng cách
#   - precision-2: chạy cloud, dùng PYANNOTE_API_KEY thay vì HF_TOKEN

import os
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import torch
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

# ── Cấu hình đọc từ .env ────────────────────────────────────────────────────
HF_TOKEN          = os.getenv("HF_TOKEN", "")
PYANNOTE_API_KEY  = os.getenv("PYANNOTE_API_KEY", "")   # chỉ dùng cho precision-2

# Model mặc định: community-1 — tốt hơn 3.1 trên mọi benchmark
# Đổi trong .env: DIARIZATION_MODEL=pyannote/speaker-diarization-3.1
DIARIZATION_MODEL = os.getenv(
    "DIARIZATION_MODEL",
    "pyannote/speaker-diarization-community-1"
)

# Detect loại model để xử lý output đúng cách
_IS_COMMUNITY = "community" in DIARIZATION_MODEL
_IS_PRECISION = "precision" in DIARIZATION_MODEL
_IS_LEGACY    = not (_IS_COMMUNITY or _IS_PRECISION)


# ────────────────────────────────────────────────────────────────────────────
# Data class
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class SpeakerSegment:
    """1 đoạn nói của 1 người: speaker label, thời điểm bắt đầu/kết thúc."""
    speaker : str
    start   : float
    end     : float
    text    : str = ""


# ────────────────────────────────────────────────────────────────────────────
# Hàm 1: Load model
# ────────────────────────────────────────────────────────────────────────────
def load_diarizer():
    """
    Load pyannote diarization pipeline theo DIARIZATION_MODEL trong .env.

    community-1 (mặc định):
        - Dùng HF_TOKEN
        - Chạy local, offline được
        - Tốt hơn 3.1 đáng kể (DER thấp hơn ~10–20% trên benchmark)
        - Output: output.speaker_diarization (Annotation)
                  output.exclusive_speaker_diarization (Annotation, MỚI)
                  exclusive: mỗi frame chỉ 1 speaker, không overlap
                  → tốt hơn cho alignment với transcript

    3.1 (legacy):
        - Dùng HF_TOKEN
        - Chạy local
        - Output: Annotation trực tiếp hoặc wrap trong dataclass

    precision-2 (cloud):
        - Dùng PYANNOTE_API_KEY từ dashboard.pyannote.ai
        - Chạy trên pyannoteAI servers — cần internet, không offline
        - Tốt nhất về chất lượng
        - Output: output.speaker_diarization

    Cách dùng trong Streamlit:
        @st.cache_resource
        def get_diarizer():
            return load_diarizer()
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        raise ImportError("Chạy: pip install pyannote.audio")

    # pyannote.audio >= 3.3 dùng kwarg `token=`, còn 3.1.x dùng `use_auth_token=`.
    # Thử lần lượt để tương thích cả 2 version.
    def _from_pretrained(model, tok):
        try:
            return Pipeline.from_pretrained(model, token=tok)
        except TypeError:
            return Pipeline.from_pretrained(model, use_auth_token=tok)

    print(f"⏳ Loading diarization pipeline: {DIARIZATION_MODEL}")

    if _IS_PRECISION:
        # precision-2: dùng pyannoteAI API key, chạy cloud
        if not PYANNOTE_API_KEY:
            raise ValueError(
                "precision-2 cần PYANNOTE_API_KEY — thêm vào .env:\n"
                "PYANNOTE_API_KEY=your_key_from_dashboard.pyannote.ai\n"
                "Hoặc đổi về community-1:\n"
                "DIARIZATION_MODEL=pyannote/speaker-diarization-community-1"
            )
        pipeline = _from_pretrained(DIARIZATION_MODEL, PYANNOTE_API_KEY)
        print("   Mode: Cloud (pyannoteAI) — cần internet")

    else:
        # community-1 hoặc 3.1: dùng HF_TOKEN, chạy local
        if not HF_TOKEN:
            raise ValueError(
                "Thiếu HF_TOKEN — thêm vào file .env:\n"
                "HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx"
            )
        pipeline = _from_pretrained(DIARIZATION_MODEL, HF_TOKEN)
        if torch.cuda.is_available():
            pipeline = pipeline.to(torch.device("cuda"))
            print("   Device: CUDA")
        else:
            print("   Device: CPU")

    # from_pretrained() trả None khi tải thất bại (gated / version không hợp).
    # Bắt sớm để báo lỗi rõ ràng thay vì crash 'NoneType is not callable' lúc chạy.
    if pipeline is None:
        raise ValueError(
            f"Không tải được pipeline '{DIARIZATION_MODEL}'.\n"
            f"  • community-1 cần pyannote.audio >= 3.3 (máy đang có bản cũ hơn).\n"
            f"  • Hoặc chưa chấp nhận điều kiện model trên HuggingFace.\n"
            f"→ KHUYẾN NGHỊ: đổi trong .env:\n"
            f"  DIARIZATION_MODEL=pyannote/speaker-diarization-3.1"
        )

    label = ("community-1" if _IS_COMMUNITY
              else "precision-2 (cloud)" if _IS_PRECISION
              else "3.1 legacy")
    print(f"✅ Diarizer loaded ({label})")
    return pipeline


# ────────────────────────────────────────────────────────────────────────────
# Hàm 2: Kiểm tra file WAV
# ────────────────────────────────────────────────────────────────────────────
def validate_wav(wav_path: str) -> tuple[bool, str]:
    """
    Kiểm tra file WAV hợp lệ trước khi đưa vào diarizer.

    Returns:
        (True, "ok")          — file hợp lệ
        (False, "lý do lỗi") — file có vấn đề
    """
    if not Path(wav_path).exists():
        return False, f"File không tồn tại: {wav_path}"

    try:
        with wave.open(wav_path, "rb") as wf:
            frames      = wf.getnframes()
            sample_rate = wf.getframerate()
            channels    = wf.getnchannels()

        if frames == 0:
            return False, "File rỗng (0 frames)"
        if sample_rate != 16000:
            return False, f"Sample rate không đúng: {sample_rate}Hz (cần 16000Hz)"
        # Stereo CHẤP NHẬN ĐƯỢC: load_audio() tự downmix sang mono (trung bình kênh).
        if channels not in (1, 2):
            return False, f"Số kênh không hỗ trợ: {channels} (cần mono hoặc stereo)"
        if channels == 2:
            print("   [diarizer] File stereo → sẽ tự downmix sang mono khi load.")

        duration = frames / sample_rate
        if duration < 0.5:
            return False, f"Audio quá ngắn: {duration:.2f}s (cần >= 0.5s)"

        return True, "ok"

    except Exception as e:
        return False, f"Lỗi đọc file WAV: {e}"


# ────────────────────────────────────────────────────────────────────────────
# Hàm 3: Load audio thành tensor
# ────────────────────────────────────────────────────────────────────────────
def load_audio(wav_path: str):
    """Đọc WAV bằng soundfile → torch tensor (1, time).

    Tự DOWNMIX stereo/đa kênh → mono bằng cách trung bình các kênh.
    sf.read trả (frames,) cho mono hoặc (frames, channels) cho đa kênh.
    """
    waveform, sample_rate = sf.read(wav_path)
    waveform = torch.tensor(waveform).float()

    if waveform.dim() == 2:                # (frames, channels) → mono
        waveform = waveform.mean(dim=1)

    waveform = waveform.unsqueeze(0)       # (1, time)
    return waveform, sample_rate


# ────────────────────────────────────────────────────────────────────────────
# Hàm 4: Trích xuất Annotation từ output pipeline
# ────────────────────────────────────────────────────────────────────────────
def _extract_annotation(result, use_exclusive: bool = False):
    """
    Trích xuất Annotation từ output pipeline — xử lý mọi dạng output.

    community-1 trả về object có:
        .speaker_diarization           — Annotation tiêu chuẩn (có overlap)
        .exclusive_speaker_diarization — Annotation không overlap (MỚI)
            Mỗi frame chỉ gán cho 1 speaker.
            Tốt hơn cho alignment với transcript Whisper.

    3.1 legacy trả về:
        Annotation trực tiếp hoặc wrap trong dataclass/dict

    Args:
        use_exclusive : True → ưu tiên exclusive_speaker_diarization
                        Chỉ có tác dụng với community-1 và precision-2
    """
    # community-1 / precision-2: ưu tiên exclusive nếu yêu cầu
    if use_exclusive and hasattr(result, "exclusive_speaker_diarization"):
        annotation = result.exclusive_speaker_diarization
        if hasattr(annotation, "itertracks"):
            print("   [diarizer] Dùng exclusive_speaker_diarization")
            return annotation

    # community-1 / precision-2: standard speaker_diarization
    if hasattr(result, "speaker_diarization"):
        annotation = result.speaker_diarization
        if hasattr(annotation, "itertracks"):
            return annotation

    # 3.1 legacy: Annotation trực tiếp
    if hasattr(result, "itertracks"):
        return result

    # 3.1 legacy: wrap trong dict
    if isinstance(result, dict):
        for key in ("diarization", "speaker_diarization", "annotation"):
            if key in result and hasattr(result[key], "itertracks"):
                return result[key]

    # Fallback: duyệt toàn bộ attribute
    for attr in dir(result):
        if attr.startswith("_"):
            continue
        try:
            candidate = getattr(result, attr)
            if hasattr(candidate, "itertracks") and callable(
                getattr(candidate, "itertracks")
            ):
                print(f"   [diarizer] Tìm thấy Annotation tại .{attr}")
                return candidate
        except Exception:
            continue

    raise RuntimeError(
        "Không thể trích xuất Annotation từ kết quả pyannote.\n"
        f"Type: {type(result)}\n"
        f"Attributes: {[a for a in dir(result) if not a.startswith('_')]}"
    )


# ────────────────────────────────────────────────────────────────────────────
# Hàm 5: Diarize 1 file
# ────────────────────────────────────────────────────────────────────────────
def diarize_file(
    pipeline,
    wav_path      : str,
    num_speakers  : int   = None,
    min_speakers  : int   = 1,
    max_speakers  : int   = 10,
    min_duration  : float = 0.5,
    merge_gap     : float = 2.5,
    use_exclusive : bool  = True,
) -> list[SpeakerSegment]:
    """
    Phân tách người nói trong file WAV.

    Args:
        pipeline      : từ load_diarizer()
        wav_path      : file WAV 16kHz mono
        num_speakers  : số người nói nếu biết (hỏi cung thường = 2)
        min_speakers  : giới hạn dưới khi tự detect
        max_speakers  : giới hạn trên khi tự detect
        min_duration  : bỏ segment < ngưỡng này (giây). Mặc định 0.5s.
        merge_gap     : gộp cùng speaker nếu gap < ngưỡng (giây). Mặc định 2.5s.
                        Tăng lên 4s nếu người nói hay dừng lâu giữa câu.
        use_exclusive : dùng exclusive_speaker_diarization của community-1.
                        True: timestamp sạch hơn, ít overlap, tốt cho alignment.
                        False: giữ overlap — dùng khi cần phân tích chi tiết hơn.

    Returns:
        Danh sách SpeakerSegment đã qua postprocess.
    """
    valid, reason = validate_wav(wav_path)
    if not valid:
        raise ValueError(f"[diarizer] File không hợp lệ: {reason}")

    print(f"🎙️  Diarizing: {Path(wav_path).name}")
    t0 = time.perf_counter()

    waveform, sample_rate = load_audio(wav_path)

    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
    else:
        kwargs["min_speakers"] = min_speakers
        kwargs["max_speakers"] = max_speakers

    result = pipeline(
        {"waveform": waveform, "sample_rate": sample_rate},
        **kwargs,
    )

    annotation = _extract_annotation(result, use_exclusive=use_exclusive)

    segments = [
        SpeakerSegment(
            speaker = speaker,
            start   = round(turn.start, 3),
            end     = round(turn.end,   3),
        )
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]

    elapsed        = round(time.perf_counter() - t0, 2)
    speakers_found = len({s.speaker for s in segments})
    print(f"   Raw: {speakers_found} người nói | {len(segments)} đoạn | {elapsed}s")

    segments = postprocess_segments(
        segments,
        min_duration = min_duration,
        merge_gap    = merge_gap,
    )

    print(f"   ✅ Sau lọc: {len({s.speaker for s in segments})} người nói "
          f"| {len(segments)} đoạn")
    return segments


# ────────────────────────────────────────────────────────────────────────────
# Hàm 6: Hậu xử lý segment
# ────────────────────────────────────────────────────────────────────────────
def postprocess_segments(
    segments          : list,
    min_duration      : float = 0.5,
    merge_gap         : float = 2.5,
    smooth_window     : int   = 3,
    min_speaker_ratio : float = 0.08,
) -> list:
    """
    Làm sạch kết quả diarization — 4 bước theo thứ tự.

    Bước 1 — DROP segment quá ngắn (< min_duration)
        Lọc tiếng hít thở (~0.2s), ừ đơn (~0.3s), động nền ngắn.
        Giữ lại câu ngắn hợp lệ: "Có." "Đúng." (~0.5–0.8s).

    Bước 2 — MERGE segment liên tiếp CÙNG speaker (gap < merge_gap)
        merge_gap 2.5s thay vì 1.5s: gộp được đoạn ngắt do ngập ngừng dài,
        hít thở giữa câu, pyannote tách nhầm 1 câu thành nhiều segment.

    Bước 3 — SMOOTH nhãn bị gán nhầm (pattern A→B→A)
        Segment B ngắn kẹp giữa A→A → đổi B thành A.
        Threshold động: segment càng ngắn càng dễ bị flip.

    Bước 4 — DROP ghost speaker (THÊM MỚI)
        Speaker chiếm < min_speaker_ratio tổng thời gian = speaker ảo từ nhiễu.
        → Gộp vào speaker liền kề gần nhất thay vì bỏ hẳn (giữ timestamp).
        → Merge lần 3 sau khi gán lại để gộp cặp mới cùng speaker.

    Args:
        min_duration      : drop segment ngắn hơn (giây). Mặc định 0.5s.
        merge_gap         : merge cùng speaker (giây). Mặc định 2.5s.
        min_speaker_ratio : ngưỡng ghost speaker. Mặc định 8%.
                            Tăng lên 0.12 nếu file có nhiều người nói phụ ngắn.
                            Giảm về 0.04 nếu cần giữ người nói ít lượt hơn.
    """
    if not segments:
        return []

    original_count = len(segments)

    # ── Bước 1: DROP segment quá ngắn ───────────────────────────────────────
    filtered = [s for s in segments if (s.end - s.start) >= min_duration]
    dropped  = original_count - len(filtered)
    if dropped > 0:
        print(f"   [postprocess] Bỏ {dropped} segment < {min_duration}s "
              f"({original_count} → {len(filtered)})")
    if not filtered:
        return []

    # ── Bước 2: MERGE segment liên tiếp cùng speaker ────────────────────────
    merged      = [filtered[0]]
    merge_count = 0

    for current in filtered[1:]:
        prev = merged[-1]
        gap  = current.start - prev.end

        if current.speaker == prev.speaker and gap <= merge_gap:
            merged[-1] = SpeakerSegment(
                speaker = prev.speaker,
                start   = prev.start,
                end     = current.end,
                text    = prev.text,
            )
            merge_count += 1
        else:
            merged.append(current)

    if merge_count > 0:
        print(f"   [postprocess] Gộp {merge_count} cặp cùng speaker "
              f"({len(filtered)} → {len(merged)})")

    # ── Bước 3: SMOOTH nhãn gán nhầm — pattern A→B→A ────────────────────────
    smoothed   = list(merged)
    flip_count = 0

    for i in range(1, len(smoothed) - 1):
        prev_spk = smoothed[i - 1].speaker
        curr_spk = smoothed[i].speaker
        next_spk = smoothed[i + 1].speaker
        curr_dur = smoothed[i].end - smoothed[i].start

        smooth_threshold = min(merge_gap, max(0.8, curr_dur * 1.5))

        if prev_spk == next_spk and curr_spk != prev_spk and curr_dur < smooth_threshold:
            smoothed[i] = SpeakerSegment(
                speaker = prev_spk,
                start   = smoothed[i].start,
                end     = smoothed[i].end,
                text    = smoothed[i].text,
            )
            flip_count += 1

    if flip_count > 0:
        print(f"   [postprocess] Sửa {flip_count} nhãn gán nhầm (A→B→A pattern)")

    # Merge lần 2 sau smooth
    final = [smoothed[0]]
    for current in smoothed[1:]:
        prev = final[-1]
        gap  = current.start - prev.end
        if current.speaker == prev.speaker and gap <= merge_gap:
            final[-1] = SpeakerSegment(
                speaker = prev.speaker,
                start   = prev.start,
                end     = current.end,
                text    = prev.text,
            )
        else:
            final.append(current)

    # ── Bước 4: DROP ghost speaker ───────────────────────────────────────────
    total_dur   = sum(s.end - s.start for s in final)
    speaker_dur = {}
    for s in final:
        speaker_dur[s.speaker] = speaker_dur.get(s.speaker, 0) + (s.end - s.start)

    ghost_speakers = {
        spk for spk, dur in speaker_dur.items()
        if total_dur > 0 and dur / total_dur < min_speaker_ratio
    }

    if ghost_speakers:
        print(f"   [postprocess] Ghost speakers: {ghost_speakers} "
              f"(< {min_speaker_ratio*100:.0f}% thời gian) — đang gộp vào speaker liền kề")

        cleaned = []
        for i, seg in enumerate(final):
            if seg.speaker not in ghost_speakers:
                cleaned.append(seg)
                continue

            # Tìm speaker liền kề không phải ghost — ưu tiên trước, fallback sau
            neighbor = None
            for j in range(i - 1, -1, -1):
                if final[j].speaker not in ghost_speakers:
                    neighbor = final[j].speaker
                    break
            if neighbor is None:
                for j in range(i + 1, len(final)):
                    if final[j].speaker not in ghost_speakers:
                        neighbor = final[j].speaker
                        break

            if neighbor:
                cleaned.append(SpeakerSegment(
                    speaker = neighbor,
                    start   = seg.start,
                    end     = seg.end,
                    text    = seg.text,
                ))
            # Không tìm được neighbor → bỏ segment (edge case hiếm)

        # Merge lần 3 sau drop ghost
        if cleaned:
            final = [cleaned[0]]
            for current in cleaned[1:]:
                prev = final[-1]
                gap  = current.start - prev.end
                if current.speaker == prev.speaker and gap <= merge_gap:
                    final[-1] = SpeakerSegment(
                        speaker = prev.speaker,
                        start   = prev.start,
                        end     = current.end,
                        text    = prev.text,
                    )
                else:
                    final.append(current)
        else:
            final = []

        print(f"   [postprocess] Sau drop ghost: {len(final)} segments")

    print(f"   [postprocess] Kết quả: {original_count} → {len(final)} segments")
    return final


# ────────────────────────────────────────────────────────────────────────────
# Hàm 7: Gộp segment thành chunk tối ưu cho Whisper
# ────────────────────────────────────────────────────────────────────────────
def merge_for_transcription(
    segments : list[SpeakerSegment],
    max_gap  : float = 3.0,
) -> list[SpeakerSegment]:
    """
    Gộp segment CÙNG SPEAKER liên tiếp thành chunk lớn hơn cho Whisper.

    KHÁC postprocess_segments():
        postprocess: lọc nhiễu, gap 2.5s → output để display/stats
        merge_for_transcription: tạo chunk ngữ cảnh, gap 3.0s → transcribe

    Chỉ gộp cùng speaker — không gộp qua ranh giới speaker.
    Không cap max_duration — đoạn dài vẫn nguyên, Whisper tự sliding window.
    """
    if not segments:
        return []

    chunks = [SpeakerSegment(
        speaker = segments[0].speaker,
        start   = segments[0].start,
        end     = segments[0].end,
    )]

    for seg in segments[1:]:
        prev = chunks[-1]
        gap  = seg.start - prev.end

        if seg.speaker == prev.speaker and gap <= max_gap:
            chunks[-1] = SpeakerSegment(
                speaker = prev.speaker,
                start   = prev.start,
                end     = seg.end,
            )
        else:
            chunks.append(SpeakerSegment(
                speaker = seg.speaker,
                start   = seg.start,
                end     = seg.end,
            ))

    avg_dur = sum(c.end - c.start for c in chunks) / len(chunks)
    print(f"   [merge_for_transcription] {len(segments)} segments → "
          f"{len(chunks)} chunks | avg {avg_dur:.1f}s/chunk")
    return chunks


# ────────────────────────────────────────────────────────────────────────────
# Hàm 8: Gán tên thật cho speaker
# ────────────────────────────────────────────────────────────────────────────
def rename_speakers(
    segments : list[SpeakerSegment],
    name_map : dict,
) -> list[SpeakerSegment]:
    """Áp dụng name_map. Trả về list MỚI, không sửa in-place."""
    return [
        SpeakerSegment(
            speaker = name_map.get(seg.speaker, seg.speaker),
            start   = seg.start,
            end     = seg.end,
            text    = seg.text,
        )
        for seg in segments
    ]


# ────────────────────────────────────────────────────────────────────────────
# Hàm 9: Thống kê theo speaker
# ────────────────────────────────────────────────────────────────────────────
def get_speaker_stats(segments: list[SpeakerSegment]) -> dict:
    """
    Tính thống kê cho từng người nói — dùng để hiển thị widget gán nhãn.

    Returns:
        {"SPEAKER_00": {"duration": 87.4, "turns": 12, "percent": 58.3}, ...}
    """
    stats = {}
    for seg in segments:
        dur = seg.end - seg.start
        if seg.speaker not in stats:
            stats[seg.speaker] = {"duration": 0.0, "turns": 0}
        stats[seg.speaker]["duration"] += dur
        stats[seg.speaker]["turns"]    += 1

    total = sum(v["duration"] for v in stats.values())
    for spk in stats:
        stats[spk]["percent"] = round(
            stats[spk]["duration"] / total * 100, 1
        ) if total > 0 else 0.0

    return dict(sorted(stats.items(), key=lambda x: -x[1]["duration"]))


# ────────────────────────────────────────────────────────────────────────────
# TEST — chạy: python core/diarizer.py <file.wav>
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print(f"  TEST: core/diarizer.py")
    print(f"  Model: {DIARIZATION_MODEL}")
    print("=" * 60)

    print(f"\n[1] Kiểm tra token...")
    if _IS_PRECISION:
        if not PYANNOTE_API_KEY:
            print("    ❌ PYANNOTE_API_KEY trống — thêm vào .env")
            sys.exit(1)
        print(f"    ✅ PYANNOTE_API_KEY = {PYANNOTE_API_KEY[:8]}{'*'*20}")
    else:
        if not HF_TOKEN:
            print("    ❌ HF_TOKEN trống — thêm vào .env")
            sys.exit(1)
        print(f"    ✅ HF_TOKEN = {HF_TOKEN[:8]}{'*'*20}")

    print("\n[2] Kiểm tra import pyannote.audio...")
    try:
        from pyannote.audio import Pipeline
        print("    ✅ pyannote.audio import thành công")
    except ImportError as e:
        print(f"    ❌ {e}")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("\n[3] Cần file WAV để test")
        print("    Dùng: python core/diarizer.py <file_16k.wav>")
        sys.exit(0)

    wav_path = sys.argv[1]
    if not Path(wav_path).exists():
        print(f"\n    ❌ File không tồn tại: {wav_path}")
        sys.exit(1)

    # ── Tiền xử lý qua CONVERTER (giống 1_Batch_Mode.py) ──────────────────────
    # Diarizer cần WAV 16kHz mono. Nếu input là stereo / sample-rate khác /
    # định dạng khác (mp3, mp4...) → convert qua ffmpeg trước khi feed.
    print(f"\n[2.5] Tiền xử lý qua converter (cần ffmpeg)...")
    from converter import convert_to_wav, get_audio_info
    info = get_audio_info(wav_path)
    needs_convert = not (info["ok"] and info["sample_rate"] == 16000 and info["channels"] == 1)
    if needs_convert:
        print(f"    Input: {info.get('sample_rate')}Hz {info.get('channels')}ch "
              f"→ convert sang 16kHz mono…")
        converted = str(Path(wav_path).with_name(Path(wav_path).stem + "_16k_mono.wav"))
        if not convert_to_wav(wav_path, converted, sample_rate=16000, normalize=True):
            print("    ❌ Convert thất bại — kiểm tra ffmpeg / FFMPEG_PATH trong .env")
            sys.exit(1)
        wav_path = converted
        print(f"    ✅ Đã convert → {wav_path}")
    else:
        print("    ✅ Input đã là 16kHz mono — bỏ qua convert")

    print(f"\n[3] Validate: {wav_path}")
    valid, reason = validate_wav(wav_path)
    if not valid:
        print(f"    ❌ {reason}")
        sys.exit(1)
    print("    ✅ File hợp lệ")

    print("\n[4] Load diarizer...")
    try:
        pipeline = load_diarizer()
    except Exception as e:
        print(f"    ❌ {e}")
        sys.exit(1)

    print(f"\n[5] Diarize...")
    try:
        segments = diarize_file(
            pipeline, wav_path,
            min_speakers  = 1,
            max_speakers  = 4,
            use_exclusive = _IS_COMMUNITY,
        )
    except Exception as e:
        print(f"    ❌ {e}")
        sys.exit(1)

    print(f"\n[6] Kết quả ({len(segments)} đoạn):")
    for i, seg in enumerate(segments[:10]):
        bar = "█" * int((seg.end - seg.start) * 2)
        print(f"    [{i+1:02d}] {seg.speaker:12s} | "
              f"{seg.start:6.2f}s → {seg.end:6.2f}s | {bar}")
    if len(segments) > 10:
        print(f"    ... và {len(segments) - 10} đoạn nữa")

    print("\n[7] Thống kê:")
    for spk, info in get_speaker_stats(segments).items():
        print(f"    {spk:12s} | {info['duration']:6.1f}s "
              f"| {info['turns']:3d} lượt | {info['percent']}%")

    print("\n✅ Test xong\n")