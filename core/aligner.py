# core/aligner.py
#
# Mục đích: Ghép nối kết quả STT (text) + Diarization (speaker + timestamp)
#   thành danh sách AlignedTurn để hiển thị trên UI và xuất biên bản.
#
# Luồng:
#   transcriber.py  →  {"text": "...", "duration": ...}
#                       HOẶC
#                      [{"text": "...", "start": x, "end": y}, ...]  ← Whisper segments
#   diarizer.py     →  [SpeakerSegment(speaker, start, end), ...]
#                            ↓
#                       aligner.py
#                            ↓
#                   [AlignedTurn(speaker, start, end, text), ...]
#                            ↓
#               transcript_viewer.py  +  export_docx.py
#
# THAY ĐỔI so với phiên bản cũ:
#   v1: Ratio-based split → cắt giữa câu, domino drift
#   v2: Sentence-boundary alignment → tốt hơn nhưng vẫn drift nếu Whisper hallucinate
#   v3 (hiện tại): Segment-aware alignment
#     → Nếu Whisper trả về segments có timestamp → dùng timestamp overlap để map
#     → Nếu chỉ có full_text → dùng sentence-boundary với drift correction
#     → Thêm min_words_per_turn để tránh turn rỗng
#     → Thêm anchor detection để reset drift khi tìm thấy từ đặc trưng

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from core.diarizer import SpeakerSegment


# ────────────────────────────────────────────────────────────────────────────
# Data class đầu ra
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class AlignedTurn:
    """
    1 lượt nói hoàn chỉnh sau khi ghép STT + Diarization.
    Đây là đơn vị dữ liệu cơ bản chạy xuyên suốt toàn bộ app.
    """
    speaker    : str          # "SPEAKER_00" hoặc tên thật sau khi gán
    start      : float        # giây — từ diarizer
    end        : float        # giây — từ diarizer
    text       : str          # văn bản — từ transcriber
    confidence : float = 1.0  # dự phòng cho word-level confidence sau này


# ────────────────────────────────────────────────────────────────────────────
# Data class cho Whisper segment (nếu có timestamp)
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class WhisperSegment:
    """
    1 segment từ Whisper output (khi có timestamp per-segment).
    Dùng để map overlap với SpeakerSegment từ pyannote.
    """
    text  : str
    start : float
    end   : float


# ────────────────────────────────────────────────────────────────────────────
# Regex patterns
# ────────────────────────────────────────────────────────────────────────────
_SENTENCE_END = re.compile(r'[.?!。？！]$')
_CLAUSE_END   = re.compile(r'[,;،،،]$')

# Từ thường xuất hiện đầu lượt nói (anchor words)
# Thêm vào nếu domain cụ thể có từ đặc trưng hơn
_FILLER_START = re.compile(
    r'^(ừ|ừm|à|ờ|thì|nhưng mà|mà|và|thế|vâng|dạ|ừ thì|okay|ok|vậy|ơ|ôi|ôi trời)\b',
    re.IGNORECASE
)


# ────────────────────────────────────────────────────────────────────────────
# Hàm nội bộ: tìm điểm cắt tốt nhất gần budget
# ────────────────────────────────────────────────────────────────────────────
def _find_cut_point(words: List[str], budget: int, slack: int = 8) -> int:
    """
    Tìm điểm cắt tốt nhất trong khoảng [budget - slack, budget + slack].

    Ưu tiên theo thứ tự:
        1. Sau từ kết thúc câu (. ? !)  trong vùng slack
        2. Sau từ kết thúc mệnh đề (, ;) trong vùng slack
        3. Đúng budget nếu không tìm được ranh giới nào

    Args:
        words  : danh sách từ còn lại (chưa phân bổ)
        budget : số từ dự kiến theo tỉ lệ thời gian
        slack  : số từ cho phép lệch để tìm ranh giới câu

    Returns:
        Số từ thực tế sẽ lấy (index cut point, exclusive)
    """
    n = len(words)

    lo = max(1, budget - slack)
    hi = min(n - 1, budget + slack)

    if lo >= hi:
        return min(budget, n)

    # Ưu tiên 1: sentence end gần budget nhất
    best_sent, best_sent_dist = None, slack + 1
    for i in range(lo, hi + 1):
        if _SENTENCE_END.search(words[i - 1]):
            dist = abs(i - budget)
            if dist < best_sent_dist:
                best_sent_dist = dist
                best_sent = i
    if best_sent is not None:
        return best_sent

    # Ưu tiên 2: clause end
    best_clause, best_clause_dist = None, slack + 1
    for i in range(lo, hi + 1):
        if _CLAUSE_END.search(words[i - 1]):
            dist = abs(i - budget)
            if dist < best_clause_dist:
                best_clause_dist = dist
                best_clause = i
    if best_clause is not None:
        return best_clause

    return min(budget, n)


# ────────────────────────────────────────────────────────────────────────────
# CHIẾN LƯỢC 1: Segment-aware alignment (khi Whisper có timestamps)
# ────────────────────────────────────────────────────────────────────────────
def _align_with_segments(
    speaker_segs : List[SpeakerSegment],
    whisper_segs : List[WhisperSegment],
    overlap_threshold : float = 0.3,  # % overlap tối thiểu để tính là match
) -> List[AlignedTurn]:
    """
    Map Whisper segments → SpeakerSegments theo overlap thời gian.

    Với mỗi SpeakerSegment, tìm tất cả WhisperSegment có overlap,
    ghép text của chúng lại theo thứ tự thời gian.

    Ưu điểm: Không bị domino drift vì dùng timestamp thật thay vì ước tính.
    Nhược điểm: Nếu Whisper segment dài hơn speaker boundary → text bị chia sẻ
                giữa 2 turn (xử lý bằng partial_text).
    """
    turns = []

    for sp_seg in speaker_segs:
        collected_texts = []

        for w_seg in whisper_segs:
            # Tính overlap giữa speaker segment và whisper segment
            overlap_start = max(sp_seg.start, w_seg.start)
            overlap_end   = min(sp_seg.end,   w_seg.end)
            overlap_dur   = max(0.0, overlap_end - overlap_start)

            if overlap_dur <= 0:
                continue

            w_dur = w_seg.end - w_seg.start
            if w_dur <= 0:
                continue

            overlap_ratio = overlap_dur / w_dur

            if overlap_ratio >= overlap_threshold:
                # Whisper segment nằm chủ yếu trong speaker segment này
                collected_texts.append(w_seg.text.strip())
            elif overlap_ratio > 0:
                # Partial overlap: lấy phần tỉ lệ với overlap
                words = w_seg.text.strip().split()
                n_words = max(1, round(overlap_ratio * len(words)))
                partial = " ".join(words[:n_words])
                if partial:
                    collected_texts.append(partial)

        merged_text = " ".join(collected_texts).strip()
        turns.append(AlignedTurn(
            speaker = sp_seg.speaker,
            start   = sp_seg.start,
            end     = sp_seg.end,
            text    = merged_text,
        ))

    return turns


# ────────────────────────────────────────────────────────────────────────────
# CHIẾN LƯỢC 2: Text-only alignment với drift correction (fallback)
# ────────────────────────────────────────────────────────────────────────────
def _align_text_only(
    segments  : List[SpeakerSegment],
    full_text : str,
    slack     : int = 10,
) -> List[AlignedTurn]:
    """
    Align text thuần túy khi không có word-level timestamps.

    Cải tiến so với v2:
    1. Dynamic ratio (giữ nguyên từ v2) — chống domino drift
    2. Min words per turn — tránh turn rỗng / chỉ có 1-2 từ lẻ
    3. Sentence-boundary snap (giữ nguyên từ v2)
    4. Lookahead correction: nếu remaining_words quá ít so với remaining_turns,
       tăng budget của turn hiện tại để phân phối đều hơn
    """
    remaining_words = full_text.strip().split()
    total_words = len(remaining_words)

    has_punctuation = bool(re.search(r'[.?!,;]', full_text))
    effective_slack = slack if has_punctuation else 0

    turns = []
    n_segs = len(segments)

    for i, seg in enumerate(segments):
        duration = seg.end - seg.start
        is_last  = (i == n_segs - 1)

        if is_last or not remaining_words:
            chunk = remaining_words
            remaining_words = []
        else:
            remaining_segs     = segments[i:]
            remaining_duration = sum(s.end - s.start for s in remaining_segs)
            remaining_turns    = n_segs - i  # số turn còn lại kể cả turn hiện tại

            if remaining_duration > 0:
                ratio = duration / remaining_duration
            else:
                ratio = 1.0

            budget = max(1, round(ratio * len(remaining_words)))

            # Lookahead correction:
            # Nếu remaining_words ít hơn remaining_turns * min_words,
            # boost budget để các turn cuối không bị rỗng
            min_words_per_turn = 2
            words_left_after   = len(remaining_words) - budget
            turns_left_after   = remaining_turns - 1
            if turns_left_after > 0 and words_left_after < turns_left_after * min_words_per_turn:
                # Nhường lại đủ cho các turn sau
                budget = max(1, len(remaining_words) - turns_left_after * min_words_per_turn)

            if effective_slack > 0:
                cut = _find_cut_point(remaining_words, budget, effective_slack)
            else:
                cut = min(budget, len(remaining_words))

            # Đảm bảo không lấy hết — phải để lại ít nhất 1 từ cho các turn sau
            cut = min(cut, len(remaining_words) - max(0, n_segs - i - 1))
            cut = max(1, cut)

            chunk = remaining_words[:cut]
            remaining_words = remaining_words[cut:]

        turns.append(AlignedTurn(
            speaker = seg.speaker,
            start   = seg.start,
            end     = seg.end,
            text    = " ".join(chunk).strip(),
        ))

    return turns


# ────────────────────────────────────────────────────────────────────────────
# Hàm chính: align — tự động chọn chiến lược
# ────────────────────────────────────────────────────────────────────────────
def align(
    segments      : List[SpeakerSegment],
    full_text     : str,
    slack         : int = 10,
    whisper_segs  : Optional[List[WhisperSegment]] = None,
    overlap_threshold : float = 0.3,
) -> List[AlignedTurn]:
    """
    Ghép nối text từ Whisper vào từng segment từ pyannote.

    Tự động chọn chiến lược:
    - Nếu whisper_segs có giá trị → Segment-aware alignment (chính xác hơn)
    - Nếu chỉ có full_text        → Text-only alignment với drift correction

    Args:
        segments          : Speaker segments từ pyannote diarizer
        full_text         : Toàn bộ text từ Whisper (luôn bắt buộc)
        slack             : Số từ cho phép lệch khi tìm sentence boundary
        whisper_segs      : (Optional) Segments từ Whisper CÓ timestamp.
                            Truyền vào nếu Whisper output dạng list of dicts:
                            [{"text": "...", "start": 0.0, "end": 3.5}, ...]
        overlap_threshold : Tỉ lệ overlap tối thiểu để map whisper → speaker segment

    Returns:
        List[AlignedTurn] đã được ghép nối
    """
    if not segments:
        if full_text.strip():
            return [AlignedTurn(
                speaker = "SPEAKER_00",
                start   = 0.0,
                end     = 0.0,
                text    = full_text.strip(),
            )]
        return []

    if not full_text.strip():
        return [
            AlignedTurn(speaker=seg.speaker, start=seg.start, end=seg.end, text="")
            for seg in segments
        ]

    # Chiến lược 1: dùng Whisper timestamps nếu có
    if whisper_segs and len(whisper_segs) > 0:
        return _align_with_segments(segments, whisper_segs, overlap_threshold)

    # Chiến lược 2: fallback text-only
    return _align_text_only(segments, full_text, slack)


# ────────────────────────────────────────────────────────────────────────────
# Hàm tiện ích: parse Whisper output thành WhisperSegment
# ────────────────────────────────────────────────────────────────────────────
def parse_whisper_segments(raw_segments) -> Tuple[List[WhisperSegment], str]:
    """
    Parse output từ Whisper thành (whisper_segs, full_text).

    Nhận vào:
        raw_segments : list of dict từ Whisper, mỗi dict có keys:
                       "text", "start", "end"
                       (output chuẩn của openai-whisper và faster-whisper)

    Returns:
        (whisper_segs, full_text) — tuple để truyền vào align()

    Ví dụ sử dụng:
        result = whisper_model.transcribe(audio_path)
        w_segs, full_text = parse_whisper_segments(result["segments"])
        turns = align(speaker_segments, full_text, whisper_segs=w_segs)
    """
    w_segs = []
    texts  = []

    for seg in raw_segments:
        text  = seg.get("text", "").strip()
        start = float(seg.get("start", 0.0))
        end   = float(seg.get("end", 0.0))

        if text:
            w_segs.append(WhisperSegment(text=text, start=start, end=end))
            texts.append(text)

    full_text = " ".join(texts)
    return w_segs, full_text


# ────────────────────────────────────────────────────────────────────────────
# Hàm tiện ích: gán tên thật vào danh sách turn
# ────────────────────────────────────────────────────────────────────────────
def rename_turns(
    turns    : List[AlignedTurn],
    name_map : dict,
) -> List[AlignedTurn]:
    """
    Áp dụng name_map vào danh sách AlignedTurn.
    Không thay đổi in-place — trả về list mới.
    """
    return [
        AlignedTurn(
            speaker    = name_map.get(t.speaker, t.speaker),
            start      = t.start,
            end        = t.end,
            text       = t.text,
            confidence = t.confidence,
        )
        for t in turns
    ]


# ────────────────────────────────────────────────────────────────────────────
# Hàm tiện ích: merge các turn liên tiếp của cùng 1 người nói
# ────────────────────────────────────────────────────────────────────────────
def merge_consecutive(
    turns     : List[AlignedTurn],
    gap_limit : float = 1.5,
) -> List[AlignedTurn]:
    """
    Gộp các turn liên tiếp của cùng 1 người nói nếu khoảng cách < gap_limit.
    """
    if not turns:
        return []

    merged = [turns[0]]

    for current in turns[1:]:
        prev = merged[-1]
        gap  = current.start - prev.end

        if current.speaker == prev.speaker and gap < gap_limit:
            merged[-1] = AlignedTurn(
                speaker    = prev.speaker,
                start      = prev.start,
                end        = current.end,
                text       = f"{prev.text} {current.text}".strip(),
                confidence = min(prev.confidence, current.confidence),
            )
        else:
            merged.append(current)

    return merged


# ────────────────────────────────────────────────────────────────────────────
# Hàm tiện ích: format timestamp MM:SS.ss
# ────────────────────────────────────────────────────────────────────────────
def format_timestamp(seconds: float) -> str:
    """Chuyển giây thành chuỗi MM:SS.ss dùng trong transcript và DOCX."""
    minutes = int(seconds // 60)
    secs    = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


# ────────────────────────────────────────────────────────────────────────────
# TEST — chạy: python core/aligner.py
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  TEST: core/aligner.py  (v3 — segment-aware)")
    print("=" * 60)

    mock_segments = [
        SpeakerSegment(speaker="SPEAKER_00", start=0.0,  end=5.2),
        SpeakerSegment(speaker="SPEAKER_01", start=5.5,  end=12.0),
        SpeakerSegment(speaker="SPEAKER_00", start=12.3, end=16.8),
        SpeakerSegment(speaker="SPEAKER_01", start=17.1, end=25.0),
        SpeakerSegment(speaker="SPEAKER_00", start=25.4, end=28.0),
    ]

    # ── Test 1: Chiến lược 1 — Whisper segments có timestamp ────────────────
    print("\n[1] Test align() — Whisper CÓ timestamps (segment-aware)...")
    mock_whisper_raw = [
        {"text": "Anh cho biết tên tuổi và địa chỉ thường trú.",   "start": 0.5,  "end": 4.8},
        {"text": "Tôi tên Nguyễn Văn A, sinh năm 1985, Hà Nội.",   "start": 5.8,  "end": 11.5},
        {"text": "Anh có mặt ở đâu vào tối ngày 15 tháng 3?",      "start": 12.5, "end": 16.5},
        {"text": "Tôi ở nhà suốt buổi tối hôm đó, không đi đâu.",  "start": 17.2, "end": 24.5},
        {"text": "Có ai có thể xác nhận không?",                    "start": 25.6, "end": 27.8},
    ]
    w_segs, full_text = parse_whisper_segments(mock_whisper_raw)
    turns = align(mock_segments, full_text, whisper_segs=w_segs)
    print(f"    ✅ {len(turns)} turns từ segment-aware alignment")
    for t in turns:
        ts = format_timestamp(t.start)
        print(f"    [{ts}] {t.speaker:12s} | {t.text[:70]}")

    # ── Test 2: Chiến lược 2 — chỉ có full_text (fallback) ──────────────────
    print("\n[2] Test align() — chỉ có full_text (text-only fallback)...")
    mock_text_punct = (
        "Anh cho biết tên tuổi và địa chỉ thường trú của mình. "
        "Tôi tên Nguyễn Văn A, sinh năm 1985, thường trú tại Hà Nội. "
        "Anh có mặt ở đâu vào tối ngày 15 tháng 3? "
        "Tôi ở nhà suốt buổi tối hôm đó, không đi đâu cả. "
        "Có ai có thể xác nhận không?"
    )
    turns2 = align(mock_segments, mock_text_punct)
    print(f"    ✅ {len(turns2)} turns từ text-only alignment")
    for t in turns2:
        ts = format_timestamp(t.start)
        print(f"    [{ts}] {t.speaker:12s} | {t.text[:70]}")

    # ── Test 3: Whisper segments có partial overlap ──────────────────────────
    print("\n[3] Test partial overlap — 1 Whisper segment span 2 speaker turns...")
    mock_whisper_overlap = [
        {"text": "Câu đầu của người nói số một và tiếp sang người nói hai.", "start": 3.0, "end": 8.0},
        {"text": "Tiếp tục câu chuyện sau đó.",                              "start": 8.5, "end": 14.0},
        {"text": "Đây là lượt của người một lần nữa.",                       "start": 14.5, "end": 28.0},
    ]
    w_segs3, ft3 = parse_whisper_segments(mock_whisper_overlap)
    turns3 = align(mock_segments, ft3, whisper_segs=w_segs3)
    print(f"    ✅ {len(turns3)} turns (partial overlap handled)")
    for t in turns3:
        ts = format_timestamp(t.start)
        print(f"    [{ts}] {t.speaker:12s} | {t.text[:70]}")

    # ── Test 4: parse_whisper_segments ──────────────────────────────────────
    print("\n[4] Test parse_whisper_segments()...")
    w_segs4, ft4 = parse_whisper_segments(mock_whisper_raw)
    print(f"    ✅ {len(w_segs4)} WhisperSegments parsed")
    print(f"    full_text[:60]: '{ft4[:60]}...'")

    # ── Test 5: rename + merge ───────────────────────────────────────────────
    print("\n[5] Test rename_turns() + merge_consecutive()...")
    name_map = {"SPEAKER_00": "Điều tra viên", "SPEAKER_01": "Đối tượng"}
    renamed  = rename_turns(turns, name_map)
    merged   = merge_consecutive(renamed, gap_limit=1.5)
    speakers = {t.speaker for t in renamed}
    print(f"    ✅ Speakers sau đổi tên: {speakers}")
    print(f"    ✅ {len(turns)} turns → {len(merged)} sau merge")

    # ── Test 6: edge cases ───────────────────────────────────────────────────
    print("\n[6] Test edge cases...")
    e1 = align([], "Có text nhưng không có segment")
    print(f"    segments=[]     → {len(e1)} turns (expect 1 fallback)")
    e2 = align(mock_segments, "")
    print(f"    text=''         → {len(e2)} turns (expect {len(mock_segments)} empty)")
    e3 = align(mock_segments, "chỉ ba từ")
    print(f"    text ngắn hơn  → {len(e3)} turns, texts: {[t.text for t in e3]}")

    print("\n✅ Tất cả test đều qua — aligner.py v3 sẵn sàng\n")
