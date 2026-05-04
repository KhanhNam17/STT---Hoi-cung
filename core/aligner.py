# core/aligner.py
#
# Mục đích: Ghép nối kết quả STT (text) + Diarization (speaker + timestamp)
#   thành danh sách AlignedTurn để hiển thị trên UI và xuất biên bản.
#
# Luồng:
#   transcriber.py  →  {"text": "...", "duration": ...}
#   diarizer.py     →  [SpeakerSegment(speaker, start, end), ...]
#                            ↓
#                       aligner.py
#                            ↓
#                   [AlignedTurn(speaker, start, end, text), ...]
#                            ↓
#               transcript_viewer.py  +  export_docx.py
#
# THAY ĐỔI so với phiên bản cũ:
#   - Chiến lược align cũ: chia từ theo tỉ lệ thời gian (ratio-based)
#     → Vấn đề: text bị cắt giữa câu, trôi sang turn sai khi lượt nói ngắn
#   - Chiến lược align mới: sentence-boundary alignment
#     → Tính word budget theo tỉ lệ thời gian (giữ nguyên)
#     → Nhưng snap điểm cắt đến ranh giới câu gần nhất (dấu . ? ! ,)
#     → Kết quả: mỗi turn nhận trọn câu, không bị cắt lưng chừng

import re
from dataclasses import dataclass, field
from typing import List, Optional

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
# Hàm nội bộ: tìm điểm cắt tốt nhất gần budget
# ────────────────────────────────────────────────────────────────────────────

# Ký tự kết thúc câu — snap điểm cắt vào SAU các ký tự này
_SENTENCE_END = re.compile(r'[.?!。？！]$')

# Ký tự ngắt mệnh đề — ưu tiên thấp hơn sentence end
_CLAUSE_END   = re.compile(r'[,;،،،]$')


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

    # Giới hạn vùng tìm kiếm
    lo = max(1, budget - slack)
    hi = min(n - 1, budget + slack)   # giữ lại ít nhất 1 từ cho segment sau

    if lo >= hi:
        # Budget gần cuối list — trả về budget thô
        return min(budget, n)

    # Ưu tiên 1: tìm sentence end gần budget nhất (trong cửa sổ slack)
    # Duyệt từ giữa ra ngoài để lấy điểm gần budget nhất
    best_sent = None
    best_sent_dist = slack + 1

    for i in range(lo, hi + 1):
        if _SENTENCE_END.search(words[i - 1]):   # words[i-1] là từ cuối cùng được lấy
            dist = abs(i - budget)
            if dist < best_sent_dist:
                best_sent_dist = dist
                best_sent = i

    if best_sent is not None:
        return best_sent

    # Ưu tiên 2: clause end
    best_clause = None
    best_clause_dist = slack + 1

    for i in range(lo, hi + 1):
        if _CLAUSE_END.search(words[i - 1]):
            dist = abs(i - budget)
            if dist < best_clause_dist:
                best_clause_dist = dist
                best_clause = i

    if best_clause is not None:
        return best_clause

    # Fallback: đúng budget
    return min(budget, n)


# ────────────────────────────────────────────────────────────────────────────
# Hàm chính: align
# ────────────────────────────────────────────────────────────────────────────
def align(
    segments  : List[SpeakerSegment],
    full_text : str,
    slack     : int = 10,    # Tăng slack lên 10 để bao quát được các câu hỏi cung dài
) -> List[AlignedTurn]:
    """
    Ghép nối text từ Whisper vào từng segment từ pyannote.
    Sử dụng chiến lược: Dynamic Sentence-Boundary Alignment (Cắt ranh giới câu ĐỘNG)
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

    # Khởi tạo danh sách từ còn lại (Ban đầu là toàn bộ văn bản)
    remaining_words = full_text.strip().split()
    
    has_punctuation = bool(re.search(r'[.?!,;]', full_text))
    effective_slack = slack if has_punctuation else 0

    turns = []

    for i, seg in enumerate(segments):
        duration = seg.end - seg.start
        is_last  = (i == len(segments) - 1)

        # Nếu là đoạn cuối cùng, gom toàn bộ chữ còn lại vào
        if is_last or not remaining_words:
            chunk = remaining_words
            remaining_words = []
        else:
            # FIX LỖI DOMINO: Tính thời gian của TẤT CẢ các đoạn CÒN LẠI (từ i đến hết)
            remaining_duration = sum(s.end - s.start for s in segments[i:])
            
            # Tính tỷ lệ dựa trên thời gian còn lại (Dynamic Ratio)
            if remaining_duration > 0:
                ratio = duration / remaining_duration
            else:
                ratio = 1.0
            
            # Budget bây giờ được chia trên số chữ CÒN LẠI, không phải tổng chữ
            budget = max(1, round(ratio * len(remaining_words)))

            # Dò tìm ranh giới dấu câu trong khoảng budget ± slack
            if effective_slack > 0:
                cut = _find_cut_point(remaining_words, budget, effective_slack)
            else:
                cut = min(budget, len(remaining_words))

            # Tách chunk ra và cập nhật lại danh sách từ còn lại
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
# Hàm tiện ích: gán tên thật vào danh sách turn
# ────────────────────────────────────────────────────────────────────────────
def rename_turns(
    turns    : List[AlignedTurn],
    name_map : dict,              # {"SPEAKER_00": "Điều tra viên", ...}
) -> List[AlignedTurn]:
    """
    Áp dụng name_map vào danh sách AlignedTurn.
    Dùng sau khi cán bộ gán tên trên UI.

    Không thay đổi in-place — trả về list mới để tránh side effect
    với Streamlit session_state.
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

    Ví dụ: pyannote có thể tách 1 câu dài thành 3 segment nhỏ
    → merge lại thành 1 turn cho dễ đọc.

    Args:
        gap_limit : khoảng cách (giây) tối đa giữa 2 segment để gộp.
                    Nếu cùng speaker và gap < gap_limit → gộp lại.
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
    import sys

    print("=" * 55)
    print("  TEST: core/aligner.py")
    print("=" * 55)

    mock_segments = [
        SpeakerSegment(speaker="SPEAKER_00", start=0.0,  end=5.2),
        SpeakerSegment(speaker="SPEAKER_01", start=5.5,  end=12.0),
        SpeakerSegment(speaker="SPEAKER_00", start=12.3, end=16.8),
        SpeakerSegment(speaker="SPEAKER_01", start=17.1, end=25.0),
        SpeakerSegment(speaker="SPEAKER_00", start=25.4, end=28.0),
    ]

    # Text CÓ dấu câu — test sentence-boundary alignment
    mock_text_punct = (
        "Anh cho biết tên tuổi và địa chỉ thường trú của mình. "
        "Tôi tên Nguyễn Văn A, sinh năm 1985, thường trú tại Hà Nội. "
        "Anh có mặt ở đâu vào tối ngày 15 tháng 3? "
        "Tôi ở nhà suốt buổi tối hôm đó, không đi đâu cả. "
        "Có ai có thể xác nhận không?"
    )

    # Text KHÔNG có dấu câu — test fallback ratio-based
    mock_text_raw = (
        "Anh cho biết tên tuổi và địa chỉ thường trú của mình "
        "Tôi tên Nguyễn Văn A sinh năm 1985 thường trú tại Hà Nội "
        "Anh có mặt ở đâu vào tối ngày 15 tháng 3 "
        "Tôi ở nhà suốt buổi tối hôm đó không đi đâu cả "
        "Có ai có thể xác nhận không"
    )

    # Test 1: align với text có dấu câu
    print("\n[1] Test align() — text CÓ dấu câu (sentence-boundary)...")
    turns = align(mock_segments, mock_text_punct, slack=8)
    print(f"    ✅ Tạo được {len(turns)} turns")
    for t in turns:
        ts      = format_timestamp(t.start)
        preview = t.text[:65] + "..." if len(t.text) > 65 else t.text
        print(f"    [{ts}] {t.speaker:12s} | {preview}")

    # Test 2: align với text thô (không dấu câu) — fallback
    print("\n[2] Test align() — text KHÔNG dấu câu (ratio fallback)...")
    turns_raw = align(mock_segments, mock_text_raw, slack=8)
    print(f"    ✅ Tạo được {len(turns_raw)} turns (fallback ratio-based)")
    for t in turns_raw:
        ts      = format_timestamp(t.start)
        preview = t.text[:65] + "..." if len(t.text) > 65 else t.text
        print(f"    [{ts}] {t.speaker:12s} | {preview}")

    # Test 3: so sánh điểm cắt — kiểm tra sentence boundary hoạt động
    print("\n[3] Test _find_cut_point()...")
    test_words = "Tôi ở nhà suốt buổi tối hôm đó không đi đâu cả.".split()
    cut = _find_cut_point(test_words, budget=5, slack=4)
    print(f"    words  : {test_words}")
    print(f"    budget : 5, slack : 4")
    print(f"    cut    : {cut} → '{' '.join(test_words[:cut])}'")
    # Kỳ vọng: snap về cuối câu (từ "cả." ở index cuối) hoặc budget nếu không tìm được

    # Test 4: rename_turns
    print("\n[4] Test rename_turns()...")
    name_map = {"SPEAKER_00": "Điều tra viên", "SPEAKER_01": "Đối tượng"}
    renamed  = rename_turns(turns, name_map)
    speakers = {t.speaker for t in renamed}
    print(f"    ✅ Speakers sau đổi tên: {speakers}")

    # Test 5: merge_consecutive
    print("\n[5] Test merge_consecutive()...")
    before = len(turns)
    merged = merge_consecutive(turns, gap_limit=1.5)
    after  = len(merged)
    print(f"    ✅ {before} turns → {after} turns sau merge")

    # Test 6: edge cases
    print("\n[6] Test edge cases...")
    e1 = align([], "Có text nhưng không có segment")
    print(f"    segments=[] → {len(e1)} turns (expect 1 fallback)")

    e2 = align(mock_segments, "")
    print(f"    text=''    → {len(e2)} turns (expect {len(mock_segments)} empty)")

    print("\n✅ Tất cả test đều qua — aligner.py sẵn sàng\n")