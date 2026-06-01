# core/aligner.py  —  v4 (WhisperX-style)
#
#
#   Dependency: stable-ts  (pip install stable-ts)
#     stable-ts chạy forced alignment bằng cross-attention của Whisper encoder,
#     không cần model riêng, không cần GPU, nhẹ hơn wav2vec2.
#
# FALLBACK:
#   Nếu stable-ts không khả dụng hoặc alignment thất bại →
#   dùng _align_text_only() (ratio-based, giữ lại từ v3) để không crash app.

import re
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Import lazy để không crash nếu stable-ts chưa cài
_stable_ts = None
def _get_stable_ts():
    global _stable_ts
    if _stable_ts is None:
        try:
            import stable_whisper as sw
            _stable_ts = sw
        except ImportError:
            _stable_ts = False
    return _stable_ts if _stable_ts is not False else None


# ────────────────────────────────────────────────────────────────────────────
# Data classes
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class AlignedTurn:
    """
    1 lượt nói hoàn chỉnh — đơn vị dữ liệu xuyên suốt toàn bộ app.

    word_starts : per-syllable/token start times (giây) lấy từ sherpa-onnx
                  get_result_as_json_string. None nếu chưa có (cho compat cũ).
                  Khi có → aligner dùng real timestamps thay vì interpolation.
    """
    speaker     : str
    start       : float
    end         : float
    text        : str
    confidence  : float = 1.0
    word_starts : list = None  # type: ignore[assignment]


@dataclass
class WordTimestamp:
    """
    1 từ có timestamp — output của forced alignment.
    Trung gian giữa Whisper text và pyannote segments.
    """
    word  : str
    start : float
    end   : float


# Giữ lại để tương thích với code cũ dùng parse_whisper_segments()
@dataclass
class WhisperSegment:
    text  : str
    start : float
    end   : float


# Import SpeakerSegment từ diarizer
try:
    from core.diarizer import SpeakerSegment
except ImportError:
    @dataclass
    class SpeakerSegment:
        speaker : str
        start   : float
        end     : float
        text    : str = ""


# ────────────────────────────────────────────────────────────────────────────
# BƯỚC 0: Forced Alignment (stable-ts)
# ────────────────────────────────────────────────────────────────────────────
def forced_align(
    wav_path  : str,
    text      : str,
    language  : str = "vi",
) -> List[WordTimestamp]:
    """
    Lấy timestamp từng từ bằng stable-ts forced alignment.

    stable-ts dùng cross-attention của Whisper encoder để "canh" text
    đã có vào waveform — KHÔNG re-transcribe, chỉ align.
    Nhanh hơn wav2vec2 (không cần model riêng), offline hoàn toàn.

    Args:
        wav_path : file WAV 16kHz mono (đã qua converter)
        text     : full transcript text từ Qualcomm Whisper
        language : mã ngôn ngữ ("vi", "en", ...)

    Returns:
        List[WordTimestamp] — mỗi từ có start/end tính bằng giây.
        Trả về [] nếu thất bại (caller sẽ fallback về ratio-based).
    """
    sw = _get_stable_ts()
    if sw is None:
        logger.warning("[aligner] stable-ts không khả dụng → dùng fallback ratio-based")
        return []

    try:
        # Load Whisper base để làm aligner — nhẹ, chỉ dùng encoder
        # Không cần large-v3-turbo vì chỉ làm alignment, không transcribe
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Cache model để tránh load lại mỗi lần
        if not hasattr(forced_align, "_model_cache"):
            forced_align._model_cache = {}

        cache_key = f"base_{device}"
        if cache_key not in forced_align._model_cache:
            logger.info(f"[aligner] Load Whisper base cho forced alignment ({device})...")
            base_model = sw.load_model("base", device=device)
            forced_align._model_cache[cache_key] = base_model

        align_model = forced_align._model_cache[cache_key]

        # Forced align: stable_whisper.align(model, audio, text)
        result = align_model.align(
            wav_path,
            text,
            language=language,
            remove_instant_words=True,   # xóa từ không align được (số, ký hiệu lạ)
            original_split=False,
        )

        words = []
        if result is None:
            logger.warning("[aligner] stable-ts align() trả về None → fallback")
            return []

        for seg in result.segments:
            for w in seg.words:
                word_text = w.word.strip()
                if word_text and w.end > w.start:
                    words.append(WordTimestamp(
                        word  = word_text,
                        start = round(float(w.start), 3),
                        end   = round(float(w.end),   3),
                    ))

        logger.info(f"[aligner] Forced alignment: {len(words)} từ có timestamp")
        return words

    except Exception as e:
        logger.warning(f"[aligner] Forced alignment thất bại: {e} → fallback")
        return []


# ────────────────────────────────────────────────────────────────────────────
# BƯỚC 1: Assign word → speaker (WhisperX assign_word_speakers)
# ────────────────────────────────────────────────────────────────────────────
def assign_word_speakers(
    words         : List[WordTimestamp],
    segments      : List[SpeakerSegment],
    fill_nearest  : bool = True,
) -> List[WordTimestamp]:
    """
    Gán speaker cho từng từ dựa trên overlap thời gian với pyannote segments.

    Logic theo WhisperX assign_word_speakers:
        Với mỗi từ [word.start, word.end]:
        → Tìm speaker segment có overlap lớn nhất.
        → Nếu không có overlap (gap giữa 2 speaker) và fill_nearest=True:
           → Gán speaker của segment gần nhất về mặt thời gian.

    Hàm này trả về list WordTimestamp mới với field 'speaker' được gán.
    (WordTimestamp không có field speaker — ta dùng subclass trick hoặc
    trả về list of tuples. Ở đây dùng namedtuple-style dict để giữ đơn giản.)

    Returns:
        List of dicts: {"word", "start", "end", "speaker"}
    """
    if not segments:
        return [{"word": w.word, "start": w.start, "end": w.end, "speaker": "SPEAKER_00"}
                for w in words]

    result = []

    for w in words:
        best_speaker = None
        best_overlap = 0.0

        for seg in segments:
            overlap_start = max(w.start, seg.start)
            overlap_end   = min(w.end,   seg.end)
            overlap       = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = seg.speaker

        # Không có overlap → fill bằng segment gần nhất
        if best_speaker is None and fill_nearest:
            min_dist = float("inf")
            for seg in segments:
                # Khoảng cách từ từ đến segment (0 nếu overlap)
                dist = max(0.0, max(seg.start - w.end, w.start - seg.end))
                if dist < min_dist:
                    min_dist = dist
                    best_speaker = seg.speaker

        result.append({
            "word"    : w.word,
            "start"   : w.start,
            "end"     : w.end,
            "speaker" : best_speaker or "SPEAKER_00",
        })

    return result


# ────────────────────────────────────────────────────────────────────────────
# BƯỚC 2: Group words → turns
# ────────────────────────────────────────────────────────────────────────────
def group_words_to_turns(
    word_speaker_list : List[dict],
    gap_limit         : float = 1.5,
    min_words         : int   = 1,
) -> List[AlignedTurn]:
    """
    Gộp danh sách từ đã gán speaker thành AlignedTurn.

    Gộp khi:
        - Cùng speaker
        - Khoảng cách giữa 2 từ liên tiếp < gap_limit giây

    Args:
        gap_limit : gộp nếu gap < ngưỡng (giây). Mặc định 1.5s.
        min_words : bỏ turn có ít hơn min_words từ (thường là noise).
    """
    if not word_speaker_list:
        return []

    turns = []
    current_words  = [word_speaker_list[0]["word"]]
    current_start  = word_speaker_list[0]["start"]
    current_end    = word_speaker_list[0]["end"]
    current_spk    = word_speaker_list[0]["speaker"]

    for w in word_speaker_list[1:]:
        gap            = w["start"] - current_end
        same_speaker   = w["speaker"] == current_spk
        close_enough   = gap < gap_limit

        if same_speaker and close_enough:
            current_words.append(w["word"])
            current_end = w["end"]
        else:
            # Flush turn hiện tại
            if len(current_words) >= min_words:
                turns.append(AlignedTurn(
                    speaker = current_spk,
                    start   = current_start,
                    end     = current_end,
                    text    = _join_words(current_words),
                ))
            # Bắt đầu turn mới
            current_words = [w["word"]]
            current_start = w["start"]
            current_end   = w["end"]
            current_spk   = w["speaker"]

    # Flush turn cuối
    if len(current_words) >= min_words:
        turns.append(AlignedTurn(
            speaker = current_spk,
            start   = current_start,
            end     = current_end,
            text    = _join_words(current_words),
        ))

    return turns


def _join_words(words: List[str]) -> str:
    """
    Nối danh sách từ thành câu.
    Tiếng Việt không dùng khoảng trắng trước dấu câu.
    """
    text = ""
    for i, w in enumerate(words):
        if i == 0:
            text = w
        elif w and w[0] in ".,?!;:":
            text = text + w
        else:
            text = text + " " + w
    return text.strip()


# ────────────────────────────────────────────────────────────────────────────
# FALLBACK: Text-only alignment (giữ nguyên từ v3 — dùng khi forced align thất bại)
# ────────────────────────────────────────────────────────────────────────────
_SENTENCE_END = re.compile(r'[.?!。？！]$')
_CLAUSE_END   = re.compile(r'[,;]$')


def _find_cut_point(words: List[str], budget: int, slack: int = 8) -> int:
    n = len(words)
    lo = max(1, budget - slack)
    hi = min(n - 1, budget + slack)
    if lo >= hi:
        return min(budget, n)

    best_sent, best_sent_dist = None, slack + 1
    for i in range(lo, hi + 1):
        if _SENTENCE_END.search(words[i - 1]):
            dist = abs(i - budget)
            if dist < best_sent_dist:
                best_sent_dist = dist
                best_sent = i
    if best_sent is not None:
        return best_sent

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


def live_turns_to_word_timestamps(
    live_turns : List["AlignedTurn"],
) -> List[WordTimestamp]:
    """
    Chuyển danh sách live Zipformer turns → per-word timestamps.

    Mục đích: thay thế forced_align (stable-ts) khi bản đó không có sẵn,
    nhưng VẪN feed được vào `assign_word_speakers` + `group_words_to_turns`
    (chung pipeline với forced-align path) thay vì rơi xuống ratio-based
    chia text trên TOÀN BỘ transcript (rất dễ misassign).

    Vì sao tốt hơn ratio-based-trên-full-text?
      - `t_end` của Zipformer là MỐC THẬT (vị trí frame khi detect silence).
      - Ta sửa `t_start` của mỗi live turn = `t_end` của turn trước đó →
        chuỗi turns liền mạch, KHÔNG overlap (loại trừ bug fake start time).
      - Trong mỗi turn, chia đều thời gian cho các từ → mỗi từ có 1 cặp
        (t_start, t_end) gần đúng nhưng không vượt khỏi biên của turn.
      - Sau đó `assign_word_speakers` map TỪNG TỪ tới pyannote segment có
        overlap lớn nhất → speaker label đúng theo thời gian thật.

    Hạn chế: nếu 1 live turn của Zipformer chứa LỜI CỦA 2 NGƯỜI NÓI
    (do không có khoảng lặng ≥ rule1 giữa họ), thì các từ trong turn đó
    vẫn được chia đều theo thời gian → ranh giới có thể lệch 1-2 từ.
    Đây là giới hạn của ASR streaming, không phải của alignment.
    """
    if not live_turns:
        return []

    # Sort theo t_end (mốc thật, không phải t_start vốn là ước lượng từ word count)
    sorted_turns = sorted(live_turns, key=lambda t: t.end)

    # Anchor: t_start của turn n = t_end của turn n-1 (chuỗi liền mạch, không overlap)
    prev_end = 0.0
    out: List[WordTimestamp] = []
    for lt in sorted_turns:
        words = lt.text.split()
        if not words:
            prev_end = lt.end
            continue

        # ── PATH 1: REAL per-word timestamps từ Zipformer (sherpa-onnx) ─────
        # Khi có → chính xác đến mức millisecond → assign_word_speakers sẽ chia
        # turn dài chứa 2+ người nói thành các sub-turn ĐÚNG ranh giới.
        real_ts = getattr(lt, "word_starts", None) or []
        if real_ts and len(real_ts) == len(words):
            for i, w in enumerate(words):
                ws = real_ts[i]
                we = real_ts[i + 1] if (i + 1) < len(real_ts) else lt.end
                # Bảo đảm we >= ws
                if we < ws:
                    we = ws + 0.05
                out.append(WordTimestamp(word=w, start=round(ws, 3), end=round(we, 3)))
            prev_end = lt.end
            continue

        # ── PATH 2: fallback synthetic interpolation (khi không có real ts) ─
        anchored_start = max(prev_end, max(0.0, lt.end - len(words) * 0.6))
        if anchored_start >= lt.end:
            anchored_start = max(0.0, lt.end - 0.5)
        turn_dur = max(0.05, lt.end - anchored_start)
        per_word = turn_dur / len(words)

        for i, w in enumerate(words):
            ws = anchored_start + i * per_word
            we = anchored_start + (i + 1) * per_word
            out.append(WordTimestamp(word=w, start=round(ws, 3), end=round(we, 3)))

        prev_end = lt.end

    return out


def _align_text_only(
    segments  : List[SpeakerSegment],
    full_text : str,
    slack     : int = 10,
) -> List[AlignedTurn]:
    """Fallback ratio-based alignment (v3). Dùng khi forced align thất bại."""
    remaining_words = full_text.strip().split()
    has_punctuation = bool(re.search(r'[.?!,;]', full_text))
    effective_slack = slack if has_punctuation else 0
    n_segs = len(segments)
    turns  = []

    for i, seg in enumerate(segments):
        duration = seg.end - seg.start
        is_last  = (i == n_segs - 1)

        if is_last or not remaining_words:
            chunk           = remaining_words
            remaining_words = []
        else:
            remaining_segs     = segments[i:]
            remaining_duration = sum(s.end - s.start for s in remaining_segs)
            remaining_turns    = n_segs - i

            ratio  = duration / remaining_duration if remaining_duration > 0 else 1.0
            budget = max(1, round(ratio * len(remaining_words)))

            min_words_per_turn = 2
            words_left_after   = len(remaining_words) - budget
            turns_left_after   = remaining_turns - 1
            if turns_left_after > 0 and words_left_after < turns_left_after * min_words_per_turn:
                budget = max(1, len(remaining_words) - turns_left_after * min_words_per_turn)

            if effective_slack > 0:
                cut = _find_cut_point(remaining_words, budget, effective_slack)
            else:
                cut = min(budget, len(remaining_words))

            cut = min(cut, len(remaining_words) - max(0, n_segs - i - 1))
            cut = max(1, cut)

            chunk           = remaining_words[:cut]
            remaining_words = remaining_words[cut:]

        turns.append(AlignedTurn(
            speaker = seg.speaker,
            start   = seg.start,
            end     = seg.end,
            text    = " ".join(chunk).strip(),
        ))

    return turns


# ────────────────────────────────────────────────────────────────────────────
# HÀM CHÍNH: align() — API công khai, giống v3 để không phá app.py
# ────────────────────────────────────────────────────────────────────────────
def align(
    segments          : List[SpeakerSegment],
    full_text         : str,
    slack             : int   = 10,
    whisper_segs      : Optional[List] = None,  # không dùng nữa, giữ để tương thích
    overlap_threshold : float = 0.3,
    # Tham số mới:
    wav_path          : Optional[str] = None,   # path file WAV để forced align
    language          : str           = "vi",
    use_forced_align  : bool          = True,   # tắt để debug / fallback
    gap_limit         : float         = 1.5,    # truyền vào group_words_to_turns
    live_turns        : Optional[List["AlignedTurn"]] = None,   # FIX align quality
) -> List[AlignedTurn]:
    """
    Ghép nối text từ Whisper vào pyannote segments.

    API tương thích với v3 — caller không cần thay đổi gì ngoài thêm wav_path.

    LUỒNG MỚI (khi wav_path được cung cấp):
        1. forced_align(wav_path, text) → word timestamps
        2. assign_word_speakers(words, segments) → word + speaker
        3. group_words_to_turns(word_speaker_list) → AlignedTurn list

    FALLBACK (khi wav_path=None hoặc forced align thất bại):
        → _align_text_only() như v3

    Args:
        segments         : Speaker segments từ pyannote
        full_text        : Full transcript text từ Whisper
        wav_path         : (MỚI) Path file WAV để forced alignment.
                           Nếu None → fallback về ratio-based.
        language         : Mã ngôn ngữ cho forced aligner ("vi", "en", ...)
        use_forced_align : False → bỏ qua forced align, dùng fallback luôn.
        gap_limit        : Giây để gộp từ cùng speaker thành 1 turn.
        slack / whisper_segs / overlap_threshold : giữ cho tương thích v3.

    Returns:
        List[AlignedTurn]
    """
    if not segments:
        if full_text.strip():
            # Khi không diarize: thử forced_align để lấy timestamp thật
            if use_forced_align and wav_path:
                words = forced_align(wav_path, full_text, language=language)
                if words:
                    real_end = words[-1].end
                    return [AlignedTurn(
                        speaker = "SPEAKER_00",
                        start   = words[0].start,
                        end     = real_end,
                        text    = full_text.strip(),
                    )]
            # Fallback: không có wav_path hoặc forced_align thất bại
            return [AlignedTurn(
                speaker = "SPEAKER_00",
                start   = 0.0,
                end     = 0.0,
                text    = full_text.strip(),
            )]
        return []

    if not full_text.strip():
        return [
            AlignedTurn(speaker=s.speaker, start=s.start, end=s.end, text="")
            for s in segments
        ]

    # ── Chiến lược 1: Forced alignment (mới) ────────────────────────────────
    if use_forced_align and wav_path:
        words = forced_align(wav_path, full_text, language=language)

        if words:
            word_speakers = assign_word_speakers(words, segments)
            turns = group_words_to_turns(word_speakers, gap_limit=gap_limit)
            if turns:
                logger.info(f"[aligner] Forced align thành công: {len(turns)} turns")
                return turns
            else:
                logger.warning("[aligner] group_words_to_turns trả về [] → fallback")
        else:
            logger.warning("[aligner] forced_align trả về [] → fallback")

    # ── Chiến lược 2: live-turns-as-pseudo-words ────────────────────────────
    # Khi stable-ts không có (env Sortformer không cài được), dùng timestamps
    # xấp xỉ từ Zipformer live turns để tạo "fake forced alignment" → chạy
    # vào cùng pipeline assign_word_speakers + group_words_to_turns. Tốt hơn
    # nhiều so với chia text theo tỷ lệ duration (mất hoàn toàn ngữ cảnh từ).
    if live_turns:
        pseudo_words = live_turns_to_word_timestamps(live_turns)
        if pseudo_words:
            word_speakers = assign_word_speakers(pseudo_words, segments)
            turns = group_words_to_turns(word_speakers, gap_limit=gap_limit)
            if turns:
                logger.info(f"[aligner] live-turns pseudo-align: {len(turns)} turns "
                            f"(stable-ts không có sẵn)")
                return turns

    # ── Fallback cuối: Ratio-based (v3) ─────────────────────────────────────
    logger.info("[aligner] Dùng ratio-based fallback (v3)")
    return _align_text_only(segments, full_text, slack)


# ────────────────────────────────────────────────────────────────────────────
# Hàm tiện ích — giữ nguyên từ v3 để tương thích
# ────────────────────────────────────────────────────────────────────────────
def rename_turns(
    turns    : List[AlignedTurn],
    name_map : dict,
) -> List[AlignedTurn]:
    """Áp dụng name_map vào danh sách AlignedTurn. Trả về list mới."""
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


def merge_consecutive(
    turns     : List[AlignedTurn],
    gap_limit : float = 1.5,
) -> List[AlignedTurn]:
    """Gộp turns liên tiếp cùng speaker nếu gap < gap_limit."""
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


def smooth_short_turns(
    turns       : List[AlignedTurn],
    max_words   : int   = 4,
    max_dur     : float = 1.2,
    gap_limit   : float = 1.5,
) -> List[AlignedTurn]:
    """Gộp các turn NGẮN bị kẹp giữa 2 turn cùng speaker (mẫu A → b → A).

    Sửa lỗi 'phân mảnh' khi nói nhanh/chồng tiếng: 1 câu của 1 người bị diarizer
    cắt vụn sang nhiều speaker (vd 'Mượn / của bố / và mượn' → 00/02/00).
    Turn b ngắn (≤ max_words từ HOẶC ≤ max_dur giây) kẹp giữa A→A → gán về A,
    rồi gộp lại. Lặp tới khi ổn định. Trả list mới.
    """
    if len(turns) < 3:
        return turns

    changed = True
    cur = list(turns)
    while changed:
        changed = False
        out = [cur[0]]
        i = 1
        while i < len(cur) - 1:
            prev, mid, nxt = out[-1], cur[i], cur[i + 1]
            n_words = len(mid.text.split())
            dur     = mid.end - mid.start
            if (prev.speaker == nxt.speaker
                    and mid.speaker != prev.speaker
                    and (n_words <= max_words or dur <= max_dur)):
                # gán mid về speaker A (prev) → sẽ được gộp ở merge sau
                out.append(AlignedTurn(speaker=prev.speaker, start=mid.start,
                                       end=mid.end, text=mid.text, confidence=mid.confidence))
                changed = True
            else:
                out.append(mid)
            i += 1
        out.append(cur[-1])
        cur = merge_consecutive(out, gap_limit=gap_limit)
    return cur


def format_timestamp(seconds: float) -> str:
    """Chuyển giây thành MM:SS.ss."""
    minutes = int(seconds // 60)
    secs    = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


def parse_whisper_segments(raw_segments) -> Tuple[List[WhisperSegment], str]:
    """Giữ để tương thích với code cũ."""
    w_segs = []
    texts  = []
    for seg in raw_segments:
        text  = seg.get("text", "").strip()
        start = float(seg.get("start", 0.0))
        end   = float(seg.get("end", 0.0))
        if text:
            w_segs.append(WhisperSegment(text=text, start=start, end=end))
            texts.append(text)
    return w_segs, " ".join(texts)


# ────────────────────────────────────────────────────────────────────────────
# TEST
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  TEST: aligner.py  v4  (WhisperX-style)")
    print("=" * 60)

    mock_segments = [
        SpeakerSegment(speaker="SPEAKER_00", start=0.0,  end=5.2),
        SpeakerSegment(speaker="SPEAKER_01", start=5.5,  end=12.0),
        SpeakerSegment(speaker="SPEAKER_00", start=12.3, end=16.8),
        SpeakerSegment(speaker="SPEAKER_01", start=17.1, end=25.0),
        SpeakerSegment(speaker="SPEAKER_00", start=25.4, end=28.0),
    ]

    # Test assign_word_speakers với mock word timestamps
    print("\n[1] Test assign_word_speakers()...")
    mock_words = [
        WordTimestamp(word="Anh",    start=0.3,  end=0.6),
        WordTimestamp(word="cho",    start=0.7,  end=0.9),
        WordTimestamp(word="biết",   start=1.0,  end=1.3),
        WordTimestamp(word="tên",    start=1.4,  end=1.6),
        WordTimestamp(word="tuổi",   start=1.7,  end=2.0),
        WordTimestamp(word="Tôi",    start=5.8,  end=6.1),
        WordTimestamp(word="tên",    start=6.2,  end=6.5),
        WordTimestamp(word="Nguyễn", start=6.6,  end=7.0),
        WordTimestamp(word="Văn",    start=7.1,  end=7.3),
        WordTimestamp(word="A",      start=7.4,  end=7.6),
        WordTimestamp(word="Anh",    start=12.5, end=12.8),
        WordTimestamp(word="có",     start=12.9, end=13.1),
        WordTimestamp(word="mặt",    start=13.2, end=13.5),
        WordTimestamp(word="ở",      start=13.6, end=13.8),
        WordTimestamp(word="đâu",    start=13.9, end=14.3),
        WordTimestamp(word="vào",    start=14.4, end=14.6),
        WordTimestamp(word="tối",    start=14.7, end=15.0),
        WordTimestamp(word="Tôi",    start=17.3, end=17.6),
        WordTimestamp(word="ở",      start=17.7, end=17.9),
        WordTimestamp(word="nhà",    start=18.0, end=18.3),
        WordTimestamp(word="suốt",   start=18.4, end=18.8),
        WordTimestamp(word="tối",    start=18.9, end=19.2),
        WordTimestamp(word="hôm",    start=19.3, end=19.5),
        WordTimestamp(word="đó",     start=19.6, end=19.9),
        WordTimestamp(word="Có",     start=25.5, end=25.8),
        WordTimestamp(word="ai",     start=25.9, end=26.1),
        WordTimestamp(word="xác",    start=26.2, end=26.5),
        WordTimestamp(word="nhận",   start=26.6, end=27.0),
        WordTimestamp(word="không",  start=27.1, end=27.5),
    ]

    word_speakers = assign_word_speakers(mock_words, mock_segments)
    print(f"    ✅ {len(word_speakers)} từ được gán speaker")
    for ws in word_speakers[:5]:
        print(f"       [{ws['start']:.1f}-{ws['end']:.1f}] {ws['speaker']:12s} | {ws['word']}")
    print(f"       ...")

    print("\n[2] Test group_words_to_turns()...")
    turns = group_words_to_turns(word_speakers, gap_limit=1.5)
    print(f"    ✅ {len(turns)} turns")
    for t in turns:
        ts = format_timestamp(t.start)
        print(f"    [{ts}] {t.speaker:12s} | {t.text}")

    print("\n[3] Test align() — fallback (không có wav_path)...")
    mock_text = (
        "Anh cho biết tên tuổi và địa chỉ thường trú. "
        "Tôi tên Nguyễn Văn A, sinh năm 1985, Hà Nội. "
        "Anh có mặt ở đâu vào tối ngày 15? "
        "Tôi ở nhà suốt buổi tối hôm đó. "
        "Có ai xác nhận không?"
    )
    turns_fallback = align(mock_segments, mock_text, wav_path=None)
    print(f"    ✅ {len(turns_fallback)} turns (ratio-based fallback)")
    for t in turns_fallback:
        ts = format_timestamp(t.start)
        print(f"    [{ts}] {t.speaker:12s} | {t.text}")

    print("\n[4] Test rename + merge...")
    name_map = {"SPEAKER_00": "Điều tra viên", "SPEAKER_01": "Đối tượng"}
    renamed  = rename_turns(turns, name_map)
    merged   = merge_consecutive(renamed, gap_limit=2.0)
    print(f"    ✅ {len(turns)} turns → {len(merged)} sau merge")
    for t in merged:
        print(f"    {t.speaker:15s} | {format_timestamp(t.start)} | {t.text[:60]}")

    print("\n[5] Test stable-ts availability...")
    sw = _get_stable_ts()
    if sw:
        print(f"    ✅ stable-ts khả dụng ({sw.__version__})")
        print(f"    ✅ Forced alignment sẵn sàng — chạy align() với wav_path để kích hoạt")
    else:
        print("    ⚠️  stable-ts chưa cài → pip install stable-ts")
        print("    ℹ️  Sẽ dùng ratio-based fallback")

    print("\n✅ Test xong — aligner.py v4 sẵn sàng\n")