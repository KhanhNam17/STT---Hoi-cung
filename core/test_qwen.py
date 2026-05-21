# core/summarizer.py
#
# ── Root cause của lỗi "Context length exceeded" ────────────────────────────
#
# Với file 1 tiếng → ~13000-15000 từ → 15 chunks × 900 từ/chunk
#
# Luồng cũ (bị lỗi):
#   Tầng 1: merge(chunk[0:5]) → summary_A  (max_tokens=800 → ~400-600 từ output)
#            merge(chunk[5:10]) → summary_B (~400-600 từ)
#            merge(chunk[10:15]) → summary_C (~400-600 từ)
#   Tầng 2: merge(A + B + C)
#            = 3 × 500 từ × 2.5 token/từ = 3750 tokens input
#            + system_prompt 340 + template 65 + output 800 = 4955 tokens → TRÀN 🔴
#
# Fix:
#   1. max_tokens KHÁC NHAU theo vai trò:
#      chunk summary     → max_tokens=120  (3-5 bullet, đủ cô đọng)
#      merge trung gian  → max_tokens=180  (giữ ngắn để tầng sau không tràn)
#      merge cuối cùng   → max_tokens=700  (bản tóm tắt hoàn chỉnh, cho phép dài)
#   2. max_chunks_per_merge = 4 (thay vì 5) để có thêm buffer
#   3. Truncate input của mỗi summary trước khi đưa vào merge nếu vẫn dài

import os
import re
import time
import requests
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

# ── Cấu hình ──────────────────────────────────────────────────────────────────
NEXA_LLM_MODEL = os.getenv("NEXA_LLM_MODEL", "NexaAI/Qwen3-4B-Instruct-2507-npu")
_LLM_URL       = "http://127.0.0.1:18183/v1/chat/completions"
_TIMEOUT       = int(os.getenv("NEXA_LLM_TIMEOUT", "300"))

# ── Token budget (Qwen3-4B context = 4096) ───────────────────────────────────
#
# Tiếng Việt thực tế: ~2.0–2.5 tokens/từ (có dấu thanh điệu)
# Dùng 2.5 làm hệ số an toàn (worst case).
#
# Budget phân bổ:
#   System prompt CHUNK  : ~72  tokens (fixed)
#   System prompt MERGE  : ~340 tokens (fixed)
#   User template        : ~65  tokens (fixed)
#   Output reserve CHUNK : 120  tokens (3–5 bullet points)
#   Output reserve MERGE_INTERMEDIATE: 180 tokens
#   Output reserve MERGE_FINAL: 700 tokens
#
# → Input budget cho CHUNK  = 4096 − 72 − 65 − 120 = 3839 tokens ≈ 1535 từ
# → Max chunk size an toàn = 700 từ (buffer thêm 50%)
# → Max input cho MERGE = 4096 − 340 − 65 − 700 = 2991 tokens
# → Với 4 summaries/merge và max_tokens_intermediate=180 (270 tokens max/summary)
#    4 × 270 = 1080 tokens → tổng = 1080 + 405 = 1485 tokens → rất an toàn ✅

_MAX_WORDS_DIRECT    = 1200   # dưới ngưỡng này: gọi 1 lần trực tiếp
_MAX_CHUNK_WORDS     = 700    # kích thước mỗi chunk (giảm từ 900 → 700)
_MAX_CHUNKS_PER_MERGE = 4     # số chunk tối đa mỗi lần merge (giảm từ 5 → 4)

# max_tokens theo từng giai đoạn
_MAX_TOKENS_CHUNK_SUMMARY    = 120   # map: 3-5 bullet points
_MAX_TOKENS_MERGE_INTERMEDIATE = 180  # reduce tầng giữa: giữ cô đọng
_MAX_TOKENS_MERGE_FINAL      = 700   # reduce tầng cuối: bản đầy đủ
_MAX_TOKENS_DIRECT           = 700   # direct: bản đầy đủ

# Giới hạn độ dài mỗi summary trước khi đưa vào merge (safeguard)
# = (budget_merge - fixed_overhead) / max_chunks_per_merge / token_ratio
_MAX_WORDS_PER_SUMMARY_INPUT = 130   # ~130 từ × 2.5 = 325 tokens/summary


# ── System Prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPT_CHUNK = """Tóm tắt đoạn hội thoại dưới đây.
Yêu cầu: Viết ngắn gọn tối đa 4 gạch đầu dòng, chỉ giữ sự kiện cốt lõi. Tuyệt đối không tóm tắt các câu chữ mang tính chất biểu mẫu hành chính, pháp lý (như khai báo lý lịch, luật tố tụng, cách ghi biên bản)."""

SYSTEM_PROMPT_MERGE_INTERMEDIATE = """Tổng hợp các gạch đầu dòng sau thành danh sách ngắn gọn hơn.
Giữ tối đa 5 điểm quan trọng nhất. Tuyệt đối không đưa các nội dung về quy trình ghi chép biên bản hành chính vào."""

SYSTEM_PROMPT_MERGE_FINAL = """Bạn là biên tập viên. Dưới đây là các tóm tắt từng phần của một cuộc trò chuyện. Hãy tổng hợp thành bài hoàn chỉnh theo cấu trúc:

1. CÁC LUẬN ĐIỂM CHÍNH: Nội dung thảo luận (bullet point).

ĐIỀU KIỆN RÀNG BUỘC (TUYỆT ĐỐI TUÂN THỦ):
- Không dùng dấu sao (**) để in đậm văn bản. Không dùng Markdown.
- Bỏ qua mọi thông tin về quy cách lập biên bản, ghi âm hay gạch chéo giấy trắng.
- Chỉ in ra kết quả tóm tắt cuối cùng. Không tự ý giải thích, không chép lại yêu cầu của hệ thống ở cuối bài."""


# ── Data class ────────────────────────────────────────────────────────────────
@dataclass
class SummaryResult:
    summary     : str
    strategy    : str    # "direct" | "map_reduce"
    n_chunks    : int
    input_words : int
    elapsed_sec : float
    ok          : bool
    error       : str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────
def _strip_think_tags(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def _count_words(text: str) -> int:
    return len(text.split())

def _truncate_to_words(text: str, max_words: int) -> str:
    """Cắt text xuống còn max_words từ, thêm '...' nếu bị cắt."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."

def _build_transcript_text(turns) -> str:
    lines = []
    for t in turns:
        if hasattr(t, "start"):
            start, speaker, text = t.start, t.speaker, t.text
        else:
            start   = t.get("start", 0)
            speaker = t.get("speaker", "")
            text    = t.get("text", "")
        ts = f"[{int(start // 60):02d}:{int(start % 60):02d}]"
        lines.append(f"{ts} {speaker}: {text}")
    return "\n".join(lines)

def _chunk_transcript(text: str, max_words: int = _MAX_CHUNK_WORDS) -> list[str]:
    """Chia transcript tại ranh giới dòng (turn), không cắt giữa lượt nói."""
    lines, chunks, current, count = text.splitlines(), [], [], 0
    for line in lines:
        w = len(line.split())
        if count + w > max_words and current:
            chunks.append("\n".join(current))
            current, count = [line], w
        else:
            current.append(line)
            count += w
    if current:
        chunks.append("\n".join(current))
    return chunks


# ── Gọi API ───────────────────────────────────────────────────────────────────
def _call_llm(
    system_prompt : str,
    user_content  : str,
    max_tokens    : int = 400,
    timeout       : int = _TIMEOUT,
) -> str:
    """
    Gọi /v1/chat/completions với max_tokens kiểm soát chặt chẽ.
    max_tokens được truyền vào từng lời gọi theo vai trò (chunk/merge/final).
    """
    payload = {
        "model"      : NEXA_LLM_MODEL,
        "messages"   : [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        "temperature": 0.2,
        "max_tokens" : max_tokens,
    }
    try:
        r = requests.post(
            _LLM_URL,
            headers = {"Content-Type": "application/json"},
            json    = payload,
            timeout = timeout,
        )
        if r.status_code != 200:
            return f"[ERROR] Server báo lỗi {r.status_code}: {r.text}"
        content = r.json()["choices"][0]["message"]["content"]
        return _strip_think_tags(content)
    except requests.exceptions.ConnectionError:
        return "[ERROR] Không kết nối được cổng 18183. Hãy chắc chắn Nexa serve đang chạy!"
    except Exception as e:
        return f"[ERROR] {e}"


# ── Map phase ─────────────────────────────────────────────────────────────────
def _summarize_chunk(chunk_text: str, idx: int, total: int) -> str:
    user = (
        f"Đoạn {idx}/{total}:\n\n"
        f"{chunk_text}\n\n"
        f"Tóm tắt ngắn gọn (tối đa 4 gạch đầu dòng)."
    )
    return _call_llm(
        system_prompt = SYSTEM_PROMPT_CHUNK,
        user_content  = user,
        max_tokens    = _MAX_TOKENS_CHUNK_SUMMARY,  # 120 tokens — đủ cho 4 bullet
    )


# ── Reduce phase ──────────────────────────────────────────────────────────────
def _merge_batch(
    summaries   : list[str],
    is_final    : bool = False,
) -> str:
    """
    Gộp một batch summaries.

    is_final=False → intermediate merge: output ngắn (max_tokens=180)
                     để tầng tiếp theo không tràn context.
    is_final=True  → final merge: output đầy đủ (max_tokens=700).

    Trước khi merge, mỗi summary được truncate về _MAX_WORDS_PER_SUMMARY_INPUT từ
    → đảm bảo tổng input không bao giờ vượt context limit.
    """
    # Truncate từng summary để đảm bảo không tràn
    safe_summaries = [
        _truncate_to_words(s, _MAX_WORDS_PER_SUMMARY_INPUT)
        for s in summaries
    ]

    # Đánh số phần để model hiểu cấu trúc
    combined = "\n\n".join(
        f"[{i+1}] {s}" for i, s in enumerate(safe_summaries)
    )

    if is_final:
        user = (
            f"Các tóm tắt từng phần:\n\n{combined}\n\n"
            f"Tổng hợp thành bài tóm tắt hoàn chỉnh."
        )
        return _call_llm(
            system_prompt = SYSTEM_PROMPT_MERGE_FINAL,
            user_content  = user,
            max_tokens    = _MAX_TOKENS_MERGE_FINAL,
        )
    else:
        user = f"Gộp các điểm sau thành danh sách súc tích:\n\n{combined}"
        return _call_llm(
            system_prompt = SYSTEM_PROMPT_MERGE_INTERMEDIATE,
            user_content  = user,
            max_tokens    = _MAX_TOKENS_MERGE_INTERMEDIATE,
        )


def _tiered_reduce(
    summaries        : list[str],
    progress_callback,
    pct_start        : int = 82,
    pct_end          : int = 97,
) -> str:
    """
    Tiered reduce với max_tokens được kiểm soát chặt theo từng tầng.

    Tầng giữa (intermediate): max_tokens=180 → output ngắn → tầng sau không tràn.
    Tầng cuối (final):        max_tokens=700 → output đầy đủ.

    Ví dụ với 15 chunks, batch=4:
        Tầng 1: [0:4]→A, [4:8]→B, [8:12]→C, [12:15]→D  (intermediate, 4 lần gọi)
        Tầng 2: [A,B,C,D] → final                        (final, 1 lần gọi)
    """
    current = summaries
    layer   = 1

    while len(current) > 1:
        pct_layer = pct_start + int((pct_end - pct_start) * (1 - len(current) / len(summaries)))

        # Tầng cuối khi batch tiếp theo sẽ cho ra 1 phần tử duy nhất
        will_be_final = (
            len(current) <= _MAX_CHUNKS_PER_MERGE
        )
        role = "final" if will_be_final else f"tầng {layer}"

        if progress_callback:
            progress_callback(
                pct_layer,
                f"Reduce {role}: {len(current)} phần → "
                f"{-(-len(current) // _MAX_CHUNKS_PER_MERGE)} batch...",
            )
        print(f"   [reduce] {role}: {len(current)} summaries, "
              f"max_tokens={'final' if will_be_final else 'intermediate'}")

        next_layer = []
        for i in range(0, len(current), _MAX_CHUNKS_PER_MERGE):
            batch     = current[i : i + _MAX_CHUNKS_PER_MERGE]
            is_final  = will_be_final and (i + _MAX_CHUNKS_PER_MERGE >= len(current))
            result    = _merge_batch(batch, is_final=is_final)

            if result.startswith("[ERROR]"):
                return result  # caller xử lý

            # Log token estimate để debug
            est_tokens = _count_words(result) * 2.5
            print(f"   [reduce] batch {i//4+1}: {_count_words(result)} từ ≈ {est_tokens:.0f} tokens")

            next_layer.append(result)

        current = next_layer
        layer  += 1

    return current[0]


# ── Hàm chính ─────────────────────────────────────────────────────────────────
def summarize(
    turns,
    progress_callback = None,
    timeout           : int = _TIMEOUT,
) -> SummaryResult:
    t0 = time.perf_counter()

    def _progress(pct: int, msg: str):
        if progress_callback:
            progress_callback(pct, msg)
        print(f"   [{pct:3d}%] {msg}")

    if not turns:
        return SummaryResult(
            summary="Không có nội dung để tóm tắt.",
            strategy="none", n_chunks=0, input_words=0,
            elapsed_sec=0.0, ok=False, error="Transcript rỗng",
        )

    transcript_text = _build_transcript_text(turns)
    n_words         = _count_words(transcript_text)
    _progress(5, f"Transcript: {n_words} từ")

    # ── Direct ────────────────────────────────────────────────────────────────
    if n_words <= _MAX_WORDS_DIRECT:
        _progress(10, "Tóm tắt trực tiếp...")
        user    = f"Nội dung:\n\n{transcript_text}\n\nTóm tắt theo cấu trúc yêu cầu."
        summary = _call_llm(SYSTEM_PROMPT_MERGE_FINAL, user, max_tokens=_MAX_TOKENS_DIRECT, timeout=timeout)

        elapsed = round(time.perf_counter() - t0, 2)
        if summary.startswith("[ERROR]"):
            return SummaryResult(summary="", strategy="direct", n_chunks=1,
                                 input_words=n_words, elapsed_sec=elapsed,
                                 ok=False, error=summary)
        _progress(98, "Hoàn tất!")
        return SummaryResult(summary=summary, strategy="direct", n_chunks=1,
                             input_words=n_words, elapsed_sec=elapsed, ok=True)

    # ── Map-Reduce ────────────────────────────────────────────────────────────
    chunks = _chunk_transcript(transcript_text, max_words=_MAX_CHUNK_WORDS)
    n      = len(chunks)
    _progress(5, f"Chia thành {n} chunks × ≤{_MAX_CHUNK_WORDS} từ...")

    # Map
    partial = []
    for i, chunk in enumerate(chunks):
        pct  = 10 + int(70 * i / n)
        _progress(pct, f"Chunk {i+1}/{n} ({_count_words(chunk)} từ)...")

        part = _summarize_chunk(chunk, i + 1, n)
        if part.startswith("[ERROR]"):
            elapsed = round(time.perf_counter() - t0, 2)
            return SummaryResult(summary="", strategy="map_reduce", n_chunks=i + 1,
                                 input_words=n_words, elapsed_sec=elapsed,
                                 ok=False, error=f"Lỗi chunk {i+1}: {part}")

        est = _count_words(part) * 2.5
        print(f"   Chunk {i+1}/{n}: {_count_words(part)} từ ≈ {est:.0f} tokens → OK")
        partial.append(part)

    # Reduce
    _progress(82, f"Reduce {n} summaries...")
    summary = _tiered_reduce(partial, progress_callback=progress_callback)

    elapsed = round(time.perf_counter() - t0, 2)
    if summary.startswith("[ERROR]"):
        return SummaryResult(summary="", strategy="map_reduce", n_chunks=n,
                             input_words=n_words, elapsed_sec=elapsed,
                             ok=False, error=f"Lỗi reduce: {summary}")

    _progress(99, "Hoàn tất!")
    return SummaryResult(summary=summary, strategy="map_reduce", n_chunks=n,
                         input_words=n_words, elapsed_sec=elapsed, ok=True)


# ── Tiện ích ──────────────────────────────────────────────────────────────────
def check_nexa_available() -> tuple[bool, str]:
    try:
        r = requests.get("http://127.0.0.1:18183/", timeout=3)
        return True, f"Nexa serve sẵn sàng (model: {NEXA_LLM_MODEL})"
    except requests.exceptions.ConnectionError:
        return False, "Nexa serve chưa chạy tại cổng 18183"
    except Exception as e:
        return False, str(e)


# ── TEST ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os

    print("=" * 65)
    print("  TEST: summarizer.py — Fixed context budget")
    print("=" * 65)
    print(f"  MAX_CHUNK_WORDS           : {_MAX_CHUNK_WORDS}")
    print(f"  MAX_CHUNKS_PER_MERGE      : {_MAX_CHUNKS_PER_MERGE}")
    print(f"  MAX_TOKENS_CHUNK          : {_MAX_TOKENS_CHUNK_SUMMARY}")
    print(f"  MAX_TOKENS_MERGE_INTERMED : {_MAX_TOKENS_MERGE_INTERMEDIATE}")
    print(f"  MAX_TOKENS_MERGE_FINAL    : {_MAX_TOKENS_MERGE_FINAL}")
    print(f"  MAX_WORDS_PER_SUMMARY_INPUT: {_MAX_WORDS_PER_SUMMARY_INPUT}")

    try:
        import docx
    except ImportError:
        print("\n❌ pip install python-docx")
        sys.exit(1)

    test_file = "data/podcast_2/Bien_ban_HS-20260520-1000.docx"
    if not os.path.exists(test_file):
        print(f"\n❌ Không tìm thấy {test_file}")
        sys.exit(1)

    print(f"\n[1] Đọc {test_file}...")
    doc     = docx.Document(test_file)
    content = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    class MockTurn:
        def __init__(self, speaker, start, text):
            self.speaker, self.start, self.text = speaker, start, text

    mock_turns = [MockTurn("Speaker", 0.0, content)]
    n = _count_words(content)
    chunks_est = -(-n // _MAX_CHUNK_WORDS)
    print(f"   {n} từ → dự kiến {chunks_est} chunks")
    print(f"   Tầng reduce dự kiến: {-(-chunks_est // _MAX_CHUNKS_PER_MERGE)} batches tầng 1 → 1 final")

    def on_progress(pct, msg):
        print(f"   [{pct:3d}%] {msg}")

    print("\n[2] Tóm tắt...\n")
    result = summarize(mock_turns, progress_callback=on_progress)

    print(f"\n[KẾT QUẢ]")
    print(f"  Trạng thái : {'✅ Thành công' if result.ok else '❌ Thất bại'}")
    print(f"  Chiến lược : {result.strategy}")
    print(f"  Số chunk   : {result.n_chunks}")
    print(f"  Thời gian  : {result.elapsed_sec}s")
    if result.ok:
        print(f"\n{'━'*65}")
        print(result.summary)
        print('━'*65)
    else:
        print(f"  Lỗi: {result.error}")