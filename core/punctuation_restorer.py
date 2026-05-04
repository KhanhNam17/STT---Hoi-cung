# core/punctuation_restorer.py
#
# Mục đích: Khôi phục dấu câu cho output của Qualcomm Whisper (Bản Rút Gọn)
# Giải pháp: Rule-based (Chạy offline, không cần internet, 0ms overhead)
#
# Cách dùng:
#   from core.punctuation_restorer import restore
#   clean_text = restore(raw_text)

import re

# ────────────────────────────────────────────────────────────────────────────
# Hằng số — từ điển từ khoá tiếng Việt
# ────────────────────────────────────────────────────────────────────────────

# Từ bắt đầu câu hỏi → thêm "?" ở cuối câu chứa chúng
_QUESTION_STARTERS = (
    r"(?:tại sao|vì sao|tại vì sao|lý do gì|lý do nào)"
    r"|(?:như thế nào|thế nào|ra sao|thế ra|vậy thì)"
    r"|(?:ở đâu|nơi nào|chỗ nào|đâu)"
    r"|(?:khi nào|lúc nào|bao giờ|bao lâu|bao nhiêu lâu)"
    r"|(?:ai|người nào|những ai)"
    r"|(?:cái gì|điều gì|việc gì|chuyện gì|thứ gì|món gì)"
    r"|(?:bao nhiêu|mấy|bao giờ)"
    r"|(?:có phải|có đúng|có không|đúng không|phải không|vậy không|thật không|thế không)"
    r"|(?:được không|có được|có thể không|có thể)"
)

# Liên từ nối mệnh đề → thêm "," phía trước nếu chưa có
_CLAUSE_CONJUNCTIONS = [
    "nhưng mà", "nhưng", "mà", "tuy nhiên", "tuy vậy", "thế nhưng",
    "vì vậy", "vì thế", "do đó", "cho nên", "nên",
    "bởi vì", "bởi thế", "bởi vậy",
    "mặc dù", "dù vậy", "dù sao",
    "thế mà", "vậy mà",
    "còn", "và", "hoặc", "hay là", "hay",
]

# Từ/cụm thường bắt đầu câu mới trong hội thoại
_SENTENCE_STARTERS = [
    "thưa", "xin chào", "xin hỏi", "vâng", "dạ", "ừ", "ừm",
    "đúng rồi", "đúng vậy", "đúng là", "đúng",
    "thật ra", "thực ra", "thực chất", "thực tế",
    "theo tôi", "theo mình", "theo anh", "theo chị",
    "tôi nghĩ", "mình nghĩ", "tôi cho rằng", "mình cho rằng",
    "tôi thấy", "mình thấy", "tôi tin", "mình tin",
    "hôm nay", "hôm qua", "ngày hôm nay", "lúc đó", "khi đó",
    "ví dụ", "chẳng hạn", "cụ thể",
    "đầu tiên", "thứ nhất", "thứ hai", "thứ ba", "cuối cùng",
    "ngoài ra", "bên cạnh đó", "đồng thời",
]


# ────────────────────────────────────────────────────────────────────────────
# Xử lý Logic (Rule-based)
# ────────────────────────────────────────────────────────────────────────────

def _normalize_spaces(text: str) -> str:
    """Chuẩn hóa khoảng trắng thừa."""
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _fix_existing_punctuation(text: str) -> str:
    """Sửa dấu câu đã có nhưng đặt sai vị trí."""
    text = re.sub(r"\s+([,\.?!;:])", r"\1", text)
    text = re.sub(r"([,\.?!;:])([^\s\d\"\'])", r"\1 \2", text)
    text = re.sub(r"\.\s*,", ".", text)
    text = re.sub(r"\.{2,}", ".", text)
    return text


def _capitalize_sentences(text: str) -> str:
    """Viết hoa chữ đầu câu sau dấu chấm, chấm hỏi, chấm than."""
    if text:
        text = text[0].upper() + text[1:]

    def _cap(m):
        return m.group(1) + " " + m.group(2).upper()

    text = re.sub(r"([\.?!])\s+([a-zàáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỷỹ])", _cap, text)
    return text


def _add_question_marks(text: str) -> str:
    """Thêm "?" vào cuối câu hỏi nếu đang thiếu."""
    sentences = re.split(r"(?<=[\.?!])\s+", text)
    result    = []
    question_re = re.compile(r"^(?:" + _QUESTION_STARTERS + r")\b", flags=re.IGNORECASE | re.UNICODE)

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if (question_re.match(sent) and not sent.endswith("?") and len(sent.split()) >= 4):
            sent = sent.rstrip(".") + "?"
        result.append(sent)

    return " ".join(result)


def _add_commas_before_conjunctions(text: str) -> str:
    """Thêm dấu phẩy trước liên từ nối mệnh đề nếu chưa có."""
    for conj in _CLAUSE_CONJUNCTIONS:
        pattern = r"(\w[\w\s]{8,}?)(?<![,\.?!])\s+(" + re.escape(conj) + r"\b)"
        text = re.sub(pattern, r"\1, \2", text, flags=re.IGNORECASE | re.UNICODE)
    return text


def _split_run_on_sentences(text: str) -> str:
    """Tách các câu dài bị dính liền nhau."""
    for starter in _SENTENCE_STARTERS:
        pattern = r"(\w(?:[^\.?!\n]){20,}?)\s+(?<![\.?!,])(" + re.escape(starter) + r"\b)"
        text = re.sub(pattern, lambda m: m.group(1).rstrip() + ". " + m.group(2).capitalize(), text, flags=re.IGNORECASE | re.UNICODE)
    return text


def _fix_whisper_artifacts(text: str) -> str:
    """Sửa các lỗi đặc thù của Whisper (lặp từ, viết hoa giữa câu)."""
    text = re.sub(r"\b(\w{3,})\s+\1\b", r"\1", text, flags=re.IGNORECASE | re.UNICODE)
    text = re.sub(
        r"(?<=[a-zàáâãèéêìíòóôõùúýăđơư])\s+((?:[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯ][a-zàáâãèéêìíòóôõùúýăđơư]{1,}\s+){2,})", 
        lambda m: " " + m.group(1).lower(), 
        text
    )
    return text


# ────────────────────────────────────────────────────────────────────────────
# Hàm chính: restore()
# ────────────────────────────────────────────────────────────────────────────

def restore(text: str) -> str:
    """
    Khôi phục dấu câu cho transcript STT.
    
    Args:
        text: văn bản thô từ Whisper
        
    Returns:
        Văn bản đã được xử lý dấu câu theo quy tắc cứng (Rule-based).
    """
    if not text or not text.strip():
        return text

    text = _normalize_spaces(text)
    text = _fix_whisper_artifacts(text)
    text = _fix_existing_punctuation(text)
    text = _add_commas_before_conjunctions(text)
    text = _add_question_marks(text)
    text = _split_run_on_sentences(text)
    text = _capitalize_sentences(text)
    text = _normalize_spaces(text)   # chuẩn hóa lại lần cuối

    return text


# ────────────────────────────────────────────────────────────────────────────
# TEST 
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        ("tại sao lại là Vestong cái hình ảnh của mình này cái hình ảnh mà mình tin và mình cảm thấy tự tin đó là cái hình ảnh chiếc áo vest của mình", "Câu hỏi thiếu dấu ?"),
        ("Tôi ở nhà suốt buổi tối hôm đó không đi đâu cả có ai có thể xác nhận không", "Hỏi cung — thiếu dấu câu hoàn toàn"),
    ]

    print("=" * 65)
    print("  TEST: core/punctuation_restorer.py (Rule-based Only)")
    print("=" * 65)

    for i, (text, label) in enumerate(test_cases, 1):
        print(f"\n[{i}] {label}")
        print(f"  INPUT : {text[:90]}...")
        result = restore(text)
        print(f"  OUTPUT: {result[:90]}...")

    print("\n✅ Test xong\n")