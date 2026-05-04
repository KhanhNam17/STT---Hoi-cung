# core/punctuation_restorer.py
#
# Mục đích: Khôi phục dấu câu cho output của Qualcomm Whisper
# Giải pháp: Rule-based (Chạy offline, không cần internet, 0ms overhead)
#
# Cải tiến: 
#   - Loại bỏ các liên từ gây ngắt câu sai (để, mà, nên)
#   - Tối ưu bộ từ khóa câu hỏi
#   - Bảo vệ viết hoa tên riêng từ Whisper

import re

# ────────────────────────────────────────────────────────────────────────────
# Hằng số — từ điển từ khoá tiếng Việt
# ────────────────────────────────────────────────────────────────────────────

# Từ bắt đầu câu hỏi → thêm "?" ở cuối câu chứa chúng
_QUESTION_STARTERS = (
    r"(?:tại sao|vì sao|tại vì sao|lý do gì|lý do nào)"
    r"|(?:như thế nào|thế nào|ra sao|thế ra|vậy thì)"
    r"|(?:ở đâu|đâu|khi nào|bao giờ|ai|cái gì|bao nhiêu|đúng không|phải không)"
)

# Liên từ nối mệnh đề chính → chỉ giữ lại các từ thực sự cần ngắt nghỉ
# Đã loại bỏ "mà", "nên", "để" để tránh lỗi ngắt câu sai ngữ pháp
_CLAUSE_CONJUNCTIONS = [
    "nhưng mà", "nhưng", "tuy nhiên", "tuy vậy", "thế nhưng",
    "vì vậy", "vì thế", "do đó", "cho nên",
    "bởi vì", "bởi thế", "bởi vậy",
    "mặc dù", "dù vậy", "dù sao",
    "thế mà", "vậy mà"
]

# Từ/cụm thường bắt đầu câu mới trong hội thoại
_SENTENCE_STARTERS = [
    "thưa", "vâng", "dạ", "đúng rồi", "đúng vậy",
    "thật ra", "thực ra", "theo tôi", "tôi nghĩ",
    "ví dụ", "chẳng hạn", "đầu tiên", "cuối cùng"
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
    if not text:
        return text
    
    text = text[0].upper() + text[1:]

    def _cap(m):
        return m.group(1) + " " + m.group(2).upper()

    text = re.sub(r"([\.?!])\s+([a-zàáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỷỹ])", 
                  _cap, text)
    return text

def _add_question_marks(text: str) -> str:
    """Thêm "?" vào cuối câu hỏi nếu đang thiếu."""
    sentences = re.split(r"(?<=[\.?!])\s+", text)
    result = []
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
    """Tách các câu dài bị dính liền nhau dựa trên từ khóa bắt đầu."""
    for starter in _SENTENCE_STARTERS:
        pattern = r"(\w(?:[^\.?!\n]){20,}?)\s+(?<![\.?!,])(" + re.escape(starter) + r"\b)"
        text = re.sub(pattern, 
                      lambda m: m.group(1).rstrip() + ". " + m.group(2).capitalize(), 
                      text, flags=re.IGNORECASE | re.UNICODE)
    return text

def _protect_proper_nouns(text: str) -> str:
    """
    Sửa lỗi đặc thù của Whisper nhưng bảo vệ tên riêng.
    Không còn ép viết thường toàn bộ cụm từ viết hoa để giữ lại 'Hà Anh Tuấn', 'Thùy Minh'.
    """
    # Chỉ xóa lặp từ (ví dụ: "cà phê cà phê" -> "cà phê")
    text = re.sub(r"\b(\w{3,})\s+\1\b", r"\1", text, flags=re.IGNORECASE | re.UNICODE)
    return text

# ────────────────────────────────────────────────────────────────────────────
# Hàm chính: restore()
# ────────────────────────────────────────────────────────────────────────────

def restore(text: str) -> str:
    """
    Hàm chính khôi phục dấu câu cho transcript.
    """
    if not text or not text.strip():
        return text

    text = _normalize_spaces(text)
    text = _protect_proper_nouns(text)
    text = _fix_existing_punctuation(text)
    text = _add_commas_before_conjunctions(text)
    text = _add_question_marks(text)
    text = _split_run_on_sentences(text)
    text = _capitalize_sentences(text)
    text = _normalize_spaces(text)

    return text

# ────────────────────────────────────────────────────────────────────────────
# TEST 
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_text = "tại sao lại là Vestong hình ảnh của Hà Anh Tuấn là chiếc áo vest nhưng mà mình cảm thấy tự tin vì vậy hôm nay mời Tuấn ly cà phê"
    print("INPUT :", test_text)
    print("OUTPUT:", restore(test_text))
