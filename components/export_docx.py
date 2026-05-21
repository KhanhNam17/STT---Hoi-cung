# components/export_docx.py
#
# Xuất transcript/tóm tắt vào template biên bản hỏi cung thật (bienbanhoicung.docx).
# Template có sẵn các placeholder:
#   {{NGAY}}   {{THANG}}   {{NAM}}       — ngày tháng năm
#   {{NOI_DUNG_TRANSCRIPT}}              — chèn nội dung hỏi đáp/tóm tắt vào đây

import io
import os
from datetime import datetime

from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

COLORS_RGB = [
    RGBColor(0xE8, 0x52, 0x0A),   # cam
    RGBColor(0x1A, 0x6F, 0xAD),   # xanh
    RGBColor(0x2E, 0x8B, 0x2E),   # lá
    RGBColor(0x8B, 0x2E, 0x8B),   # tím
    RGBColor(0x8B, 0x69, 0x14),   # vàng đậm
]

DEFAULT_TEMPLATE = "components/templates/bienbanhoicung.docx"

# ── 1. Hàm xuất Bản Tóm Tắt (MỚI) ───────────────────────────────────────────
def export_summary_to_docx(
    summary_text: str,
    session_name: str = "BB_01",
    template_path: str = DEFAULT_TEMPLATE,
) -> bytes:
    """
    Điền đoạn văn bản tóm tắt thuần (plain text) vào template biên bản.
    Format chuẩn: Times New Roman, cỡ 14, căn đều 2 bên.
    """
    if not os.path.exists(template_path):
        raise FileNotFoundError(
            f"Không tìm thấy file mẫu tại: {template_path}\n"
            "Đảm bảo file bienbanhoicung.docx nằm trong thư mục templates/"
        )

    doc  = Document(template_path)
    now  = datetime.now()

    # 1. Điền ngày tháng năm
    replacements = {
        "{{NGAY}}":  now.strftime("%d"),
        "{{THANG}}": now.strftime("%m"),
        "{{NAM}}":   now.strftime("%Y"),
    }

    for p in doc.paragraphs:
        for key, val in replacements.items():
            if key in p.text:
                for run in p.runs:
                    if key in run.text:
                        run.text = run.text.replace(key, val)

    # 2. Tìm placeholder nội dung transcript và chèn Tóm tắt
    target_p = None
    for p in doc.paragraphs:
        if "{{NOI_DUNG_TRANSCRIPT}}" in p.text:
            target_p = p
            # Xoá chữ placeholder
            for run in p.runs:
                run.text = run.text.replace("{{NOI_DUNG_TRANSCRIPT}}", "")
            break

    # 3. Chèn text tóm tắt
    if target_p is not None:
        # Tách đoạn tóm tắt thành các dòng
        paragraphs = summary_text.split('\n')
        
        # Chèn xuôi chiều (insert_paragraph_before liên tục sẽ đẩy text lên đúng thứ tự)
        for text_line in paragraphs:
            if text_line.strip():
                new_p = target_p.insert_paragraph_before(text_line.strip())
                new_p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                for run in new_p.runs:
                    run.font.name = 'Times New Roman'
                    run.font.size = Pt(14)
    else:
        # Fallback nếu không tìm thấy thẻ
        doc.add_paragraph("--- NỘI DUNG TÓM TẮT ---").alignment = WD_ALIGN_PARAGRAPH.CENTER
        for text_line in summary_text.split('\n'):
            if text_line.strip():
                new_p = doc.add_paragraph(text_line.strip())
                new_p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                for run in new_p.runs:
                    run.font.name = 'Times New Roman'
                    run.font.size = Pt(14)

    # 4. Xuất ra bytes
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ── 2. Hàm xuất Full Transcript gốc (Giữ nguyên dự phòng) ───────────────────
def export_to_docx(
    turns,
    session_name: str = "BB_01",
    template_path: str = DEFAULT_TEMPLATE,
) -> bytes:
    """
    Điền toàn bộ transcript (từng lượt nói) vào template biên bản hỏi cung.
    """
    if not os.path.exists(template_path):
        raise FileNotFoundError(
            f"Không tìm thấy file mẫu tại: {template_path}\n"
        )

    doc  = Document(template_path)
    now  = datetime.now()

    replacements = {
        "{{NGAY}}":  now.strftime("%d"),
        "{{THANG}}": now.strftime("%m"),
        "{{NAM}}":   now.strftime("%Y"),
    }

    for p in doc.paragraphs:
        for key, val in replacements.items():
            if key in p.text:
                for run in p.runs:
                    if key in run.text:
                        run.text = run.text.replace(key, val)

    target_p = None
    for p in doc.paragraphs:
        if "{{NOI_DUNG_TRANSCRIPT}}" in p.text:
            target_p = p
            for run in p.runs:
                run.text = run.text.replace("{{NOI_DUNG_TRANSCRIPT}}", "")
            break

    speakers  = list(dict.fromkeys(t.speaker for t in turns))
    color_map = {
        spk: COLORS_RGB[i % len(COLORS_RGB)]
        for i, spk in enumerate(speakers)
    }

    if target_p is not None:
        ref_p = target_p._p          
        for turn in turns:           
            new_p_elem = _build_turn_xml(turn, color_map)
            ref_p.addnext(new_p_elem)
            ref_p = new_p_elem       
    else:
        doc.add_paragraph("--- NỘI DUNG HỎI VÀ ĐÁP ---").alignment = WD_ALIGN_PARAGRAPH.CENTER
        for turn in turns:
            new_p = doc.add_paragraph()
            _write_turn(new_p, turn, color_map)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _build_turn_xml(turn, color_map: dict):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    color = color_map.get(turn.speaker, RGBColor(0, 0, 0))
    try:
        hex_color = f"{color.red:02X}{color.green:02X}{color.blue:02X}"
    except AttributeError:
        hex_color = "000000"

    ts = f"[{int(turn.start // 60):02d}:{int(turn.start % 60):02d}]  "

    def _run(text, bold=False, size_pt=11, hex_c=None):
        r = OxmlElement("w:r")
        rPr = OxmlElement("w:rPr")
        rFonts = OxmlElement("w:rFonts")
        rFonts.set(qn("w:ascii"), "Times New Roman")
        rFonts.set(qn("w:hAnsi"), "Times New Roman")
        rPr.append(rFonts)
        sz = OxmlElement("w:sz");   sz.set(qn("w:val"), str(int(size_pt * 2)))
        szCs = OxmlElement("w:szCs"); szCs.set(qn("w:val"), str(int(size_pt * 2)))
        rPr.append(sz); rPr.append(szCs)
        if bold:
            rPr.append(OxmlElement("w:b"))
        if hex_c:
            c_el = OxmlElement("w:color"); c_el.set(qn("w:val"), hex_c)
            rPr.append(c_el)
        r.append(rPr)
        t_el = OxmlElement("w:t")
        t_el.set(qn("xml:space"), "preserve")
        t_el.text = text
        r.append(t_el)
        return r

    p = OxmlElement("w:p")
    pPr = OxmlElement("w:pPr")
    jc = OxmlElement("w:jc"); jc.set(qn("w:val"), "both")
    pPr.append(jc)
    p.append(pPr)

    p.append(_run(ts,              bold=False, size_pt=10, hex_c="999999"))
    p.append(_run(f"{turn.speaker}:  ", bold=True,  size_pt=11, hex_c=hex_color))
    p.append(_run(turn.text,       bold=False, size_pt=11))
    return p


def _write_turn(p, turn, color_map: dict):
    ts = f"[{int(turn.start // 60):02d}:{int(turn.start % 60):02d}]  "
    color = color_map.get(turn.speaker, RGBColor(0, 0, 0))

    r_ts = p.add_run(ts)
    r_ts.font.size = Pt(10)
    r_ts.font.color.rgb = RGBColor(0x99, 0x99, 0x99)

    r_spk = p.add_run(f"{turn.speaker}:  ")
    r_spk.bold = True
    r_spk.font.size = Pt(11)
    r_spk.font.color.rgb = color

    r_txt = p.add_run(turn.text)
    r_txt.font.size = Pt(11)