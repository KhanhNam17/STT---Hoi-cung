import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.converter import convert_from_bytes, get_audio_info
from core.transcriber import load_model, transcribe_file
from core.diarizer import load_diarizer, diarize_file, get_speaker_stats
from core.aligner import align, merge_consecutive, rename_turns
from core.punctuation_restorer import restore

from components.transcript_viewer import preview as show_preview, full as show_full
from components.speaker_editor import speaker_editor
from components.export_docx import export_to_docx

@st.cache_resource(show_spinner=False)
def _get_stt():
    _, app = load_model()
    return app

@st.cache_resource(show_spinner=False)
def _get_diarizer():
    return load_diarizer()

_defaults = {
    "b_file_bytes"   : None,
    "b_file_name"    : "",
    "b_file_size"    : 0,
    "b_wav_path"     : None,
    "b_audio_info"   : {},
    "b_result_text"  : "",
    "b_turns"        : [],
    "b_stats"        : {},
    "b_name_map"     : {},
    "b_session_name" : "",
    "b_processed"    : False,
    "b_show_full"    : False,   # toggle xem toàn bộ transcript
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="font-size:11px;letter-spacing:3px;text-transform:uppercase;
            color:#8b949e;padding-bottom:12px;border-bottom:1px solid #30363d;
            margin-bottom:20px;">
    📁  Xử lý File Audio/Video — Batch Mode
</div>
""", unsafe_allow_html=True)

# ── Hàng 1: Upload + Hàng đợi ────────────────────────────────────────────────
col_up, col_q = st.columns(2, gap="medium")

with col_up:
    st.markdown('<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;'
                'color:#8b949e;margin-bottom:10px;">① Tải lên File Audio / Video</div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        label            = "file_upload",
        type             = ["mp3", "wav", "mp4", "mkv", "m4a", "aac"],
        label_visibility = "collapsed",
        help             = "Hỗ trợ: MP3 · WAV · MP4 · MKV · M4A · AAC",
    )

    # Lưu bytes ngay — Streamlit xóa uploaded object sau mỗi rerun
    if uploaded is not None:
        if uploaded.name != st.session_state.b_file_name:
            st.session_state.b_file_bytes   = uploaded.read()
            st.session_state.b_file_name    = uploaded.name
            st.session_state.b_file_size    = uploaded.size
            st.session_state.b_processed    = False
            st.session_state.b_turns        = []
            st.session_state.b_stats        = {}
            st.session_state.b_wav_path     = None
            st.session_state.b_audio_info   = {}
            st.session_state.b_result_text  = ""
            st.session_state.b_show_full    = False

    st.markdown("""
    <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:10px;">
        <span style="font-size:10px;padding:2px 10px;border-radius:20px;background:#1e2530;
                     border:1px solid #30363d;color:#8b949e;font-family:monospace;">.mp3</span>
        <span style="font-size:10px;padding:2px 10px;border-radius:20px;background:#1e2530;
                     border:1px solid #30363d;color:#8b949e;font-family:monospace;">.wav</span>
        <span style="font-size:10px;padding:2px 10px;border-radius:20px;background:#1e2530;
                     border:1px solid #30363d;color:#8b949e;font-family:monospace;">.mp4</span>
        <span style="font-size:10px;padding:2px 10px;border-radius:20px;background:#1e2530;
                     border:1px solid #30363d;color:#8b949e;font-family:monospace;">.mkv</span>
        <span style="font-size:10px;padding:2px 10px;border-radius:20px;background:#1e2530;
                     border:1px solid #30363d;color:#8b949e;font-family:monospace;">.m4a</span>
    </div>""", unsafe_allow_html=True)

with col_q:
    st.markdown('<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;'
                'color:#8b949e;margin-bottom:10px;">③ Hàng đợi xử lý</div>',
                unsafe_allow_html=True)

    if not st.session_state.b_file_name:
        st.markdown("""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
                    padding:32px;text-align:center;color:#484f58;font-size:13px;">
            Chưa có file nào được tải lên
        </div>""", unsafe_allow_html=True)
    else:
        done    = st.session_state.b_processed
        s_color = "#2e8b2e" if done else "#e8520a"
        s_bg    = "rgba(46,139,46,0.15)" if done else "rgba(232,82,10,0.15)"
        s_label = "✓ Xong" if done else "⏳ Chờ xử lý"
        mb      = st.session_state.b_file_size / 1_000_000

        st.markdown(f"""
        <div style="background:#161b22;border:1px solid #30363d;
                    border-radius:8px;overflow:hidden;">
            <div style="background:#1e2530;padding:8px 14px;font-size:10px;
                        letter-spacing:1.5px;text-transform:uppercase;color:#8b949e;
                        border-bottom:1px solid #30363d;">Danh sách file</div>
            <div style="padding:12px 14px;display:flex;align-items:center;gap:12px;">
                <span style="font-size:18px;">🎵</span>
                <span style="flex:1;font-size:12px;color:#e6edf3;overflow:hidden;
                             text-overflow:ellipsis;white-space:nowrap;">
                    {st.session_state.b_file_name}</span>
                <span style="font-size:10px;color:#8b949e;font-family:monospace;
                             white-space:nowrap;">{mb:.1f} MB</span>
                <span style="font-size:10px;font-weight:700;padding:2px 10px;
                             border-radius:10px;white-space:nowrap;
                             background:{s_bg};color:{s_color};
                             border:1px solid {s_color}55;">{s_label}</span>
            </div>
        </div>""", unsafe_allow_html=True)

        if st.session_state.b_audio_info.get("ok"):
            ai = st.session_state.b_audio_info
            st.markdown(
                f'<div style="margin-top:6px;font-size:11px;color:#484f58;'
                f'font-family:monospace;padding-left:2px;">'
                f'⏱ {ai["duration_str"]} &nbsp;·&nbsp; '
                f'🔊 {ai["sample_rate"]} Hz &nbsp;·&nbsp; Mono</div>',
                unsafe_allow_html=True,
            )

st.markdown("<hr style='border-color:#30363d;margin:20px 0 16px;'>",
            unsafe_allow_html=True)

# ── Hàng 2: Cấu hình ─────────────────────────────────────────────────────────
st.markdown('<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;'
            'color:#8b949e;margin-bottom:14px;">② Cấu hình xử lý</div>',
            unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4, gap="small")

with c1:
    language = st.selectbox(
        "Ngôn ngữ",
        options     = ["vi", "en", "auto"],
        format_func = lambda x: {"vi": "🇻🇳 Tiếng Việt",
                                  "en": "🇺🇸 English",
                                  "auto": "🌐 Tự động"}[x],
    )

with c2:
    enable_diar = st.toggle("Phân tách người nói", value=True)
    st.markdown(
        f'<span style="font-size:11px;color:{"#2e8b2e" if enable_diar else "#8b949e"};">'
        f'{"● Có" if enable_diar else "○ Không"}</span>',
        unsafe_allow_html=True,
    )

with c3:
    st.markdown("""
    <div style="margin-top:4px;">
        <div style="font-size:12px;color:#8b949e;margin-bottom:4px;">Model</div>
        <div style="font-size:13px;color:#e8520a;font-weight:600;">
            Whisper v3 Turbo
        </div>
    </div>""", unsafe_allow_html=True)

with c4:
    export_fmt = st.selectbox(
        "Định dạng xuất",
        options     = ["docx", "txt"],
        format_func = lambda x: {"docx": "📄 DOCX (mặc định)", "txt": "📋 TXT"}[x],
    )

col_spk, col_sess = st.columns([1, 2])
with col_spk:
    num_speakers = st.number_input(
        "Số người nói",
        min_value = 1, max_value = 8, value = 2,
        disabled  = not enable_diar,
        help      = "Để 2 cho phiên hỏi cung thông thường (điều tra viên + đối tượng)",
    ) if enable_diar else None

with col_sess:
    session_name = st.text_input(
        "Mã phiên / Tên hồ sơ",
        value       = st.session_state.b_session_name or f"HS-{time.strftime('%Y%m%d-%H%M')}",
        placeholder = "VD: HS-20250422-0930",
    )

# ── Nút xử lý ────────────────────────────────────────────────────────────────
st.markdown("<div style='margin:20px 0 8px;'></div>", unsafe_allow_html=True)

has_file = bool(st.session_state.b_file_bytes)
run_btn  = st.button(
    "▶️  Bắt đầu xử lý",
    type                = "primary",
    use_container_width = True,
    disabled            = not has_file,
)
if not has_file:
    st.caption("⬆️  Tải lên file audio/video để kích hoạt")

# ── Pipeline ──────────────────────────────────────────────────────────────────
if run_btn and has_file:
    st.session_state.b_processed = False
    st.session_state.b_turns     = []
    st.session_state.b_stats     = {}
    st.session_state.b_show_full = False
    st.session_state.b_session_name = session_name

    progress = st.progress(0, text="Đang chuẩn bị...")

    try:
        # B1: Convert
        progress.progress(5, text="Bước 1: Tiền xử lý file audio...")
        wav_path = convert_from_bytes(
            file_bytes        = st.session_state.b_file_bytes,
            original_filename = st.session_state.b_file_name,
            output_dir        = "data/uploads",
        )
        if not wav_path:
            st.error("❌ Convert thất bại — kiểm tra FFMPEG_PATH trong .env")
            st.stop()

        audio_info = get_audio_info(wav_path)
        st.session_state.b_wav_path   = wav_path
        st.session_state.b_audio_info = audio_info
        progress.progress(25, text="✅ Bước 1 — Convert xong!")

        # B2: Transcribe — truyền language từ UI vào model
        progress.progress(26, text="Bước 2: Nhận dạng giọng nói...")
        stt_app = _get_stt()

        # Cập nhật ngôn ngữ theo lựa chọn của người dùng
        # language="auto" → bỏ prefix, để Whisper tự detect
        lang_code = None if language == "auto" else language
        if hasattr(stt_app, "tokenizer") and hasattr(stt_app.tokenizer, "set_prefix_tokens"):
            stt_app.tokenizer.set_prefix_tokens(
                language = lang_code,
                task     = "transcribe",
            )

        result = transcribe_file(stt_app, wav_path)

        # Tích hợp dấu câu 
        progress.progress(45, text="Đang phục hồi dấu câu...")
        raw_text = result["text"]
        clean_text = restore(raw_text)

        st.session_state.b_result_text = clean_text
        progress.progress(50, text="✅ Bước 2 — Nhận dạng và phục hồi dấu câu xong!")

        # B3: Diarize
        segments = []
        if enable_diar:
            progress.progress(51, text="Bước 3: Phân tách người nói...")
            pipeline = _get_diarizer()
            segments = diarize_file(
                pipeline,
                wav_path,
                num_speakers = num_speakers,
                min_duration = 0.8,
                merge_gap    = 1.5,
            )
            st.session_state.b_stats = get_speaker_stats(segments)
            progress.progress(75, text="✅ Bước 3 — Phân tách người nói xong!")
        else:
            progress.progress(75, text="⏭️  Bước 3 — Bỏ qua phân tách người nói")

        # B4: Align
        progress.progress(76, text="Bước 4: Ghép nối transcript...")
        turns  = align(segments, clean_text)
        merged = merge_consecutive(turns, gap_limit=1.5)
        st.session_state.b_turns     = merged
        st.session_state.b_processed = True

        # Dọn file WAV tạm sau khi đã xử lý xong
        try:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
                st.session_state.b_wav_path = None
        except OSError:
            pass  # không crash nếu không xóa được

        n_turns = len(merged)
        n_words = len(result["text"].split())
        dur     = audio_info.get("duration_str", "?")
        lat     = result.get("latency", 0)

        progress.progress(
            100,
            text=(f"✅  Hoàn tất — {dur} audio · "
                  f"{n_turns} lượt nói · {n_words} từ · "
                  f"STT {lat:.1f}s"),
        )

    except Exception as e:
        st.error(f"❌ Lỗi pipeline: {e}")
        st.exception(e)
        st.stop()

# ── Gán nhãn người nói ────────────────────────────────────────────────────────
if st.session_state.b_processed and st.session_state.b_stats:
    st.markdown("<hr style='border-color:#30363d;margin:24px 0 16px;'>",
                unsafe_allow_html=True)
    st.markdown('<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;'
                'color:#8b949e;margin-bottom:14px;">Gán tên người nói</div>',
                unsafe_allow_html=True)

    name_map = speaker_editor(st.session_state.b_stats, key_prefix="b_spk")

    if st.button("✅ Xác nhận tên người nói"):
        st.session_state.b_turns    = rename_turns(st.session_state.b_turns, name_map)
        st.session_state.b_name_map = name_map
        st.rerun()

# ── Transcript review ─────────────────────────────────────────────────────────
if st.session_state.b_processed and st.session_state.b_turns:
    st.markdown("<hr style='border-color:#30363d;margin:24px 0 16px;'>",
                unsafe_allow_html=True)

    turns = st.session_state.b_turns

    # Header + nút toggle xem đầy đủ
    col_title, col_toggle = st.columns([3, 1])
    with col_title:
        st.markdown(
            '<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;'
            'color:#8b949e;margin-bottom:14px;">④ Kết quả — Transcript</div>',
            unsafe_allow_html=True,
        )
    with col_toggle:
        if st.button(
            "📄 Xem toàn bộ" if not st.session_state.b_show_full else "🔼 Thu gọn",
            use_container_width=True,
        ):
            st.session_state.b_show_full = not st.session_state.b_show_full
            st.rerun()

    if st.session_state.b_show_full:
        # Toàn bộ transcript, có thể chỉnh sửa từng dòng
        show_full(turns, editable=True)
        # Lưu lại text đã sửa (text_area cập nhật trực tiếp vào t.text qua editable mode)
        if st.button("💾 Lưu chỉnh sửa", type="primary"):
            st.session_state.b_turns = turns
            st.success("Đã lưu chỉnh sửa")
    else:
        # Preview 4 lượt đầu
        show_preview(turns, max_turns=4)

    st.markdown("<div style='margin:16px 0 8px;'></div>", unsafe_allow_html=True)

    # ── Xuất file ────────────────────────────────────────────────────────────
    stem = Path(st.session_state.b_file_name).stem

    # Layout nút xuất: 2 cột nếu chọn TXT, full width nếu chỉ DOCX
    if export_fmt == "txt":
        col_docx, col_txt = st.columns(2)
    else:
        col_docx = st.container()
        col_txt  = None

    with col_docx:
        try:
            docx_bytes = export_to_docx(
                turns        = st.session_state.b_turns,
                session_name = st.session_state.b_session_name,
            )
            st.download_button(
                label               = "📄  Xuất biên bản DOCX",
                data                = docx_bytes,
                file_name           = f"bien_ban_{stem}.docx",
                mime                = "application/vnd.openxmlformats-officedocument"
                                      ".wordprocessingml.document",
                type                = "primary",
                use_container_width = True,
            )
        except FileNotFoundError as e:
            st.error(str(e))

    if col_txt is not None:
        with col_txt:
            lines = [
                f"[{int(t.start//60):02d}:{int(t.start%60):02d}] {t.speaker}: {t.text}"
                for t in st.session_state.b_turns
            ]
            st.download_button(
                label               = "📋  Xuất TXT Transcript",
                data                = "\n".join(lines).encode("utf-8"),
                file_name           = f"transcript_{stem}.txt",
                mime                = "text/plain",
                use_container_width = True,
            )