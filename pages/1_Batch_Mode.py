import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.converter import convert_from_bytes, get_audio_info
from core.transcriber import load_model, transcribe_file
from core.diarizer import get_speaker_stats

# Backend diarization cho BATCH (độc lập với Live Mode).
#   BATCH_DIARIZATION_BACKEND = pyannote (mặc định) | sortformer | diart | nexa
#     pyannote   — offline pyannote 3.1, chính xác, ổn định (mặc định)
#     sortformer — NVIDIA Streaming Sortformer (SOTA, như WhisperLiveKit); cần NeMo
#     diart      — diart offline (kém hơn cho file)
#     nexa       — Nexa NPU CLI
# Live Mode vẫn đọc DIARIZATION_BACKEND riêng.
_DIAR_BACKEND = os.getenv("BATCH_DIARIZATION_BACKEND", "pyannote").strip().lower()
if _DIAR_BACKEND == "sortformer":
    # Dùng BRIDGE (subprocess sang sortformer_env) — KHÔNG import NeMo ở app env.
    from core.diarization.sortformer_bridge import load_diarizer_sortformer as load_diarizer
    from core.diarization.sortformer_bridge import diarize_file_sortformer as diarize_file
elif _DIAR_BACKEND == "diart":
    from core.diarization.streaming import load_diarizer_diart as load_diarizer
    from core.diarization.streaming import diarize_file_diart as diarize_file
elif _DIAR_BACKEND == "nexa":
    from core.diarizer_nexa import load_diarizer_nexa as load_diarizer
    from core.diarizer_nexa import diarize_file_nexa as diarize_file
else:   # pyannote — offline, chính xác nhất cho file
    _DIAR_BACKEND = "pyannote"
    from core.diarizer import load_diarizer, diarize_file
print(f"[BatchMode] Diarization backend = {_DIAR_BACKEND}")
from core.aligner import align, merge_consecutive, rename_turns, parse_whisper_segments
from core.punctuation_restorer import restore
from core.test_qwen import summarize

from components.transcript_viewer import preview as show_preview, full as show_full
from components.speaker_editor import speaker_editor
from components.export_docx import export_summary_to_docx

import warnings
import logging

warnings.filterwarnings("ignore")

logging.getLogger("pyannote").setLevel(logging.ERROR)
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("stable_whisper").setLevel(logging.ERROR)

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
    "b_show_full"    : False,
    "b_summary_text" : None, 
    "b_metrics"      : {},  # <-- Đã thêm biến lưu metrics
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.set_page_config(
    page_title = "Batch Mode — Hỏi Cung",
    page_icon  = "🎙️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="font-size:11px;letter-spacing:3px;text-transform:uppercase;
            color:var(--text-color);opacity:0.7;padding-bottom:12px;border-bottom:1px solid rgba(128,128,128,0.2);
            margin-bottom:20px;">
    📁  Xử lý Dữ liệu Ghi âm/Ghi hình
</div>
""", unsafe_allow_html=True)

# ── Hàng 1: Upload + Hàng đợi ────────────────────────────────────────────────
col_up, col_q = st.columns(2, gap="medium")

with col_up:
    st.markdown('<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;'
                'color:var(--text-color);opacity:0.7;margin-bottom:10px;">① Tải lên Dữ liệu ghi âm / ghi hình </div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        label            = "file_upload",
        type             = ["mp3", "wav", "mp4", "mkv", "m4a", "aac"],
        label_visibility = "collapsed",
    )

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
            st.session_state.b_summary_text = None
            st.session_state.b_metrics      = {}

    st.markdown("""
    <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:10px;">
        <span style="font-size:10px;padding:2px 10px;border-radius:20px;background:var(--secondary-background-color);
                     border:1px solid rgba(128,128,128,0.2);color:var(--text-color);opacity:0.8;font-family:monospace;">.mp3</span>
        <span style="font-size:10px;padding:2px 10px;border-radius:20px;background:var(--secondary-background-color);
                     border:1px solid rgba(128,128,128,0.2);color:var(--text-color);opacity:0.8;font-family:monospace;">.wav</span>
        <span style="font-size:10px;padding:2px 10px;border-radius:20px;background:var(--secondary-background-color);
                     border:1px solid rgba(128,128,128,0.2);color:var(--text-color);opacity:0.8;font-family:monospace;">.mp4</span>
        <span style="font-size:10px;padding:2px 10px;border-radius:20px;background:var(--secondary-background-color);
                     border:1px solid rgba(128,128,128,0.2);color:var(--text-color);opacity:0.8;font-family:monospace;">.mkv</span>
        <span style="font-size:10px;padding:2px 10px;border-radius:20px;background:var(--secondary-background-color);
                     border:1px solid rgba(128,128,128,0.2);color:var(--text-color);opacity:0.8;font-family:monospace;">.m4a</span>
    </div>""", unsafe_allow_html=True)

with col_q:
    st.markdown('<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;'
                'color:var(--text-color);opacity:0.7;margin-bottom:10px;">③ Hàng đợi xử lý</div>',
                unsafe_allow_html=True)

    if not st.session_state.b_file_name:
        st.markdown("""
        <div style="background:var(--secondary-background-color);border:1px solid rgba(128,128,128,0.2);border-radius:8px;
                    padding:32px;text-align:center;color:var(--text-color);opacity:0.7;font-size:13px;">
            Chưa có file nào được tải lên
        </div>""", unsafe_allow_html=True)
    else:
        done    = st.session_state.b_processed
        s_color = "#2e8b2e" if done else "#e8520a"
        s_bg    = "rgba(46,139,46,0.15)" if done else "rgba(232,82,10,0.15)"
        s_label = "✓ Xong" if done else "⏳ Chờ xử lý"
        mb      = st.session_state.b_file_size / 1_000_000

        st.markdown(f"""
        <div style="background:var(--background-color);border:1px solid rgba(128,128,128,0.2);
                    border-radius:8px;overflow:hidden;box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="background:var(--secondary-background-color);padding:8px 14px;font-size:10px;
                        letter-spacing:1.5px;text-transform:uppercase;color:var(--text-color);opacity:0.8;
                        border-bottom:1px solid rgba(128,128,128,0.2);">Danh sách file</div>
            <div style="padding:12px 14px;display:flex;align-items:center;gap:12px;">
                <span style="font-size:18px;">🎵</span>
                <span style="flex:1;font-size:12px;color:var(--text-color);font-weight:bold;overflow:hidden;
                             text-overflow:ellipsis;white-space:nowrap;">
                    {st.session_state.b_file_name}</span>
                <span style="font-size:10px;color:var(--text-color);opacity:0.7;font-family:monospace;
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
                f'<div style="margin-top:6px;font-size:11px;color:var(--text-color);opacity:0.8;'
                f'font-family:monospace;padding-left:2px;">'
                f'⏱ {ai["duration_str"]} &nbsp;·&nbsp; '
                f'🔊 {ai["sample_rate"]} Hz &nbsp;·&nbsp; Mono</div>',
                unsafe_allow_html=True,
            )

st.markdown("<hr style='border-color:rgba(128,128,128,0.2);margin:20px 0 16px;'>",
            unsafe_allow_html=True)

# ── Hàng 2: Cấu hình ─────────────────────────────────────────────────────────
st.markdown('<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;'
            'color:var(--text-color);opacity:0.7;margin-bottom:14px;">② Cấu hình xử lý</div>',
            unsafe_allow_html=True)

enable_diar = True

c1, c2 = st.columns(2, gap="medium")

with c1:
    language = st.selectbox(
        "Ngôn ngữ",
        options     = ["vi", "en", "auto"],
        format_func = lambda x: {"vi": "🇻🇳 Tiếng Việt",
                                  "en": "🇺🇸 English",
                                  "auto": "🌐 Tự động"}[x],
    )


with c2:
    st.text_input(
        "Định dạng xuất",
        value="📄 DOCX (mặc định)",
        disabled=True,
        help="Hệ thống mặc định xuất biên bản theo chuẩn DOCX"
    )
export_fmt = "docx"

col_spk, col_sess = st.columns([1, 2])
with col_spk:
    num_speakers = st.number_input(
        "Số người nói",
        min_value = 2, max_value = 8, value = 2,
        help      = "Mặc định là 2 (Điều tra viên + Đối tượng)",
    ) 

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
    st.session_state.b_metrics      = {}

    progress = st.progress(0, text="Đang chuẩn bị...")
    t_pipeline_start = time.perf_counter()


    try:
        # B1: Convert
        t0 = time.perf_counter()
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
        t_b1 = time.perf_counter() - t0
        progress.progress(25, text="✅ Bước 1 — Convert xong!")

        # B2: Transcribe
        t0 = time.perf_counter()
        progress.progress(26, text="Bước 2: Nhận dạng giọng nói...")
        stt_app = _get_stt()

        lang_code = None if language == "auto" else language
        if hasattr(stt_app, "tokenizer") and hasattr(stt_app.tokenizer, "set_prefix_tokens"):
            stt_app.tokenizer.set_prefix_tokens(
                language = lang_code,
                task     = "transcribe",
            )

        result = transcribe_file(stt_app, wav_path)
        raw_text = result["text"]
        st.session_state.b_result_text = raw_text
        t_b2 = time.perf_counter() - t0
        progress.progress(50, text="✅ Bước 2 — Nhận dạng và lấy mốc thời gian!")

        # B3: Diarize
        t0 = time.perf_counter()
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
            t_b3 = time.perf_counter() - t0
            progress.progress(75, text="✅ Bước 3 — Phân tách người nói xong!")
        else:
            t_b3 = 0.0
            progress.progress(75, text="⏭️  Bước 3 — Bỏ qua phân tách người nói")

        # B4: Align
        t0 = time.perf_counter()
        progress.progress(76, text="Bước 4: Ghép nối transcript...")
        turns  = align(segments, full_text=raw_text, wav_path=wav_path, language=language if language != "auto" else "vi")
        merged = merge_consecutive(turns, gap_limit=1.5)
        from core.aligner import smooth_short_turns
        merged = smooth_short_turns(merged, max_words=4, max_dur=1.2, gap_limit=1.5)
        t_b4 = time.perf_counter() - t0

        # B5: Punctuation
        t0 = time.perf_counter()
        progress.progress(90, text="Bước 5: Phục hồi dấu câu và chuẩn hoá...")
        for t in merged:
            t.text = restore(t.text)
        st.session_state.b_turns     = merged
        st.session_state.b_processed = True
        t_b5 = time.perf_counter() - t0

        try:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
                st.session_state.b_wav_path = None
        except OSError:
            pass 

        t_total = time.perf_counter() - t_pipeline_start
        dur_sec = audio_info.get("duration_sec", 1)
        dur_str = audio_info.get("duration_str", "?")
        n_words = len(raw_text.split())

        # <-- Thay đổi 1: LƯU METRICS VÀO SESSION STATE VÀ KHÔNG IN BẢNG BÁO CÁO -->
        st.session_state.b_metrics = {
            "dur_str": dur_str,
            "dur_sec": dur_sec,
            "n_words": n_words,
            "t_b1": t_b1,
            "t_b2": t_b2,
            "t_b3": t_b3,
            "t_b4": t_b4,
            "t_b5": t_b5,
            "t_stt_total": t_total
        }
        
        print(f"\n✅ STT hoàn tất ({dur_str}). Đang chờ quá trình Tóm tắt để in báo cáo tổng...\n")

        progress.progress(
            100,
            text=f"✅ Hoàn tất! Đã xử lý STT cho {dur_str} audio. Hãy tiếp tục gán tên và tóm tắt."
        )

    except Exception as e:
        st.error(f"❌ Lỗi pipeline: {e}")
        st.exception(e)
        st.stop()

# ── Gán nhãn người nói ────────────────────────────────────────────────────────
if st.session_state.b_processed and st.session_state.b_stats:
    st.markdown("<hr style='border-color:rgba(128,128,128,0.2);margin:24px 0 16px;'>",
                unsafe_allow_html=True)
    st.markdown('<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;'
                'color:var(--text-color);opacity:0.7;margin-bottom:14px;">Gán tên người nói</div>',
                unsafe_allow_html=True)

    name_map = speaker_editor(st.session_state.b_stats, key_prefix="b_spk")

    if st.button("✅ Xác nhận tên người nói"):
        st.session_state.b_turns    = rename_turns(st.session_state.b_turns, name_map)
        st.session_state.b_name_map = name_map
        st.rerun()

# ── Transcript review ─────────────────────────────────────────────────────────
if st.session_state.b_processed and st.session_state.b_turns:
    st.markdown("<hr style='border-color:rgba(128,128,128,0.2);margin:24px 0 16px;'>",
                unsafe_allow_html=True)

    turns = st.session_state.b_turns

    col_title, col_toggle = st.columns([3, 1])
    with col_title:
        st.markdown(
            '<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;'
            'color:var(--text-color);opacity:0.7;margin-bottom:14px;">④ Kết quả — Transcript</div>',
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
        show_full(turns, editable=True)
        if st.button("💾 Lưu chỉnh sửa", type="primary"):
            st.session_state.b_turns = turns
            st.success("Đã lưu chỉnh sửa")
    else:
        show_preview(turns, max_turns=4)

    st.markdown("<div style='margin:16px 0 8px;'></div>", unsafe_allow_html=True)

    # ── Tóm tắt & Xuất file ──────────────────────────────────────────────────
    st.markdown("<hr style='border-color:rgba(128,128,128,0.2);margin:24px 0 16px;'>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--text-color);opacity:0.7;margin-bottom:14px;">⑤ Tóm tắt & Xuất Biên Bản</div>', unsafe_allow_html=True)

    safe_session_name = st.session_state.b_session_name.replace(" ", "_")

    # <-- Thay đổi 2: TẠO THANH TIẾN TRÌNH VÀ IN BẢNG BÁO CÁO Ở ĐÂY -->
    if st.button("🤖 Chạy Tóm tắt Nội dung", use_container_width=True):
        summary_progress = st.progress(0, text="Khởi động NPU tóm tắt...")

        def update_progress(pct, msg):
            safe_pct = max(0, min(100, pct)) 
            summary_progress.progress(safe_pct, text=f"NPU: {msg}")

        result = summarize(st.session_state.b_turns, progress_callback=update_progress)
        
        if result.ok:
            st.session_state.b_summary_text = result.summary.strip()
            summary_progress.progress(100, text=f"✅ Đã tóm tắt thành công! ({result.elapsed_sec}s)")
            st.success("✅ Đã tóm tắt thành công!")

            m = st.session_state.b_metrics
            if m:
                t_b6 = result.elapsed_sec
                t_total_final = m["t_stt_total"] + t_b6
                dur_sec = m["dur_sec"]
                rtf = t_total_final / dur_sec if dur_sec > 0 else 0

                print("\n" + "═"*55)
                print(" 📊 BÁO CÁO HIỆU NĂNG HỆ THỐNG NPU (TOÀN QUY TRÌNH) ".center(55))
                print("═"*55)
                print(f" 🎵 Audio gốc : {m['dur_str']} | Tổng từ: {m['n_words']} từ")
                print(f" 🗣 Lượt nói  : {len(st.session_state.b_turns)} turns")
                print("─"*55)
                print(f" B1 | Convert file     : {m['t_b1']:>7.2f}s")
                print(f" B2 | Transcribe (Whisper)    : {m['t_b2']:>7.2f}s")
                print(f" B3 | Diarize          : {m['t_b3']:>7.2f}s")
                print(f" B4 | Align & Gom câu  : {m['t_b4']:>7.2f}s")
                print(f" B5 | Punctuation      : {m['t_b5']:>7.2f}s")
                print(f" B6 | Tóm tắt (Qwen)   : {t_b6:>7.2f}s")
                print("─"*55)
                print(f" ⏱  TỔNG THỜI GIAN   : {t_total_final:>7.2f}s")
                print(f" 🚀 HỆ SỐ RTF        : {rtf:>7.3f}x")
                print("═"*55 + "\n")
        else:
            summary_progress.empty() 
            st.error(f"❌ Lỗi khi tóm tắt: {result.error}")

    st.caption("ℹ️ Tóm tắt là TUỲ CHỌN (cần Qwen/NPU). Bỏ qua vẫn xuất được DOCX "
               "chứa transcript đầy đủ theo từng người nói.")

    if st.session_state.b_summary_text:
        st.text_area("Bản xem trước Tóm tắt (Sẽ được chèn vào Word):",
                     st.session_state.b_summary_text,
                     height=200)

    # Xuất được DOCX chỉ cần CÓ TRANSCRIPT — không bắt buộc phải có tóm tắt.
    is_ready = bool(st.session_state.b_turns)

    export_data = b""
    if is_ready:
        if st.session_state.b_summary_text:
            # Có tóm tắt → xuất bản tóm tắt
            export_data = export_summary_to_docx(
                summary_text = st.session_state.b_summary_text,
                session_name = st.session_state.b_session_name,
            )
        else:
            # Không tóm tắt → xuất full transcript (đa speaker) làm baseline
            from components.export_docx import export_to_docx
            export_data = export_to_docx(
                turns        = st.session_state.b_turns,
                session_name = st.session_state.b_session_name,
            )

    _has_summary = bool(st.session_state.b_summary_text)
    st.download_button(
        label               = "📄 Xuất biên bản DOCX (tóm tắt)" if _has_summary
                               else "📄 Xuất biên bản DOCX (transcript đầy đủ)",
        data                = export_data,
        file_name           = f"Bien_ban_{safe_session_name}.docx",
        mime                = "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        type                = "primary",
        disabled            = not is_ready,
        use_container_width = True,
    )