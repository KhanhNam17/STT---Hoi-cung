# pages/2_Live_Mode.py
#
# ══════════════════════════════════════════════════════════════════════════════
# KIẾN TRÚC REAL-TIME MICROSERVICES (STT + NPU DIARIZATION)
# ══════════════════════════════════════════════════════════════════════════════
#   Làn 1 (Main Thread): Thu âm → Zipformer → Ra chữ với nhãn "..." ngay lập tức.
#   Làn 2 (NPU Thread):  NPUDiarizer v3 (core/npu_diarizer.py) — Anchor Bank.
#   Đồng bộ:             Retro-update tự động vá tên người nói vào transcript cũ.
#
# FIX LOG v3:
#   [1] Bỏ num_speakers cứng → Pyannote tự detect
#   [2] NPUDiarizer module độc lập (core/npu_diarizer.py)
#   [3] Fix hiển thị speaker: normalize label, deduplicate color map
#   [4] Bỏ widget "Số người nói dự kiến"
#   [5] Panel log hoạt động real-time
#   [6] Anchor Bank v2: không quên speaker dù im lặng
#   [7] Finalize: _live_normalize_segments — không drop speaker nói ít
#   [8] NPUDiarizer v3: persistent map + energy reconcile + clip accumulator
# ══════════════════════════════════════════════════════════════════════════════

import os
import time
import queue as _queue_module
import threading
import warnings
import tempfile
import html as _html
import io
import wave

import numpy as np
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("SPEECHBRAIN_DISABLE_K2", "1")
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# 0. LAZY IMPORT
# ══════════════════════════════════════════════════════════════════════════════
def _try_import(name):
    try:
        import importlib
        return importlib.import_module(name)
    except ImportError:
        return None

sherpa_onnx = _try_import("sherpa_onnx")
sd          = _try_import("sounddevice")

LIVE_ENV_READY = all([sherpa_onnx, sd])

# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIG
# ══════════════════════════════════════════════════════════════════════════════
MODEL_DIR    = os.getenv("LIVE_MODEL_DIR",   "models/zipformer")
TOKENS_PATH  = os.getenv("LIVE_TOKENS",      f"{MODEL_DIR}/config.json")
ENCODER_PATH = os.getenv("LIVE_ENCODER",     f"{MODEL_DIR}/encoder-epoch-31-avg-11-chunk-32-left-128.fp16.onnx")
DECODER_PATH = os.getenv("LIVE_DECODER",     f"{MODEL_DIR}/decoder-epoch-31-avg-11-chunk-32-left-128.fp16.onnx")
JOINER_PATH  = os.getenv("LIVE_JOINER",      f"{MODEL_DIR}/joiner-epoch-31-avg-11-chunk-32-left-128.fp16.onnx")

SAMPLE_RATE  = int(os.getenv("LIVE_SAMPLE_RATE",  "16000"))
CHUNK_FRAMES = int(os.getenv("LIVE_CHUNK_FRAMES", "3200"))   # 200ms/chunk

NPU_API_URL  = os.getenv("NPU_API_URL", "http://127.0.0.1:18182/v1/audio/diarize")

# ══════════════════════════════════════════════════════════════════════════════
# 2. LOG SYSTEM — Ghi thẳng ra Terminal
# ══════════════════════════════════════════════════════════════════════════════
def _system_log(msg: str) -> None:
    """Ghi log thẳng ra cửa sổ Terminal đang chạy env_live"""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. SHARED RESOURCES
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_shared_resources():
    return {
        "audio_q": _queue_module.Queue(),
    }

_shared      = get_shared_resources()
_AUDIO_QUEUE = _shared["audio_q"]

# ══════════════════════════════════════════════════════════════════════════════
# 4. PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title = "Live Mode — Hỏi Cung",
    page_icon  = "🎙️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    footer { visibility: hidden; }
    .stButton > button[kind="primary"],
    .stDownloadButton > button[kind="primary"] {
        background: #e8520a !important; border: none !important;
        font-weight: 600 !important; color: white !important;
    }
    .stButton > button[kind="primary"]:hover { background: #c44008 !important; }
    .stProgress > div > div > div { background: #e8520a !important; }
    hr { border-color: rgba(128,128,128,0.2) !important; }
    @keyframes rec-blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
    .rec-dot {
        display: inline-block; width: 8px; height: 8px; border-radius: 50%;
        background: #e8520a; animation: rec-blink 1.2s ease-in-out infinite; margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 24px;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;
                    letter-spacing:3px;text-transform:uppercase;color:#e8520a;
                    margin-bottom:8px;">// HỆ THỐNG</div>
        <div style="font-size:18px;font-weight:700;color:var(--text-color);">Trợ lý Hỏi Cung</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                    color:var(--text-color);opacity:0.6;margin-top:6px;">STT · NPU DIARIZATION · DOCX</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.page_link("app.py",                label="🏠  Trang chủ")
    st.page_link("pages/1_Batch_Mode.py", label="📁  Xử lý File Audio/Video")
    st.page_link("pages/2_Live_Mode.py",  label="🎙️  Live Mode")

st.markdown("""
<div style="font-size:11px;letter-spacing:3px;text-transform:uppercase;
            color:var(--text-color);opacity:0.7;padding-bottom:12px;
            border-bottom:1px solid rgba(128,128,128,0.2);margin-bottom:20px;">
    🎙️  Thu âm Trực tiếp — Live Mode (NPU Accelerated)
</div>
""", unsafe_allow_html=True)

if not LIVE_ENV_READY:
    st.error("⚠️ Thiếu thư viện âm thanh. Hãy kiểm tra môi trường.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# 5. CACHE MODELS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def _get_zipformer():
    _system_log("⏳ Đang nạp mô hình Zipformer STT...")
    r = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=TOKENS_PATH, encoder=ENCODER_PATH, decoder=DECODER_PATH, joiner=JOINER_PATH,
        num_threads=2, sample_rate=SAMPLE_RATE, feature_dim=80,
        enable_endpoint_detection=True, rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2, rule3_min_utterance_length=20,
    )
    _system_log("✅ Zipformer nạp xong")
    return r

with st.spinner("Đang nạp mô hình STT..."):
    recognizer = _get_zipformer()

# ══════════════════════════════════════════════════════════════════════════════
# 6. NPU DIARIZER INSTANCE
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def _get_npu_diarizer():
    """NPUDiarizer v3 — Persistent map + Energy reconcile + Clip accumulator"""
    from core.npu_diarizer import NPUDiarizer
    cfg = {
        "api_url":              NPU_API_URL,
        "sample_rate":          SAMPLE_RATE,
        "step_sec":             5.0,
        "recent_sec":           30.0,
        "anchor_sec":           6.0,    # v3: giảm từ 8s → 6s, đổi lại bằng min_anchor
        "min_anchor_sec":       1.5,    # v3: tích lũy đủ 1.5s trước khi lưu anchor
        "max_total_anchor_sec": 36.0,   # v3: 6 spk × 6s
        "max_speakers":         10,
        "min_segment_duration": 0.3,
        "min_rms_for_anchor":   60,     # v3: hạ từ 80 → 60, tránh bỏ clip hợp lệ
        "energy_frame_ms":      50,     # v3: frame energy profile
        "min_similarity":       0.35,   # v3: ngưỡng cosine similarity reconcile
    }
    d = NPUDiarizer(config=cfg, log_callback=_system_log)
    return d

_diarizer = _get_npu_diarizer()
_diarizer._log_cb = _system_log  # Đảm bảo luồng callback trỏ ra Terminal

# ══════════════════════════════════════════════════════════════════════════════
# 7. SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_defaults = {
    "l_recording":    False,
    "l_finished":     False,
    "l_diar_done":    False,
    "l_renamed":      False,
    "l_stream_ref":   None,
    "l_asr_stream":   None,
    "l_turns":        [],
    "l_stats":        {},
    "l_name_map":     {},
    "l_raw_audio":    [],
    "l_n_chunks":     0,
    "l_partial":      "",
    "l_session_name": "",
    "l_show_full":    False,
    "l_summary":      "",
    "l_summary_done": False,
    "l_prev_win_count": 0,
    "l_show_log":     False,
    "_ui_log_lines":  [],   # buffer log cho UI panel (tối đa 200 dòng)
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Patch _system_log để ghi đồng thời vào UI buffer (thread-safe qua list append)
_orig_system_log = _system_log
_ui_log_lock = threading.Lock()

def _system_log(msg: str) -> None:   # type: ignore[no-redef]
    _orig_system_log(msg)
    ts   = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}" if not msg.startswith("[") else msg
    with _ui_log_lock:
        buf = st.session_state.get("_ui_log_lines", [])
        buf.append(line)
        if len(buf) > 200:
            del buf[:-200]
        st.session_state["_ui_log_lines"] = buf

# ══════════════════════════════════════════════════════════════════════════════
# 8. AUDIO CALLBACK (Làn 0)
# ══════════════════════════════════════════════════════════════════════════════
def _audio_callback(indata, frames, time_info, status):
    if status:
        _system_log(f"[Mic] Status: {status}")
    chunk_i16 = (indata[:, 0] * 32767).astype(np.int16)
    _AUDIO_QUEUE.put(chunk_i16.copy())
    _diarizer.push_chunk(chunk_i16)

# ══════════════════════════════════════════════════════════════════════════════
# 9. SPEAKER ASSIGNMENT & RETRO-UPDATE
# ══════════════════════════════════════════════════════════════════════════════
def _assign_speaker(t_start: float, t_end: float) -> str:
    return _diarizer.assign_speaker(t_start, t_end)

def _retro_update_speakers() -> int:
    turns   = st.session_state.get("l_turns", [])
    updated = 0
    for t in turns:
        spk = _assign_speaker(t.start, t.end)
        if spk != "..." and spk != t.speaker:
            t.speaker = spk
            updated += 1
    if updated:
        _system_log(f"[Retro] Vá {updated} câu với nhãn speaker mới")
    return updated

# ══════════════════════════════════════════════════════════════════════════════
# 10. STT DRAIN (Làn 1)
# ══════════════════════════════════════════════════════════════════════════════
def _drain_queue() -> tuple[str, list]:
    from core.aligner import AlignedTurn

    if st.session_state.l_asr_stream is None:
        st.session_state.l_asr_stream = recognizer.create_stream()

    z_stream  = st.session_state.l_asr_stream
    completed = []

    while not _AUDIO_QUEUE.empty():
        try:
            chunk = _AUDIO_QUEUE.get_nowait()
        except _queue_module.Empty:
            break

        st.session_state.l_raw_audio.append(chunk)
        st.session_state.l_n_chunks += 1

        samples = chunk.astype(np.float32) / 32768.0
        z_stream.accept_waveform(SAMPLE_RATE, samples)

        while recognizer.is_ready(z_stream):
            recognizer.decode_stream(z_stream)

        if recognizer.is_endpoint(z_stream):
            text = recognizer.get_result(z_stream).strip()
            if text:
                t_end   = round((st.session_state.l_n_chunks * CHUNK_FRAMES) / SAMPLE_RATE, 2)
                est_sec = max(0.5, len(text.split()) / 2.5)
                t_start = round(max(0.0, t_end - est_sec), 2)
                speaker = _assign_speaker(t_start, t_end)

                _system_log(
                    f"[STT] [{t_start:.1f}s→{t_end:.1f}s] "
                    f"speaker={speaker} | \"{text[:60]}{'...' if len(text)>60 else ''}\""
                )

                completed.append(AlignedTurn(
                    speaker=speaker,
                    start=t_start,
                    end=t_end,
                    text=text,
                ))
            recognizer.reset(z_stream)

    cur_wins = len(_diarizer.get_speaker_windows())
    if cur_wins > st.session_state.l_prev_win_count:
        _retro_update_speakers()
        st.session_state.l_prev_win_count = cur_wins

    partial = recognizer.get_result(z_stream).strip()
    return partial, completed

# ══════════════════════════════════════════════════════════════════════════════
# 11. SPEAKER COLOR MAP
# ══════════════════════════════════════════════════════════════════════════════
SPEAKER_COLORS = ["#e8520a", "#4da6e8", "#4ec94e", "#c97fe8", "#e8c14d",
                  "#e84d6e", "#4de8d4", "#e8b04d"]

def _build_color_map(turns) -> dict:
    seen = {}
    for t in turns:
        label = (t.speaker or "...").strip()
        # Không cấp màu chính thức vĩnh viễn cho những nhãn đang PENDING
        if label not in seen and label != "..." and not label.startswith("PENDING"):
            seen[label] = SPEAKER_COLORS[len(seen) % len(SPEAKER_COLORS)]
    seen["..."] = "#888888"
    return seen

# ══════════════════════════════════════════════════════════════════════════════
# 12. LIVE NORMALIZE
# ══════════════════════════════════════════════════════════════════════════════
def _live_normalize_segments(
    segments:     list,
    min_duration: float = 0.3,
    merge_gap:    float = 2.5,
) -> list:
    from core.diarizer import SpeakerSegment
    if not segments:
        return []

    filtered = [s for s in segments if (s.end - s.start) >= min_duration]
    dropped  = len(segments) - len(filtered)
    if dropped:
        _system_log(f"[Normalize] Drop {dropped} segment < {min_duration}s")
    if not filtered:
        return []

    merged = [filtered[0]]
    for cur in filtered[1:]:
        prev = merged[-1]
        if cur.speaker == prev.speaker and (cur.start - prev.end) <= merge_gap:
            merged[-1] = SpeakerSegment(
                speaker=prev.speaker,
                start=prev.start,
                end=cur.end,
                text=prev.text,
            )
        else:
            merged.append(cur)

    _system_log(
        f"[Normalize] {len(filtered)} → {len(merged)} segments | "
        f"speakers={sorted({s.speaker for s in merged})}"
    )
    return merged

# ══════════════════════════════════════════════════════════════════════════════
# 13. UI CONTROL PANEL
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;'
    'color:var(--text-color);opacity:0.7;margin-bottom:16px;">① Điều khiển phiên ghi âm</div>',
    unsafe_allow_html=True,
)

session_name = st.text_input(
    "Mã phiên / Tên hồ sơ",
    value=st.session_state.l_session_name or f"LIVE-{time.strftime('%Y%m%d-%H%M')}",
    disabled=st.session_state.l_recording,
)
if session_name != st.session_state.l_session_name:
    st.session_state.l_session_name = session_name

st.markdown("<div style='margin:12px 0 8px;'></div>", unsafe_allow_html=True)

btn_col1, btn_col2, status_col = st.columns([1, 1, 2], gap="small")

with btn_col1:
    if not st.session_state.l_recording:
        if st.button("🔴 Bắt đầu Ghi âm", type="primary", use_container_width=True):
            st.session_state.l_turns        = []
            st.session_state.l_stats        = {}
            st.session_state.l_name_map     = {}
            st.session_state.l_raw_audio    = []
            st.session_state.l_n_chunks     = 0
            st.session_state.l_partial      = ""
            st.session_state.l_finished     = False
            st.session_state.l_diar_done    = False
            st.session_state.l_renamed      = False
            st.session_state.l_summary      = ""
            st.session_state.l_summary_done = False
            st.session_state.l_asr_stream   = None
            st.session_state.l_prev_win_count = 0

            # ─── DỌN SẠCH UI RÁC TỪ PHIÊN TRƯỚC ───
            st.session_state.l_show_full = False
            keys_to_delete = [
                k for k in st.session_state.keys() 
                if k.startswith("turn_") or k.startswith("l_spk_")
            ]
            for k in keys_to_delete:
                del st.session_state[k]

            with _AUDIO_QUEUE.mutex:
                _AUDIO_QUEUE.queue.clear()

            _diarizer.reset()
            _diarizer.start()

            _system_log(
                f"🟢 Bắt đầu phiên ghi âm: {st.session_state.l_session_name} | "
                f"SR={SAMPLE_RATE}Hz | chunk={CHUNK_FRAMES}frames ({CHUNK_FRAMES/SAMPLE_RATE*1000:.0f}ms)"
            )

            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=CHUNK_FRAMES,
                callback=_audio_callback,
            )
            stream.start()
            st.session_state.l_stream_ref = stream
            st.session_state.l_recording  = True
            st.rerun()
    else:
        if st.button("⏹️ Dừng Ghi âm", use_container_width=True):
            if st.session_state.l_stream_ref:
                st.session_state.l_stream_ref.stop()
                st.session_state.l_stream_ref.close()
                st.session_state.l_stream_ref = None

            _diarizer.stop()

            dur_total = (st.session_state.l_n_chunks * CHUNK_FRAMES) / SAMPLE_RATE
            _system_log(
                f"🔴 Dừng ghi âm | tổng={dur_total:.1f}s | "
                f"câu STT={len(st.session_state.l_turns)} | "
                f"speakers_detected={sorted(_diarizer.stats['detected_speakers'])}"
            )

            st.session_state.l_asr_stream  = None
            st.session_state.l_recording   = False
            st.session_state.l_finished    = True
            st.rerun()

with btn_col2:
    btn_finalize_disabled = not st.session_state.l_finished or st.session_state.l_diar_done
    if st.button("✅ Hoàn thiện Transcript", disabled=btn_finalize_disabled, use_container_width=True):
        with st.spinner("Đang chuẩn hóa transcript..."):
            _retro_update_speakers()

            from core.diarizer import SpeakerSegment, get_speaker_stats
            from core.aligner import align, merge_consecutive
            from core.punctuation_restorer import restore

            npu_windows = _diarizer.get_speaker_windows()
            raw_segs    = [
                SpeakerSegment(speaker=w.label, start=w.start, end=w.end)
                for w in npu_windows
            ]

            segments = _live_normalize_segments(raw_segs, min_duration=0.3, merge_gap=2.5)

            spk_set = sorted({s.speaker for s in segments})
            _system_log(
                f"[Finalize] Segments: {len(raw_segs)} raw → {len(segments)} sau normalize | "
                f"speakers={spk_set}"
            )

            raw_full_text = " ".join(t.text for t in st.session_state.l_turns if t.text.strip())
            full_text     = restore(raw_full_text.lower())

            aligned = align(
                segments=segments,
                full_text=full_text,
                gap_limit=1.5,
            )
            merged = merge_consecutive(aligned, gap_limit=1.5)

            st.session_state.l_turns     = merged
            st.session_state.l_stats     = get_speaker_stats(segments)
            st.session_state.l_diar_done = True

            _system_log(
                f"[Finalize] Xong | speakers={list(st.session_state.l_stats.keys())} | "
                f"turns={len(merged)}"
            )

        st.rerun()

with status_col:
    if st.session_state.l_recording:
        dur_sec = (st.session_state.l_n_chunks * CHUNK_FRAMES) / SAMPLE_RATE
        wins    = _diarizer.get_speaker_windows()
        spk_set = sorted({w.label for w in wins} - {"..."})
        spk_txt = f" · {len(spk_set)} spk" if spk_set else ""
        st.markdown(
            f'<div style="background:rgba(232,82,10,0.1);border:1px solid #e8520a;'
            f'border-radius:4px;padding:8px 14px;display:inline-flex;align-items:center;gap:8px;margin-top:4px;">'
            f'<span class="rec-dot"></span>'
            f'<span style="font-family:monospace;font-size:11px;font-weight:600;color:#e8520a;">'
            f'REC &nbsp;{int(dur_sec//60):02d}:{int(dur_sec%60):02d} '
            f'&nbsp;·&nbsp; {len(st.session_state.l_turns)} câu{spk_txt}</span></div>',
            unsafe_allow_html=True,
        )
    elif st.session_state.l_finished and not st.session_state.l_diar_done:
        st.markdown(
            '<div style="font-family:monospace;font-size:11px;color:var(--text-color);opacity:0.8;'
            'padding:8px 14px;background:var(--secondary-background-color);'
            'border:1px solid rgba(128,128,128,0.3);border-radius:4px;display:inline-block;margin-top:4px;">'
            '⏸ Đã dừng &nbsp;·&nbsp; Bấm Hoàn thiện Transcript</div>',
            unsafe_allow_html=True,
        )
    elif st.session_state.l_diar_done and not st.session_state.l_summary_done:
        n_spk = len(st.session_state.l_stats)
        st.markdown(
            f'<div style="font-family:monospace;font-size:11px;color:#4da6e8;'
            f'padding:8px 14px;background:rgba(77,166,232,0.1);border:1px solid #4da6e8;'
            f'border-radius:4px;display:inline-block;margin-top:4px;">'
            f'✓ Đã hoàn thiện &nbsp;·&nbsp; {n_spk} người nói &nbsp;·&nbsp; Bấm Tạo Tóm tắt bên dưới</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# LOG PANEL — hiển thị trên UI (toggle) + NPU stats
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='margin:16px 0 4px;'></div>", unsafe_allow_html=True)

_log_header, _log_toggle = st.columns([4, 1])
with _log_header:
    st.markdown(
        '<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;'
        'color:var(--text-color);opacity:0.5;">📋 Log hệ thống (NPU Diarizer)</div>',
        unsafe_allow_html=True,
    )
with _log_toggle:
    if st.button(
        "🔽 Ẩn" if st.session_state.get("l_show_log") else "▶ Log",
        use_container_width=True, key="btn_toggle_log",
    ):
        st.session_state.l_show_log = not st.session_state.get("l_show_log", False)
        st.rerun()

if st.session_state.get("l_show_log"):
    import html as _html_mod

    def _colorize_log(line: str) -> str:
        esc = _html_mod.escape(line)
        if "[API ✓]" in line:
            return f'<span style="color:#79c0ff">{esc}</span>'
        if "[API ✗]" in line or "ERROR" in line or "❌" in line:
            return f'<span style="color:#ff7b72">{esc}</span>'
        if "[Anchor] ✅" in line:
            return f'<span style="color:#56d364">{esc}</span>'
        if "[Anchor]" in line or "[Pending]" in line:
            return f'<span style="color:#8b949e">{esc}</span>'
        if "[Reconcile]" in line:
            return f'<span style="color:#d2a8ff">{esc}</span>'
        if "[Filter]" in line or "[Retro]" in line or "[Normalize]" in line:
            return f'<span style="color:#8b949e">{esc}</span>'
        if "⚠️" in line or "TIMEOUT" in line:
            return f'<span style="color:#f0883e">{esc}</span>'
        return f'<span style="color:#7ee787">{esc}</span>'

    # _system_log ghi ra terminal → đọc lại từ logging handler
    # Dùng buffer in-memory nếu có, fallback hiển thị thông báo
    log_lines = st.session_state.get("_ui_log_lines", [])
    if log_lines:
        html_lines = "\n".join(_colorize_log(l) for l in log_lines[-80:])
        st.markdown(
            f'<div style="background:#0d1117;border:1px solid rgba(232,82,10,0.3);'
            f'border-radius:6px;padding:10px 14px;font-family:monospace;font-size:10px;'
            f'max-height:200px;overflow-y:auto;line-height:1.7;white-space:pre-wrap;">'
            f'{html_lines}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("Log ghi ra Terminal. Xem cửa sổ `streamlit run` để đọc log chi tiết.")

    # NPU stats
    if st.session_state.l_recording or st.session_state.l_finished:
        s = _diarizer.get_stats()
        c = st.columns(5)
        c[0].metric("API calls",   s["total_api_calls"])
        c[1].metric("Thành công",  s["successful_calls"])
        c[2].metric("Latency TB",  f"{s['avg_latency_sec']}s")
        c[3].metric("Speakers",    len(s["detected_speakers"]))
        c[4].metric("Anchor bank", s["anchor_bank_size"])

# ══════════════════════════════════════════════════════════════════════════════
# 14. GÁN TÊN SPEAKER & UI TRANSCRIPT
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.l_diar_done and st.session_state.l_stats:
    st.markdown("<hr style='border-color:rgba(128,128,128,0.2);margin:24px 0 16px;'>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;'
        'color:var(--text-color);opacity:0.7;margin-bottom:14px;">② Gán tên người nói</div>',
        unsafe_allow_html=True,
    )
    from components.speaker_editor import speaker_editor
    name_map = speaker_editor(st.session_state.l_stats, key_prefix="l_spk")
    if st.button("✅ Xác nhận tên người nói"):
        from core.aligner import rename_turns
        st.session_state.l_turns    = rename_turns(st.session_state.l_turns, name_map)
        st.session_state.l_name_map = name_map
        st.session_state.l_renamed  = True
        st.rerun()

st.markdown("<hr style='border-color:rgba(128,128,128,0.2);margin:24px 0 16px;'>", unsafe_allow_html=True)
col_title, col_toggle = st.columns([3, 1])
with col_title:
    st.markdown(
        '<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;'
        'color:var(--text-color);opacity:0.7;margin-bottom:14px;">③ Nội dung phiên hỏi cung</div>',
        unsafe_allow_html=True,
    )
with col_toggle:
    if st.session_state.l_turns:
        label = "📄 Xem toàn bộ" if not st.session_state.l_show_full else "🔼 Thu gọn"
        if st.button(label, use_container_width=True):
            st.session_state.l_show_full = not st.session_state.l_show_full
            st.rerun()

turns = st.session_state.l_turns

if st.session_state.l_recording:
    cmap   = _build_color_map(turns)
    recent = turns[-6:] if len(turns) > 6 else turns
    rows   = ""
    for t in recent:
        label = (t.speaker or "...").strip()
        
        # ─── XỬ LÝ GIAO DIỆN RIÊNG CHO NHÃN PENDING ───
        if label.startswith("PENDING"):
            color = "#e8520a"  # Màu cam nổi bật cho người lạ
            display_label = f"⏳ {label}"
            font_style = "font-style: italic;" # In nghiêng
        else:
            color = cmap.get(label, "#888888")
            display_label = label
            font_style = ""
            
        rows += (
            f'<div style="display:flex;gap:10px;align-items:baseline;padding:6px 0;'
            f'border-bottom:1px solid rgba(128,128,128,0.1);">'
            f'<span style="font-family:monospace;font-size:10px;color:{color};'
            f'white-space:nowrap;min-width:48px;">[{int(t.start//60):02d}:{int(t.start%60):02d}]</span>'
            f'<span style="font-size:11px;font-weight:700;color:{color};min-width:90px;'
            f'white-space:nowrap;{font_style}">{_html.escape(display_label)}:</span>'
            f'<span style="font-size:13px;color:var(--text-color);line-height:1.6;">'
            f'{_html.escape(t.text[:180])}</span></div>'
        )

    partial = st.session_state.l_partial
    if partial:
        rows += (
            f'<div style="display:flex;gap:10px;align-items:baseline;padding:6px 0;opacity:0.55;">'
            f'<span style="font-family:monospace;font-size:10px;color:#e8520a;'
            f'white-space:nowrap;min-width:48px;">[···]</span>'
            f'<span style="font-size:11px;font-weight:700;color:#e8520a;min-width:90px;'
            f'white-space:nowrap;">🎙️:</span>'
            f'<span style="font-size:13px;color:var(--text-color);opacity:0.7;font-style:italic;">'
            f'{_html.escape(partial)}</span></div>'
        )

    empty_msg = '<div style="padding:40px;text-align:center;color:#484f58;font-size:13px;">🎙️ Đang lắng nghe...</div>'
    st.markdown(
        f'<div style="background:var(--secondary-background-color);'
        f'border:1px solid rgba(128,128,128,0.2);border-radius:8px;padding:14px 16px;">'
        f'{rows if rows else empty_msg}</div>',
        unsafe_allow_html=True,
    )

elif turns:
    from components.transcript_viewer import full as show_full, preview as show_preview
    if st.session_state.l_show_full:
        show_full(turns, editable=True)
        if st.button("💾 Lưu chỉnh sửa", type="primary"):
            st.session_state.l_turns = turns
            st.success("Đã lưu chỉnh sửa")
    else:
        show_preview(turns, max_turns=4)
else:
    st.markdown(
        '<div style="background:var(--secondary-background-color);'
        'border:1px solid rgba(128,128,128,0.2);border-radius:8px;padding:40px;'
        'text-align:center;color:var(--text-color);opacity:0.6;font-size:13px;">'
        'Bắt đầu ghi âm để thấy transcript xuất hiện ở đây...</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# 15. TÓM TẮT & XUẤT DOCX
# ══════════════════════════════════════════════════════════════════════════════
if turns and st.session_state.l_diar_done:
    st.markdown("<hr style='border-color:rgba(128,128,128,0.2);margin:32px 0 16px;'>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;'
        'color:var(--text-color);opacity:0.7;margin-bottom:14px;">⑤ Tóm tắt & Xuất Biên Bản</div>',
        unsafe_allow_html=True,
    )

    if st.button("🤖 Chạy Tóm tắt Nội dung (NPU Qwen)", use_container_width=True):
        summary_progress = st.progress(0, text="Khởi động NPU tóm tắt...")
        def update_progress(pct, msg):
            summary_progress.progress(max(0, min(100, pct)), text=f"NPU: {msg}")
        try:
            from core.test_qwen import summarize
            result = summarize(st.session_state.l_turns, progress_callback=update_progress)
            if result.ok:
                st.session_state.l_summary      = result.summary.strip()
                st.session_state.l_summary_done = True
                summary_progress.progress(100, text=f"✅ Tóm tắt xong! ({result.elapsed_sec}s)")
                st.success("✅ Đã tóm tắt thành công!")
            else:
                summary_progress.empty()
                st.error(f"❌ Lỗi khi tóm tắt: {result.error}")
        except Exception as e:
            st.error(f"❌ Lỗi hệ thống: {e}")

    if st.session_state.l_summary_done and st.session_state.l_summary:
        st.text_area(
            "Bản xem trước Tóm tắt (Sẽ được chèn vào Word):",
            st.session_state.l_summary,
            height=200,
        )

    st.markdown("<div style='margin:16px 0 8px;'></div>", unsafe_allow_html=True)
    is_ready    = bool(st.session_state.l_summary)
    export_data = b""

    if is_ready:
        try:
            from components.export_docx import export_summary_to_docx
            export_data = export_summary_to_docx(
                summary_text=st.session_state.l_summary,
                session_name=st.session_state.l_session_name,
            )
        except FileNotFoundError as e:
            st.error(str(e))

    st.download_button(
        label="📄 Xuất biên bản DOCX",
        data=export_data,
        file_name=f"Bien_ban_{st.session_state.l_session_name.replace(' ', '_')}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        type="primary",
        disabled=not is_ready,
        use_container_width=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# 16. STREAMLIT EVENT POLL 200ms
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.l_recording:
    partial, completed = _drain_queue()
    st.session_state.l_partial = partial
    if completed:
        st.session_state.l_turns.extend(completed)
    time.sleep(0.2)
    st.rerun()