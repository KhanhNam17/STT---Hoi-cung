import os
import time
import queue as _queue_module
import threading
import warnings
import html as _html
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

sherpa_onnx    = _try_import("sherpa_onnx")
sd             = _try_import("sounddevice")
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

WESPEAKER_MODEL  = os.getenv("EMBED_MODEL",      "models/wespeaker")
WESPEAKER_DEVICE = os.getenv("WESPEAKER_DEVICE", "cpu").lower()

VAD_WINDOW_SEC = 0.5   
EMB_CHUNK_SEC  = 2.0   # FIX-2: Tăng lên 2s để Vector đủ dài, tránh sinh pending rác

# ══════════════════════════════════════════════════════════════════════════════
# 2. LOG
# ══════════════════════════════════════════════════════════════════════════════
def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# 3. SHARED QUEUE (STT làn 1 dùng)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def _get_shared():
    return {"audio_q": _queue_module.Queue()}

_AUDIO_QUEUE = _get_shared()["audio_q"]

# ══════════════════════════════════════════════════════════════════════════════
# 4. PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Live Mode — Hỏi Cung",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
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
        background: #e8520a; animation: rec-blink 1.2s ease-in-out infinite;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 24px;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;
                    letter-spacing:3px;text-transform:uppercase;color:#e8520a;
                    margin-bottom:8px;">// HE THONG</div>
        <div style="font-size:18px;font-weight:700;color:var(--text-color);">Tro ly Hoi Cung</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                    color:var(--text-color);opacity:0.6;margin-top:6px;">STT · UTTERR DIARIZATION · DOCX</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.page_link("app.py",                label="🏠  Trang chu")
    st.page_link("pages/1_Batch_Mode.py", label="📁  Xu ly File Audio/Video")
    st.page_link("pages/2_Live_Mode.py",  label="🎙️  Live Mode")

st.markdown("""
<div style="font-size:11px;letter-spacing:3px;text-transform:uppercase;
            color:var(--text-color);opacity:0.7;padding-bottom:12px;
            border-bottom:1px solid rgba(128,128,128,0.2);margin-bottom:20px;">
    🎙️  Thu am Truc tiep — Live Mode (Utterr Diarization)
</div>
""", unsafe_allow_html=True)

if not LIVE_ENV_READY:
    missing = [n for n, m in [("sherpa_onnx", sherpa_onnx), ("sounddevice", sd)] if not m]
    st.error(f"⚠️ Thieu thu vien: {', '.join(missing)}")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# 5. CACHE MODELS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def _load_zipformer():
    _log("⏳ Dang nap Zipformer STT...")
    r = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens                     = TOKENS_PATH,
        encoder                    = ENCODER_PATH,
        decoder                    = DECODER_PATH,
        joiner                     = JOINER_PATH,
        num_threads                = 2,
        sample_rate                = SAMPLE_RATE,
        feature_dim                = 80,
        enable_endpoint_detection  = True,
        rule1_min_trailing_silence = 2.4,
        rule2_min_trailing_silence = 1.2,
        rule3_min_utterance_length = 20,
    )
    _log("✅ Zipformer nap xong")
    return r

@st.cache_resource(show_spinner=False)
def _load_diarizer() -> dict:
    from core.utterr_diarize_core import SileroVAD, WeSpeakerEncoder, SpeakerHandler

    _log("⏳ Dang nap SileroVAD...")
    vad = SileroVAD()
    vad.load()
    _log("✅ SileroVAD nap xong")

    _log(f"⏳ Dang nap WeSpeaker ({WESPEAKER_DEVICE.upper()})...")
    encoder = WeSpeakerEncoder(device=WESPEAKER_DEVICE, model_path=WESPEAKER_MODEL)
    encoder.load()
    _log("✅ WeSpeaker nap xong")

    handler = SpeakerHandler()
    handler.set_embedding_callback(
        lambda: _log(
            f"[Diar] 🆕 Speaker promote | active={handler.active_spks} | "
            f"pending={len(handler.pending_embs)}"
        )
    )

    diarizer = {
        "vad":     vad,
        "encoder": encoder,
        "handler": handler,
        "windows": [],
        "lock":    threading.Lock(),
        "running": False,
        "thread":  None,
        "audio_q": _queue_module.Queue(),
        "stats": {
            "total_windows":       0,
            "speech_windows":      0,
            "embeddings_computed": 0,
            "active_speakers":     set(),
            "pending_count":       0,
        },
    }
    _log("✅ Utterr Diarizer san sang")
    return diarizer

with st.spinner("Dang nap mo hinh..."):
    recognizer = _load_zipformer()
    _diarizer  = _load_diarizer()

# ══════════════════════════════════════════════════════════════════════════════
# 6. DIARIZER API
# ══════════════════════════════════════════════════════════════════════════════

def diarizer_push_chunk(d: dict, audio_i16: np.ndarray) -> None:
    if d["running"]:
        d["audio_q"].put(audio_i16.copy())


def diarizer_assign_speaker(d: dict, t_start: float, t_end: float) -> str:
    """Tìm speaker có overlap nhiều nhất với [t_start, t_end]."""
    best_label, best_ov = "...", 0.0
    with d["lock"]:
        for w in d["windows"]:
            ov = max(0.0, min(w["end"], t_end) - max(w["start"], t_start))
            if ov > best_ov:
                best_ov    = ov
                best_label = w["label"]
    return best_label


def diarizer_get_windows(d: dict) -> list:
    with d["lock"]:
        return list(d["windows"])


def diarizer_get_stats(d: dict) -> dict:
    s = dict(d["stats"])
    s["active_speakers"] = sorted(d["stats"]["active_speakers"])
    s["pending_count"]   = len(d["handler"].pending_embs)
    s["handler_active"]  = sorted(d["handler"].active_spks)
    return s


def diarizer_reset(d: dict) -> None:
    with d["lock"]:
        d["windows"].clear()
    d["handler"].reset()
    while not d["audio_q"].empty():
        try: d["audio_q"].get_nowait()
        except _queue_module.Empty: break
    d["stats"] = {
        "total_windows":       0,
        "speech_windows":      0,
        "embeddings_computed": 0,
        "active_speakers":     set(),
        "pending_count":       0,
    }
    _log("🔄 Diarizer reset")


def diarizer_start(d: dict) -> None:
    if d["running"]:
        return
    d["running"] = True
    t = threading.Thread(
        target=_diarizer_worker, args=(d,),
        name="utterr-diarizer", daemon=True,
    )
    t.start()
    d["thread"] = t
    _log("🟢 Diarizer worker bat dau")


def diarizer_stop(d: dict) -> None:
    if not d["running"]:
        return
    d["running"] = False
    d["audio_q"].put(None)   # poison pill
    if d["thread"]:
        d["thread"].join(timeout=5.0)

    # Recluster sau dừng để cải thiện accuracy
    n_active = len(d["handler"].active_spks)
    if n_active >= 2:
        _log(f"[Diar] Recluster sau dung ({n_active} speakers)...")
        d["handler"].recluster_spks(target_clusters=n_active)
        _log(f"[Diar] Recluster xong | active={d['handler'].active_spks}")

    _log("🔴 Diarizer worker dung")


# ══════════════════════════════════════════════════════════════════════════════
# 7. DIARIZER WORKER (Làn 2 — chạy ngầm)
# ══════════════════════════════════════════════════════════════════════════════

def _diarizer_worker(d: dict) -> None:
    sr         = SAMPLE_RATE
    vad_frames = int(sr * VAD_WINDOW_SEC)
    emb_frames = int(sr * EMB_CHUNK_SEC)

    vad: "SileroVAD"        = d["vad"]
    enc: "WeSpeakerEncoder" = d["encoder"]
    hnd: "SpeakerHandler"   = d["handler"]

    vad_buf    = []
    emb_buf    = []
    t_cursor   = 0.0
    emb_start  = 0.0

    _log(f"[Worker] VAD_win={VAD_WINDOW_SEC}s | EMB_chunk={EMB_CHUNK_SEC}s")

    while True:
        try:
            chunk_i16 = d["audio_q"].get(timeout=2.0)
        except _queue_module.Empty:
            if not d["running"]:
                break
            continue

        if chunk_i16 is None:
            _log("[Worker] Thoat")
            break

        chunk_dur = len(chunk_i16) / sr
        chunk_f32 = chunk_i16.astype(np.float32) / 32768.0

        vad_buf.append(chunk_f32)
        d["stats"]["total_windows"] += 1

        # VAD: xử lý khi tích đủ window
        if sum(len(c) for c in vad_buf) < vad_frames:
            t_cursor += chunk_dur
            continue

        vad_audio = np.concatenate(vad_buf)
        vad_buf   = []
        is_speech = vad._detect_speech(vad_audio, sr=sr)

        if not is_speech:
            if emb_buf:
                _flush_emb_buf(d, emb_buf, emb_start, t_cursor, enc, hnd, sr)
                emb_buf   = []
                emb_start = t_cursor
            t_cursor += len(vad_audio) / sr
            continue

        d["stats"]["speech_windows"] += 1
        if not emb_buf:
            emb_start = t_cursor

        emb_buf.append(vad_audio)
        t_cursor += len(vad_audio) / sr

        if sum(len(c) for c in emb_buf) >= emb_frames:
            _flush_emb_buf(d, emb_buf, emb_start, t_cursor, enc, hnd, sr)
            emb_buf   = []
            emb_start = t_cursor

    if emb_buf:
        _flush_emb_buf(d, emb_buf, emb_start, t_cursor, enc, hnd, sr)


def _flush_emb_buf(
    d: dict, emb_buf: list, t_start: float, t_end: float,
    enc, hnd, sr: int,
) -> None:
    audio = np.concatenate(emb_buf)
    emb   = enc._compute_emb(audio, sr=sr)
    if emb is None:
        return

    d["stats"]["embeddings_computed"] += 1

    spk_id, sim = hnd.classify_spk(emb, seg_time=t_start)

    if spk_id == "pending":
        label = f"PENDING_{len(hnd.pending_embs) - 1:02d}"
        d["stats"]["pending_count"] = len(hnd.pending_embs)
    else:
        label = f"SPEAKER_{int(spk_id):02d}"
        d["stats"]["active_speakers"].add(label)

    _log(
        f"[Diar] [{t_start:.1f}→{t_end:.1f}s] "
        f"spk={spk_id} ({label}) | sim={sim:.3f} | "
        f"active={sorted(hnd.active_spks)} | pending={len(hnd.pending_embs)}"
    )

    with d["lock"]:
        if d["windows"] and d["windows"][-1]["label"] == label:
            gap = t_start - d["windows"][-1]["end"]
            if gap <= 1.0:
                d["windows"][-1]["end"] = t_end
                return

        d["windows"].append({
            "start": round(t_start, 3),
            "end":   round(t_end,   3),
            "label": label,
        })

    if spk_id != "pending":
        _retro_patch_pending(d, label)


def _retro_patch_pending(d: dict, confirmed_label: str) -> None:
    """
    Quét ngược d["windows"], đổi tất cả PENDING_ gần nhất → confirmed_label.
    FIX-4: Gộp toàn bộ vào một khối Lock duy nhất để triệt tiêu Race Condition.
    """
    patched = 0
    with d["lock"]:
        # Đếm số window đã mang confirmed_label TRƯỚC window vừa thêm
        count_existing = sum(
            1 for w in d["windows"][:-1] if w["label"] == confirmed_label
        )

        # Chỉ patch khi đây là lần xuất hiện đầu tiên của Speaker này
        if count_existing > 0:
            return

        for w in reversed(d["windows"]):
            if w["label"].startswith("PENDING_"):
                w["label"] = confirmed_label
                patched   += 1
            elif w["label"].startswith("SPEAKER_"):
                break   # Chạm vùng an toàn

    if patched:
        _log(f"[Retro] Va {patched} window PENDING → {confirmed_label}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. AUDIO CALLBACK (sounddevice C thread)
# ══════════════════════════════════════════════════════════════════════════════
def _audio_callback(indata, frames, time_info, status):
    if status:
        _log(f"[Mic] {status}")
    chunk_i16 = (indata[:, 0] * 32767).astype(np.int16)
    _AUDIO_QUEUE.put(chunk_i16.copy())
    diarizer_push_chunk(_diarizer, chunk_i16)


# ══════════════════════════════════════════════════════════════════════════════
# 9. RETRO-UPDATE TRANSCRIPT
# ══════════════════════════════════════════════════════════════════════════════
def _retro_update_speakers() -> int:
    turns   = st.session_state.get("l_turns", [])
    updated = 0
    for t in turns:
        spk = diarizer_assign_speaker(_diarizer, t.start, t.end)
        if spk != "..." and spk != t.speaker:
            t.speaker = spk
            updated  += 1
    if updated:
        _log(f"[Retro] Va {updated} cau voi nhan speaker moi")
    return updated


# ══════════════════════════════════════════════════════════════════════════════
# 10. STT DRAIN (Làn 1 — Zipformer, 200ms poll)
# ══════════════════════════════════════════════════════════════════════════════
def _drain_queue() -> tuple:
    from core.aligner import AlignedTurn

    if st.session_state.l_asr_stream is None:
        st.session_state.l_asr_stream    = recognizer.create_stream()
        st.session_state.l_utterance_start = 0.0

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
                t_end   = round(
                    (st.session_state.l_n_chunks * CHUNK_FRAMES) / SAMPLE_RATE, 2
                )
                t_start = round(st.session_state.l_utterance_start, 2)
                speaker = diarizer_assign_speaker(_diarizer, t_start, t_end)

                _log(
                    f"[STT] [{t_start:.1f}s→{t_end:.1f}s] "
                    f"speaker={speaker} | "
                    f"\"{text[:60]}{'...' if len(text) > 60 else ''}\""
                )

                completed.append(AlignedTurn(
                    speaker = speaker,
                    start   = t_start,
                    end     = t_end,
                    text    = text,
                ))

            recognizer.reset(z_stream)
            st.session_state.l_utterance_start = round(
                (st.session_state.l_n_chunks * CHUNK_FRAMES) / SAMPLE_RATE, 2
            )

    cur_wins = len(diarizer_get_windows(_diarizer))
    if cur_wins > st.session_state.l_prev_win_count:
        _retro_update_speakers()
        st.session_state.l_prev_win_count = cur_wins

    partial = recognizer.get_result(z_stream).strip()
    return partial, completed


# ══════════════════════════════════════════════════════════════════════════════
# 11. SPEAKER COLOR MAP
# ══════════════════════════════════════════════════════════════════════════════
SPEAKER_COLORS = [
    "#e8520a", "#4da6e8", "#4ec94e", "#c97fe8",
    "#e8c14d", "#e84d6e", "#4de8d4", "#e8b04d",
]

def _build_color_map(turns) -> dict:
    seen = {}
    for t in turns:
        label = (t.speaker or "...").strip()
        if label not in seen and label != "...":
            seen[label] = SPEAKER_COLORS[len(seen) % len(SPEAKER_COLORS)]
    seen["..."] = "#888888"
    for label in list(seen.keys()):
        if label.startswith("PENDING_"):
            seen[label] = "#aa6600"
    return seen


# ══════════════════════════════════════════════════════════════════════════════
# 12. NORMALIZE SEGMENTS SAU KHI DỪNG
# ══════════════════════════════════════════════════════════════════════════════
def _live_normalize(segments: list, min_duration=0.3, merge_gap=2.5) -> list:
    from core.diarizer import SpeakerSegment
    if not segments:
        return []

    filtered = [s for s in segments if (s.end - s.start) >= min_duration]
    if not filtered:
        return []

    merged = [filtered[0]]
    for cur in filtered[1:]:
        prev = merged[-1]
        if cur.speaker == prev.speaker and (cur.start - prev.end) <= merge_gap:
            merged[-1] = SpeakerSegment(
                speaker=prev.speaker, start=prev.start,
                end=cur.end, text=prev.text,
            )
        else:
            merged.append(cur)

    _log(
        f"[Normalize] {len(filtered)} → {len(merged)} segs | "
        f"speakers={sorted({s.speaker for s in merged})}"
    )
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# 13. SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_defaults = {
    "l_recording":        False,
    "l_finished":         False,
    "l_diar_done":        False,
    "l_renamed":          False,
    "l_stream_ref":       None,
    "l_asr_stream":       None,
    "l_utterance_start":  0.0,    
    "l_turns":            [],
    "l_stats":            {},
    "l_name_map":         {},
    "l_raw_audio":        [],
    "l_n_chunks":         0,
    "l_partial":          "",
    "l_session_name":     "",
    "l_show_full":        False,
    "l_summary":          "",
    "l_summary_done":     False,
    "l_prev_win_count":   0,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# 14. UI CONTROL PANEL
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;'
    'color:var(--text-color);opacity:0.7;margin-bottom:16px;">① Dieu khien phien ghi am</div>',
    unsafe_allow_html=True,
)

session_name = st.text_input(
    "Ma phien / Ten ho so",
    value    = st.session_state.l_session_name or f"LIVE-{time.strftime('%Y%m%d-%H%M')}",
    disabled = st.session_state.l_recording,
)
if session_name != st.session_state.l_session_name:
    st.session_state.l_session_name = session_name

st.markdown("<div style='margin:12px 0 8px;'></div>", unsafe_allow_html=True)
btn_col1, btn_col2, status_col = st.columns([1, 1, 2], gap="small")

# ── Bắt đầu / Dừng ───────────────────────────────────────────────────────────
with btn_col1:
    if not st.session_state.l_recording:
        if st.button("🔴 Bắt đầu ghi âm", type="primary", use_container_width=True):
            for k in ["l_turns", "l_raw_audio"]:
                st.session_state[k] = []
            for k in ["l_stats", "l_name_map"]:
                st.session_state[k] = {}
            for k in ["l_n_chunks", "l_prev_win_count"]:
                st.session_state[k] = 0
            st.session_state.l_partial          = ""
            st.session_state.l_finished         = False
            st.session_state.l_diar_done        = False
            st.session_state.l_renamed          = False
            st.session_state.l_summary          = ""
            st.session_state.l_summary_done     = False
            st.session_state.l_asr_stream       = None
            st.session_state.l_utterance_start  = 0.0   
            st.session_state.l_show_full        = False
            st.session_state.l_prev_win_count   = 0

            with _AUDIO_QUEUE.mutex:
                _AUDIO_QUEUE.queue.clear()

            diarizer_reset(_diarizer)
            diarizer_start(_diarizer)

            _log(
                f"🟢 Bat dau: {st.session_state.l_session_name} | "
                f"SR={SAMPLE_RATE}Hz | chunk={CHUNK_FRAMES}frames"
            )

            stream = sd.InputStream(
                samplerate = SAMPLE_RATE, channels = 1,
                dtype      = "float32",   blocksize = CHUNK_FRAMES,
                callback   = _audio_callback,
            )
            stream.start()
            st.session_state.l_stream_ref = stream
            st.session_state.l_recording  = True
            st.rerun()

    else:
        if st.button("⏹️ Dung Ghi am", use_container_width=True):
            if st.session_state.l_stream_ref:
                st.session_state.l_stream_ref.stop()
                st.session_state.l_stream_ref.close()
                st.session_state.l_stream_ref = None

            diarizer_stop(_diarizer)

            dur = (st.session_state.l_n_chunks * CHUNK_FRAMES) / SAMPLE_RATE
            s   = diarizer_get_stats(_diarizer)
            _log(
                f"🔴 Dung | dur={dur:.1f}s | cau={len(st.session_state.l_turns)} | "
                f"active={s['handler_active']} | embs={s['embeddings_computed']}"
            )

            st.session_state.l_asr_stream = None
            st.session_state.l_recording  = False
            st.session_state.l_finished   = True
            st.rerun()

# ── Hoàn thiện transcript ─────────────────────────────────────────────────────
with btn_col2:
    btn_fin_disabled = (
        not st.session_state.l_finished or st.session_state.l_diar_done
    )
    if st.button("✅ Hoàn thiện Transcript", disabled=btn_fin_disabled, use_container_width=True):
        with st.spinner("Đang chuẩn hóa transcript..."):
            
            # MÁY HÚT BỤI
            with _diarizer["lock"]:
                last_valid_spk = "SPEAKER_00" 
                for w in _diarizer["windows"]:
                    if w["label"].startswith("SPEAKER_"):
                        last_valid_spk = w["label"]
                    elif w["label"].startswith("PENDING_"):
                        w["label"] = last_valid_spk

            _retro_update_speakers()

            from core.diarizer import SpeakerSegment, get_speaker_stats
            from core.aligner  import align, merge_consecutive
            from core.punctuation_restorer import restore

            wins     = diarizer_get_windows(_diarizer)
            raw_segs = [
                SpeakerSegment(speaker=w["label"], start=w["start"], end=w["end"])
                for w in wins
            ]
            segments = _live_normalize(raw_segs, min_duration=0.3, merge_gap=2.5)

            spk_set = sorted({s.speaker for s in segments})
            _log(f"[Finalize] {len(raw_segs)} raw → {len(segments)} segs | speakers={spk_set}")

            raw_text  = " ".join(t.text for t in st.session_state.l_turns if t.text.strip())
            full_text = restore(raw_text.lower())

            # FIX-6: Tự động ghi mảng Audio ra file WAV tạm thời để ép Forced Alignment hoạt động
            temp_wav_path = f"temp_align_{int(time.time())}.wav"
            if st.session_state.l_raw_audio:
                with wave.open(temp_wav_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2) # 16-bit
                    wf.setframerate(SAMPLE_RATE)
                    for chunk in st.session_state.l_raw_audio:
                        wf.writeframes(chunk.tobytes())
            
            # Truyền đường dẫn WAV thực tế vào hàm align()
            aligned = align(segments=segments, full_text=full_text, gap_limit=1.5, wav_path=temp_wav_path)
            merged  = merge_consecutive(aligned, gap_limit=1.5)

            st.session_state.l_turns     = merged
            st.session_state.l_stats     = get_speaker_stats(segments)
            st.session_state.l_diar_done = True

            # Dọn dẹp file tạm
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)

            _log(
                f"[Finalize] Xong | speakers={list(st.session_state.l_stats.keys())} | "
                f"turns={len(merged)}"
            )
        st.rerun()

# ── Status bar ────────────────────────────────────────────────────────────────
with status_col:
    if st.session_state.l_recording:
        dur_sec = (st.session_state.l_n_chunks * CHUNK_FRAMES) / SAMPLE_RATE
        s       = diarizer_get_stats(_diarizer)
        spk_txt = f" · {len(s['handler_active'])} spk" if s["handler_active"] else ""
        pnd_txt = f" · {s['pending_count']} pending" if s["pending_count"] > 0 else ""
        st.markdown(
            f'<div style="background:rgba(232,82,10,0.1);border:1px solid #e8520a;'
            f'border-radius:4px;padding:8px 14px;display:inline-flex;align-items:center;'
            f'gap:8px;margin-top:4px;">'
            f'<span class="rec-dot"></span>'
            f'<span style="font-family:monospace;font-size:11px;font-weight:600;color:#e8520a;">'
            f'REC &nbsp;{int(dur_sec//60):02d}:{int(dur_sec%60):02d}'
            f' &nbsp;·&nbsp; {len(st.session_state.l_turns)} cau'
            f'{spk_txt}{pnd_txt}</span></div>',
            unsafe_allow_html=True,
        )
    elif st.session_state.l_finished and not st.session_state.l_diar_done:
        st.markdown(
            '<div style="font-family:monospace;font-size:11px;padding:8px 14px;'
            'background:var(--secondary-background-color);'
            'border:1px solid rgba(128,128,128,0.3);border-radius:4px;'
            'display:inline-block;margin-top:4px;">'
            '⏸ Da dung &nbsp;·&nbsp; Bam Hoan thien Transcript</div>',
            unsafe_allow_html=True,
        )
    elif st.session_state.l_diar_done:
        n_spk = len(st.session_state.l_stats)
        st.markdown(
            f'<div style="font-family:monospace;font-size:11px;color:#4da6e8;'
            f'padding:8px 14px;background:rgba(77,166,232,0.1);'
            f'border:1px solid #4da6e8;border-radius:4px;'
            f'display:inline-block;margin-top:4px;">'
            f'✓ Hoan thien &nbsp;·&nbsp; {n_spk} nguoi noi</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# 15. GÁN TÊN SPEAKER
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.l_diar_done and st.session_state.l_stats:
    st.markdown(
        "<hr style='border-color:rgba(128,128,128,0.2);margin:24px 0 16px;'>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;'
        'color:var(--text-color);opacity:0.7;margin-bottom:14px;">② Gan ten nguoi noi</div>',
        unsafe_allow_html=True,
    )
    from components.speaker_editor import speaker_editor
    name_map = speaker_editor(st.session_state.l_stats, key_prefix="l_spk")
    if st.button("✅ Xac nhan ten nguoi noi"):
        from core.aligner import rename_turns
        st.session_state.l_turns    = rename_turns(st.session_state.l_turns, name_map)
        st.session_state.l_name_map = name_map
        st.session_state.l_renamed  = True
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# 16. TRANSCRIPT VIEWER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<hr style='border-color:rgba(128,128,128,0.2);margin:24px 0 16px;'>",
    unsafe_allow_html=True,
)
col_title, col_toggle = st.columns([3, 1])
with col_title:
    st.markdown(
        '<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;'
        'color:var(--text-color);opacity:0.7;margin-bottom:14px;">③ Noi dung phien hoi cung</div>',
        unsafe_allow_html=True,
    )
with col_toggle:
    if st.session_state.l_turns:
        lbl = "📄 Xem toan bo" if not st.session_state.l_show_full else "🔼 Thu gon"
        if st.button(lbl, use_container_width=True):
            st.session_state.l_show_full = not st.session_state.l_show_full
            st.rerun()

turns = st.session_state.l_turns

if st.session_state.l_recording:
    cmap   = _build_color_map(turns)
    recent = turns[-6:] if len(turns) > 6 else turns
    rows   = ""
    for t in recent:
        label = (t.speaker or "...").strip()
        color = cmap.get(label, "#888888")
        rows += (
            f'<div style="display:flex;gap:10px;align-items:baseline;padding:6px 0;'
            f'border-bottom:1px solid rgba(128,128,128,0.1);">'
            f'<span style="font-family:monospace;font-size:10px;color:{color};'
            f'white-space:nowrap;min-width:48px;">'
            f'[{int(t.start//60):02d}:{int(t.start%60):02d}]</span>'
            f'<span style="font-size:11px;font-weight:700;color:{color};'
            f'min-width:90px;white-space:nowrap;">{_html.escape(label)}:</span>'
            f'<span style="font-size:13px;color:var(--text-color);line-height:1.6;">'
            f'{_html.escape(t.text[:180])}</span></div>'
        )

    partial = st.session_state.l_partial
    if partial:
        rows += (
            f'<div style="display:flex;gap:10px;align-items:baseline;'
            f'padding:6px 0;opacity:0.55;">'
            f'<span style="font-family:monospace;font-size:10px;color:#e8520a;'
            f'min-width:48px;">[···]</span>'
            f'<span style="font-size:11px;font-weight:700;color:#e8520a;'
            f'min-width:90px;">🎙️:</span>'
            f'<span style="font-size:13px;color:var(--text-color);opacity:0.7;'
            f'font-style:italic;">{_html.escape(partial)}</span></div>'
        )

    empty = (
        '<div style="padding:40px;text-align:center;'
        'color:#484f58;font-size:13px;">🎙️ Dang lang nghe...</div>'
    )
    st.markdown(
        f'<div style="background:var(--secondary-background-color);'
        f'border:1px solid rgba(128,128,128,0.2);border-radius:8px;padding:14px 16px;">'
        f'{rows if rows else empty}</div>',
        unsafe_allow_html=True,
    )

elif turns:
    from components.transcript_viewer import full as show_full, preview as show_preview
    if st.session_state.l_show_full:
        show_full(turns, editable=True)
        if st.button("💾 Luu chinh sua", type="primary"):
            st.session_state.l_turns = turns
            st.success("Da luu chinh sua")
    else:
        show_preview(turns, max_turns=4)

else:
    st.markdown(
        '<div style="background:var(--secondary-background-color);'
        'border:1px solid rgba(128,128,128,0.2);border-radius:8px;padding:40px;'
        'text-align:center;color:var(--text-color);opacity:0.6;font-size:13px;">'
        'Bat dau ghi am de thay transcript xuat hien o day...</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# 17. TÓM TẮT & XUẤT DOCX
# ══════════════════════════════════════════════════════════════════════════════
if turns and st.session_state.l_diar_done:
    st.markdown(
        "<hr style='border-color:rgba(128,128,128,0.2);margin:32px 0 16px;'>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;'
        'color:var(--text-color);opacity:0.7;margin-bottom:14px;">⑤ Tom tat & Xuat Bien Ban</div>',
        unsafe_allow_html=True,
    )

    if st.button("🤖 Chay Tom tat Noi dung (NPU Qwen)", use_container_width=True):
        progress = st.progress(0, text="Khoi dong NPU tom tat...")
        try:
            from core.test_qwen import summarize
            result = summarize(
                st.session_state.l_turns,
                progress_callback=lambda p, m: progress.progress(
                    max(0, min(100, p)), text=f"NPU: {m}"
                ),
            )
            if result.ok:
                st.session_state.l_summary      = result.summary.strip()
                st.session_state.l_summary_done = True
                progress.progress(100, text=f"✅ Tom tat xong! ({result.elapsed_sec}s)")
                st.success("✅ Da tom tat thanh cong!")
            else:
                progress.empty()
                st.error(f"❌ {result.error}")
        except Exception as e:
            st.error(f"❌ Loi he thong: {e}")

    if st.session_state.l_summary_done and st.session_state.l_summary:
        st.text_area("Ban xem truoc Tom tat:", st.session_state.l_summary, height=200)

    is_ready    = bool(st.session_state.l_summary)
    export_data = b""
    if is_ready:
        try:
            from components.export_docx import export_summary_to_docx
            export_data = export_summary_to_docx(
                summary_text = st.session_state.l_summary,
                session_name = st.session_state.l_session_name,
            )
        except FileNotFoundError as e:
            st.error(str(e))

    st.download_button(
        label    = "📄 Xuat bien ban DOCX",
        data     = export_data,
        file_name= f"Bien_ban_{st.session_state.l_session_name.replace(' ','_')}.docx",
        mime     = "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        type     = "primary",
        disabled = not is_ready,
        use_container_width = True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# 18. POLL 200ms (chỉ khi đang ghi)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.l_recording:
    partial, completed = _drain_queue()
    st.session_state.l_partial = partial
    if completed:
        st.session_state.l_turns.extend(completed)
    time.sleep(0.2)
    st.rerun()