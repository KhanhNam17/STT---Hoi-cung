# pages/2_Live_Mode.py
#
# ══════════════════════════════════════════════════════════════════════════════
# KIẾN TRÚC REAL-TIME MICROSERVICES (STT + NPU DIARIZATION)
# ══════════════════════════════════════════════════════════════════════════════
#   Làn 1 (Main Thread): Thu âm → Zipformer → Ra chữ với nhãn "..." ngay lập tức.
#   Làn 2 (NPU Thread): Đọc âm thanh → Gọi API NPU Server → Cập nhật nhãn siêu tốc.
#   Đồng bộ: Kỹ thuật Retro-update tự động vá tên người nói vào các đoạn text cũ.

import os
import time
import queue as _queue_module
import threading
import warnings
import logging
import tempfile
import html as _html
import io
import wave
import base64
import requests

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from streamlit.runtime.scriptrunner import add_script_run_ctx

load_dotenv()

os.environ.setdefault("SPEECHBRAIN_DISABLE_K2", "1")
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# 0. LAZY IMPORT — không crash nếu chạy sai env
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


# ══════════════════════════════════════════════════════════════════════════════
# 2. GLOBAL QUEUES & LOCK (Thread-safe)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_shared_resources():
    return {
        "audio_q": _queue_module.Queue(),
        "audio_peak": 0.0,        # FIX #7: VU meter peak (0..1)
        "sortformer_warm": False, # FIX #5: đã pre-warm subprocess chưa
        # FIX #5c: incremental Sortformer cache + stop signal
        "sortformer_cache": None, # dict: {segments, duration, elapsed, computed_at}
        "sortformer_stop":  True, # True = không chạy thread; set False khi start recording
        "sortformer_run_lock": threading.Lock(),  # serialize NeMo calls (Windows tempdir race)
        # FIX #13b: streaming WAV writer (chỉ giữ file handle ở _shared, dùng từ main thread)
        "wav_writer": None,       # wave.Wave_write hoặc None
        "wav_path": None,         # đường dẫn file đang ghi
        "wav_lock": threading.Lock(),
        # FIX #5b: trạng thái finalize chạy nền
        "finalize_state": {       # main thread đọc, worker thread ghi
            "running": False,
            "pct": 0,
            "msg": "",
            "result": None,       # dict: {"merged":..., "stats":..., "warning":..., "backend":...}
            "error": None,
        },
        "finalize_lock": threading.Lock(),
    }

_shared = get_shared_resources()
_AUDIO_QUEUE = _shared["audio_q"]


# ══════════════════════════════════════════════════════════════════════════════
# 2b. HELPERS — friendly speaker labels, mic devices, pre-warm
# ══════════════════════════════════════════════════════════════════════════════
def _friendly_speaker(raw_label: str, attendees: list[str] | None = None) -> str:
    """SPEAKER_00 → 'Người nói 1' hoặc tên attendee nếu có sẵn (#9 + #10).
    Giữ nguyên nếu đã là tên người (đã rename hoặc raw không match)."""
    if not raw_label or raw_label == "...":
        return "…"
    # raw_label dạng SPEAKER_00, SPEAKER_01 ...
    if raw_label.upper().startswith("SPEAKER_"):
        try:
            idx = int(raw_label.split("_", 1)[1])
        except (ValueError, IndexError):
            return raw_label
        if attendees:
            if 0 <= idx < len(attendees) and attendees[idx].strip():
                return attendees[idx].strip()
        return f"Người nói {idx + 1}"
    return raw_label  # đã là tên người, giữ nguyên


def _spk_css_class(label: str) -> str:
    """Map speaker label → CSS class spk-0..spk-5 (cố định màu theo thứ tự xuất hiện)."""
    seen = st.session_state.setdefault("l_spk_color_map", {})
    if label not in seen:
        seen[label] = f"spk-{len(seen) % 6}" if label != "…" else "spk-x"
    return seen[label]


@st.cache_data(show_spinner=False)
def _list_input_devices() -> list[tuple[int, str]]:
    """Liệt kê mic devices của hệ thống (#6). Cache để không quét đi quét lại."""
    if sd is None:
        return []
    try:
        devices = sd.query_devices()
    except Exception:
        return []
    out = []
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            out.append((i, f'{d["name"]}  ({int(d.get("default_samplerate", 0))}Hz)'))
    return out


# ── FIX #13b: streaming WAV writer ─────────────────────────────────────────────
def _wav_open(sample_rate: int = SAMPLE_RATE) -> str:
    """Mở file WAV mới ở temp, lưu writer vào _shared. Trả về đường dẫn."""
    _wav_close()   # đóng file cũ nếu có
    tmp = tempfile.NamedTemporaryFile(suffix="_live.wav", delete=False)
    tmp.close()
    wf = wave.open(tmp.name, "wb")
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate)
    with _shared["wav_lock"]:
        _shared["wav_writer"] = wf
        _shared["wav_path"]   = tmp.name
    print(f"[WAV] mở stream → {tmp.name}", flush=True)
    return tmp.name


def _wav_append(chunk: np.ndarray) -> None:
    """Ghi 1 chunk int16 vào file WAV đang mở. No-op nếu chưa mở."""
    with _shared["wav_lock"]:
        wf = _shared.get("wav_writer")
        if wf is None:
            return
        try:
            wf.writeframes(chunk.tobytes())
        except Exception as e:
            print(f"[WAV] writeframes lỗi: {e}", flush=True)


def _wav_close() -> str | None:
    """Đóng file WAV. Trả về path (hoặc None nếu chưa mở)."""
    with _shared["wav_lock"]:
        wf   = _shared.get("wav_writer")
        path = _shared.get("wav_path")
        if wf is None:
            return None
        try: wf.close()
        except: pass
        _shared["wav_writer"] = None
    print(f"[WAV] đóng stream ← {path}", flush=True)
    return path


def _wav_path() -> str | None:
    return _shared.get("wav_path")


def _wav_snapshot() -> str | None:
    """Lấy snapshot WAV (header hợp lệ) của recording đang ghi — KHÔNG đóng writer.
    Trick: writer chỉ append PCM vào sau byte 44, nên ta:
      1) flush + fsync writer hiện tại
      2) đọc số byte PCM đã ghi (_datawritten)
      3) đọc trực tiếp PCM từ disk
      4) đóng gói lại thành WAV mới có header hợp lệ → trả path tạm
    Dùng cho incremental Sortformer khi user đang thu.
    """
    with _shared["wav_lock"]:
        wf = _shared.get("wav_writer")
        path = _shared.get("wav_path")
        if wf is None or path is None:
            return None
        try:
            wf._file.flush()
            os.fsync(wf._file.fileno())
        except Exception:
            pass
        data_written = getattr(wf, "_datawritten", 0)

    if data_written < SAMPLE_RATE * 2 * 5:   # < 5s audio → quá ngắn
        return None
    try:
        with open(path, "rb") as f:
            f.seek(44)
            pcm = f.read(data_written)
    except Exception as e:
        print(f"[snapshot] đọc PCM lỗi: {e}", flush=True)
        return None
    if not pcm:
        return None

    tmp = tempfile.NamedTemporaryFile(suffix="_snap.wav", delete=False)
    tmp.close()
    try:
        with wave.open(tmp.name, "wb") as wf2:
            wf2.setnchannels(1); wf2.setsampwidth(2); wf2.setframerate(SAMPLE_RATE)
            wf2.writeframes(pcm)
    except Exception as e:
        print(f"[snapshot] ghi WAV mới lỗi: {e}", flush=True)
        try: os.unlink(tmp.name)
        except: pass
        return None
    return tmp.name


def _run_sortformer_incremental_thread(sample_rate: int = SAMPLE_RATE) -> None:
    """Background: chạy Sortformer liên tục trên snapshot WAV.
    Sau mỗi pass, cache `_shared['sortformer_cache']` để finalize tận dụng.

    Sortformer trên CPU chạy ~3× real-time → nếu đợi đến lúc bấm Hoàn thiện
    mới chạy thì user phải ngồi chờ 5 phút cho 100s audio. Chạy NỀN từ lúc thu
    để khi bấm Hoàn thiện thì kết quả đã có sẵn (hoặc gần xong).
    """
    print("[SortformerInc] background thread start", flush=True)
    # Warm-up: đợi 15s đầu để có ít audio rồi mới chạy pass 1
    for _ in range(15):
        if _shared.get("sortformer_stop"): return
        time.sleep(1.0)

    while not _shared.get("sortformer_stop"):
        snap = _wav_snapshot()
        if not snap:
            time.sleep(3.0); continue
        try:
            with wave.open(snap, "rb") as wf:
                dur = wf.getnframes() / float(wf.getframerate() or sample_rate)
            if dur < 10.0:
                continue
            t0 = time.time()
            print(f"[SortformerInc] pass start: {dur:.1f}s audio …", flush=True)
            from core.diarization.sortformer_bridge import diarize_file_sortformer
            # FIX: serialize NeMo calls — đồng thời inc + finalize gây WinError 267 trên temp dir
            with _shared["sortformer_run_lock"]:
                segments = diarize_file_sortformer(wav_path=snap, num_speakers=None)
            elapsed = time.time() - t0
            _shared["sortformer_cache"] = {
                "segments":    segments,
                "duration":    dur,
                "elapsed":     elapsed,
                "computed_at": time.time(),
            }
            n_spk = len(set(getattr(s, "speaker", "?") for s in segments))
            print(f"[SortformerInc] ✓ cached: {len(segments)} segs, {n_spk} spk, "
                  f"{dur:.1f}s audio, took {elapsed:.1f}s", flush=True)
        except Exception as e:
            print(f"[SortformerInc] pass lỗi: {e}", flush=True)
            time.sleep(5.0)
        finally:
            try: os.unlink(snap)
            except: pass
        # Không sleep giữa các pass — recording vẫn đang dài ra, chạy tiếp ngay.
    print("[SortformerInc] background thread exit", flush=True)


# ── FIX #5b: async finalize worker (chạy nền, không block UI) ─────────────────
def _finalize_worker(wav_path: str, turns_snapshot: list, attendees_text: str,
                     final_backend_env: str, debug: bool) -> None:
    """Worker thread: chạy toàn bộ pipeline Hoàn thiện, đẩy progress + kết quả qua _shared.

    KHÔNG được gọi st.* (không có script context). Communication qua _shared['finalize_state'].
    """
    state = _shared["finalize_state"]
    lock  = _shared["finalize_lock"]

    def _prog(pct: int, msg: str):
        with lock:
            state["pct"] = max(state["pct"], int(pct))
            state["msg"] = msg
        print(f"[Finalize {pct:3d}%] {msg}", flush=True)

    try:
        _prog(5, "📼 Đang chuẩn bị âm thanh cuộc họp...")
        from core.diarizer import SpeakerSegment, get_speaker_stats
        from core.aligner import align, merge_consecutive, smooth_short_turns, rename_turns
        from core.punctuation_restorer import restore
        from collections import Counter as _Counter

        def _spk_dist(segs):
            return dict(_Counter(getattr(s, "speaker", s[2] if isinstance(s, tuple) else "?") for s in segs))

        if not wav_path or not os.path.exists(wav_path):
            raise RuntimeError("Không tìm thấy file âm thanh đã thu.")
        with wave.open(wav_path, "rb") as _wf:
            audio_dur_sec = _wf.getnframes() / float(_wf.getframerate() or SAMPLE_RATE)

        # Sortformer-only: không còn pyannote/diart fallback. 4-speaker cap là giới hạn cứng.
        segments = None
        used_backend = "sortformer"
        warning = None

        # 1) Ưu tiên cache từ thread incremental đã chạy nền lúc thu
        cache = _shared.get("sortformer_cache")
        if cache and cache.get("duration", 0) >= audio_dur_sec * 0.5:
            coverage = cache["duration"] / audio_dur_sec
            if coverage < 0.85 and audio_dur_sec < 600:
                _prog(20, f"⏳ Đợi pass Sortformer incremental cuối hoàn tất (đang cover {coverage*100:.0f}%)...")
                t_wait = time.time()
                while time.time() - t_wait < 90.0:
                    time.sleep(2.0)
                    new_cache = _shared.get("sortformer_cache")
                    if new_cache and new_cache.get("computed_at", 0) > cache.get("computed_at", 0):
                        cache = new_cache
                        coverage = cache["duration"] / audio_dur_sec
                        if coverage >= 0.85:
                            break
            segments = cache["segments"]
            used_backend = f"sortformer-incremental ({coverage*100:.0f}%)"
            _prog(40, f"⚡ Tái sử dụng Sortformer incremental (cover {coverage*100:.0f}% audio, "
                      f"đã tính {cache['elapsed']:.0f}s nền) ...")

        # 2) Không có cache hợp lệ → chạy Sortformer trên toàn bộ recording
        if segments is None:
            _prog(15, "🧠 Sortformer đang phân tách người nói (SOTA, tối đa 4 người)...")
            from core.diarization.sortformer_bridge import diarize_file_sortformer
            with _shared["sortformer_run_lock"]:
                segments = diarize_file_sortformer(wav_path=wav_path, num_speakers=None)
            used_backend = "sortformer"

        # 3) Cảnh báo (không fail) khi attendees > 4 — Sortformer cap 4.
        expected_attendees = [n for n in attendees_text.splitlines() if n.strip()]
        if len(expected_attendees) > 4:
            warning = (f"Bạn nhập {len(expected_attendees)} thành viên nhưng Sortformer chỉ "
                       f"hỗ trợ tối đa 4 người nói. Các thành viên dư sẽ bị gộp.")

        if not segments:
            raise RuntimeError(
                "Sortformer không trả ra segment nào. Kiểm tra file âm thanh có hợp lệ "
                "(WAV 16kHz mono, >= vài giây speech) và NeMo cài đúng trong sortformer_env."
            )

        n_spk_final = len(set(_spk_dist(segments)))
        _prog(60, f"✓ Phân tách xong: {n_spk_final} người nói ({used_backend}). Khôi phục dấu câu...")

        raw = " ".join(t.text for t in turns_snapshot if t.text.strip())
        full_text = restore(raw.lower())

        _prog(80, "🧩 Đang ép khớp thời gian (forced-align)...")
        # FIX: truyền live_turns vào để align có fallback "pseudo-word-timestamps"
        # tốt hơn ratio-based khi stable-ts không có sẵn trong env (env Sortformer).
        aligned = align(segments=segments, full_text=full_text, wav_path=wav_path,
                        language="vi", use_forced_align=True, gap_limit=1.5,
                        live_turns=turns_snapshot)
        _prog(92, "🪡 Đang gộp lượt nói liên tiếp...")
        merged = merge_consecutive(aligned, gap_limit=1.5)
        merged = smooth_short_turns(merged, max_words=4, max_dur=1.2, gap_limit=1.5)

        # Auto-rename
        attendees_list = [n.strip() for n in attendees_text.splitlines() if n.strip()]
        raw_speakers_sorted = sorted({t.speaker for t in merged if t.speaker.upper().startswith("SPEAKER_")},
                                     key=lambda s: int(s.split("_")[1]) if s.split("_")[1].isdigit() else 999)
        rename_map = {}
        for i, raw_s in enumerate(raw_speakers_sorted):
            rename_map[raw_s] = attendees_list[i] if (attendees_list and i < len(attendees_list)) else f"Người nói {i + 1}"
        if rename_map:
            merged = rename_turns(merged, rename_map)
            segments = [SpeakerSegment(speaker=rename_map.get(s.speaker, s.speaker), start=s.start, end=s.end)
                        for s in segments]

        with lock:
            state["result"] = {
                "merged":       merged,
                "stats":        get_speaker_stats(segments),
                "warning":      warning,
                "backend":      used_backend,
                "n_speakers":   n_spk_final,
            }
            state["pct"] = 100
            state["msg"] = f"✅ Hoàn tất! {len(merged)} lượt · {n_spk_final} người nói · {used_backend}"
            state["running"] = False
    except Exception as e:
        import traceback
        traceback.print_exc()
        with lock:
            state["error"] = f"{type(e).__name__}: {e}"
            state["running"] = False


def _sortformer_disabled() -> bool:
    """Cờ tắt hoàn toàn Sortformer — đặt SORTFORMER_DISABLED=1 trong .env nếu NeMo
    crash trên máy này (DLL/CUDA/ABI mismatch). App sẽ skip cả prewarm, incremental
    thread, lẫn finalize path → đi thẳng pyannote. Giúp port app sang máy khác
    không có NeMo hoạt động."""
    return os.getenv("SORTFORMER_DISABLED", "0").strip() in ("1", "true", "yes")


def _prewarm_sortformer_async():
    """FIX #5: pre-load Sortformer subprocess ở background → bấm 'Hoàn thiện' không chờ cold-start."""
    if _sortformer_disabled():
        print("[Prewarm] ⏭️ Bỏ qua: SORTFORMER_DISABLED=1", flush=True)
        return
    if _shared.get("sortformer_warm"):
        return
    _shared["sortformer_warm"] = True  # đặt true ngay để tránh spawn nhiều thread

    def _worker():
        try:
            from core.diarization.sortformer_bridge import diarize_file_sortformer
            # Bơm 1s silence vào để model load + JIT compile mà không tốn CPU thật
            import wave, tempfile as _tf
            tmp = _tf.NamedTemporaryFile(suffix="_warm.wav", delete=False)
            try:
                with wave.open(tmp.name, "wb") as wf:
                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
                    wf.writeframes((np.zeros(16000, dtype=np.int16)).tobytes())
                diarize_file_sortformer(wav_path=tmp.name, num_speakers=None)
                print("[Prewarm] ✅ Sortformer subprocess sẵn sàng.", flush=True)
            finally:
                try: os.unlink(tmp.name)
                except: pass
        except Exception as e:
            print(f"[Prewarm] ⚠️ Sortformer chưa pre-warm được ({e}). Sẽ load khi bấm Hoàn thiện.",
                  flush=True)

    t = threading.Thread(target=_worker, daemon=True, name="sortformer-prewarm")
    add_script_run_ctx(t); t.start()

# ══════════════════════════════════════════════════════════════════════════════
# 3. PAGE CONFIG & CSS 
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title = "Smart Meeting — Live",
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
        background: #e8520a !important; border: none !important; font-weight: 600 !important; color: white !important;
    }
    .stButton > button[kind="primary"]:hover { background: #c44008 !important; }
    .stProgress > div > div > div { background: #e8520a !important; }
    hr { border-color: rgba(128,128,128,0.2) !important; }
    @keyframes rec-blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
    .rec-dot {
        display: inline-block; width: 8px; height: 8px; border-radius: 50%;
        background: #e8520a; animation: rec-blink 1.2s ease-in-out infinite; margin-right: 6px;
    }
    /* Speaker pill (live transcript) */
    .spk-pill {
        display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:10px;
        padding:1px 8px; border-radius:10px; font-weight:600; letter-spacing:0.5px;
        white-space:nowrap; margin-right:8px;
    }
    .spk-0 { background:rgba(232,82,10,0.15);  color:#e8520a; border:1px solid rgba(232,82,10,0.3);}
    .spk-1 { background:rgba(77,166,232,0.15); color:#4da6e8; border:1px solid rgba(77,166,232,0.3);}
    .spk-2 { background:rgba(46,139,46,0.15);  color:#2e8b2e; border:1px solid rgba(46,139,46,0.3);}
    .spk-3 { background:rgba(180,90,200,0.15); color:#b45ac8; border:1px solid rgba(180,90,200,0.3);}
    .spk-4 { background:rgba(220,160,40,0.15); color:#dca028; border:1px solid rgba(220,160,40,0.3);}
    .spk-5 { background:rgba(160,80,40,0.15);  color:#a05028; border:1px solid rgba(160,80,40,0.3);}
    .spk-x { background:rgba(128,128,128,0.15);color:#888;    border:1px solid rgba(128,128,128,0.3);}
    /* VU meter */
    .vu-track { background:rgba(128,128,128,0.15); border-radius:3px; height:6px; overflow:hidden; }
    .vu-fill  { background:linear-gradient(90deg,#2e8b2e 0%,#dca028 70%,#e8520a 95%); height:100%; transition:width 80ms linear; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 24px;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;
                    letter-spacing:3px;text-transform:uppercase;color:#e8520a;
                    margin-bottom:8px;">// SMART MEETING</div>
        <div style="font-size:18px;font-weight:700;color:var(--text-color);">Trợ lý Cuộc họp</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                    color:var(--text-color);opacity:0.6;margin-top:6px;">STT · DIARIZATION · SUMMARY</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.page_link("app.py",               label="🏠  Trang chủ")
    st.page_link("pages/2_Live_Mode.py", label="🎙️  Smart Meeting (Live)")

st.markdown("""
<div style="font-size:11px;letter-spacing:3px;text-transform:uppercase;
            color:var(--text-color);opacity:0.7;padding-bottom:12px;border-bottom:1px solid rgba(128,128,128,0.2);
            margin-bottom:20px;">
    🎙️  Smart Meeting — Ghi âm Cuộc họp Trực tiếp
</div>
""", unsafe_allow_html=True)

if not LIVE_ENV_READY:
    st.error("⚠️ Thiếu thư viện âm thanh. Hãy kiểm tra môi trường.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# 4. CACHE MODELS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def _get_zipformer():
    print("⏳ Loading Zipformer...")
    # FIX #14 (v2): cắt aggressive hơn cho hội thoại rapid-fire.
    # rule1 silence-ngắn  1.2 → 0.7  (catch interruption khi gap ngắn)
    # rule2 silence-dài   0.8 → 0.4  (kết thúc câu hoàn chỉnh nhanh hơn)
    # rule3 độ dài utt   14  → 10   (force flush nếu utterance kéo dài quá)
    # → giảm hiện tượng "monster turn" 100+ từ nuốt nhiều speaker.
    r = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=TOKENS_PATH, encoder=ENCODER_PATH, decoder=DECODER_PATH, joiner=JOINER_PATH,
        num_threads=2, sample_rate=SAMPLE_RATE, feature_dim=80,
        enable_endpoint_detection=True, rule1_min_trailing_silence=0.7,
        rule2_min_trailing_silence=0.4, rule3_min_utterance_length=10,
    )
    return r

with st.spinner("Đang nạp mô hình STT..."): recognizer = _get_zipformer()

# FIX #5: pre-warm Sortformer subprocess ở background khi app khởi động
_prewarm_sortformer_async()


# ══════════════════════════════════════════════════════════════════════════════
# 5. SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_defaults = {
    "l_recording": False, "l_finished": False, "l_diar_done": False, "l_renamed": False,
    "l_stream_ref": None, "l_asr_stream": None,
    "l_turns": [], "l_stats": {}, "l_name_map": {}, "l_raw_audio": [],
    "l_n_chunks": 0, "l_partial": "", "l_session_name": "", "l_show_full": False,
    "l_summary": "", "l_summary_done": False,
    "l_file_mode": False,
    # FIX #6 #8 #9 #13 #16
    "l_mic_device": None,         # mic device index (None = default)
    "l_paused": False,            # đang tạm dừng?
    "l_attendees": "",            # danh sách thành viên (textarea, 1 tên/dòng)
    "l_finalize_count": 0,        # đếm lần Hoàn thiện đã chạy
    "l_finalize_backend": "auto", # auto/sortformer/pyannote
    "l_ram_warned": False,        # đã cảnh báo RAM chưa
    "l_spk_color_map": {},        # map nhãn → CSS class màu
}
for k, v in _defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# 6. CALLBACK & THREAD WORKER (Cốt lõi xử lý NPU API)
# ══════════════════════════════════════════════════════════════════════════════
def _audio_callback(indata, frames, time_info, status):
    """Làn 0: Thu âm từ Micro. Fan-out cho STT và Diarization.

    LƯU Ý: hàm này chạy trong thread của sounddevice — KHÔNG được truy cập
    st.session_state (không có ScriptRunContext → fail âm thầm). Vì vậy diart
    streamer được lấy từ _shared (singleton cache_resource), không phải session_state.
    """
    chunk = (indata[:, 0] * 32767).astype(np.int16)
    # FIX #7: cập nhật peak level cho VU meter (smoothed)
    try:
        peak = float(np.max(np.abs(indata[:, 0]))) if indata.size else 0.0
        prev = _shared.get("audio_peak", 0.0)
        _shared["audio_peak"] = max(peak, prev * 0.75)  # decay nhanh khi im lặng
    except Exception:
        pass
    _AUDIO_QUEUE.put(chunk.copy())   # ASR thread → Zipformer; phân speaker ở finalize


def _stream_file_thread(wav_path: str, sample_rate: int = 16000, realtime: bool = True):
    """Bơm 1 file WAV vào pipeline GIỐNG HỆT mic — để test trước khi thu thật.

    Chạy trong thread nền → KHÔNG truy cập st.session_state. Đẩy chunk vào _AUDIO_QUEUE
    để ASR (Zipformer) consume. Tự downmix stereo → mono. realtime=True phát 1× như mic.
    """
    import wave as _wave
    try:
        wf = _wave.open(wav_path, "rb")
    except Exception as e:
        print(f"[FileStream] ❌ Không mở được file: {e}", flush=True)
        _shared["file_done"] = True
        return

    n_ch = wf.getnchannels()
    sw   = wf.getsampwidth()
    sr   = wf.getframerate()
    if sw != 2 or sr != sample_rate:
        print(f"[FileStream] ⚠️ Cần WAV 16kHz/16-bit. File: {sr}Hz {sw*8}-bit {n_ch}ch. "
              f"Convert: ffmpeg -i in.wav -ar 16000 -ac 1 out.wav", flush=True)
        wf.close(); _shared["file_done"] = True
        return

    print(f"[FileStream] ▶ Stream {wav_path} ({n_ch}ch {sr}Hz, {'real-time' if realtime else 'fast'})",
          flush=True)
    chunk_dt = CHUNK_FRAMES / sample_rate
    while _shared.get("file_streaming"):
        frames = wf.readframes(CHUNK_FRAMES)
        if not frames:
            break
        chunk = np.frombuffer(frames, dtype=np.int16)
        if n_ch == 2:
            chunk = chunk.reshape(-1, 2).mean(axis=1).astype(np.int16)
        _AUDIO_QUEUE.put(chunk.copy())
        if realtime:
            time.sleep(chunk_dt)

    wf.close()
    _shared["file_done"] = True
    print("[FileStream] ✅ Đã đọc hết file", flush=True)


# (Đã xoá: _run_diart_thread — NPU HTTP backend cho diart, không dùng nữa.)

# ══════════════════════════════════════════════════════════════════════════════
# (Đã loại bỏ live speaker assignment — Sortformer ở Hoàn thiện gán hết.)


# ══════════════════════════════════════════════════════════════════════════════
# 8. XỬ LÝ LÀN 1 (STT STREAMING)
# ══════════════════════════════════════════════════════════════════════════════
def _prettify_live(text: str) -> str:
    """Lấy output thô của Zipformer (TOÀN UPPERCASE, không dấu câu) → câu đọc được.
    Dùng cho mỗi turn live khi commit, KHÔNG cho partial (partial flicker quá nhanh).
    """
    if not text:
        return text
    try:
        from core.punctuation_restorer import restore
        return restore(text.lower())
    except Exception:
        # Fallback: lower + capitalize chữ đầu để không bị wall-of-uppercase
        return text.lower().capitalize()


def _zipformer_result_with_ts(z_stream) -> tuple[str, list]:
    """Lấy (text, per_token_timestamps_giây) từ sherpa-onnx stream.

    Sherpa-onnx trả mỗi token (syllable Vietnamese) 1 timestamp THẬT, không
    phải ước đoán. Dùng để map vào pyannote/Sortformer segments chính xác
    hơn ratio-based interpolation.

    Fallback: nếu binding không hỗ trợ JSON method → trả text + list rỗng.
    """
    # Thử JSON API trước (newer binding)
    try:
        import json
        s = recognizer.get_result_as_json_string(z_stream)
        if s:
            data = json.loads(s)
            text = (data.get("text") or "").strip()
            ts = data.get("timestamps") or []
            return text, list(ts)
    except Exception:
        pass

    # Fallback: text-only (legacy binding)
    try:
        result = recognizer.get_result(z_stream)
        text = result if isinstance(result, str) else getattr(result, "text", str(result))
        return text.strip(), []
    except Exception:
        return "", []


def _drain_queue() -> tuple[str, list]:
    from core.aligner import AlignedTurn

    if st.session_state.l_asr_stream is None:
        st.session_state.l_asr_stream = recognizer.create_stream()
    z_stream  = st.session_state.l_asr_stream
    completed = []

    while not _AUDIO_QUEUE.empty():
        try: chunk = _AUDIO_QUEUE.get_nowait()
        except _queue_module.Empty: break

        # FIX #13b: ghi trực tiếp ra file WAV thay vì giữ list trong RAM
        _wav_append(chunk)
        st.session_state.l_n_chunks += 1

        samples = chunk.astype(np.float32) / 32768.0
        z_stream.accept_waveform(SAMPLE_RATE, samples)

        while recognizer.is_ready(z_stream):
            recognizer.decode_stream(z_stream)

        if recognizer.is_endpoint(z_stream):
            # FIX: lấy text + REAL per-token timestamps từ sherpa-onnx
            text, ts = _zipformer_result_with_ts(z_stream)
            if text:
                # t_end là thời điểm CHUNK cuối — chính xác. t_start: ưu tiên timestamp THẬT.
                t_end = round((st.session_state.l_n_chunks * CHUNK_FRAMES) / SAMPLE_RATE, 2)
                if ts:
                    t_start = round(max(0.0, float(ts[0])), 2)
                else:
                    # Fallback ước lượng theo word count (binding cũ không có timestamps)
                    est_sec = max(0.5, len(text.split()) / 2.5)
                    t_start = round(max(0.0, t_end - est_sec), 2)

                # Speaker placeholder lúc live — Sortformer ở Hoàn thiện ghi đè.
                pretty_text = _prettify_live(text)
                # Map timestamps gốc về tọa độ syllable-level dùng cho aligner
                # (sherpa-onnx token = Vietnamese syllable, gần khớp 1-1 với word.split())
                word_starts = [float(t) for t in ts] if ts else None
                completed.append(AlignedTurn(
                    speaker     = "...",
                    start       = t_start,
                    end         = t_end,
                    text        = pretty_text,
                    word_starts = word_starts,
                ))
                ts_tag = f"+{len(ts)}ts" if ts else "no-ts"
                print(f"[ASR turn] [{t_start:6.1f}-{t_end:6.1f}] "
                      f"({len(text.split())} từ, {ts_tag}): {text}", flush=True)
            recognizer.reset(z_stream)

    # FIX #1: lower-case partial (chưa restore vì partial flicker quá nhanh, restore tốn CPU)
    partial, _partial_ts = _zipformer_result_with_ts(z_stream)
    partial = partial.strip()
    if partial:
        partial = partial.lower()
        partial = partial[:1].upper() + partial[1:] if partial else partial
    return partial, completed


def _flush_final_asr() -> None:
    """Khi DỪNG: ép Zipformer trả nốt phần text ĐANG DỞ (partial chưa tới endpoint).

    Zipformer chỉ commit 1 turn khi phát hiện endpoint (im lặng ~2.4s). Nói liên
    tục → phần lớn text nằm ở 'partial', chưa vào l_turns. Nếu không flush, đoạn
    cuối (đang hiện ở dòng [···]) BỊ MẤT khỏi transcript cuối cùng.
    """
    from core.aligner import AlignedTurn

    z = st.session_state.get("l_asr_stream")
    if z is None:
        return

    # 1) Xử lý nốt audio còn trong queue
    while not _AUDIO_QUEUE.empty():
        try:
            chunk = _AUDIO_QUEUE.get_nowait()
        except _queue_module.Empty:
            break
        _wav_append(chunk)   # FIX #13b
        st.session_state.l_n_chunks += 1
        z.accept_waveform(SAMPLE_RATE, chunk.astype(np.float32) / 32768.0)

    # 2) Báo hết stream → decode nốt → lấy text cuối
    try:
        z.input_finished()
    except Exception:
        pass
    while recognizer.is_ready(z):
        recognizer.decode_stream(z)

    text, ts = _zipformer_result_with_ts(z)
    if text:
        t_end = round((st.session_state.l_n_chunks * CHUNK_FRAMES) / SAMPLE_RATE, 2)
        if ts:
            t_start = round(max(0.0, float(ts[0])), 2)
        else:
            est_sec = max(0.5, len(text.split()) / 2.5)
            t_start = round(max(0.0, t_end - est_sec), 2)
        st.session_state.l_turns.append(AlignedTurn(
            speaker     = "...",
            start       = t_start,
            end         = t_end,
            text        = _prettify_live(text),
            word_starts = [float(t) for t in ts] if ts else None,
        ))
        print(f"[Flush] Commit đoạn cuối ({len(text.split())} từ): {text[:60]}…", flush=True)

    st.session_state.l_partial = ""


# ══════════════════════════════════════════════════════════════════════════════
# 9. UI CONTROL PANEL
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--text-color);opacity:0.7;margin-bottom:16px;">① Thiết lập cuộc họp</div>', unsafe_allow_html=True)

cfg_c1, cfg_c2 = st.columns([2, 1], gap="medium")
with cfg_c1:
    session_name = st.text_input("Tên cuộc họp", value=st.session_state.l_session_name or f"MEETING-{time.strftime('%Y%m%d-%H%M')}", disabled=st.session_state.l_recording)
    if session_name != st.session_state.l_session_name: st.session_state.l_session_name = session_name
with cfg_c2:
    # FIX #6: mic selector
    devices = _list_input_devices()
    if devices:
        device_labels = ["🎙️ Mặc định hệ thống"] + [f"#{i} · {name}" for i, name in devices]
        device_ids    = [None] + [i for i, _ in devices]
        try:
            current_idx = device_ids.index(st.session_state.l_mic_device) if st.session_state.l_mic_device in device_ids else 0
        except ValueError:
            current_idx = 0
        sel = st.selectbox("Thiết bị micro", device_labels, index=current_idx, disabled=st.session_state.l_recording)
        st.session_state.l_mic_device = device_ids[device_labels.index(sel)]
    else:
        st.caption("🎙️ Mic: mặc định")

# FIX #9: pre-load thành viên (optional)
with st.expander("👥 Danh sách thành viên (tuỳ chọn — gán tên người nói trước)", expanded=False):
    st.caption("Nhập mỗi thành viên một dòng theo thứ tự dự kiến. Mô hình sẽ tự gán "
               "`Người nói 1 → tên dòng 1`, `Người nói 2 → tên dòng 2`, v.v. sau khi hoàn thiện.")
    attendees_text = st.text_area("Tên thành viên", value=st.session_state.l_attendees,
                                  placeholder="Chủ tọa\nThư ký\nThành viên A\nThành viên B",
                                  height=110, disabled=st.session_state.l_recording,
                                  label_visibility="collapsed")
    if attendees_text != st.session_state.l_attendees:
        st.session_state.l_attendees = attendees_text

st.markdown("<div style='margin:12px 0 8px;'></div>", unsafe_allow_html=True)
btn_col1, btn_col2, status_col = st.columns([1, 1, 2], gap="small")

with btn_col1:
    if not st.session_state.l_recording:
        if st.button("🔴 Bắt đầu Cuộc họp", type="primary", use_container_width=True):
            st.session_state.l_turns = []; st.session_state.l_stats = {}; st.session_state.l_name_map = {}
            st.session_state.l_raw_audio = []; st.session_state.l_n_chunks = 0; st.session_state.l_partial = ""
            st.session_state.l_finished = False; st.session_state.l_diar_done = False; st.session_state.l_renamed = False
            st.session_state.l_summary = ""; st.session_state.l_summary_done = False
            st.session_state.l_asr_stream = None
            st.session_state.l_file_mode = False   # đây là phiên thu mic, không phải file-test
            st.session_state.l_paused = False
            st.session_state.l_finalize_count = 0
            st.session_state.l_ram_warned = False
            st.session_state.l_spk_color_map = {}

            with _AUDIO_QUEUE.mutex: _AUDIO_QUEUE.queue.clear()
            # FIX #13b: mở stream WAV mới (xoá file cũ nếu có)
            old = _wav_close()
            if old:
                try: os.unlink(old)
                except: pass
            _wav_open(SAMPLE_RATE)

            # Live diarization đã loại bỏ — Smart Meeting chỉ phân tách người nói
            # ở bước Hoàn thiện bằng Sortformer. Lúc thu chỉ chạy Zipformer STT
            # + ghi WAV ra disk + (optionally) Sortformer incremental nền.
            print("[LiveMode] Live diarization OFF — speakers will be assigned at finalize", flush=True)

            # Spawn incremental Sortformer thread nền (nếu chưa disabled)
            if not _sortformer_disabled():
                _shared["sortformer_stop"]  = False
                _shared["sortformer_cache"] = None
                sf_inc = threading.Thread(target=_run_sortformer_incremental_thread,
                                          kwargs={"sample_rate": SAMPLE_RATE},
                                          daemon=True, name="sortformer-inc")
                add_script_run_ctx(sf_inc); sf_inc.start()
                print("[LiveMode] Incremental Sortformer thread spawned", flush=True)

            stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                blocksize=CHUNK_FRAMES, callback=_audio_callback,
                device=st.session_state.l_mic_device,   # FIX #6
            )
            stream.start()
            st.session_state.l_stream_ref = stream
            st.session_state.l_recording = True
            st.rerun()
    else:
        # FIX #8: nút Pause/Resume (khi đang thu)
        if st.session_state.l_paused:
            if st.button("▶️ Tiếp tục", type="primary", use_container_width=True):
                # Resume: mở lại InputStream
                try:
                    stream = sd.InputStream(
                        samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=CHUNK_FRAMES, callback=_audio_callback,
                        device=st.session_state.l_mic_device,
                    )
                    stream.start()
                    st.session_state.l_stream_ref = stream
                    st.session_state.l_paused = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Không mở được mic để tiếp tục: {e}")
        else:
            if st.button("⏸️ Tạm dừng", use_container_width=True):
                if st.session_state.l_stream_ref:
                    try: st.session_state.l_stream_ref.stop()
                    except: pass
                    try: st.session_state.l_stream_ref.close()
                    except: pass
                    st.session_state.l_stream_ref = None
                st.session_state.l_paused = True
                st.rerun()

with btn_col2:
    if st.session_state.l_recording:
        if st.button("⏹️ Kết thúc Cuộc họp", use_container_width=True, key="btn_stop"):
            _shared["file_streaming"] = False   # dừng file-test thread nếu đang chạy
            if st.session_state.l_stream_ref:
                st.session_state.l_stream_ref.stop()
                st.session_state.l_stream_ref.close()
                st.session_state.l_stream_ref = None

            _flush_final_asr()   # ép commit đoạn text đang dở (partial) → không mất

            st.session_state.l_asr_stream = None
            st.session_state.l_recording = False
            st.session_state.l_finished = True
            st.session_state.l_paused = False
            _wav_close()   # FIX #13b: đóng file WAV → flush header
            _shared["sortformer_stop"] = True   # FIX #5c: dừng thread incremental sau pass hiện tại
            st.rerun()
    else:
        # Khi đã DỪNG → hiển thị Hoàn thiện. FIX #16: cho phép Re-run.
        if st.session_state.l_finished and not st.session_state.l_diar_done:
            _finalize_label = "✅ Hoàn thiện Transcript"
            _finalize_disabled = False
        elif st.session_state.l_diar_done:
            _finalize_label = "🔄 Chạy lại Hoàn thiện"
            _finalize_disabled = False
        else:
            _finalize_label = "✅ Hoàn thiện Transcript"
            _finalize_disabled = True
        # Disable nút khi worker đang chạy (FIX #5b)
        _finalize_state = _shared["finalize_state"]
        _is_running     = bool(_finalize_state.get("running"))
        if _is_running:
            _finalize_disabled = True
            _finalize_label = "⏳ Đang hoàn thiện..."

        if st.button(_finalize_label, disabled=_finalize_disabled, type="primary", use_container_width=True, key="btn_finalize"):
            # FIX #16: reset diar_done để re-run được
            st.session_state.l_diar_done = False
            st.session_state.l_renamed = False
            st.session_state.l_finalize_count += 1

            # Reset finalize_state + spawn worker (FIX #5b)
            with _shared["finalize_lock"]:
                _shared["finalize_state"].update({
                    "running": True, "pct": 0, "msg": "Khởi động pipeline hoàn thiện...",
                    "result": None, "error": None,
                })
            _worker = threading.Thread(
                target=_finalize_worker,
                kwargs={
                    "wav_path":          _wav_path(),
                    "turns_snapshot":    list(st.session_state.l_turns),
                    "attendees_text":    st.session_state.l_attendees,
                    "final_backend_env": os.getenv("LIVE_FINAL_BACKEND", "sortformer"),
                    "debug":             os.getenv("FINALIZE_DEBUG", "0") == "1",
                },
                daemon=True, name="finalize-worker",
            )
            add_script_run_ctx(_worker)
            _worker.start()
            st.rerun()

with status_col:
    if st.session_state.l_recording:
        dur_sec = (st.session_state.l_n_chunks * CHUNK_FRAMES) / SAMPLE_RATE
        # FIX #7: VU meter (peak level từ _shared, đã smoothed trong callback)
        peak = float(_shared.get("audio_peak", 0.0))
        peak_pct = min(100, int(peak * 130))   # boost hệ số để hiển thị rõ hơn
        if st.session_state.l_paused:
            badge = (f'<div style="background:rgba(128,128,128,0.1);border:1px solid #888;border-radius:4px;'
                     f'padding:8px 14px;display:inline-flex;align-items:center;gap:8px;margin-top:4px;">'
                     f'<span style="font-family:monospace;font-size:11px;font-weight:600;color:#888;">'
                     f'⏸ TẠM DỪNG &nbsp;{int(dur_sec//60):02d}:{int(dur_sec%60):02d} &nbsp;·&nbsp; {len(st.session_state.l_turns)} lượt</span></div>')
        else:
            badge = (f'<div style="background:rgba(232,82,10,0.1);border:1px solid #e8520a;border-radius:4px;'
                     f'padding:8px 14px;display:inline-flex;align-items:center;gap:8px;margin-top:4px;">'
                     f'<span class="rec-dot"></span>'
                     f'<span style="font-family:monospace;font-size:11px;font-weight:600;color:#e8520a;">'
                     f'REC &nbsp;{int(dur_sec//60):02d}:{int(dur_sec%60):02d} &nbsp;·&nbsp; {len(st.session_state.l_turns)} lượt</span></div>')
        vu = (f'<div style="margin-top:6px;display:flex;align-items:center;gap:8px;">'
              f'<span style="font-family:monospace;font-size:9px;color:var(--text-color);opacity:0.6;'
              f'min-width:24px;">🎤</span>'
              f'<div class="vu-track" style="flex:1;max-width:260px;"><div class="vu-fill" style="width:{peak_pct}%;"></div></div>'
              f'<span style="font-family:monospace;font-size:9px;color:var(--text-color);opacity:0.5;'
              f'min-width:34px;">{peak_pct:>3}%</span></div>')
        st.markdown(badge + vu, unsafe_allow_html=True)
    elif st.session_state.l_finished and not st.session_state.l_diar_done:
        st.markdown(f'<div style="font-family:monospace;font-size:11px;color:var(--text-color);opacity:0.8;padding:8px 14px;background:var(--secondary-background-color);border:1px solid rgba(128,128,128,0.3);border-radius:4px;display:inline-block;margin-top:4px;">⏹ Đã kết thúc &nbsp;·&nbsp; Bấm <b>Hoàn thiện Transcript</b></div>', unsafe_allow_html=True)
    elif st.session_state.l_diar_done and not st.session_state.l_summary_done:
        st.markdown(f'<div style="font-family:monospace;font-size:11px;color:#4da6e8;padding:8px 14px;background:rgba(77,166,232,0.1);border:1px solid #4da6e8;border-radius:4px;display:inline-block;margin-top:4px;">✓ Đã hoàn thiện &nbsp;·&nbsp; {len(st.session_state.l_stats)} người nói &nbsp;·&nbsp; Có thể tạo Tóm tắt bên dưới</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 9A. FIX #5b: ASYNC FINALIZE POLLING — render progress + commit result
# ══════════════════════════════════════════════════════════════════════════════
_fs = _shared["finalize_state"]
with _shared["finalize_lock"]:
    _fs_running = bool(_fs.get("running"))
    _fs_pct     = int(_fs.get("pct", 0))
    _fs_msg     = str(_fs.get("msg", ""))
    _fs_result  = _fs.get("result")
    _fs_error   = _fs.get("error")

if _fs_running:
    # Worker đang chạy — render progress + tự rerun mỗi 500ms
    st.progress(max(0, min(100, _fs_pct)), text=_fs_msg or "Đang xử lý...")
    time.sleep(0.5)
    st.rerun()
elif _fs_result is not None and not st.session_state.l_diar_done:
    # Worker xong → commit kết quả vào session_state
    res = _fs_result
    st.session_state.l_turns        = res["merged"]
    st.session_state.l_stats        = res["stats"]
    st.session_state.l_diar_done    = True
    if res.get("warning"):
        st.warning(f"⚠️ {res['warning']}")
    st.success(f"✅ Hoàn tất — {len(res['merged'])} lượt · {res['n_speakers']} người nói · backend: {res['backend']}")
    # Clear result để không re-commit
    with _shared["finalize_lock"]:
        _fs["result"] = None
    st.rerun()
elif _fs_error and not st.session_state.l_diar_done:
    st.error(f"❌ Lỗi khi hoàn thiện transcript: {_fs_error}")
    with _shared["finalize_lock"]:
        _fs["error"] = None

# ══════════════════════════════════════════════════════════════════════════════
# 9B. TEST BẰNG FILE (chạy thử pipeline với file trước khi thu mic)
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.l_recording and not st.session_state.l_finished:
    with st.expander("🧪 Test bằng file (chạy thử trước khi thu mic)", expanded=False):
        st.caption("Bơm 1 file WAV qua ĐÚNG pipeline như mic: STT + diarization + gán speaker. "
                   "File nên là WAV 16kHz (stereo sẽ tự downmix). Chạy ở tốc độ thực 1×.")
        up = st.file_uploader("Chọn file WAV để test", type=["wav"], key="l_test_file")
        if st.button("▶️ Stream file qua pipeline", disabled=up is None, use_container_width=True):
            # Lưu file upload ra temp
            tmp = tempfile.NamedTemporaryFile(suffix="_test.wav", delete=False)
            tmp.write(up.getbuffer()); tmp.flush(); tmp.close()

            # Reset state (giống bắt đầu thu)
            st.session_state.l_turns = []; st.session_state.l_stats = {}; st.session_state.l_name_map = {}
            st.session_state.l_raw_audio = []; st.session_state.l_n_chunks = 0; st.session_state.l_partial = ""
            st.session_state.l_finished = False; st.session_state.l_diar_done = False; st.session_state.l_renamed = False
            st.session_state.l_summary = ""; st.session_state.l_summary_done = False
            st.session_state.l_asr_stream = None
            # Reset cờ mới (#8 #13 #16)
            st.session_state.l_paused = False
            st.session_state.l_finalize_count = 0
            st.session_state.l_ram_warned = False
            st.session_state.l_spk_color_map = {}

            with _AUDIO_QUEUE.mutex: _AUDIO_QUEUE.queue.clear()
            old = _wav_close()
            if old:
                try: os.unlink(old)
                except: pass
            _wav_open(SAMPLE_RATE)

            # File-test mode: chỉ Sortformer incremental ở nền (giống mic mode)
            if not _sortformer_disabled():
                _shared["sortformer_stop"]  = False
                _shared["sortformer_cache"] = None
                sf_inc = threading.Thread(target=_run_sortformer_incremental_thread,
                                          kwargs={"sample_rate": SAMPLE_RATE},
                                          daemon=True, name="sortformer-inc")
                add_script_run_ctx(sf_inc); sf_inc.start()
                print("[LiveMode] Incremental Sortformer thread spawned (file-test mode)", flush=True)

            # Spawn thread đọc file → bơm vào pipeline
            _shared["file_streaming"] = True
            _shared["file_done"] = False
            ft = threading.Thread(target=_stream_file_thread,
                                  args=(tmp.name, SAMPLE_RATE, True),
                                  daemon=True, name="file-stream")
            add_script_run_ctx(ft); ft.start()

            st.session_state.l_file_mode = True
            st.session_state.l_recording = True
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# 10. GÁN TÊN SPEAKER & UI TRANSCRIPT & SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.l_diar_done and st.session_state.l_stats:
    st.markdown("<hr style='border-color:rgba(128,128,128,0.2);margin:24px 0 16px;'>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--text-color);opacity:0.7;margin-bottom:14px;">② Gán tên người nói</div>', unsafe_allow_html=True)
    from components.speaker_editor import speaker_editor
    name_map = speaker_editor(st.session_state.l_stats, key_prefix="l_spk")
    if st.button("✅ Xác nhận tên người nói"):
        from core.aligner import rename_turns
        st.session_state.l_turns = rename_turns(st.session_state.l_turns, name_map)
        st.session_state.l_name_map = name_map
        st.session_state.l_renamed = True
        st.rerun()

st.markdown("<hr style='border-color:rgba(128,128,128,0.2);margin:24px 0 16px;'>", unsafe_allow_html=True)
col_title, col_toggle = st.columns([3, 1])
with col_title: st.markdown('<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--text-color);opacity:0.7;margin-bottom:14px;">③ Nội dung cuộc họp</div>', unsafe_allow_html=True)
with col_toggle:
    if st.session_state.l_turns:
        if st.button("📄 Xem toàn bộ" if not st.session_state.l_show_full else "🔼 Thu gọn", use_container_width=True):
            st.session_state.l_show_full = not st.session_state.l_show_full
            st.rerun()

turns = st.session_state.l_turns
attendees_list = [n.strip() for n in st.session_state.l_attendees.splitlines() if n.strip()]
if st.session_state.l_recording:
    # ── Khi THU: hiện nhãn người nói (live, có thể flicker nhẹ) + text, KHÔNG hiện timestamp giả.
    recent = turns[-8:] if len(turns) > 8 else turns
    rows = ""
    for t in recent:
        # FIX #2 + #10: nhãn người nói thân thiện (Người nói 1, hoặc tên attendee)
        spk_friendly = _friendly_speaker(t.speaker, attendees_list)
        spk_class    = _spk_css_class(spk_friendly)
        # FIX #4: KHÔNG hiện [mm:ss] giả lúc live; sẽ hiện sau khi Hoàn thiện
        rows += (f'<div style="display:flex;gap:10px;align-items:baseline;padding:6px 0;'
                 f'border-bottom:1px solid rgba(128,128,128,0.1);">'
                 f'<span class="spk-pill {spk_class}">{_html.escape(spk_friendly)}</span>'
                 f'<span style="font-size:13px;color:var(--text-color);line-height:1.6;">'
                 f'{_html.escape(t.text)}</span></div>')

    partial = st.session_state.l_partial
    if partial:
        rows += (f'<div style="display:flex;gap:10px;align-items:baseline;padding:6px 0;opacity:0.55;">'
                 f'<span class="spk-pill spk-x">···</span>'
                 f'<span style="font-size:13px;color:var(--text-color);opacity:0.7;'
                 f'font-style:italic;">{_html.escape(partial)}</span></div>')

    empty_msg = '<div style="padding:40px;text-align:center;color:#484f58;font-size:13px;">🎙️ Đang lắng nghe...</div>'
    st.markdown(f'<div style="background:var(--secondary-background-color);border:1px solid rgba(128,128,128,0.2);border-radius:8px;padding:14px 16px;">{rows if rows else empty_msg}</div>', unsafe_allow_html=True)
    st.caption("⏺ Phiên âm realtime · người nói sẽ được phân tách bằng Sortformer khi bấm **Hoàn thiện Transcript**.")

elif turns:
    from components.transcript_viewer import full as show_full, preview as show_preview
    if st.session_state.l_show_full:
        show_full(turns, editable=True)
        if st.button("💾 Lưu chỉnh sửa", type="primary"):
            st.session_state.l_turns = turns
            st.success("Đã lưu chỉnh sửa")
    else: show_preview(turns, max_turns=4)
else:
    st.markdown('<div style="background:var(--secondary-background-color);border:1px solid rgba(128,128,128,0.2);border-radius:8px;padding:40px;text-align:center;color:var(--text-color);opacity:0.6;font-size:13px;">Bắt đầu ghi âm để thấy transcript xuất hiện ở đây...</div>', unsafe_allow_html=True)

if turns and st.session_state.l_diar_done:
    st.markdown("<hr style='border-color:rgba(128,128,128,0.2);margin:32px 0 16px;'>", unsafe_allow_html=True)

    # FIX #12: Copy-to-clipboard transcript dạng plain
    full_transcript_text = "\n".join(
        f"[{int(t.start//60):02d}:{int(t.start%60):02d}] {t.speaker}: {t.text}"
        for t in turns
    )
    cp_c1, cp_c2 = st.columns([3, 1])
    with cp_c1:
        st.markdown('<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--text-color);opacity:0.7;margin-bottom:14px;">④ Sao chép & Tóm tắt & Xuất Biên Bản</div>', unsafe_allow_html=True)
    with cp_c2:
        with st.popover("📋 Sao chép Transcript", use_container_width=True):
            st.caption("Bấm vào ô bên dưới, Ctrl+A → Ctrl+C để chép.")
            st.code(full_transcript_text, language=None)

    # FIX #15: Tóm tắt với graceful fallback
    sum_c1, sum_c2 = st.columns([2, 1])
    with sum_c1:
        if st.button("🤖 Tạo Tóm tắt Cuộc họp (NPU Qwen)", use_container_width=True):
            summary_progress = st.progress(0, text="Khởi động NPU tóm tắt...")
            def update_progress(pct, msg): summary_progress.progress(max(0, min(100, pct)), text=f"NPU: {msg}")
            try:
                from core.test_qwen import summarize
                result = summarize(st.session_state.l_turns, progress_callback=update_progress)
                if result.ok:
                    st.session_state.l_summary = result.summary.strip(); st.session_state.l_summary_done = True
                    summary_progress.progress(100, text=f"✅ Đã tóm tắt thành công! ({result.elapsed_sec}s)"); st.success("✅ Đã tóm tắt thành công!")
                else:
                    summary_progress.empty()
                    st.warning(f"⚠️ NPU Qwen lỗi: {result.error}. Bạn vẫn có thể xuất DOCX không tóm tắt, hoặc bấm 'Tóm tắt rule-based' bên cạnh.")
            except Exception as e:
                st.warning(f"⚠️ Không gọi được Qwen ({type(e).__name__}: {e}). Thử 'Tóm tắt rule-based' bên cạnh, hoặc xuất DOCX không tóm tắt.")
    with sum_c2:
        if st.button("📝 Tóm tắt rule-based", use_container_width=True,
                     help="Không cần NPU — chỉ trích các lượt dài nhất + thống kê người nói."):
            try:
                # Rule-based fallback: thống kê + 5 lượt dài nhất
                from collections import Counter as _C
                spk_dur = {}
                for t in st.session_state.l_turns:
                    d = max(0.0, float(t.end) - float(t.start))
                    spk_dur[t.speaker] = spk_dur.get(t.speaker, 0.0) + d
                total = sum(spk_dur.values()) or 1.0
                lines = ["**TỔNG QUAN CUỘC HỌP**", "", f"Tổng thời lượng: {int(total//60)} phút {int(total%60)} giây", ""]
                lines.append("**Thời gian phát biểu:**")
                for spk, d in sorted(spk_dur.items(), key=lambda x: -x[1]):
                    pct = 100.0 * d / total
                    lines.append(f"- {spk}: {int(d//60)}m{int(d%60):02d}s ({pct:.0f}%)")
                lines += ["", "**Một số phát biểu nổi bật:**"]
                top_turns = sorted(st.session_state.l_turns, key=lambda t: -len(t.text.split()))[:5]
                for t in top_turns:
                    lines.append(f"- *{t.speaker}*: {t.text[:200]}{'…' if len(t.text) > 200 else ''}")
                st.session_state.l_summary = "\n".join(lines)
                st.session_state.l_summary_done = True
                st.success("✅ Đã tạo tóm tắt rule-based.")
            except Exception as e:
                st.error(f"❌ Lỗi: {e}")

    st.caption("ℹ️ Tóm tắt là TUỲ CHỌN. Bỏ qua vẫn xuất được DOCX chứa transcript đầy đủ theo từng người nói.")

    if st.session_state.l_summary_done and st.session_state.l_summary:
        st.text_area("Bản xem trước Tóm tắt (sẽ được chèn vào Word):", st.session_state.l_summary, height=200)

    st.markdown("<div style='margin:16px 0 8px;'></div>", unsafe_allow_html=True)

    # Xuất được DOCX chỉ cần CÓ TRANSCRIPT — không bắt buộc phải có tóm tắt.
    is_ready = bool(st.session_state.l_turns)
    export_data = b""
    if is_ready:
        try:
            if st.session_state.l_summary:
                from components.export_docx import export_summary_to_docx
                export_data = export_summary_to_docx(
                    summary_text = st.session_state.l_summary,
                    session_name = st.session_state.l_session_name,
                )
            else:
                from components.export_docx import export_to_docx
                export_data = export_to_docx(
                    turns        = st.session_state.l_turns,
                    session_name = st.session_state.l_session_name,
                )
        except FileNotFoundError as e:
            st.error(str(e))

    _has_sum = bool(st.session_state.l_summary)
    st.download_button(
        label="📄 Xuất biên bản DOCX (tóm tắt)" if _has_sum else "📄 Xuất biên bản DOCX (transcript đầy đủ)",
        data=export_data,
        file_name=f"Bien_ban_{st.session_state.l_session_name.replace(' ', '_')}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        type="primary", disabled=not is_ready, use_container_width=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# 11. STREAMLIT EVENT POLL 200ms
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.l_recording and not st.session_state.l_paused:
    partial, completed = _drain_queue()
    st.session_state.l_partial = partial
    if completed: st.session_state.l_turns.extend(completed)

    # FIX #13b: âm thanh đã được stream ra disk — không lo RAM. Cảnh báo theo thời lượng.
    dur_min = (st.session_state.l_n_chunks * CHUNK_FRAMES / SAMPLE_RATE) / 60.0
    if dur_min > 60 and not st.session_state.l_ram_warned:
        st.session_state.l_ram_warned = True
        st.toast(f"ℹ️ Cuộc họp đã chạy {dur_min:.0f} phút. "
                 f"Bản finalize sẽ tốn ~{int(dur_min * 0.5)}–{int(dur_min)}s xử lý.", icon="ℹ️")

    # ── DEBUG verbose terminal (bật bằng LIVE_DEBUG=1 trong .env) ───────────
    if os.getenv("LIVE_DEBUG", "0") == "1":
        _now = time.time()
        if _now - st.session_state.get("_dbg_last", 0) >= 1.0:
            st.session_state["_dbg_last"] = _now
            dur = (st.session_state.l_n_chunks * CHUNK_FRAMES) / SAMPLE_RATE
            print(f"[LIVE {dur:6.1f}s] turns={len(st.session_state.l_turns)} "
                  f"| partial({len(partial.split())} từ): {partial[-80:]}", flush=True)

    # File-test mode: khi file đọc xong VÀ queue ASR đã drain → tự động dừng
    if (st.session_state.l_file_mode
            and _shared.get("file_done")
            and _AUDIO_QUEUE.empty()):
        _shared["file_streaming"] = False
        _flush_final_asr()
        st.session_state.l_asr_stream = None
        st.session_state.l_recording = False
        st.session_state.l_finished = True
        st.session_state.l_file_mode = False
        _wav_close()                              # FIX #13b: đóng WAV
        _shared["sortformer_stop"] = True         # FIX #5c: dừng inc thread
        st.rerun()

    time.sleep(0.2)
    st.rerun()