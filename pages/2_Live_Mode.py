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

# API Endpoint kết nối với Nexa NPU Server
NPU_API_URL  = os.getenv("NPU_API_URL", "http://127.0.0.1:18182/v1/audio/diarize")

# Backend cho diarization real-time: "npu" (HTTP server) hoặc "diart" (Python library local)
DIARIZATION_BACKEND = os.getenv("DIARIZATION_BACKEND", "npu").strip().lower()

# Ẩn log ồn ào của NPU thread (đặt NPU_DEBUG=1 để bật lại khi cần debug)
_NPU_DEBUG = os.getenv("NPU_DEBUG", "0").strip() == "1"
def _npu_log(msg: str):
    if _NPU_DEBUG:
        print(msg, flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# 2. GLOBAL QUEUES & LOCK (Thread-safe)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_shared_resources():
    return {
        "audio_q": _queue_module.Queue(),
        "diar_q": _queue_module.Queue(),
        "spk_windows": [],
        "spk_lock": threading.Lock()
    }

_shared = get_shared_resources()
_AUDIO_QUEUE = _shared["audio_q"]
_DIAR_QUEUE  = _shared["diar_q"]
_speaker_windows = _shared["spk_windows"]
_speaker_lock = _shared["spk_lock"]

# ══════════════════════════════════════════════════════════════════════════════
# 3. PAGE CONFIG & CSS 
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
            color:var(--text-color);opacity:0.7;padding-bottom:12px;border-bottom:1px solid rgba(128,128,128,0.2);
            margin-bottom:20px;">
    🎙️  Thu âm Trực tiếp — Live Mode (NPU Accelerated)
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
    r = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=TOKENS_PATH, encoder=ENCODER_PATH, decoder=DECODER_PATH, joiner=JOINER_PATH,
        num_threads=2, sample_rate=SAMPLE_RATE, feature_dim=80,
        enable_endpoint_detection=True, rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2, rule3_min_utterance_length=20,
    )
    return r

with st.spinner("Đang nạp mô hình STT..."): recognizer = _get_zipformer()


@st.cache_resource(show_spinner=False)
def _get_whisper_finalizer():
    """Whisper (chính xác) để re-transcribe khi 'Hoàn thiện' — load 1 lần, cache.
    Chỉ dùng khi LIVE_FINAL_STT=whisper."""
    print("⏳ Loading Whisper (final-pass STT)...", flush=True)
    from core.transcriber import load_model
    _, app = load_model()
    return app

# ══════════════════════════════════════════════════════════════════════════════
# 5. SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_defaults = {
    "l_recording": False, "l_finished": False, "l_diar_done": False, "l_renamed": False,
    "l_stream_ref": None, "l_asr_stream": None,
    "l_turns": [], "l_stats": {}, "l_name_map": {}, "l_raw_audio": [], "l_diart_ref": None,
    "l_n_chunks": 0, "l_partial": "", "l_session_name": "", "l_show_full": False,
    "l_summary": "", "l_summary_done": False, "l_prev_window_count": 0, "l_target_spk": 2,
    "l_file_mode": False,
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
    _AUDIO_QUEUE.put(chunk.copy())   # ASR thread sẽ append vào l_raw_audio

    # Diarization fan-out theo backend live:
    #   diart → feed streamer | npu → queue HTTP | none/off → KHÔNG diarize live
    #   (none: speaker gán hết ở final-pass Sortformer khi bấm Hoàn thiện)
    if DIARIZATION_BACKEND == "diart":
        diart_ref = _shared.get("diart_ref")
        if diart_ref is not None:
            diart_ref.feed(chunk.copy())
    elif DIARIZATION_BACKEND == "npu":
        _DIAR_QUEUE.put(chunk.copy())
    # else (none/off): bỏ qua — không diarize trong lúc thu


def _stream_file_thread(wav_path: str, sample_rate: int = 16000, realtime: bool = True):
    """Bơm 1 file WAV vào pipeline GIỐNG HỆT mic — để test trước khi thu thật.

    Chạy trong thread nền → KHÔNG truy cập st.session_state. Đẩy chunk vào cùng
    _AUDIO_QUEUE (ASR) + diart/_DIAR_QUEUE (diarization) như _audio_callback.
    Tự downmix stereo → mono. realtime=True phát ở tốc độ 1× như mic thật.
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
        if DIARIZATION_BACKEND == "diart":
            ref = _shared.get("diart_ref")
            if ref is not None:
                ref.feed(chunk.copy())
        else:
            _DIAR_QUEUE.put(chunk.copy())
        if realtime:
            time.sleep(chunk_dt)

    wf.close()
    _shared["file_done"] = True
    print("[FileStream] ✅ Đã đọc hết file", flush=True)


def _run_diart_thread(sample_rate: int = 16000, num_speakers: int = 2):
    """Làn 2: Gửi tối đa 60s âm thanh gần nhất lên NPU Server để ID đồng nhất và không bị tràn RAM"""
    import io
    import wave
    import base64
    import requests
    import numpy as np

    STEP_SEC       = 5.0
    step_frames    = int(sample_rate * STEP_SEC)
    
    # GIỚI HẠN NGỮ CẢNH: Chỉ gửi tối đa 60 giây gần nhất để chống lỗi 400 Bad Request
    MAX_CONTEXT_SEC = 60.0 
    max_context_frames = int(sample_rate * MAX_CONTEXT_SEC)
    
    full_audio = []
    accumulated_frames = 0
    last_processed_frames = 0

    _npu_log("[NPU Thread] Bắt đầu hoạt động")
    while True:
        try:
            chunk = _DIAR_QUEUE.get(timeout=2.0)
        except _queue_module.Empty:
            continue
            
        if chunk is None:
            _npu_log("[NPU Thread] Nhận tín hiệu kết thúc")
            break
            
        full_audio.append(chunk)
        accumulated_frames += len(chunk)
        
        if (accumulated_frames - last_processed_frames) >= step_frames:
            # 1. Ghép toàn bộ âm thanh hiện có
            audio_i16 = np.concatenate(full_audio)
            offset_sec = 0.0
            
            # 2. CẮT TỈA: Nếu âm thanh dài hơn 60s, chỉ lấy 60s cuối cùng
            if len(audio_i16) > max_context_frames:
                # Lưu lại mốc thời gian bị cắt đi để bù trừ (offset) sau này
                offset_sec = (len(audio_i16) - max_context_frames) / sample_rate
                audio_i16 = audio_i16[-max_context_frames:]
            
            try:
                wav_io = io.BytesIO()
                with wave.open(wav_io, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_i16.tobytes())
                
                b64_audio = base64.b64encode(wav_io.getvalue()).decode('utf-8')
                data_url = f"data:audio/wav;base64,{b64_audio}"
                
                payload = {
                    "file": data_url, 
                    "audio": data_url,
                    "model": "NexaAI/Pyannote-NPU",
                    "num_speakers": num_speakers
                }
                
                # TĂNG TIMEOUT LÊN 45 GIÂY để Python không vội vàng ngắt kết nối
                response = requests.post(NPU_API_URL, json=payload, timeout=45.0)
                
                if response.status_code == 200:
                    results = response.json()
                    segments = results.get("Segments", [])
                    
                    if segments:
                        new_windows = []
                        for seg in segments:
                            # 3. BÙ TRỪ THỜI GIAN: Cộng thêm khoảng thời gian đã cắt (offset) 
                            # để tọa độ map chính xác với STT của hệ thống
                            abs_start = round(offset_sec + seg['StartTime'], 2)
                            abs_end   = round(offset_sec + seg['EndTime'], 2)
                            label     = seg['SpeakerLabel']
                            new_windows.append((abs_start, abs_end, label))
                            
                        with _speaker_lock:
                            # Không clear toàn bộ nữa, chỉ cập nhật/ghi đè các đoạn trong khoảng 60s gần nhất
                            # Lọc bỏ các windows cũ nằm trong vùng thời gian đang xét
                            current_windows = [w for w in _speaker_windows if w[1] <= offset_sec]
                            current_windows.extend(new_windows)
                            
                            _speaker_windows.clear()
                            _speaker_windows.extend(current_windows)
                            
                else:
                    _npu_log(f"[NPU Thread] Lỗi API ({response.status_code}): {response.text[:100]}")
                    
                last_processed_frames = accumulated_frames
                
            except requests.exceptions.Timeout:
                _npu_log("[NPU Thread] NPU đang quá tải, phản hồi chậm hơn 45 giây...")
            except Exception as e:
                _npu_log(f"[NPU Thread] Lỗi xử lý/Kết nối: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. LOGIC GÁN NHÃN NGƯỢC (Retro-Update)
# ══════════════════════════════════════════════════════════════════════════════
def _assign_speaker(t_start: float, t_end: float) -> str:
    """Gán speaker cho 1 câu nói. 3 tầng (mạnh → yếu):
       1) overlap lớn nhất với speaker window
       2) speaker tại ĐIỂM GIỮA câu (khi timestamp ước lượng lệch)
       3) speaker của window GẦN NHẤT theo thời gian (fill_nearest)
    Tránh trả '...' khi diart đã có dữ liệu → live labels đỡ bị kẹt 1 speaker."""
    with _speaker_lock:
        windows = list(_speaker_windows)
    if not windows:
        return "..."

    # 1) max overlap
    best_speaker, best_overlap = None, 0.0
    for (ws, we, spk) in windows:
        overlap = max(0.0, min(t_end, we) - max(t_start, ws))
        if overlap > best_overlap:
            best_overlap, best_speaker = overlap, spk
    if best_speaker is not None:
        return best_speaker

    # 2) speaker tại điểm giữa câu
    mid = (t_start + t_end) / 2.0
    for (ws, we, spk) in windows:
        if ws <= mid <= we:
            return spk

    # 3) window gần nhất theo khoảng cách thời gian
    best_speaker, best_dist = None, float("inf")
    for (ws, we, spk) in windows:
        dist = max(0.0, max(ws - t_end, t_start - we))
        if dist < best_dist:
            best_dist, best_speaker = dist, spk
    return best_speaker or "..."

def _retro_update_speakers():
    """Vá tên người nói vào các đoạn Transcript cũ"""
    turns = st.session_state.get("l_turns", [])
    updated = 0
    for t in turns:
        spk = _assign_speaker(t.start, t.end)
        # Nếu đã tìm ra nhãn thật và khác với nhãn hiện tại, tiến hành đổi tên
        if spk != "..." and spk != t.speaker:
            t.speaker = spk
            updated += 1
    return updated

# ══════════════════════════════════════════════════════════════════════════════
# 8. XỬ LÝ LÀN 1 (STT STREAMING)
# ══════════════════════════════════════════════════════════════════════════════
def _drain_queue() -> tuple[str, list]:
    from core.aligner import AlignedTurn

    if st.session_state.l_asr_stream is None:
        st.session_state.l_asr_stream = recognizer.create_stream()
    z_stream  = st.session_state.l_asr_stream
    completed = []

    while not _AUDIO_QUEUE.empty():
        try: chunk = _AUDIO_QUEUE.get_nowait()
        except _queue_module.Empty: break

        st.session_state.l_raw_audio.append(chunk)
        st.session_state.l_n_chunks += 1

        samples = chunk.astype(np.float32) / 32768.0
        z_stream.accept_waveform(SAMPLE_RATE, samples)

        while recognizer.is_ready(z_stream):
            recognizer.decode_stream(z_stream)

        if recognizer.is_endpoint(z_stream):
            text = recognizer.get_result(z_stream).strip()
            if text:
                t_end = round((st.session_state.l_n_chunks * CHUNK_FRAMES) / SAMPLE_RATE, 2)
                # Ước lượng độ dài câu bằng số từ (2.5 từ/giây)
                est_sec = max(0.5, len(text.split()) / 2.5)
                t_start = round(max(0.0, t_end - est_sec), 2)
                
                # Cố gắng gán tên ngay lúc này (có thể là "..." nếu NPU chưa phản hồi kịp)
                speaker = _assign_speaker(t_start, t_end)

                completed.append(AlignedTurn(
                    speaker = speaker,
                    start   = t_start,
                    end     = t_end,
                    text    = text,
                ))
                print(f"[ASR turn] [{t_start:6.1f}-{t_end:6.1f}] {speaker} "
                      f"({len(text.split())} từ): {text}", flush=True)
            recognizer.reset(z_stream)

    # Nếu NPU ngầm vừa phát hiện thêm cửa sổ thời gian mới, tự động vá (Retro-update)
    n_windows = len(_speaker_windows)
    if n_windows > st.session_state.l_prev_window_count:
        _retro_update_speakers()
        st.session_state.l_prev_window_count = n_windows

    partial = recognizer.get_result(z_stream).strip()
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
        st.session_state.l_raw_audio.append(chunk)
        st.session_state.l_n_chunks += 1
        z.accept_waveform(SAMPLE_RATE, chunk.astype(np.float32) / 32768.0)

    # 2) Báo hết stream → decode nốt → lấy text cuối
    try:
        z.input_finished()
    except Exception:
        pass
    while recognizer.is_ready(z):
        recognizer.decode_stream(z)

    text = recognizer.get_result(z).strip()
    if text:
        t_end   = round((st.session_state.l_n_chunks * CHUNK_FRAMES) / SAMPLE_RATE, 2)
        est_sec = max(0.5, len(text.split()) / 2.5)
        t_start = round(max(0.0, t_end - est_sec), 2)
        st.session_state.l_turns.append(AlignedTurn(
            speaker = _assign_speaker(t_start, t_end),
            start   = t_start,
            end     = t_end,
            text    = text,
        ))
        print(f"[Flush] Commit đoạn cuối ({len(text.split())} từ): {text[:60]}…", flush=True)

    st.session_state.l_partial = ""


# ══════════════════════════════════════════════════════════════════════════════
# 9. UI CONTROL PANEL
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--text-color);opacity:0.7;margin-bottom:16px;">① Điều khiển phiên ghi âm</div>', unsafe_allow_html=True)

c1, c2 = st.columns([2, 1], gap="medium")
with c1:
    session_name = st.text_input("Mã phiên / Tên hồ sơ", value=st.session_state.l_session_name or f"LIVE-{time.strftime('%Y%m%d-%H%M')}", disabled=st.session_state.l_recording)
    if session_name != st.session_state.l_session_name: st.session_state.l_session_name = session_name
with c2:
    num_spk = st.number_input("Số người nói dự kiến", min_value=1, max_value=8, value=st.session_state.l_target_spk, disabled=st.session_state.l_recording)
    if num_spk != st.session_state.l_target_spk: st.session_state.l_target_spk = num_spk

st.markdown("<div style='margin:12px 0 8px;'></div>", unsafe_allow_html=True)
btn_col1, btn_col2, status_col = st.columns([1, 1, 2], gap="small")

with btn_col1:
    if not st.session_state.l_recording:
        if st.button("🔴 Bắt đầu Ghi âm", type="primary", use_container_width=True):
            st.session_state.l_turns = []; st.session_state.l_stats = {}; st.session_state.l_name_map = {}
            st.session_state.l_raw_audio = []; st.session_state.l_n_chunks = 0; st.session_state.l_partial = ""
            st.session_state.l_finished = False; st.session_state.l_diar_done = False; st.session_state.l_renamed = False
            st.session_state.l_summary = ""; st.session_state.l_summary_done = False
            st.session_state.l_asr_stream = None; st.session_state.l_prev_window_count = 0
            st.session_state.l_file_mode = False   # đây là phiên thu mic, không phải file-test

            with _AUDIO_QUEUE.mutex: _AUDIO_QUEUE.queue.clear()
            with _DIAR_QUEUE.mutex:  _DIAR_QUEUE.queue.clear()
            with _speaker_lock:      _speaker_windows.clear()

            # ── Dispatch backend ────────────────────────────────────────
            if DIARIZATION_BACKEND == "diart":
                # diart streaming: chạy local, không cần NPU server
                from core.diarization.streaming import DiartStreamingDiarizer

                def _sync_to_speaker_windows(new_windows):
                    """Diart callback → sync vào _speaker_windows để UI/retro-update đọc."""
                    with _speaker_lock:
                        _speaker_windows.clear()
                        _speaker_windows.extend(new_windows)

                diart_streamer = DiartStreamingDiarizer(
                    sample_rate  = SAMPLE_RATE,
                    num_speakers = st.session_state.l_target_spk,
                    on_update    = _sync_to_speaker_windows,
                )
                diart_streamer.start()
                _shared["diart_ref"] = diart_streamer      # cho audio callback (thread-safe)
                st.session_state.l_diart_ref = diart_streamer  # cho main thread (stop)
                print(f"[LiveMode] ✅ Backend = diart (local streaming)", flush=True)
            elif DIARIZATION_BACKEND == "npu":
                # NPU HTTP server (Nexa)
                diart_t = threading.Thread(
                    target=_run_diart_thread,
                    args=(SAMPLE_RATE, st.session_state.l_target_spk),
                    daemon=True, name="npu-worker",
                )
                add_script_run_ctx(diart_t)
                diart_t.start()
                print(f"[LiveMode] ✅ Backend = npu (HTTP {NPU_API_URL})", flush=True)
            else:
                # none/off: KHÔNG diarize lúc thu. Speaker gán hết ở final-pass
                # (Sortformer) khi bấm Hoàn thiện → chỉ 1 diarizer, không lẫn nhãn.
                print(f"[LiveMode] ✅ Backend = none (live không diarize; "
                      f"final-pass = {os.getenv('LIVE_FINAL_BACKEND','recluster')})", flush=True)

            stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=CHUNK_FRAMES, callback=_audio_callback)
            stream.start()
            st.session_state.l_stream_ref = stream
            st.session_state.l_recording = True
            st.rerun()
    else:
        if st.button("⏹️ Dừng Ghi âm", use_container_width=True):
            _shared["file_streaming"] = False   # dừng file-test thread nếu đang chạy
            if st.session_state.l_stream_ref:
                st.session_state.l_stream_ref.stop()
                st.session_state.l_stream_ref.close()
                st.session_state.l_stream_ref = None

            _flush_final_asr()   # ép commit đoạn text đang dở (partial) → không mất

            # Dừng backend diarization tương ứng
            if DIARIZATION_BACKEND == "diart" and st.session_state.get("l_diart_ref"):
                # stop() đợi diart drain + xử lý nốt; trả windows tích luỹ cuối
                final_windows = st.session_state.l_diart_ref.stop(timeout=60.0)
                with _speaker_lock:
                    _speaker_windows.clear()
                    _speaker_windows.extend(final_windows)
                st.session_state.l_diart_ref = None
                _shared["diart_ref"] = None
            else:
                _DIAR_QUEUE.put(None)   # sentinel cho NPU worker thread

            st.session_state.l_asr_stream = None
            st.session_state.l_recording = False
            st.session_state.l_finished = True
            st.rerun()

with btn_col2:
    # Nút Hoàn thiện: Tái sử dụng kết quả siêu chuẩn của NPU để làm sạch văn bản
    btn_finalize_disabled = not st.session_state.l_finished or st.session_state.l_diar_done
    if st.button("✅ Hoàn thiện Transcript", disabled=btn_finalize_disabled, use_container_width=True):
        with st.spinner("Đang ép khớp thời gian (Align) và khôi phục dấu câu..."):
            _retro_update_speakers() 
            
            import wave
            from core.diarizer import SpeakerSegment, postprocess_segments, get_speaker_stats
            from core.aligner import align, merge_consecutive
            from core.punctuation_restorer import restore

            audio_np = np.concatenate(st.session_state.l_raw_audio).astype(np.int16)
            tmp_wav = tempfile.NamedTemporaryFile(suffix="_live.wav", delete=False)
            with wave.open(tmp_wav.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_np.tobytes())

            # ── Final pass cho diart: re-cluster toàn cục để sửa labels flickery ─
            # Live diart chỉ thấy past+window → cluster online có thể nhầm.
            # Final pass dùng pyannote/embedding + agglomerative clustering trên
            # TOÀN BỘ recording để gán speaker IDs nhất quán.
            from collections import Counter as _Counter
            def _spk_dist(segs):
                return dict(_Counter(getattr(s, "speaker", s[2] if isinstance(s, tuple) else "?") for s in segs))

            # LIVE_FINAL_BACKEND=sortformer → bỏ qua live windows, diarize lại TOÀN BỘ
            # recording bằng Sortformer (SOTA, như WhisperLiveKit). Chất lượng cao nhất.
            _FINAL_BACKEND = os.getenv("LIVE_FINAL_BACKEND", "recluster").strip().lower()

            if _FINAL_BACKEND == "sortformer":
                try:
                    # BRIDGE: gọi sortformer_env qua subprocess (không import NeMo ở app env)
                    from core.diarization.sortformer_bridge import diarize_file_sortformer
                    segments = diarize_file_sortformer(
                        wav_path     = tmp_wav.name,
                        num_speakers = st.session_state.l_target_spk,
                    )
                    print(f"[Finalize] Sortformer final-pass: {len(segments)} segs, "
                          f"speakers={_spk_dist(segments)}", flush=True)
                except Exception as e:
                    print(f"[LiveMode] Sortformer final-pass lỗi → fallback live labels: {e}", flush=True)
                    segments = [SpeakerSegment(speaker=spk, start=ws, end=we)
                                for ws, we, spk in _speaker_windows]
            elif DIARIZATION_BACKEND == "diart":
                from core.pipeline.postprocess import final_recluster
                try:
                    print(f"[Finalize] _speaker_windows: {len(_speaker_windows)} windows, "
                          f"speakers={_spk_dist(list(_speaker_windows))}", flush=True)
                    segments = final_recluster(
                        wav_path     = tmp_wav.name,
                        windows      = list(_speaker_windows),
                        num_speakers = st.session_state.l_target_spk,
                    )
                    # KHÔNG chạy postprocess_segments ở đây: final_recluster đã cluster
                    # sạch rồi. postprocess_segments (smoothing A→B→A + ghost-drop) được
                    # chỉnh cho output pyannote thô → nó XOÁ speaker thiểu số (host-dominant
                    # → mất guest). Chỉ merge nhẹ đoạn liền kề cùng speaker là đủ.
                    print(f"[Finalize] Sau recluster: {len(segments)} segs, "
                          f"speakers={_spk_dist(segments)}", flush=True)
                except Exception as e:
                    print(f"[LiveMode] Final re-cluster lỗi, fallback live labels: {e}", flush=True)
                    segments = [SpeakerSegment(speaker=spk, start=ws, end=we)
                                for ws, we, spk in _speaker_windows]
            else:
                # Lọc nhiễu cửa sổ NPU (path cũ)
                raw_segs = [SpeakerSegment(speaker=spk, start=ws, end=we) for ws, we, spk in _speaker_windows]
                segments = postprocess_segments(raw_segs, min_duration=0.5, merge_gap=2.5)

            # ── Văn bản cho transcript cuối ──────────────────────────────────
            # LIVE_FINAL_STT=whisper → re-transcribe recording bằng Whisper (chính
            # xác hơn Zipformer live nhiều: câu đầy đủ + dấu câu). Mặc định whisper.
            _FINAL_STT = os.getenv("LIVE_FINAL_STT", "whisper").strip().lower()
            if _FINAL_STT == "whisper":
                try:
                    from core.transcriber import transcribe_file
                    app = _get_whisper_finalizer()
                    wres = transcribe_file(app, tmp_wav.name)
                    full_text = (wres.get("text") or "").strip()
                    print(f"[Finalize] Whisper re-transcribe: {len(full_text.split())} từ", flush=True)
                    if not full_text:  # Whisper rỗng → fallback text live
                        raise ValueError("Whisper trả text rỗng")
                except Exception as e:
                    print(f"[Finalize] Whisper lỗi → dùng text Zipformer live: {e}", flush=True)
                    raw = " ".join(t.text for t in st.session_state.l_turns if t.text.strip())
                    full_text = restore(raw.lower())
            else:
                raw = " ".join(t.text for t in st.session_state.l_turns if t.text.strip())
                full_text = restore(raw.lower())

            # DEBUG verbose (bật bằng FINALIZE_DEBUG=1): in text nguồn để so sánh
            if os.getenv("FINALIZE_DEBUG", "0") == "1":
                print(f"[Finalize] === FULL_TEXT ({_FINAL_STT}) ===\n{full_text[:600]}\n=== /FULL_TEXT ===",
                      flush=True)

            # Chạy Aligner để ép thời gian và nối câu
            aligned = align(
                segments=segments,
                full_text=full_text,
                wav_path=tmp_wav.name,
                language="vi",
                use_forced_align=True,
                gap_limit=1.5
            )
            merged = merge_consecutive(aligned, gap_limit=1.5)
            # Gộp turn ngắn bị kẹp (A→b→A) → bớt phân mảnh khi nói nhanh/chồng tiếng
            from core.aligner import smooth_short_turns
            merged = smooth_short_turns(merged, max_words=4, max_dur=1.2, gap_limit=1.5)
            print(f"[Finalize] Sau align+merge+smooth: {len(merged)} turns, "
                  f"speakers={_spk_dist(merged)}", flush=True)
            if os.getenv("FINALIZE_DEBUG", "0") == "1":
                for _i, _t in enumerate(merged[:6]):
                    print(f"   [turn {_i}] {_t.speaker}: {_t.text[:90]}", flush=True)

            st.session_state.l_turns = merged
            st.session_state.l_stats = get_speaker_stats(segments)
            st.session_state.l_diar_done = True
            
            try: os.unlink(tmp_wav.name)
            except: pass
            
        st.rerun()

with status_col:
    if st.session_state.l_recording:
        dur_sec = (st.session_state.l_n_chunks * CHUNK_FRAMES) / SAMPLE_RATE
        st.markdown(f'<div style="background:rgba(232,82,10,0.1);border:1px solid #e8520a;border-radius:4px;padding:8px 14px;display:inline-flex;align-items:center;gap:8px;margin-top:4px;"><span class="rec-dot"></span><span style="font-family:monospace;font-size:11px;font-weight:600;color:#e8520a;">REC &nbsp;{int(dur_sec//60):02d}:{int(dur_sec%60):02d} &nbsp;·&nbsp; {len(st.session_state.l_turns)} câu</span></div>', unsafe_allow_html=True)
    elif st.session_state.l_finished and not st.session_state.l_diar_done:
        st.markdown(f'<div style="font-family:monospace;font-size:11px;color:var(--text-color);opacity:0.8;padding:8px 14px;background:var(--secondary-background-color);border:1px solid rgba(128,128,128,0.3);border-radius:4px;display:inline-block;margin-top:4px;">⏸ Đã dừng &nbsp;·&nbsp; Bấm Hoàn thiện Transcript</div>', unsafe_allow_html=True)
    elif st.session_state.l_diar_done and not st.session_state.l_summary_done:
        st.markdown(f'<div style="font-family:monospace;font-size:11px;color:#4da6e8;padding:8px 14px;background:rgba(77,166,232,0.1);border:1px solid #4da6e8;border-radius:4px;display:inline-block;margin-top:4px;">✓ Đã hoàn thiện &nbsp;·&nbsp; {len(st.session_state.l_stats)} người nói &nbsp;·&nbsp; Bấm Tạo Tóm tắt bên dưới</div>', unsafe_allow_html=True)

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
            st.session_state.l_asr_stream = None; st.session_state.l_prev_window_count = 0

            with _AUDIO_QUEUE.mutex: _AUDIO_QUEUE.queue.clear()
            with _DIAR_QUEUE.mutex:  _DIAR_QUEUE.queue.clear()
            with _speaker_lock:      _speaker_windows.clear()

            # Khởi động backend diarization (giống mic)
            if DIARIZATION_BACKEND == "diart":
                from core.diarization.streaming import DiartStreamingDiarizer
                def _sync_to_speaker_windows(new_windows):
                    with _speaker_lock:
                        _speaker_windows.clear(); _speaker_windows.extend(new_windows)
                ds = DiartStreamingDiarizer(
                    sample_rate=SAMPLE_RATE,
                    num_speakers=st.session_state.l_target_spk,
                    on_update=_sync_to_speaker_windows,
                )
                ds.start()
                _shared["diart_ref"] = ds
                st.session_state.l_diart_ref = ds
            else:
                t = threading.Thread(target=_run_diart_thread,
                                     args=(SAMPLE_RATE, st.session_state.l_target_spk),
                                     daemon=True, name="npu-worker")
                add_script_run_ctx(t); t.start()

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
with col_title: st.markdown('<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--text-color);opacity:0.7;margin-bottom:14px;">③ Nội dung phiên hỏi cung</div>', unsafe_allow_html=True)
with col_toggle:
    if st.session_state.l_turns:
        if st.button("📄 Xem toàn bộ" if not st.session_state.l_show_full else "🔼 Thu gọn", use_container_width=True):
            st.session_state.l_show_full = not st.session_state.l_show_full
            st.rerun()

turns = st.session_state.l_turns
if st.session_state.l_recording:
    # ── Khi THU: chỉ hiện [mốc thời gian] + văn bản realtime, KHÔNG gán người nói.
    # Phân tách người nói làm ở bước Hoàn thiện (Sortformer) — chính xác hơn,
    # không giới hạn cứng số người, không nhãn sai nhấp nháy lúc live.
    recent = turns[-8:] if len(turns) > 8 else turns
    rows = ""
    for t in recent:
        ts = f"{int(t.start//60):02d}:{int(t.start%60):02d}"
        rows += (f'<div style="display:flex;gap:12px;align-items:baseline;padding:6px 0;'
                 f'border-bottom:1px solid rgba(128,128,128,0.1);">'
                 f'<span style="font-family:monospace;font-size:10px;color:#e8520a;'
                 f'white-space:nowrap;min-width:44px;">[{ts}]</span>'
                 f'<span style="font-size:13px;color:var(--text-color);line-height:1.6;">'
                 f'{_html.escape(t.text)}</span></div>')

    partial = st.session_state.l_partial
    if partial:
        rows += (f'<div style="display:flex;gap:12px;align-items:baseline;padding:6px 0;opacity:0.55;">'
                 f'<span style="font-family:monospace;font-size:10px;color:#e8520a;'
                 f'white-space:nowrap;min-width:44px;">[···]</span>'
                 f'<span style="font-size:13px;color:var(--text-color);opacity:0.7;'
                 f'font-style:italic;">{_html.escape(partial)}</span></div>')

    empty_msg = '<div style="padding:40px;text-align:center;color:#484f58;font-size:13px;">🎙️ Đang lắng nghe...</div>'
    st.markdown(f'<div style="background:var(--secondary-background-color);border:1px solid rgba(128,128,128,0.2);border-radius:8px;padding:14px 16px;">{rows if rows else empty_msg}</div>', unsafe_allow_html=True)
    st.caption("⏺ Phiên âm realtime (chưa phân người nói). Người nói sẽ được phân tách khi bấm **Hoàn thiện**.")

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
    st.markdown('<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--text-color);opacity:0.7;margin-bottom:14px;">⑤ Tóm tắt & Xuất Biên Bản</div>', unsafe_allow_html=True)

    if st.button("🤖 Chạy Tóm tắt Nội dung (NPU Qwen)", use_container_width=True):
        summary_progress = st.progress(0, text="Khởi động NPU tóm tắt...")
        def update_progress(pct, msg): summary_progress.progress(max(0, min(100, pct)), text=f"NPU: {msg}")
        try:
            from core.test_qwen import summarize
            result = summarize(st.session_state.l_turns, progress_callback=update_progress)
            if result.ok:
                st.session_state.l_summary = result.summary.strip(); st.session_state.l_summary_done = True
                summary_progress.progress(100, text=f"✅ Đã tóm tắt thành công! ({result.elapsed_sec}s)"); st.success("✅ Đã tóm tắt thành công!")
            else:
                summary_progress.empty(); st.error(f"❌ Lỗi khi tóm tắt: {result.error}")
        except Exception as e: st.error(f"❌ Lỗi hệ thống: {e}")

    st.caption("ℹ️ Tóm tắt là TUỲ CHỌN (cần Qwen/NPU). Bỏ qua vẫn xuất được DOCX "
               "chứa transcript đầy đủ theo từng người nói.")

    if st.session_state.l_summary_done and st.session_state.l_summary:
        st.text_area("Bản xem trước Tóm tắt (Sẽ được chèn vào Word):", st.session_state.l_summary, height=200)

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
if st.session_state.l_recording:
    partial, completed = _drain_queue()
    st.session_state.l_partial = partial
    if completed: st.session_state.l_turns.extend(completed)

    # ── DEBUG verbose terminal (bật bằng LIVE_DEBUG=1 trong .env) ───────────
    if os.getenv("LIVE_DEBUG", "0") == "1":
        _now = time.time()
        if _now - st.session_state.get("_dbg_last", 0) >= 1.0:
            st.session_state["_dbg_last"] = _now
            dur = (st.session_state.l_n_chunks * CHUNK_FRAMES) / SAMPLE_RATE
            with _speaker_lock:
                n_win = len(_speaker_windows)
                spk_set = sorted({w[2] for w in _speaker_windows})
            print(f"[LIVE {dur:6.1f}s] turns={len(st.session_state.l_turns)} "
                  f"| diart_windows={n_win} speakers={spk_set} "
                  f"| partial({len(partial.split())} từ): {partial[-80:]}", flush=True)

    # File-test mode: khi file đọc xong VÀ queue ASR đã drain → tự động dừng
    # (giống bấm nút "Dừng Ghi âm" cho mic).
    if (st.session_state.l_file_mode
            and _shared.get("file_done")
            and _AUDIO_QUEUE.empty()):
        _shared["file_streaming"] = False
        _flush_final_asr()   # commit đoạn text cuối trước khi finalize
        if DIARIZATION_BACKEND == "diart" and st.session_state.get("l_diart_ref"):
            final_windows = st.session_state.l_diart_ref.stop(timeout=60.0)
            with _speaker_lock:
                _speaker_windows.clear()
                _speaker_windows.extend(final_windows)
            st.session_state.l_diart_ref = None
            _shared["diart_ref"] = None
        else:
            _DIAR_QUEUE.put(None)
        st.session_state.l_asr_stream = None
        st.session_state.l_recording = False
        st.session_state.l_finished = True
        st.session_state.l_file_mode = False
        st.rerun()

    time.sleep(0.2)
    st.rerun()