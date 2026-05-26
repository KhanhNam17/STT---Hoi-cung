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

# ══════════════════════════════════════════════════════════════════════════════
# 5. SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_defaults = {
    "l_recording": False, "l_finished": False, "l_diar_done": False, "l_renamed": False,
    "l_stream_ref": None, "l_asr_stream": None,
    "l_turns": [], "l_stats": {}, "l_name_map": {}, "l_raw_audio": [],
    "l_n_chunks": 0, "l_partial": "", "l_session_name": "", "l_show_full": False,
    "l_summary": "", "l_summary_done": False, "l_prev_window_count": 0, "l_target_spk": 2
}
for k, v in _defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# 6. CALLBACK & THREAD WORKER (Cốt lõi xử lý NPU API)
# ══════════════════════════════════════════════════════════════════════════════
def _audio_callback(indata, frames, time_info, status):
    """Làn 0: Thu âm từ Micro. Fan-out ra 2 Queue cho STT và Diarization"""
    chunk = (indata[:, 0] * 32767).astype(np.int16)
    _AUDIO_QUEUE.put(chunk.copy())
    _DIAR_QUEUE.put(chunk.copy()) 

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

    print("[NPU Thread] Bắt đầu hoạt động")
    while True:
        try:
            chunk = _DIAR_QUEUE.get(timeout=2.0)
        except _queue_module.Empty:
            continue
            
        if chunk is None:
            print("[NPU Thread] Nhận tín hiệu kết thúc")
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
                    print(f"[NPU Thread] Lỗi API ({response.status_code}): {response.text[:100]}")
                    
                last_processed_frames = accumulated_frames
                
            except requests.exceptions.Timeout:
                print("[NPU Thread] NPU đang quá tải, phản hồi chậm hơn 45 giây...")
            except Exception as e:
                print(f"[NPU Thread] Lỗi xử lý/Kết nối: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. LOGIC GÁN NHÃN NGƯỢC (Retro-Update)
# ══════════════════════════════════════════════════════════════════════════════
def _assign_speaker(t_start: float, t_end: float) -> str:
    """Tìm Speaker có thời gian đè lên (overlap) nhiều nhất với câu nói hiện tại"""
    best_speaker = "..."
    best_overlap = 0.0
    with _speaker_lock:
        for (ws, we, spk) in _speaker_windows:
            overlap = max(0.0, min(t_end, we) - max(t_start, ws))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = spk
    return best_speaker

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
            recognizer.reset(z_stream)

    # Nếu NPU ngầm vừa phát hiện thêm cửa sổ thời gian mới, tự động vá (Retro-update)
    n_windows = len(_speaker_windows)
    if n_windows > st.session_state.l_prev_window_count:
        _retro_update_speakers()
        st.session_state.l_prev_window_count = n_windows

    partial = recognizer.get_result(z_stream).strip()
    return partial, completed

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

            with _AUDIO_QUEUE.mutex: _AUDIO_QUEUE.queue.clear()
            with _DIAR_QUEUE.mutex:  _DIAR_QUEUE.queue.clear()
            with _speaker_lock:      _speaker_windows.clear()

            # Kích hoạt Luồng ngầm NPU siêu nhẹ
            diart_t = threading.Thread(
                target=_run_diart_thread,
                args=(SAMPLE_RATE,st.session_state.l_target_spk),
                daemon=True, name="npu-worker"
            )
            add_script_run_ctx(diart_t)
            diart_t.start()

            stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=CHUNK_FRAMES, callback=_audio_callback)
            stream.start()
            st.session_state.l_stream_ref = stream
            st.session_state.l_recording = True
            st.rerun()
    else:
        if st.button("⏹️ Dừng Ghi âm", use_container_width=True):
            if st.session_state.l_stream_ref:
                st.session_state.l_stream_ref.stop()
                st.session_state.l_stream_ref.close()
                st.session_state.l_stream_ref = None

            _DIAR_QUEUE.put(None) # Phóng tín hiệu kết liễu luồng ngầm
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

            # Lọc nhiễu cửa sổ NPU
            raw_segs = [SpeakerSegment(speaker=spk, start=ws, end=we) for ws, we, spk in _speaker_windows]
            segments = postprocess_segments(raw_segs, min_duration=0.5, merge_gap=2.5)

            # Gom text và phục hồi dấu phẩy
            raw_full_text = " ".join(t.text for t in st.session_state.l_turns if t.text.strip())
            full_text = restore(raw_full_text.lower())

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
    SPEAKER_COLORS = ["#e8520a", "#4da6e8", "#4ec94e", "#c97fe8", "#e8c14d"]
    speakers = list(dict.fromkeys(t.speaker for t in turns if t.speaker != "..."))
    cmap = {s: SPEAKER_COLORS[i % len(SPEAKER_COLORS)] for i, s in enumerate(speakers)}
    cmap["..."] = "#888" # Màu mặc định cho nhãn chưa xác định
    
    recent = turns[-6:] if len(turns) > 6 else turns
    rows = ""
    for t in recent:
        color = cmap.get(t.speaker, "#888")
        rows += f'<div style="display:flex;gap:10px;align-items:baseline;padding:6px 0;border-bottom:1px solid rgba(128,128,128,0.1);"><span style="font-family:monospace;font-size:10px;color:{color};white-space:nowrap;min-width:48px;">[{int(t.start//60):02d}:{int(t.start%60):02d}]</span><span style="font-size:11px;font-weight:700;color:{color};min-width:80px;white-space:nowrap;">{_html.escape(t.speaker)}:</span><span style="font-size:13px;color:var(--text-color);line-height:1.6;">{_html.escape(t.text[:180])}</span></div>'
    
    partial = st.session_state.l_partial
    if partial:
        rows += f'<div style="display:flex;gap:10px;align-items:baseline;padding:6px 0;opacity:0.55;"><span style="font-family:monospace;font-size:10px;color:#e8520a;white-space:nowrap;min-width:48px;">[···]</span><span style="font-size:11px;font-weight:700;color:#e8520a;min-width:80px;white-space:nowrap;">🎙️:</span><span style="font-size:13px;color:var(--text-color);opacity:0.7;font-style:italic;">{_html.escape(partial)}</span></div>'
    
    empty_msg = '<div style="padding:40px;text-align:center;color:#484f58;font-size:13px;">🎙️ Đang lắng nghe...</div>'
    st.markdown(f'<div style="background:var(--secondary-background-color);border:1px solid rgba(128,128,128,0.2);border-radius:8px;padding:14px 16px;">{rows if rows else empty_msg}</div>', unsafe_allow_html=True)

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

    if st.session_state.l_summary_done and st.session_state.l_summary:
        st.text_area("Bản xem trước Tóm tắt (Sẽ được chèn vào Word):", st.session_state.l_summary, height=200)

    st.markdown("<div style='margin:16px 0 8px;'></div>", unsafe_allow_html=True)
    is_ready = bool(st.session_state.l_summary)
    export_data = b""

    if is_ready:
        try:
            from components.export_docx import export_summary_to_docx
            export_data = export_summary_to_docx(summary_text = st.session_state.l_summary, session_name = st.session_state.l_session_name)
        except FileNotFoundError as e: st.error(str(e))

    st.download_button(label="📄 Xuất biên bản DOCX", data=export_data, file_name=f"Bien_ban_{st.session_state.l_session_name.replace(' ', '_')}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", type="primary", disabled=not is_ready, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 11. STREAMLIT EVENT POLL 200ms
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.l_recording:
    partial, completed = _drain_queue()
    st.session_state.l_partial = partial
    if completed: st.session_state.l_turns.extend(completed)
    time.sleep(0.2)
    st.rerun()