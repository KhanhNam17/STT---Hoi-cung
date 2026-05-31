# test_pyannote_npu_v2.py
#
# ══════════════════════════════════════════════════════════════════════════════
# KIẾN TRÚC: VAD + TURN-BASED + SPEAKER BANK
# ══════════════════════════════════════════════════════════════════════════════
#
# Khác biệt so với v1 (rolling window 30s):
#
#   v1: mỗi 5s → gửi 30s audio → Pyannote batch cluster → label reset mỗi call
#   v2: VAD detect turn → gửi 1 turn (~1-8s) → Pyannote 1-2 speaker/call
#       → Speaker Bank giữ embedding → map local label → global label nhất quán
#
# Pipeline:
#   Mic (100ms chunk)
#     │
#     ├─► Waveform display (real-time)
#     │
#     └─► VAD engine (Silero-style energy VAD)
#           │
#           ├── Đang im lặng → tích lũy silence
#           └── Vừa kết thúc turn (silence > 0.6s) → Turn queue
#                                                          │
#                                                    API Worker
#                                                          │
#                                               Pyannote-NPU (turn ngắn)
#                                                          │
#                                               Speaker Bank reconcile
#                                               (energy profile similarity)
#                                                          │
#                                               Global label (SPEAKER_A, B,...)
#                                                          │
#                                               Segments display
# ══════════════════════════════════════════════════════════════════════════════

import sys
import time
import queue
import threading
import wave
import io
import base64
import requests
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# ══════════════════════════════════════════════════════════════════════════════
# CẤU HÌNH
# ══════════════════════════════════════════════════════════════════════════════
API_URL       = "http://127.0.0.1:18182/v1/audio/diarize"
MODEL_NAME    = "NexaAI/Pyannote-NPU"
SAMPLE_RATE   = 16000
CHUNK_DUR     = 0.1          # 100ms mỗi chunk mic

# VAD
VAD_ENERGY_THRESHOLD = 150   # RMS tối thiểu để coi là "có tiếng"
VAD_SILENCE_SEC      = 0.6   # Im lặng bao lâu thì cắt turn
VAD_MIN_TURN_SEC     = 0.4   # Turn ngắn hơn thế này → bỏ qua (nhiễu)
VAD_MAX_TURN_SEC     = 12.0  # Turn dài hơn thế này → buộc cắt (tránh timeout)

# Speaker Bank
ANCHOR_SEC          = 6.0    # Giữ tối đa 6s clip per speaker
MIN_ANCHOR_SEC      = 0.8    # Tích lũy ít nhất 0.8s trước khi lưu vào bank
ENERGY_FRAME_MS     = 50     # Frame RMS cho energy profile
MIN_SIM_THRESHOLD   = 0.30   # Cosine similarity tối thiểu để nhận ra speaker cũ
MAX_SPEAKERS        = 10

# Display
DISPLAY_SEC   = 30.0         # Độ rộng cửa sổ hiển thị
MAX_DISP_ROWS = 8            # Số hàng tối đa trên timeline

COLORS = [
    "#ff7f0e", "#1f77b4", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
    "#bcbd22", "#7f7f7f",
]
_GLOBAL_LABELS = [f"SPEAKER_{chr(65+i)}" for i in range(26)]


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Turn:
    """Một lượt nói hoàn chỉnh từ VAD"""
    audio_i16:  np.ndarray   # INT16 audio
    abs_start:  float        # Timestamp tuyệt đối (giây từ lúc bắt đầu)
    duration:   float        # Độ dài (giây)

@dataclass
class SpeakerEntry:
    """Một speaker trong bank"""
    global_label:   str
    clip_i16:       np.ndarray     # Audio clip đại diện (tối đa ANCHOR_SEC)
    energy_profile: np.ndarray     # RMS profile chuẩn hóa
    first_seen:     float
    total_speech:   float = 0.0
    turn_count:     int   = 0
    # Pending: tích lũy clip trước khi đủ MIN_ANCHOR_SEC
    pending_clips:  List[np.ndarray] = field(default_factory=list)
    is_confirmed:   bool = False    # True khi đã đủ clip để lưu vào bank

@dataclass
class DisplaySegment:
    """Segment để hiển thị trên timeline"""
    abs_start:    float
    abs_end:      float
    global_label: str


# ══════════════════════════════════════════════════════════════════════════════
# SHARED STATE (thread-safe)
# ══════════════════════════════════════════════════════════════════════════════

# Waveform buffer (rolling)
_wave_buf  = np.zeros(int(SAMPLE_RATE * DISPLAY_SEC), dtype=np.float32)
_wave_lock = threading.Lock()
_t_now     = 0.0   # Thời gian hiện tại (giây)

# VAD state
_vad_speech_buf:  List[np.ndarray] = []   # Tích lũy audio của turn đang nói
_vad_silence_buf: float = 0.0              # Giây im lặng liên tiếp
_vad_turn_start:  float = 0.0             # Timestamp bắt đầu turn
_vad_in_speech:   bool  = False

# Turn queue → API worker
_turn_queue: queue.Queue = queue.Queue()

# Speaker bank (global_label → SpeakerEntry)
_speaker_bank: Dict[str, SpeakerEntry] = {}
_bank_lock = threading.Lock()

# Persistent local→global map (xuyên suốt session)
_local_map: Dict[str, str] = {}
_lmap_lock = threading.Lock()

# Display segments
_disp_segs: List[DisplaySegment] = []
_seg_lock  = threading.Lock()

# Color map (global_label → hex color)
_color_map: Dict[str, str] = {}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS: ENERGY PROFILE & COSINE SIMILARITY
# ══════════════════════════════════════════════════════════════════════════════

def _energy_profile(audio_i16: np.ndarray) -> Optional[np.ndarray]:
    """RMS theo từng ENERGY_FRAME_MS frame, normalize thành unit vector."""
    frame_len = max(1, int(SAMPLE_RATE * ENERGY_FRAME_MS / 1000))
    audio_f32 = audio_i16.astype(np.float32)
    n_frames  = len(audio_f32) // frame_len
    if n_frames == 0:
        return None
    frames  = audio_f32[:n_frames * frame_len].reshape(n_frames, frame_len)
    profile = np.sqrt(np.mean(frames**2, axis=1))
    norm    = np.linalg.norm(profile)
    return (profile / norm) if norm > 1e-6 else None


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, xử lý khác độ dài bằng interpolation."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    if len(a) != len(b):
        n = min(len(a), len(b))
        a = np.interp(np.linspace(0,1,n), np.linspace(0,1,len(a)), a)
        b = np.interp(np.linspace(0,1,n), np.linspace(0,1,len(b)), b)
    dot  = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 1e-8 else 0.0


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# LUỒNG 1: THU ÂM + VAD
# ══════════════════════════════════════════════════════════════════════════════

def audio_callback(indata, frames, time_info, status):
    """
    Chạy trong real-time audio thread.
    Làm 2 việc:
      1. Cập nhật waveform buffer để hiển thị
      2. VAD: detect turn boundary, đẩy Turn vào _turn_queue
    """
    global _t_now, _vad_in_speech, _vad_silence_buf, _vad_turn_start

    if status:
        print(f"[Mic] {status}", file=sys.stderr)

    chunk_f32 = indata[:, 0].copy()
    chunk_dur = frames / SAMPLE_RATE

    # ── 1. Cập nhật waveform buffer ──────────────────────────────────────────
    with _wave_lock:
        _wave_buf[:-frames] = _wave_buf[frames:]
        _wave_buf[-frames:] = chunk_f32
        _t_now += chunk_dur

    # ── 2. VAD ───────────────────────────────────────────────────────────────
    chunk_i16 = (chunk_f32 * 32767).astype(np.int16)
    rms = float(np.sqrt(np.mean(chunk_i16.astype(np.float32)**2)))

    if rms >= VAD_ENERGY_THRESHOLD:
        # Có tiếng → vào speech mode
        if not _vad_in_speech:
            _vad_in_speech  = True
            _vad_turn_start = _t_now - chunk_dur
            _vad_speech_buf.clear()
            _vad_silence_buf = 0.0
        _vad_speech_buf.append(chunk_i16)
        _vad_silence_buf = 0.0

    else:
        # Im lặng
        if _vad_in_speech:
            _vad_speech_buf.append(chunk_i16)   # Giữ trailing silence
            _vad_silence_buf += chunk_dur

            # Đủ silence → kết thúc turn
            if _vad_silence_buf >= VAD_SILENCE_SEC:
                turn_audio = np.concatenate(_vad_speech_buf)
                turn_dur   = len(turn_audio) / SAMPLE_RATE

                # Cắt nếu quá dài (buộc cắt turn)
                if turn_dur > VAD_MAX_TURN_SEC:
                    max_frames = int(VAD_MAX_TURN_SEC * SAMPLE_RATE)
                    turn_audio = turn_audio[-max_frames:]
                    turn_dur   = VAD_MAX_TURN_SEC

                if turn_dur >= VAD_MIN_TURN_SEC:
                    t = Turn(
                        audio_i16 = turn_audio,
                        abs_start = _vad_turn_start,
                        duration  = turn_dur,
                    )
                    _turn_queue.put(t)
                else:
                    pass  # Quá ngắn → bỏ qua (nhiễu)

                _vad_in_speech   = False
                _vad_silence_buf = 0.0
                _vad_speech_buf.clear()

            # Buộc cắt nếu turn quá dài ngay cả khi chưa có silence
        if _vad_in_speech and _vad_speech_buf:
            accumulated = sum(len(c) for c in _vad_speech_buf) / SAMPLE_RATE
            if accumulated >= VAD_MAX_TURN_SEC:
                turn_audio = np.concatenate(_vad_speech_buf)
                t = Turn(
                    audio_i16 = turn_audio,
                    abs_start = _vad_turn_start,
                    duration  = accumulated,
                )
                _turn_queue.put(t)
                _vad_speech_buf.clear()
                _vad_turn_start = _t_now


# ══════════════════════════════════════════════════════════════════════════════
# LUỒNG 2: API WORKER + SPEAKER BANK
# ══════════════════════════════════════════════════════════════════════════════

def _wav_b64(audio_i16: np.ndarray) -> str:
    """Encode INT16 audio thành WAV base64 data URL."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_i16.tobytes())
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def _call_api(audio_i16: np.ndarray) -> Tuple[List[dict], float]:
    """Gọi NPU API, trả về (segments, latency)."""
    data_url = _wav_b64(audio_i16)
    payload  = {
        "file":  data_url,
        "audio": data_url,
        "model": MODEL_NAME,
    }
    t0   = time.perf_counter()
    resp = requests.post(API_URL, json=payload, timeout=30.0)
    lat  = time.perf_counter() - t0
    if resp.status_code == 200:
        return resp.json().get("Segments", []), lat
    return [], lat


def _get_next_global_label() -> Optional[str]:
    """Lấy global label tiếp theo chưa được dùng."""
    used = set(_speaker_bank.keys()) | set(
        e.global_label for e in _speaker_bank.values()
    )
    for lbl in _GLOBAL_LABELS:
        if lbl not in used:
            return lbl
    return None


def _reconcile_turn(
    segments:   List[dict],
    turn_audio: np.ndarray,
    turn_dur:   float,
) -> Dict[str, str]:
    """
    Map local_label (từ API call này) → global_label (ổn định).

    Tầng 1: Dùng persistent _local_map nếu đã biết.
    Tầng 2: So sánh energy profile của audio segment với bank.
    Tầng 3: Nếu không match → speaker mới → assign global label mới.

    One-to-one constraint: mỗi global label chỉ nhận 1 local label.
    """
    local_to_global: Dict[str, str] = {}
    used_globals: set = set()

    with _lmap_lock:
        # Tầng 1: persistent map
        for seg in segments:
            local = seg["SpeakerLabel"]
            if local in _local_map:
                g = _local_map[local]
                if g not in used_globals:
                    local_to_global[local] = g
                    used_globals.add(g)

    # Collect local labels chưa được map
    unmapped = [
        seg for seg in segments
        if seg["SpeakerLabel"] not in local_to_global
    ]
    if not unmapped:
        return local_to_global

    frame_len = max(1, int(SAMPLE_RATE * ENERGY_FRAME_MS / 1000))

    with _bank_lock:
        confirmed_entries = {
            g: e for g, e in _speaker_bank.items()
            if e.is_confirmed and e.energy_profile is not None
        }

    for seg in unmapped:
        local = seg["SpeakerLabel"]
        if local in local_to_global:
            continue

        # Trích audio của segment này từ turn_audio
        f0 = max(0, int(seg["StartTime"] * SAMPLE_RATE))
        f1 = min(len(turn_audio), int(seg["EndTime"] * SAMPLE_RATE))
        if f1 - f0 < int(0.2 * SAMPLE_RATE):
            continue

        seg_audio   = turn_audio[f0:f1]
        seg_profile = _energy_profile(seg_audio)
        if seg_profile is None:
            continue

        # Tầng 2: so sánh với confirmed speakers trong bank
        best_g, best_sim = None, MIN_SIM_THRESHOLD
        for g, entry in confirmed_entries.items():
            if g in used_globals:
                continue
            sim = _cosine_sim(seg_profile, entry.energy_profile)
            if sim > best_sim:
                best_sim = sim
                best_g   = g

        if best_g:
            # Match speaker cũ
            local_to_global[local] = best_g
            used_globals.add(best_g)
            with _lmap_lock:
                _local_map[local] = best_g
            _log(f"  [Bank] {local} → {best_g} (sim={best_sim:.2f}) ✓ speaker cũ")
        else:
            # Speaker mới
            new_g = _get_next_global_label()
            if new_g is None:
                _log("  [Bank] Đã đủ 26 speakers")
                continue
            local_to_global[local] = new_g
            used_globals.add(new_g)
            with _lmap_lock:
                _local_map[local] = new_g
            # Gán màu ngay
            if new_g not in _color_map:
                _color_map[new_g] = COLORS[len(_color_map) % len(COLORS)]
            _log(f"  [Bank] {local} → {new_g} ★ speaker mới")

    return local_to_global


def _update_speaker_bank(
    segments:        List[dict],
    local_to_global: Dict[str, str],
    turn_audio:      np.ndarray,
    turn_abs_start:  float,
):
    """
    Cập nhật Speaker Bank sau mỗi turn:
    - Tích lũy clip vào pending cho mỗi global label
    - Khi đủ MIN_ANCHOR_SEC → confirm speaker, lưu energy_profile
    - Cập nhật total_speech và turn_count
    """
    anchor_frames     = int(ANCHOR_SEC * SAMPLE_RATE)
    min_anchor_frames = int(MIN_ANCHOR_SEC * SAMPLE_RATE)

    with _bank_lock:
        for seg in segments:
            local  = seg["SpeakerLabel"]
            global_lbl = local_to_global.get(local)
            if global_lbl is None:
                continue

            seg_dur = seg["EndTime"] - seg["StartTime"]
            f0 = max(0, int(seg["StartTime"] * SAMPLE_RATE))
            f1 = min(len(turn_audio), int(seg["EndTime"] * SAMPLE_RATE))
            if f1 <= f0:
                continue

            clip = turn_audio[f0:f1]
            rms  = float(np.sqrt(np.mean(clip.astype(np.float32)**2)))

            # Tạo entry nếu chưa có
            if global_lbl not in _speaker_bank:
                _speaker_bank[global_lbl] = SpeakerEntry(
                    global_label   = global_lbl,
                    clip_i16       = np.array([], dtype=np.int16),
                    energy_profile = None,
                    first_seen     = turn_abs_start + seg["StartTime"],
                )
                if global_lbl not in _color_map:
                    _color_map[global_lbl] = COLORS[len(_color_map) % len(COLORS)]

            entry = _speaker_bank[global_lbl]
            entry.total_speech += seg_dur
            entry.turn_count   += 1

            if entry.is_confirmed:
                # Đã confirm → chỉ cập nhật stats, không cần thêm clip
                continue

            # Tích lũy pending clip
            if rms >= 60:  # Lọc khoảng lặng
                entry.pending_clips.append(clip)

            total_pending = sum(len(c) for c in entry.pending_clips)

            if total_pending >= min_anchor_frames:
                # Đủ → confirm speaker
                combined = np.concatenate(entry.pending_clips)
                if len(combined) > anchor_frames:
                    combined = combined[:anchor_frames]

                profile = _energy_profile(combined)
                if profile is not None:
                    entry.clip_i16       = combined
                    entry.energy_profile = profile
                    entry.is_confirmed   = True
                    entry.pending_clips  = []
                    _log(
                        f"  [Bank] ✅ Confirm {global_lbl}: "
                        f"{len(combined)/SAMPLE_RATE:.2f}s | "
                        f"RMS={rms:.0f} | turns={entry.turn_count}"
                    )


def api_worker():
    """
    Lấy Turn từ queue, gọi NPU API, reconcile label, cập nhật bank + display.
    """
    while True:
        try:
            turn: Turn = _turn_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        if turn is None:
            break

        t_start_abs = turn.abs_start
        turn_dur    = turn.duration
        audio       = turn.audio_i16

        _log(
            f"[Turn] abs={t_start_abs:.1f}s | dur={turn_dur:.2f}s | "
            f"RMS={np.sqrt(np.mean(audio.astype(np.float32)**2)):.0f}"
        )

        # ── Gọi API ──────────────────────────────────────────────────────────
        try:
            segments, latency = _call_api(audio)
        except requests.exceptions.Timeout:
            _log(f"  [API] TIMEOUT")
            continue
        except requests.exceptions.ConnectionError:
            _log(f"  [API] CONNECTION ERROR — NPU server không phản hồi")
            continue
        except Exception as e:
            _log(f"  [API] ERROR: {e}")
            continue

        if not segments:
            _log(f"  [API] latency={latency:.3f}s | 0 segments")
            continue

        num_spk = len({s["SpeakerLabel"] for s in segments})
        _log(
            f"  [API] latency={latency:.3f}s | "
            f"NumSpeakers(call)={num_spk} | segments={len(segments)}"
        )

        # ── Reconcile local → global ──────────────────────────────────────────
        local_to_global = _reconcile_turn(segments, audio, turn_dur)

        # ── Cập nhật Speaker Bank ─────────────────────────────────────────────
        _update_speaker_bank(segments, local_to_global, audio, t_start_abs)

        # ── Cập nhật Display Segments ─────────────────────────────────────────
        new_segs: List[DisplaySegment] = []
        for seg in segments:
            local  = seg["SpeakerLabel"]
            global_lbl = local_to_global.get(local)
            if global_lbl is None:
                continue
            dur = seg["EndTime"] - seg["StartTime"]
            if dur < 0.25:
                continue
            abs_s = t_start_abs + seg["StartTime"]
            abs_e = t_start_abs + seg["EndTime"]
            new_segs.append(DisplaySegment(abs_s, abs_e, global_lbl))

        with _seg_lock:
            global _disp_segs
            # Giữ segment trong DISPLAY_SEC gần nhất
            cutoff = _t_now - DISPLAY_SEC
            _disp_segs = [
                s for s in _disp_segs if s.abs_end > cutoff
            ]
            _disp_segs.extend(new_segs)

        # Log bank status
        with _bank_lock:
            confirmed = [g for g, e in _speaker_bank.items() if e.is_confirmed]
            pending   = [g for g, e in _speaker_bank.items() if not e.is_confirmed]
        _log(
            f"  [Bank] confirmed={confirmed} | pending={pending} | "
            f"global_speakers={sorted(_color_map.keys())}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# LUỒNG CHÍNH: MATPLOTLIB VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Khởi động API worker
    threading.Thread(target=api_worker, daemon=True).start()

    plt.style.use("dark_background")
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(13, 8),
        gridspec_kw={"height_ratios": [2, 3, 1]}
    )
    fig.canvas.manager.set_window_title(
        "NPU Diarizer v2 — VAD + Speaker Bank"
    )
    fig.subplots_adjust(hspace=0.35, left=0.12, right=0.97)

    # ── Ax1: Waveform ─────────────────────────────────────────────────────────
    x_t = np.linspace(-DISPLAY_SEC, 0, int(SAMPLE_RATE * DISPLAY_SEC))
    (line_wave,) = ax1.plot(x_t, _wave_buf, color="#00ffcc", linewidth=0.6)
    ax1.set_ylim(-1.0, 1.0)
    ax1.set_xlim(-DISPLAY_SEC, 0)
    ax1.set_title("Audio Waveform", color="#aaaaaa", fontsize=10, pad=4)
    ax1.axis("off")

    # VAD indicator line (đường đỏ khi đang nói)
    vad_line = ax1.axvline(x=0, color="#ff4444", linewidth=1.5, alpha=0.0)

    # ── Ax2: Speaker Timeline ─────────────────────────────────────────────────
    ax2.set_title("Speaker Diarization Timeline (Turn-based + Speaker Bank)",
                  color="#aaaaaa", fontsize=10, pad=4)
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.15, axis="x")
    ax2.set_facecolor("#111111")

    # ── Ax3: Speaker Bank Status ──────────────────────────────────────────────
    ax3.set_title("Speaker Bank Status", color="#aaaaaa", fontsize=9, pad=4)
    ax3.axis("off")
    ax3.set_facecolor("#0a0a0a")

    # Y-position map: global_label → row index (ổn định xuyên suốt)
    _y_map: Dict[str, int] = {}

    def _get_y(label: str) -> int:
        if label not in _y_map:
            _y_map[label] = len(_y_map) % MAX_DISP_ROWS
        return _y_map[label]

    # ── Update function ───────────────────────────────────────────────────────
    def update_plot(frame):
        with _wave_lock:
            wave_data = _wave_buf.copy()
            t_now     = _t_now

        # 1. Cập nhật waveform
        line_wave.set_ydata(wave_data)
        ax1.set_xlim(t_now - DISPLAY_SEC, t_now)

        # 2. VAD indicator — highlight khi đang nói
        if _vad_in_speech:
            vad_line.set_alpha(0.6)
            vad_line.set_xdata([t_now, t_now])
        else:
            vad_line.set_alpha(0.0)

        # 3. Cập nhật timeline
        ax2.clear()
        ax2.set_xlim(t_now - DISPLAY_SEC, t_now)
        n_rows = max(len(_y_map), 1)
        ax2.set_ylim(-0.6, n_rows - 0.4)
        ax2.grid(True, alpha=0.15, axis="x")
        ax2.set_facecolor("#111111")

        with _seg_lock:
            segs = list(_disp_segs)

        for ds in segs:
            y   = _get_y(ds.global_label)
            col = _color_map.get(ds.global_label, "#888888")
            w   = ds.abs_end - ds.abs_start
            if w <= 0:
                continue
            rect = Rectangle(
                (ds.abs_start, y - 0.35), w, 0.7,
                facecolor=col, alpha=0.85,
                edgecolor="white", linewidth=0.4
            )
            ax2.add_patch(rect)
            if w > 0.8:
                ax2.text(
                    ds.abs_start + w / 2, y,
                    ds.global_label,
                    color="white", weight="bold",
                    fontsize=7.5, ha="center", va="center"
                )

        # Y-axis labels (tên speaker)
        if _y_map:
            ax2.set_yticks(list(_y_map.values()))
            ax2.set_yticklabels(
                list(_y_map.keys()),
                fontsize=8, color="white"
            )

        # 4. Speaker Bank status bar
        ax3.clear()
        ax3.axis("off")
        ax3.set_facecolor("#0a0a0a")

        with _bank_lock:
            bank_snapshot = {
                g: (e.is_confirmed, round(e.total_speech, 1), e.turn_count)
                for g, e in _speaker_bank.items()
            }

        x_cursor = 0.02
        for g, (confirmed, total_s, turns) in sorted(bank_snapshot.items()):
            col    = _color_map.get(g, "#888888")
            status = "✓" if confirmed else "…"
            text   = f"{g} {status}  {total_s}s / {turns}t"
            ax3.text(
                x_cursor, 0.5, text,
                transform=ax3.transAxes,
                color=col, fontsize=8.5, weight="bold",
                va="center", ha="left",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="#1a1a1a",
                    edgecolor=col, linewidth=1.2,
                    alpha=0.9,
                )
            )
            x_cursor += 0.13
            if x_cursor > 0.95:
                break

        return line_wave,

    # ── Bắt đầu thu âm ───────────────────────────────────────────────────────
    stream = sd.InputStream(
        samplerate  = SAMPLE_RATE,
        channels    = 1,
        dtype       = "float32",
        blocksize   = int(SAMPLE_RATE * CHUNK_DUR),
        callback    = audio_callback,
    )

    _log("🎙️  Bắt đầu thu âm — Nói chuyện để test diarization...")
    _log(f"    VAD threshold={VAD_ENERGY_THRESHOLD} | silence={VAD_SILENCE_SEC}s")
    _log(f"    Min turn={VAD_MIN_TURN_SEC}s | Max turn={VAD_MAX_TURN_SEC}s")
    _log(f"    Min anchor={MIN_ANCHOR_SEC}s | Sim threshold={MIN_SIM_THRESHOLD}")
    _log("    Nhấn Ctrl+C hoặc đóng cửa sổ để dừng\n")

    with stream:
        ani = animation.FuncAnimation(
            fig, update_plot,
            interval=100, blit=False, cache_frame_data=False
        )
        try:
            plt.show()
        except KeyboardInterrupt:
            pass

    _turn_queue.put(None)
    _log("🛑 Đã dừng.")

    # In tóm tắt cuối
    print("\n" + "═"*60)
    print("SPEAKER BANK SUMMARY")
    print("═"*60)
    with _bank_lock:
        for g, e in sorted(_speaker_bank.items()):
            status = "CONFIRMED" if e.is_confirmed else "PENDING"
            print(f"  {g}: {status} | speech={e.total_speech:.1f}s | turns={e.turn_count} | first={e.first_seen:.1f}s")
    print("═"*60)


if __name__ == "__main__":
    main()