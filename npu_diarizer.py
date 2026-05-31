# core/npu_diarizer.py
#
# ══════════════════════════════════════════════════════════════════════════════
# MODULE: NPU DIARIZER — Rolling Buffer + Anchor Bank
# ══════════════════════════════════════════════════════════════════════════════
#
# Kiến trúc:
#   NPUDiarizer(config={...}, log_callback=fn)
#   .start() / .stop() / .reset()
#   .push_chunk(audio_i16)
#   .assign_speaker(t_start, t_end) → label
#   .get_speaker_windows()          → List[SpeakerWindow]
#   .get_anchor_summary()           → dict (debug)
#
# Anchor Bank — giải quyết "sliding window amnesia":
#   Mỗi speaker được biết lần đầu → lưu ANCHOR_SEC giây audio đại diện
#   Mỗi API call = [ghép anchor tất cả speakers] + [30s audio gần nhất]
#   → Pyannote luôn "nhớ" mọi speaker dù họ im lặng bao lâu
#
# Label reconciliation:
#   Pyannote không đảm bảo SPEAKER_00 call 1 = SPEAKER_00 call 2.
#   → So sánh embedding giả (audio energy profile) để map label về người thật.
#   NOTE: Với NPU REST API hiện tại chưa expose embedding thật,
#         dùng heuristic overlap-time làm reconciliation proxy.
# ══════════════════════════════════════════════════════════════════════════════

import io
import time
import wave
import base64
import queue
import threading
import logging
import numpy as np
import requests
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any, Tuple

logger = logging.getLogger("npu_diarizer")


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class SpeakerWindow:
    start: float
    end:   float
    label: str

    def overlap_with(self, t0: float, t1: float) -> float:
        return max(0.0, min(self.end, t1) - max(self.start, t0))


@dataclass
class AnchorEntry:
    """Audio đại diện cho một speaker đã gặp"""
    label:          str           # global label (SPEAKER_A, SPEAKER_B, ...)
    audio_i16:      np.ndarray    # clip âm thanh (ANCHOR_SEC giây)
    first_seen_sec: float         # timestamp tuyệt đối lần đầu gặp
    total_speech_sec: float = 0.0 # tổng thời gian nói (để sort ưu tiên)


# ─── Config mặc định ──────────────────────────────────────────────────────────

DEFAULT_CONFIG: Dict[str, Any] = {
    # API
    "api_url":         "http://127.0.0.1:18182/v1/audio/diarize",
    "model_name":      "NexaAI/Pyannote-NPU",
    "api_timeout_sec": 45.0,

    # Audio
    "sample_rate":     16000,

    # Rolling buffer — chỉ gửi N giây gần nhất (không kể anchor)
    "step_sec":        5.0,
    "recent_sec":      30.0,   # Giảm xuống 30s vì anchor đã cover context cũ

    # Anchor Bank
    "anchor_sec":      8.0,    # Độ dài clip anchor mỗi speaker (giây)
    "max_anchor_gap_sec": 4.0, # Khoảng lặng tối đa trong clip anchor (loại bỏ khoảng lặng)
    "max_total_anchor_sec": 40.0, # Tổng anchor tối đa để tránh lỗi 400 (max_speakers * anchor_sec)
    "max_speakers":    10,     # Giới hạn số speaker trong anchor bank

    # Quality filter
    "min_segment_duration": 0.3,
    "min_rms_for_anchor":   80,   # RMS tối thiểu để chọn clip làm anchor (lọc khoảng lặng)
}

# Label toàn cục: SPEAKER_A, B, C, ... (không đổi giữa các API call)
_GLOBAL_LABELS = [f"SPEAKER_{chr(65+i)}" for i in range(26)]  # A-Z


class NPUDiarizer:
    """
    Diarizer NPU với Anchor Bank — không quên speaker dù họ im lặng.

    Pipeline mỗi API call:
        payload audio = [anchor_A(8s)] + [anchor_B(8s)] + ... + [recent(30s)]
        offset_sec    = tổng anchor duration (để bù timestamp về thời gian thật)
    """

    def __init__(
        self,
        config:       Optional[Dict[str, Any]]       = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}
        self._log_cb = log_callback

        # Thread-safe state
        self._audio_q:  queue.Queue = queue.Queue()
        self._windows:  List[SpeakerWindow] = []
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Anchor Bank: local_label → AnchorEntry với global_label
        # local_label: "SPEAKER_00" từ API call đầu tiên detect ra speaker này
        # global_label: "SPEAKER_A"  — ổn định xuyên suốt session
        self._anchor_bank:  Dict[str, AnchorEntry] = {}   # global_label → AnchorEntry
        self._label_map:    Dict[str, str]          = {}   # local_label → global_label
        self._anchor_lock = threading.Lock()

        # Stats
        self.stats: Dict[str, Any] = {
            "total_api_calls":   0,
            "successful_calls":  0,
            "failed_calls":      0,
            "avg_latency_sec":   0.0,
            "last_latency_sec":  0.0,
            "detected_speakers": set(),
            "anchor_bank_size":  0,
        }

        self._log(
            f"NPUDiarizer v2 (Anchor Bank) | "
            f"URL={self.cfg['api_url']} | "
            f"step={self.cfg['step_sec']}s | "
            f"recent={self.cfg['recent_sec']}s | "
            f"anchor={self.cfg['anchor_sec']}s/speaker"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            self._log("⚠️  đã chạy, bỏ qua start()")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="npu-diarizer",
            daemon=True,
        )
        self._thread.start()
        self._log("🟢 Worker bắt đầu")

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self._audio_q.put(None)
        if self._thread:
            self._thread.join(timeout=6.0)
        self._log("🔴 Worker dừng")

    def push_chunk(self, audio_i16: np.ndarray) -> None:
        if self._running:
            self._audio_q.put(audio_i16.copy())

    def get_speaker_windows(self) -> List[SpeakerWindow]:
        with self._lock:
            return list(self._windows)

    def assign_speaker(self, t_start: float, t_end: float) -> str:
        best_label   = "..."
        best_overlap = 0.0
        with self._lock:
            for w in self._windows:
                ov = w.overlap_with(t_start, t_end)
                if ov > best_overlap:
                    best_overlap = ov
                    best_label   = w.label
        return best_label

    def get_anchor_summary(self) -> Dict[str, Any]:
        """Debug: trả về thông tin anchor bank hiện tại"""
        with self._anchor_lock:
            return {
                label: {
                    "first_seen_sec":   e.first_seen_sec,
                    "total_speech_sec": round(e.total_speech_sec, 2),
                    "anchor_frames":    len(e.audio_i16),
                }
                for label, e in self._anchor_bank.items()
            }

    def get_stats(self) -> Dict[str, Any]:
        s = dict(self.stats)
        s["detected_speakers"] = sorted(self.stats["detected_speakers"])
        with self._anchor_lock:
            s["anchor_bank_size"] = len(self._anchor_bank)
        return s

    def reset(self) -> None:
        with self._lock:
            self._windows.clear()
        with self._anchor_lock:
            self._anchor_bank.clear()
            self._label_map.clear()
        while not self._audio_q.empty():
            try:
                self._audio_q.get_nowait()
            except queue.Empty:
                break
        self.stats = {
            "total_api_calls":   0,
            "successful_calls":  0,
            "failed_calls":      0,
            "avg_latency_sec":   0.0,
            "last_latency_sec":  0.0,
            "detected_speakers": set(),
            "anchor_bank_size":  0,
        }
        self._log("🔄 Reset (anchor bank xóa)")

    # ──────────────────────────────────────────────────────────────────────────
    # Worker loop
    # ──────────────────────────────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        sr           = self.cfg["sample_rate"]
        step_frames  = int(sr * self.cfg["step_sec"])
        recent_frames = int(sr * self.cfg["recent_sec"])

        full_audio:       List[np.ndarray] = []
        accumulated_frames = 0
        last_sent_frames   = 0

        self._log(
            f"[Worker] step={self.cfg['step_sec']}s | "
            f"recent_ctx={self.cfg['recent_sec']}s | "
            f"anchor={self.cfg['anchor_sec']}s/spk"
        )

        while True:
            try:
                chunk = self._audio_q.get(timeout=2.0)
            except queue.Empty:
                if not self._running:
                    break
                continue

            if chunk is None:
                self._log("[Worker] Poison pill → thoát")
                break

            full_audio.append(chunk)
            accumulated_frames += len(chunk)

            if (accumulated_frames - last_sent_frames) < step_frames:
                continue

            # ── Chuẩn bị recent audio (30s cuối) ────────────────────────────
            recent_concat = np.concatenate(full_audio)
            if len(recent_concat) > recent_frames:
                recent_concat = recent_concat[-recent_frames:]

            session_time = accumulated_frames / sr

            # ── Build payload với anchor bank ────────────────────────────────
            audio_payload, anchor_duration = self._build_payload_audio(
                recent_concat, sr
            )

            dur_sent = len(audio_payload) / sr
            self._log(
                f"[Worker] t={session_time:.1f}s | "
                f"anchor={anchor_duration:.1f}s | "
                f"recent={len(recent_concat)/sr:.1f}s | "
                f"total_sent={dur_sent:.1f}s | "
                f"bank_size={len(self._anchor_bank)}"
            )

            self._call_api(audio_payload, anchor_duration, session_time, sr)
            last_sent_frames = accumulated_frames

        self._log("[Worker] Thoát")

    # ──────────────────────────────────────────────────────────────────────────
    # Anchor Bank — build payload
    # ──────────────────────────────────────────────────────────────────────────

    def _build_payload_audio(
        self,
        recent_i16: np.ndarray,
        sr: int,
    ) -> Tuple[np.ndarray, float]:
        """
        Ghép: [anchor_A] + [anchor_B] + ... + [recent]
        Trả về: (audio_concat, anchor_total_duration_sec)
        """
        with self._anchor_lock:
            # Sort theo first_seen để thứ tự anchor nhất quán
            entries = sorted(
                self._anchor_bank.values(),
                key=lambda e: e.first_seen_sec,
            )
            # Giới hạn tổng anchor để tránh lỗi 400
            max_anchor_frames = int(sr * self.cfg["max_total_anchor_sec"])
            anchor_parts: List[np.ndarray] = []
            total_anchor_frames = 0

            for entry in entries:
                if total_anchor_frames + len(entry.audio_i16) > max_anchor_frames:
                    self._log(
                        f"[Anchor] Bỏ {entry.label} — đã đủ "
                        f"{total_anchor_frames/sr:.1f}s anchor"
                    )
                    break
                anchor_parts.append(entry.audio_i16)
                total_anchor_frames += len(entry.audio_i16)

        if anchor_parts:
            anchors_concat  = np.concatenate(anchor_parts)
            anchor_duration = total_anchor_frames / sr
            payload_audio   = np.concatenate([anchors_concat, recent_i16])
            self._log(
                f"[Anchor] Ghép {len(anchor_parts)} anchor "
                f"({anchor_duration:.1f}s) + recent ({len(recent_i16)/sr:.1f}s)"
            )
        else:
            payload_audio   = recent_i16
            anchor_duration = 0.0

        return payload_audio, anchor_duration

    # ──────────────────────────────────────────────────────────────────────────
    # NPU API call
    # ──────────────────────────────────────────────────────────────────────────

    def _call_api(
        self,
        audio_i16:      np.ndarray,
        anchor_duration: float,
        session_time:    float,
        sr:             int,
    ) -> None:
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio_i16.tobytes())

        b64      = base64.b64encode(wav_io.getvalue()).decode("utf-8")
        data_url = f"data:audio/wav;base64,{b64}"
        payload  = {
            "file":  data_url,
            "audio": data_url,
            "model": self.cfg["model_name"],
            # KHÔNG gửi num_speakers → Pyannote tự detect
        }

        t0 = time.perf_counter()
        self.stats["total_api_calls"] += 1

        try:
            resp    = requests.post(
                self.cfg["api_url"], json=payload,
                timeout=self.cfg["api_timeout_sec"],
            )
            latency = time.perf_counter() - t0
            self.stats["last_latency_sec"] = round(latency, 3)
            n = self.stats["successful_calls"] + self.stats["failed_calls"] + 1
            self.stats["avg_latency_sec"] = round(
                (self.stats["avg_latency_sec"] * (n-1) + latency) / n, 3
            )

            if resp.status_code == 200:
                self.stats["successful_calls"] += 1
                data         = resp.json()
                segments     = data.get("Segments", [])
                num_detected = data.get("NumSpeakers", 0)
                self._log(
                    f"[API ✓] latency={latency:.3f}s | "
                    f"NumSpeakers={num_detected} | segments={len(segments)}"
                )
                self._process_response(
                    segments, audio_i16, anchor_duration, session_time, sr
                )
            else:
                self.stats["failed_calls"] += 1
                self._log(
                    f"[API ✗] HTTP {resp.status_code} | "
                    f"body={resp.text[:120]}"
                )

        except requests.exceptions.Timeout:
            self.stats["failed_calls"] += 1
            self._log(f"[API ✗] TIMEOUT ({self.cfg['api_timeout_sec']}s)")
        except requests.exceptions.ConnectionError:
            self.stats["failed_calls"] += 1
            self._log("[API ✗] CONNECTION ERROR")
        except Exception as e:
            self.stats["failed_calls"] += 1
            self._log(f"[API ✗] {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # Response processing — reconciliation + anchor update
    # ──────────────────────────────────────────────────────────────────────────

    def _process_response(
        self,
        segments:        list,
        full_audio_i16:  np.ndarray,  # audio đã gửi (anchor + recent)
        anchor_duration: float,
        session_time:    float,
        sr:              int,
    ) -> None:
        """
        1. Tách phần segments nằm trong anchor vs recent
        2. Reconcile local_label → global_label bằng anchor overlap
        3. Cập nhật anchor bank với speakers mới
        4. Cập nhật windows với timestamp tuyệt đối
        """
        min_dur      = self.cfg["min_segment_duration"]
        recent_start = anchor_duration  # trong payload, recent bắt đầu ở đây

        # ── Bước 1: phân loại segments ───────────────────────────────────────
        anchor_segs: List[dict] = []   # segments nằm trong phần anchor
        recent_segs: List[dict] = []   # segments nằm trong phần recent

        for seg in segments:
            s, e, label = seg["StartTime"], seg["EndTime"], seg["SpeakerLabel"]
            if e <= recent_start + 0.1:
                anchor_segs.append(seg)
            elif s >= recent_start - 0.1:
                recent_segs.append(seg)
            else:
                # Segment vắt qua ranh giới anchor/recent → tính là recent
                recent_segs.append(seg)

        # ── Bước 2: reconcile local → global label ───────────────────────────
        # Chiến lược: local label của anchor segment → map về global label
        # dựa trên vị trí thời gian trong anchor bank (anchor A ở [0, 8s], B ở [8s, 16s], ...)
        local_to_global = self._reconcile_labels(anchor_segs, anchor_duration, sr)

        self._log(
            f"[Reconcile] map={local_to_global} | "
            f"anchor_segs={len(anchor_segs)} | recent_segs={len(recent_segs)}"
        )

        # ── Bước 3: cập nhật anchor bank với speakers mới ────────────────────
        self._update_anchor_bank(
            recent_segs, local_to_global,
            full_audio_i16, anchor_duration, session_time, sr
        )

        # ── Bước 4: cập nhật windows (timestamp tuyệt đối) ───────────────────
        new_windows: List[SpeakerWindow] = []
        recent_audio_start = session_time - (len(full_audio_i16)/sr - anchor_duration)

        for seg in recent_segs:
            local_label  = seg["SpeakerLabel"]
            global_label = local_to_global.get(local_label, local_label)

            # Chuyển timestamp trong payload → timestamp tuyệt đối trong session
            payload_start = seg["StartTime"]
            payload_end   = seg["EndTime"]
            abs_start = round(recent_audio_start + (payload_start - recent_start), 3)
            abs_end   = round(recent_audio_start + (payload_end   - recent_start), 3)

            if abs_end - abs_start < min_dur:
                self._log(
                    f"[Filter] {global_label} [{abs_start:.2f}→{abs_end:.2f}] "
                    f"({abs_end-abs_start:.2f}s < {min_dur}s)"
                )
                continue

            new_windows.append(SpeakerWindow(abs_start, abs_end, global_label))
            self.stats["detected_speakers"].add(global_label)

        speakers_in_recent = sorted({w.label for w in new_windows})
        self._log(
            f"[Windows] {len(new_windows)} windows | "
            f"speakers_in_recent={speakers_in_recent} | "
            f"all_known={sorted(self._anchor_bank.keys())}"
        )

        # Merge vào self._windows: giữ cũ ngoài vùng recent, thêm mới
        with self._lock:
            kept = [w for w in self._windows if w.end < recent_audio_start]
            kept.extend(new_windows)
            self._windows.clear()
            self._windows.extend(kept)

    def _reconcile_labels(
        self,
        anchor_segs:     List[dict],
        anchor_duration: float,
        sr:              int,
    ) -> Dict[str, str]:
        """
        Map local_label (từ API call này) → global_label (ổn định).

        Chiến lược:
        - Anchor trong payload được xếp theo thứ tự: A ở [0,8s), B ở [8s,16s), ...
        - Nếu segment nằm chủ yếu trong khoảng của anchor X → local label = global X
        - Speakers xuất hiện trong recent nhưng không match anchor nào → gán global label mới
        """
        with self._anchor_lock:
            ordered_globals = sorted(
                self._anchor_bank.keys(),
                key=lambda g: self._anchor_bank[g].first_seen_sec,
            )

        # Tính time boundary của từng anchor trong payload
        anchor_boundaries: List[Tuple[float, float, str]] = []
        cursor = 0.0
        anchor_sec = self.cfg["anchor_sec"]
        for g_label in ordered_globals:
            anchor_boundaries.append((cursor, cursor + anchor_sec, g_label))
            cursor += anchor_sec

        local_to_global: Dict[str, str] = {}

        for seg in anchor_segs:
            local   = seg["SpeakerLabel"]
            if local in local_to_global:
                continue
            s, e = seg["StartTime"], seg["EndTime"]
            best_g, best_ov = None, 0.0
            for (ab_s, ab_e, g) in anchor_boundaries:
                ov = max(0.0, min(e, ab_e) - max(s, ab_s))
                if ov > best_ov:
                    best_ov = ov
                    best_g  = g
            if best_g:
                local_to_global[local] = best_g

        return local_to_global

    def _update_anchor_bank(
        self,
        recent_segs:     List[dict],
        local_to_global: Dict[str, str],
        full_audio_i16:  np.ndarray,
        anchor_duration: float,
        session_time:    float,
        sr:              int,
    ) -> None:
        """
        Với mỗi speaker trong recent_segs:
        - Nếu đã có anchor → cập nhật total_speech_sec
        - Nếu chưa có anchor → tạo anchor mới từ audio recent tương ứng
        """
        anchor_sec    = self.cfg["anchor_sec"]
        anchor_frames = int(sr * anchor_sec)
        min_rms       = self.cfg["min_rms_for_anchor"]
        recent_offset = anchor_duration  # vị trí trong payload audio

        # recent audio trong payload
        recent_payload_start_frame = int(anchor_duration * sr)
        recent_audio = full_audio_i16[recent_payload_start_frame:]

        with self._anchor_lock:
            for seg in recent_segs:
                local_label  = seg["SpeakerLabel"]
                global_label = local_to_global.get(local_label)

                # Speaker mới chưa có trong map → assign global label mới
                if global_label is None:
                    existing = set(self._anchor_bank.keys())
                    for candidate in _GLOBAL_LABELS:
                        if candidate not in existing:
                            global_label = candidate
                            break
                    if global_label is None:
                        self._log("[Anchor] Đã đầy 26 speakers, bỏ qua")
                        continue
                    local_to_global[local_label] = global_label
                    self._log(
                        f"[Anchor] Speaker mới: {local_label} → {global_label}"
                    )

                # Cập nhật total_speech
                seg_dur = seg["EndTime"] - seg["StartTime"]
                if global_label in self._anchor_bank:
                    self._anchor_bank[global_label].total_speech_sec += seg_dur
                    continue  # anchor đã có, không cần cập nhật clip

                # Tạo anchor mới: lấy clip audio tương ứng segment này
                # Timestamp trong payload → frame trong recent_audio
                seg_start_in_payload = seg["StartTime"] - recent_offset
                seg_end_in_payload   = seg["EndTime"]   - recent_offset
                f_start = max(0, int(seg_start_in_payload * sr))
                f_end   = min(len(recent_audio), int(seg_end_in_payload * sr))

                if f_end <= f_start:
                    continue

                clip = recent_audio[f_start:f_end]

                # Kiểm tra RMS — tránh lấy khoảng lặng làm anchor
                rms = np.sqrt(np.mean(clip.astype(np.float32)**2))
                if rms < min_rms:
                    self._log(
                        f"[Anchor] {global_label}: clip RMS={rms:.0f} < {min_rms}, "
                        f"bỏ qua (có thể là khoảng lặng)"
                    )
                    continue

                # Giới hạn độ dài anchor
                if len(clip) > anchor_frames:
                    clip = clip[:anchor_frames]

                abs_first_seen = session_time - (len(recent_audio)/sr) + seg_start_in_payload

                self._anchor_bank[global_label] = AnchorEntry(
                    label=global_label,
                    audio_i16=clip,
                    first_seen_sec=abs_first_seen,
                    total_speech_sec=seg_dur,
                )
                self._log(
                    f"[Anchor] ✅ Lưu anchor {global_label}: "
                    f"{len(clip)/sr:.2f}s | RMS={rms:.0f} | "
                    f"first_seen={abs_first_seen:.1f}s"
                )

    # ──────────────────────────────────────────────────────────────────────────
    # Logging
    # ──────────────────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        ts   = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        logger.info(line)
        if self._log_cb:
            try:
                self._log_cb(line)
            except Exception:
                pass
