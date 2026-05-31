# core/npu_diarizer.py  — v4 (Memory Bank + Pending Queue)
#
# ══════════════════════════════════════════════════════════════════════════════
# KIẾN TRÚC MỚI THEO CHUẨN SMART MEETING:
# 1. Trượt cửa sổ (Rolling Buffer): Chỉ gửi 30s gần nhất, không gửi kèm Anchor.
# 2. Local Feature Extraction: Tự động trích xuất vân tay (FFT) trên Client.
# 3. Two-stage Memory Bank: 
#    - PENDING: Danh sách chờ, lọc nhiễu, tiếng ho, tạp âm.
#    - CONFIRMED: Người nói chính thức (Global Label).
# 4. Hungarian Matching: Khớp nhãn thời gian thực bằng Cosine Distance.
# ══════════════════════════════════════════════════════════════════════════════

import io
import time
import wave
import base64
import queue
import threading
import logging
import collections
import numpy as np
import requests
from dataclasses import dataclass
from typing import List, Optional, Callable, Dict, Any
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger("npu_diarizer")

# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class SpeakerWindow:
    start: float
    end:   float
    label: str

    def overlap_with(self, t0: float, t1: float) -> float:
        return max(0.0, min(self.end, t1) - max(self.start, t0))

# ─── BỘ NHỚ 2 TẦNG (SPEAKER MEMORY BANK) ──────────────────────────────────────

class SpeakerMemoryBank:
    def __init__(self, delta_new=0.2, rho_update=1.0, log_cb=None):
        # delta_new=0.2 tương đương Cosine Similarity > 0.80 cho FFT
        self.centroids: Dict[str, np.ndarray] = {}  
        self.pending_speakers: Dict[str, dict] = {} 
        self.promote_hits = 3                       
        self.pending_timeout = 10.0                 
        self.pending_counter = 0
        self.speaker_counter = 0
        
        self.delta_new = delta_new
        self.rho_update = rho_update
        self._log_cb = log_cb

    def _log(self, msg: str):
        if self._log_cb: self._log_cb(msg)

    def cosine_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6: return 1.0
        return 1.0 - (np.dot(v1, v2) / (n1 * n2))

    def compute_embedding(self, clip_i16: np.ndarray) -> Optional[np.ndarray]:
        """Trích xuất vân tay FFT nội bộ"""
        if len(clip_i16) < 160: return None
        clip_f32 = clip_i16.astype(np.float32)
        window = np.hanning(len(clip_f32))
        fft_mag = np.abs(np.fft.rfft(clip_f32 * window))
        fft_log = np.log1p(fft_mag)
        
        bins = np.array_split(fft_log, 64)
        emb = np.array([np.mean(b) for b in bins])
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 1e-8 else None

    def process_local_embeddings(self, local_segments: List[dict], current_audio_time: float) -> List[dict]:
        # 1. Dọn dẹp Pending quá hạn
        expired_pendings = [
            pid for pid, data in self.pending_speakers.items() 
            if (current_audio_time - data['last_seen']) > self.pending_timeout
        ]
        for pid in expired_pendings:
            del self.pending_speakers[pid]
            self._log(f"[MemoryBank] 🗑️ Xóa rác {pid}")

        valid_segs = [s for s in local_segments if s.get('embedding') is not None]
        if not valid_segs: return []

        # 2. Gom nhóm theo Local ID (từ NPU trả về)
        local_speakers = collections.defaultdict(list)
        for seg in valid_segs:
            local_speakers[seg.get('SpeakerLabel', 'UNKNOWN')].append(seg)

        local_embs_dict = {}
        for loc_label, segs in local_speakers.items():
            avg_emb = np.mean([np.array(s['embedding']) for s in segs], axis=0)
            local_embs_dict[loc_label] = avg_emb / np.linalg.norm(avg_emb)

        local_labels_list = list(local_embs_dict.keys())
        local_embs_list = list(local_embs_dict.values())

        # Khởi tạo người đầu tiên
        if not self.centroids and not self.pending_speakers:
            for loc_label, emb in zip(local_labels_list, local_embs_list):
                new_id = f"SPEAKER_{self.speaker_counter:02d}"
                self.centroids[new_id] = emb
                self.speaker_counter += 1
                for seg in local_speakers[loc_label]:
                    seg['global_id'] = new_id
            return valid_segs

        # 3. So khớp với Confirmed Speakers (Hungarian)
        global_ids = list(self.centroids.keys())
        unmatched_locals = []
        mapped_locals = set()

        if global_ids:
            global_embs = list(self.centroids.values())
            cost_matrix = np.zeros((len(local_embs_list), len(global_embs)))
            for i, loc_v in enumerate(local_embs_list):
                for j, glob_v in enumerate(global_embs):
                    cost_matrix[i, j] = self.cosine_distance(loc_v, glob_v)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for i, local_idx in enumerate(row_ind):
                global_idx = col_ind[i]
                if cost_matrix[local_idx, global_idx] <= self.delta_new:
                    mapped_locals.add(local_idx)
                    assigned_id = global_ids[global_idx]
                    loc_label = local_labels_list[local_idx]
                    
                    total_dur = sum([s['EndTime'] - s['StartTime'] for s in local_speakers[loc_label]])
                    if total_dur >= self.rho_update:
                        self._update_centroid(assigned_id, local_embs_list[local_idx])
                    
                    for seg in local_speakers[loc_label]:
                        seg['global_id'] = assigned_id

        # Lọc ra những người không khớp
        for i in range(len(local_embs_list)):
            if i not in mapped_locals:
                unmatched_locals.append((local_labels_list[i], local_embs_list[i]))

        # 4. Xử lý qua Pending Queue
        assigned_pendings = set() 
        for loc_label, emb in unmatched_locals:
            best_match_pid, best_dist = None, float('inf')

            for pid, p_data in self.pending_speakers.items():
                if pid in assigned_pendings: continue
                dist = self.cosine_distance(emb, p_data['embedding'])
                if dist < self.delta_new and dist < best_dist:
                    best_dist = dist
                    best_match_pid = pid

            if best_match_pid:
                assigned_pendings.add(best_match_pid)
                p_data = self.pending_speakers[best_match_pid]
                p_data['hits'] += 1
                p_data['last_seen'] = current_audio_time
                
                # Cập nhật Vector Pending
                p_data['embedding'] = (p_data['embedding'] * 0.8) + (emb * 0.2)
                p_data['embedding'] /= np.linalg.norm(p_data['embedding'])

                if p_data['hits'] >= self.promote_hits:
                    assigned_id = f"SPEAKER_{self.speaker_counter:02d}"
                    self.centroids[assigned_id] = p_data['embedding']
                    self.speaker_counter += 1
                    del self.pending_speakers[best_match_pid]
                    self._log(f"[MemoryBank] 🌟 THĂNG CẤP {best_match_pid} -> {assigned_id}")
                else:
                    assigned_id = best_match_pid

            else:
                assigned_id = f"PENDING_{self.pending_counter:02d}"
                self.pending_counter += 1
                self.pending_speakers[assigned_id] = {
                    'embedding': emb, 'hits': 1, 'last_seen': current_audio_time
                }

            for seg in local_speakers[loc_label]:
                seg['global_id'] = assigned_id

        return valid_segs

    def _update_centroid(self, global_id, new_vector, alpha=0.1):
        updated = (1 - alpha) * self.centroids[global_id] + alpha * new_vector
        norm = np.linalg.norm(updated)
        if norm > 1e-6:
            self.centroids[global_id] = updated / norm


# ─── Config mặc định ──────────────────────────────────────────────────────────

DEFAULT_CONFIG: Dict[str, Any] = {
    "api_url":           "http://127.0.0.1:18182/v1/audio/diarize",
    "model_name":        "NexaAI/Pyannote-NPU",
    "api_timeout_sec":   45.0,
    "sample_rate":       16000,
    "step_sec":          5.0,
    "recent_sec":        30.0,
    "min_segment_duration": 0.3,
}

class NPUDiarizer:
    def __init__(self, config: Optional[Dict[str, Any]] = None, log_callback: Optional[Callable[[str], None]] = None):
        self.cfg     = {**DEFAULT_CONFIG, **(config or {})}
        self._log_cb = log_callback

        self._audio_q: queue.Queue = queue.Queue()
        self._windows: List[SpeakerWindow] = []
        self._lock     = threading.Lock()
        self._thread:  Optional[threading.Thread] = None
        self._running  = False

        # Khởi tạo Memory Bank
        self.memory_bank = SpeakerMemoryBank(log_cb=self._log)

        self.stats: Dict[str, Any] = {
            "total_api_calls":   0, "successful_calls":  0, "failed_calls":      0,
            "avg_latency_sec":   0.0, "last_latency_sec":  0.0,
            "detected_speakers": set(), "anchor_bank_size":  0,
        }

        self._log(f"NPUDiarizer v4 | Memory Bank Active | step={self.cfg['step_sec']}s")

    def start(self) -> None:
        if self._running: return
        self._running = True
        self._thread  = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._running: return
        self._running = False
        self._audio_q.put(None)
        if self._thread: self._thread.join(timeout=2.0)

    def push_chunk(self, audio_i16: np.ndarray) -> None:
        if self._running: self._audio_q.put(audio_i16.copy())

    def get_speaker_windows(self) -> List[SpeakerWindow]:
        with self._lock: return list(self._windows)

    def assign_speaker(self, t_start: float, t_end: float) -> str:
        best_label, best_ov = "...", 0.0
        with self._lock:
            for w in self._windows:
                ov = w.overlap_with(t_start, t_end)
                if ov > best_ov:
                    best_ov, best_label = ov, w.label
        return best_label

    def get_stats(self) -> Dict[str, Any]:
        s = dict(self.stats)
        s["detected_speakers"] = sorted(self.stats["detected_speakers"])
        s["anchor_bank_size"] = len(self.memory_bank.centroids)
        return s

    def reset(self) -> None:
        with self._lock:
            self._windows.clear()
        self.memory_bank = SpeakerMemoryBank(log_cb=self._log)
        while not self._audio_q.empty():
            try: self._audio_q.get_nowait()
            except queue.Empty: break
        self.stats = {
            "total_api_calls": 0, "successful_calls": 0, "failed_calls": 0,
            "avg_latency_sec": 0.0, "last_latency_sec": 0.0,
            "detected_speakers": set(), "anchor_bank_size": 0,
        }

    def _worker_loop(self) -> None:
        sr            = self.cfg["sample_rate"]
        step_frames   = int(sr * self.cfg["step_sec"])
        recent_frames = int(sr * self.cfg["recent_sec"])

        full_audio: List[np.ndarray] = []
        accumulated_frames = 0
        last_sent_frames   = 0

        while True:
            try:
                chunk = self._audio_q.get(timeout=2.0)
            except queue.Empty:
                if not self._running: break
                continue

            if chunk is None: break

            full_audio.append(chunk)
            accumulated_frames += len(chunk)

            if (accumulated_frames - last_sent_frames) < step_frames:
                continue

            recent_concat = np.concatenate(full_audio)
            if len(recent_concat) > recent_frames:
                recent_concat = recent_concat[-recent_frames:]
                full_audio = [recent_concat] # Xóa rác RAM

            session_time = accumulated_frames / sr
            self._call_api(recent_concat, session_time, sr)
            last_sent_frames = accumulated_frames

    def _call_api(self, audio_i16: np.ndarray, session_time: float, sr: int) -> None:
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
            wf.writeframes(audio_i16.tobytes())

        b64 = base64.b64encode(wav_io.getvalue()).decode("utf-8")
        payload = {"file": f"data:audio/wav;base64,{b64}", "audio": f"data:audio/wav;base64,{b64}", "model": self.cfg["model_name"]}

        t0 = time.perf_counter()
        self.stats["total_api_calls"] += 1

        try:
            resp = requests.post(self.cfg["api_url"], json=payload, timeout=self.cfg["api_timeout_sec"])
            latency = time.perf_counter() - t0
            self.stats["last_latency_sec"] = round(latency, 3)
            n = self.stats["successful_calls"] + self.stats["failed_calls"] + 1
            self.stats["avg_latency_sec"] = round((self.stats["avg_latency_sec"] * (n-1) + latency) / n, 3)

            if resp.status_code == 200:
                self.stats["successful_calls"] += 1
                segments = resp.json().get("Segments", [])
                self._process_response(segments, audio_i16, session_time, sr)
            else:
                self.stats["failed_calls"] += 1
                self._log(f"[API ✗] HTTP {resp.status_code}")
        except Exception as e:
            self.stats["failed_calls"] += 1
            self._log(f"[API ✗] {e}")

    def _process_response(self, segments: list, audio_i16: np.ndarray, session_time: float, sr: int) -> None:
        recent_duration = len(audio_i16) / sr
        recent_audio_start = session_time - recent_duration

        # 1. Trích xuất Vân tay (Embeddings) cho các Segments NPU trả về
        local_segs = []
        for seg in segments:
            f0 = max(0, int(seg["StartTime"] * sr))
            f1 = min(len(audio_i16), int(seg["EndTime"] * sr))
            if f1 <= f0: continue
            
            emb = self.memory_bank.compute_embedding(audio_i16[f0:f1])
            if emb is not None:
                seg['embedding'] = emb
                local_segs.append(seg)

        # 2. Xử lý qua Memory Bank 2 Tầng
        mapped_segs = self.memory_bank.process_local_embeddings(local_segs, session_time)

        # 3. Tạo Windows mới
        new_windows = []
        for seg in mapped_segs:
            abs_start = round(recent_audio_start + seg["StartTime"], 3)
            abs_end   = round(recent_audio_start + seg["EndTime"], 3)
            
            if abs_end - abs_start < self.cfg["min_segment_duration"]: continue
            
            global_label = seg.get("global_id", "...")
            new_windows.append(SpeakerWindow(abs_start, abs_end, global_label))
            
            if "PENDING" not in global_label:
                self.stats["detected_speakers"].add(global_label)

        # 4. Cập nhật giao diện
        with self._lock:
            kept = [w for w in self._windows if w.end < recent_audio_start]
            kept.extend(new_windows)
            self._windows = kept

    def _log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        logger.info(line)
        if self._log_cb:
            try: self._log_cb(line)
            except Exception: pass