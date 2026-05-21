# core/npu_workers.py
#
# 2 Worker Process thường trú trên NPU.
# Mỗi worker load model 1 lần khi start, không bao giờ unload.
# Giao tiếp qua multiprocessing.Queue.
#
# Cách dùng:
#   manager = NPUWorkerManager()
#   manager.start()
#   text = manager.transcribe(wav_path)          # gửi job → nhận kết quả
#   segments = manager.diarize(wav_path, n=2)    # gửi job → nhận kết quả
#   manager.stop()

import os
import time
import traceback
import multiprocessing as mp
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Sentinel để shutdown worker ──────────────────────────────────────────────
_STOP = "__STOP__"


# ────────────────────────────────────────────────────────────────────────────
# Worker A: Whisper NPU
# ────────────────────────────────────────────────────────────────────────────
def _whisper_worker(job_queue: mp.Queue, result_queue: mp.Queue):
    """
    Process con — chạy mãi mãi, chỉ load Whisper 1 lần.
    Nhận: {"wav_path": str, "language": str}
    Trả : {"text": str, "duration": float, "latency": float} hoặc {"error": str}
    """
    print("[WhisperWorker] Đang khởi động...", flush=True)
    try:
        from core.transcriber import load_model, transcribe_file
        _, app = load_model()
        print("[WhisperWorker] ✅ Model loaded — sẵn sàng nhận job", flush=True)
        result_queue.put({"status": "ready"})
    except Exception as e:
        result_queue.put({"status": "error", "error": str(e)})
        return

    while True:
        job = job_queue.get()           # block cho đến khi có job
        if job == _STOP:
            print("[WhisperWorker] Nhận lệnh stop", flush=True)
            break

        wav_path = job.get("wav_path", "")
        language = job.get("language", "vi")

        try:
            # Cập nhật ngôn ngữ nếu cần
            lang_code = None if language == "auto" else language
            if hasattr(app, "tokenizer") and hasattr(app.tokenizer, "set_prefix_tokens"):
                app.tokenizer.set_prefix_tokens(language=lang_code, task="transcribe")

            result = transcribe_file(app, wav_path)
            result_queue.put({"ok": True, **result})

        except Exception as e:
            result_queue.put({"ok": False, "error": traceback.format_exc()})

    print("[WhisperWorker] Đã dừng", flush=True)


# ────────────────────────────────────────────────────────────────────────────
# Worker B: Nexa Diarize NPU
# ────────────────────────────────────────────────────────────────────────────
def _nexa_worker(job_queue: mp.Queue, result_queue: mp.Queue):
    """
    Process con — chạy mãi mãi, chỉ load Nexa 1 lần.
    Nhận: {"wav_path": str, "num_speakers": int, "min_duration": float, "merge_gap": float}
    Trả : {"segments": list[dict]} hoặc {"error": str}
            segments là list of {"speaker", "start", "end"} — serializable
    """
    print("[NexaWorker] Đang khởi động...", flush=True)
    try:
        from core.diarizer_nexa import load_diarizer_nexa, diarize_file_nexa
        pipeline = load_diarizer_nexa()
        print("[NexaWorker] ✅ Nexa loaded — sẵn sàng nhận job", flush=True)
        result_queue.put({"status": "ready"})
    except Exception as e:
        result_queue.put({"status": "error", "error": str(e)})
        return

    while True:
        job = job_queue.get()
        if job == _STOP:
            print("[NexaWorker] Nhận lệnh stop", flush=True)
            break

        wav_path     = job.get("wav_path", "")
        num_speakers = job.get("num_speakers", 2)
        min_duration = job.get("min_duration", 0.8)
        merge_gap    = job.get("merge_gap", 1.5)

        try:
            segs = diarize_file_nexa(
                pipeline,
                wav_path,
                num_speakers = num_speakers,
                min_duration = min_duration,
                merge_gap    = merge_gap,
            )
            # Serialize SpeakerSegment → dict (không truyền dataclass qua Queue)
            serialized = [
                {"speaker": s.speaker, "start": s.start, "end": s.end}
                for s in segs
            ]
            result_queue.put({"ok": True, "segments": serialized})

        except Exception as e:
            result_queue.put({"ok": False, "error": traceback.format_exc()})

    print("[NexaWorker] Đã dừng", flush=True)


# ────────────────────────────────────────────────────────────────────────────
# Manager — API công khai dùng trong Streamlit
# ────────────────────────────────────────────────────────────────────────────
class NPUWorkerManager:
    """
    Quản lý 2 worker process NPU.

    Dùng trong Streamlit:
        # Khởi tạo 1 lần (để ngoài @st.cache_resource để dùng singleton)
        @st.cache_resource
        def get_npu_manager():
            m = NPUWorkerManager()
            m.start()
            return m

        manager = get_npu_manager()
        text     = manager.transcribe(wav_path, language="vi")
        segments = manager.diarize(wav_path, num_speakers=2)
    """

    def __init__(self, startup_timeout: int = 120):
        self._startup_timeout = startup_timeout

        # Queues cho Whisper
        self._w_job    = mp.Queue()
        self._w_result = mp.Queue()

        # Queues cho Nexa
        self._n_job    = mp.Queue()
        self._n_result = mp.Queue()

        self._whisper_proc = None
        self._nexa_proc    = None
        self._ready        = False

    def start(self):
        """
        Khởi động 2 worker process và chờ cả 2 load xong model.
        Gọi 1 lần khi app khởi động — blocking cho đến khi ready.
        """
        print("[Manager] Khởi động 2 NPU worker...", flush=True)

        # Dùng 'spawn' để tránh fork conflict với NPU driver
        ctx = mp.get_context("spawn")

        self._whisper_proc = ctx.Process(
            target = _whisper_worker,
            args   = (self._w_job, self._w_result),
            daemon = True,
            name   = "WhisperNPU",
        )
        self._nexa_proc = ctx.Process(
            target = _nexa_worker,
            args   = (self._n_job, self._n_result),
            daemon = True,
            name   = "NexaNPU",
        )

        self._whisper_proc.start()
        self._nexa_proc.start()

        # Chờ cả 2 worker báo ready
        print(f"[Manager] Chờ worker load model (timeout={self._startup_timeout}s)...", flush=True)
        t0 = time.time()

        w_ready = n_ready = False
        while not (w_ready and n_ready):
            elapsed = time.time() - t0
            if elapsed > self._startup_timeout:
                raise TimeoutError(
                    f"Worker chưa ready sau {self._startup_timeout}s. "
                    "Kiểm tra log của WhisperNPU và NexaNPU."
                )

            # Poll Whisper result queue
            if not w_ready:
                try:
                    msg = self._w_result.get_nowait()
                    if msg.get("status") == "ready":
                        w_ready = True
                        print("[Manager] ✅ WhisperWorker ready", flush=True)
                    elif msg.get("status") == "error":
                        raise RuntimeError(f"WhisperWorker lỗi: {msg['error']}")
                except Exception as e:
                    if "empty" not in type(e).__name__.lower():
                        raise

            # Poll Nexa result queue
            if not n_ready:
                try:
                    msg = self._n_result.get_nowait()
                    if msg.get("status") == "ready":
                        n_ready = True
                        print("[Manager] ✅ NexaWorker ready", flush=True)
                    elif msg.get("status") == "error":
                        raise RuntimeError(f"NexaWorker lỗi: {msg['error']}")
                except Exception as e:
                    if "empty" not in type(e).__name__.lower():
                        raise

            time.sleep(0.5)

        self._ready = True
        print("[Manager] 🚀 Cả 2 NPU worker đã sẵn sàng!", flush=True)

    def transcribe(self, wav_path: str, language: str = "vi", timeout: int = 3600) -> dict:
        """
        Gửi job transcribe → chờ kết quả.

        Returns:
            {"text": str, "duration": float, "latency": float, "rtf": float}
        Raises:
            RuntimeError nếu worker báo lỗi hoặc timeout.
        """
        if not self._ready:
            raise RuntimeError("Manager chưa start(). Gọi manager.start() trước.")

        self._w_job.put({"wav_path": wav_path, "language": language})
        result = self._w_result.get(timeout=timeout)

        if not result.get("ok"):
            raise RuntimeError(f"WhisperWorker error:\n{result.get('error')}")
        return result

    def diarize(
        self,
        wav_path     : str,
        num_speakers : int   = 2,
        min_duration : float = 0.8,
        merge_gap    : float = 1.5,
        timeout      : int   = 3600,
    ) -> list:
        """
        Gửi job diarize → chờ kết quả.

        Returns:
            list of SpeakerSegment (được reconstruct từ dict)
        """
        if not self._ready:
            raise RuntimeError("Manager chưa start(). Gọi manager.start() trước.")

        self._n_job.put({
            "wav_path"    : wav_path,
            "num_speakers": num_speakers,
            "min_duration": min_duration,
            "merge_gap"   : merge_gap,
        })
        result = self._n_result.get(timeout=timeout)

        if not result.get("ok"):
            raise RuntimeError(f"NexaWorker error:\n{result.get('error')}")

        # Reconstruct SpeakerSegment từ dict
        from core.diarizer import SpeakerSegment
        return [
            SpeakerSegment(speaker=s["speaker"], start=s["start"], end=s["end"])
            for s in result["segments"]
        ]

    def stop(self):
        """Dừng cả 2 worker gracefully."""
        if self._whisper_proc and self._whisper_proc.is_alive():
            self._w_job.put(_STOP)
            self._whisper_proc.join(timeout=10)

        if self._nexa_proc and self._nexa_proc.is_alive():
            self._n_job.put(_STOP)
            self._nexa_proc.join(timeout=10)

        self._ready = False
        print("[Manager] Đã dừng tất cả worker", flush=True)

    def is_alive(self) -> bool:
        w = self._whisper_proc and self._whisper_proc.is_alive()
        n = self._nexa_proc    and self._nexa_proc.is_alive()
        return bool(w and n)