# core/diarization/sortformer.py
#
# Model: nvidia/diar_streaming_sortformer_4spk-v2  (tối đa 4 người nói).

# API thực tế của Sortformer là diarize(BUFFER ĐẦY ĐỦ) — không phải feed từng
# chunk như diart. Vì vậy:
#   • OFFLINE/BATCH  → diarize_file_sortformer(): 1 lần gọi, chất lượng cao nhất.
#   • LIVE final-pass → cũng gọi 1 lần trên toàn bộ recording khi dừng.
#   • LIVE provisional → SortformerStreamingDiarizer re-diarize định kỳ buffer
#     (gần đúng; speaker index có thể đổi giữa các lần gọi → dùng cho hiển thị tạm).

import os
import queue
import threading
import time

import numpy as np

# Bootstrap path khi chạy trực tiếp
import sys

# Khi chạy như subprocess trên Windows, stdout dùng cp1252 → emoji/tiếng Việt
# trong print() gây UnicodeEncodeError và CRASH cả tiến trình. Ép UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

# SpeakerSegment: thử dùng kiểu chung của app; nếu chạy trong env Sortformer
# RIÊNG (không có pyannote/dotenv/torch-2.2 của app) thì định nghĩa local.
# Tránh kéo theo core.diarizer (→ dotenv → cả stack app) khi chạy độc lập.
try:
    from core.diarization.diar_types import SpeakerSegment
except Exception:
    from dataclasses import dataclass

    @dataclass
    class SpeakerSegment:
        speaker: str
        start: float
        end: float
        text: str = ""


# Preset streaming theo model card (đơn vị: frame 80ms)
_PRESETS = {
    "ultra_low": dict(chunk_len=3,   chunk_right_context=1, fifo_len=188, spkcache_update_period=144, spkcache_len=188),
    "low":       dict(chunk_len=6,   chunk_right_context=7, fifo_len=188, spkcache_update_period=144, spkcache_len=188),
    "high":      dict(chunk_len=124, chunk_right_context=1, fifo_len=124, spkcache_update_period=124, spkcache_len=188),
    "very_high": dict(chunk_len=340, chunk_right_context=40, fifo_len=40, spkcache_update_period=300, spkcache_len=188),
}

_CACHED_MODEL = None


def load_sortformer(latency: str = "low"):
    """Load + cấu hình Streaming Sortformer. Cache theo process.

    latency: 'ultra_low' (0.32s) | 'low' (1.04s) | 'high' (10s) | 'very_high' (30s)
             Càng thấp latency → real-time hơn nhưng RTF cao hơn.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    try:
        from nemo.collections.asr.models import SortformerEncLabelModel
    except ImportError as e:
        raise RuntimeError(
            "Chưa cài NeMo. Cài:\n"
            "  pip install Cython packaging\n"
            '  pip install "nemo_toolkit[asr]"\n'
            f"(chi tiết: {e})"
        ) from e

    name = os.getenv("SORTFORMER_MODEL", "nvidia/diar_streaming_sortformer_4spk-v2")
    print(f"⏳ Loading Sortformer: {name} (latency={latency})", flush=True)
    model = SortformerEncLabelModel.from_pretrained(name)
    model.eval()

    p = _PRESETS.get(latency, _PRESETS["low"])
    m = model.sortformer_modules
    m.chunk_len              = p["chunk_len"]
    m.chunk_right_context    = p["chunk_right_context"]
    m.fifo_len               = p["fifo_len"]
    m.spkcache_update_period = p["spkcache_update_period"]
    m.spkcache_len           = p["spkcache_len"]
    try:
        m._check_streaming_parameters()
    except Exception as e:
        print(f"[sortformer] cảnh báo check params: {e}", flush=True)

    print("✅ Sortformer loaded", flush=True)
    _CACHED_MODEL = model
    return model


def _parse_segment(item) -> tuple[float, float, int]:
    """Sortformer có thể trả tuple (start, end, spk) hoặc chuỗi 'start end spk'."""
    if isinstance(item, (tuple, list)) and len(item) >= 3:
        return float(item[0]), float(item[1]), int(item[2])
    if isinstance(item, str):
        parts = item.replace(",", " ").split()
        # tìm 2 số đầu + nhãn speaker cuối
        nums = [pp for pp in parts if pp.replace(".", "", 1).isdigit()]
        start, end = float(nums[0]), float(nums[1])
        spk_tok = parts[-1]
        digits = "".join(c for c in spk_tok if c.isdigit())
        return start, end, int(digits or 0)
    raise ValueError(f"Không parse được segment Sortformer: {item!r}")


def _segments_from_result(result, num_speakers=None, min_duration=0.5) -> list[SpeakerSegment]:
    """result = diar_model.diarize(...) → list theo file; lấy [0]."""
    raw = result[0] if (isinstance(result, (list, tuple)) and result) else result
    segs = []
    for item in raw:
        try:
            start, end, spk = _parse_segment(item)
        except Exception:
            continue
        if end - start < min_duration:
            continue
        segs.append(SpeakerSegment(speaker=f"SPEAKER_{spk:02d}", start=round(start, 3), end=round(end, 3)))

    segs.sort(key=lambda s: s.start)

    # KHÔNG ép num_speakers: Sortformer là model 4-spk TỰ DÒ số người nói rất tốt.
    # Ép gộp về N sẽ làm MẤT speaker thật (vd gộp 2 anh em sinh đôi → 1).
    # num_speakers chỉ giữ lại cho tương thích chữ ký, không dùng để collapse.
    return segs


# ────────────────────────────────────────────────────────────────────────────
# OFFLINE — drop-in cho diarize_file_nexa / diarize_file (Batch Mode)
# ────────────────────────────────────────────────────────────────────────────
def load_diarizer_sortformer():
    """Stub tương thích load_diarizer() — trả model đã load."""
    return load_sortformer(latency=os.getenv("SORTFORMER_LATENCY", "low"))


def diarize_file_sortformer(
    pipeline=None,
    wav_path: str = "",
    num_speakers: int | None = None,
    min_duration: float = 0.5,
    merge_gap: float = 2.5,
) -> list:
    """Diarize 1 file bằng Sortformer (1 lần gọi). Cần WAV 16kHz mono."""
    model = pipeline if pipeline is not None and hasattr(pipeline, "diarize") \
        else load_sortformer(latency=os.getenv("SORTFORMER_LATENCY", "low"))
    print(f"🎙️  Diarizing (Sortformer): {os.path.basename(wav_path)}", flush=True)
    t0 = time.perf_counter()
    result = model.diarize(audio=wav_path, batch_size=1)
    segs = _segments_from_result(result, num_speakers=num_speakers, min_duration=min_duration)
    print(f"   ✅ Sortformer: {len({s.speaker for s in segs})} speakers | "
          f"{len(segs)} đoạn | {time.perf_counter()-t0:.1f}s", flush=True)
    return segs


# ────────────────────────────────────────────────────────────────────────────
# LIVE — wrapper interface giống DiartStreamingDiarizer (feed/get_windows/stop)
# Thực hiện gần đúng: re-diarize buffer định kỳ. Dùng cho provisional display.
# Chất lượng "thật" đến từ final-pass (diarize_file_sortformer trên recording).
# ────────────────────────────────────────────────────────────────────────────
class SortformerStreamingDiarizer:
    def __init__(self, sample_rate=16000, num_speakers=None, on_update=None,
                 rediarize_every=3.0, latency="low"):
        self.sample_rate    = sample_rate
        self.num_speakers    = num_speakers
        self._on_update      = on_update
        self._rediarize_every = rediarize_every
        self._latency        = latency

        self._buf: list[np.ndarray] = []
        self._buf_lock = threading.Lock()
        self._windows: list[tuple[float, float, str]] = []
        self._win_lock = threading.Lock()
        self._running   = False
        self._accepting = False
        self._thread = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._accepting = True
        self._thread = threading.Thread(target=self._worker, daemon=True, name="sortformer-stream")
        self._thread.start()

    def feed(self, chunk_int16: np.ndarray):
        if not self._accepting:
            return
        with self._buf_lock:
            self._buf.append(chunk_int16.copy())

    def stop(self, timeout=120.0):
        self._accepting = False
        self._running = False
        if self._thread:
            self._thread.join(timeout=timeout)
        # final pass — diarize toàn bộ buffer 1 lần (chất lượng cao nhất)
        self._rediarize()
        return self.get_windows()

    def get_windows(self):
        with self._win_lock:
            return list(self._windows)

    def get_segments(self):
        return [SpeakerSegment(speaker=s, start=st, end=en) for st, en, s in self.get_windows()]

    def _worker(self):
        try:
            self._model = load_sortformer(latency=self._latency)
        except Exception as e:
            print(f"[Sortformer] ❌ load lỗi: {e}", flush=True)
            self._running = False
            return
        print("[Sortformer] ✅ ready — re-diarize mỗi "
              f"{self._rediarize_every}s", flush=True)
        last = time.perf_counter()
        while self._running:
            time.sleep(0.2)
            if time.perf_counter() - last >= self._rediarize_every:
                self._rediarize()
                last = time.perf_counter()

    def _rediarize(self):
        with self._buf_lock:
            if not self._buf:
                return
            audio = np.concatenate(self._buf).astype(np.float32) / 32768.0
        try:
            model = getattr(self, "_model", None) or load_sortformer(latency=self._latency)
            result = model.diarize(audio=audio, batch_size=1, sample_rate=self.sample_rate)
            segs = _segments_from_result(result, num_speakers=self.num_speakers)
            wins = [(s.start, s.end, s.speaker) for s in segs]
            with self._win_lock:
                self._windows = wins
            if self._on_update:
                try:
                    self._on_update(wins)
                except Exception:
                    pass
        except Exception as e:
            print(f"[Sortformer] re-diarize lỗi: {e}", flush=True)


# ────────────────────────────────────────────────────────────────────────────
# CLI: python core/diarization/sortformer.py <file_16k_mono.wav> [num_spk] [--json OUT]
#   --json OUT : ghi segments ra JSON [{"speaker","start","end"}, ...] — dùng cho
#                subprocess bridge (app gọi env sortformer rồi đọc lại JSON).
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json as _json

    args = sys.argv[1:]
    json_out = None
    if "--json" in args:
        i = args.index("--json")
        json_out = args[i + 1]
        args = args[:i] + args[i + 2:]

    if not args:
        print("Dùng: python core/diarization/sortformer.py <file_16k_mono.wav> [num_speakers] [--json OUT]")
        sys.exit(0)

    wav  = args[0]
    nspk = int(args[1]) if len(args) > 1 else None
    segs = diarize_file_sortformer(wav_path=wav, num_speakers=nspk)

    if json_out:
        # Chế độ bridge: chỉ ghi JSON, in 1 dòng tóm tắt ra stderr-style
        with open(json_out, "w", encoding="utf-8") as f:
            _json.dump([{"speaker": s.speaker, "start": s.start, "end": s.end} for s in segs], f)
        print(f"[sortformer] Wrote {len(segs)} segments → {json_out}", flush=True)
        sys.exit(0)

    print(f"\n=== KẾT QUẢ: {len(segs)} đoạn ===")
    for s in segs[:40]:
        print(f"  [{s.start:7.2f} → {s.end:7.2f}] {s.speaker}")
    dist = {}
    for s in segs:
        dist[s.speaker] = dist.get(s.speaker, 0.0) + (s.end - s.start)
    total = sum(dist.values()) or 1
    print("\nThống kê:")
    for spk, d in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {spk}: {d:.1f}s ({d/total*100:.1f}%)")
