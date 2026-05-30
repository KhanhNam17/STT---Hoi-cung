# core/diarization/streaming.py
#
# Real-time diarization backend dùng diart.
# Thay thế _run_diart_thread (NPU HTTP) bằng diart Python library chạy local.
#
# Tại sao diart?
#   - Streaming-native: sliding window 5s, step 500ms — không phải POST nguyên 60s mỗi 5s
#   - Latency thấp (~500ms thay vì 5s với NPU HTTP)
#   - Built on pyannote models — chất lượng tương đương
#   - Online clustering — speaker labels nhất quán theo thời gian
#   - Không cần Nexa server chạy ngoài → portable hơn
#
# Cách dùng:
#   diarizer = DiartStreamingDiarizer(sample_rate=16000)
#   diarizer.start()
#   # trong audio callback:
#   diarizer.feed(int16_chunk)
#   # đọc windows hiện có:
#   windows = diarizer.get_windows()         # [(start, end, speaker), ...]
#   # khi dừng (trả snapshot windows cuối):
#   final_windows = diarizer.stop()
#
# Yêu cầu:
#   pip install diart
#   .env: HF_TOKEN=hf_... (diart load pyannote/segmentation-3.0 + pyannote/embedding)

import os

# LƯU Ý PHỤ THUỘC: cần speechbrain < 1.0 (pin 0.5.16 trong requirements.txt).
# speechbrain 1.0/1.1 chuyển 'speechbrain.pretrained' thành shim deprecated, shim này
# lazy-import 'integrations.k2_fsa' và crash khi k2 chưa cài (k2 gần như không cài
# được trên Windows). pyannote.audio 3.1.1 không pin version speechbrain nên downgrade
# 0.5.16 là hợp lệ. Không có env var nào tắt được lazy import — buộc phải đúng version.

import sys
import queue
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np

# Cho phép chạy trực tiếp `python core/diarization/streaming.py <file.wav>`
# bằng cách thêm project root vào sys.path TRƯỚC khi import package `core`.
if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from core.diarization.diar_types import SpeakerSegment


# ────────────────────────────────────────────────────────────────────────────
# Cấu hình diart — đọc từ .env, có defaults hợp lý cho phòng hỏi cung
# ────────────────────────────────────────────────────────────────────────────
DIART_LATENCY        = float(os.getenv("DIART_LATENCY",        "0.5"))   # giây — độ trễ tối thiểu
DIART_STEP           = float(os.getenv("DIART_STEP",           "0.5"))   # giây — bước trượt
DIART_DURATION       = float(os.getenv("DIART_DURATION",       "5.0"))   # giây — cửa sổ phân tích
DIART_TAU_ACTIVE     = float(os.getenv("DIART_TAU_ACTIVE",     "0.5"))   # ngưỡng VAD
DIART_RHO_UPDATE     = float(os.getenv("DIART_RHO_UPDATE",     "0.3"))
DIART_DELTA_NEW      = float(os.getenv("DIART_DELTA_NEW",      "1.0"))   # ngưỡng tách speaker mới


@dataclass
class _Window:
    start: float
    end: float
    speaker: str


class DiartStreamingDiarizer:
    """Wrapper bao diart streaming pipeline trong thread + queue.

    Match cùng interface với _run_diart_thread cũ (NPU HTTP):
        - đẩy chunk int16 vào qua .feed()
        - đọc speaker windows [(start, end, label), ...] qua .get_windows()
        - kết liễu qua .stop()

    Speaker labels từ diart là chuỗi `speaker0`, `speaker1`, ... — wrapper sẽ
    convert sang format `SPEAKER_00`, `SPEAKER_01`, ... để khớp với phần còn lại
    của pipeline (postprocess_segments, speaker_editor đều dùng format này).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        num_speakers: int | None = None,
        on_update: Callable[[list[tuple[float, float, str]]], None] | None = None,
    ):
        self.sample_rate   = sample_rate
        self.num_speakers  = num_speakers
        self._on_update    = on_update

        self._audio_q: queue.Queue = queue.Queue()
        self._windows: list[_Window] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False        # worker thread còn sống không
        self._accepting = False      # feed() có nhận chunk mới không
        self._started_at: float | None = None
        self._n_predictions = 0      # đếm số lần hook diart fire (chẩn đoán)
        self._final_annotation = None  # annotation tích luỹ đầy đủ từ inference()

        # Đặt sau khi diart load — embeddings cho final re-cluster ở Phase 2/3
        self._segmentation_model = None
        self._embedding_model = None

    # ── Lifecycle ──────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._accepting = True
        self._started_at = time.perf_counter()
        self._thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="diart-streaming",
        )
        self._thread.start()

    def feed(self, chunk_int16: np.ndarray) -> None:
        """Push int16 mono chunk vào diart. Non-blocking."""
        if not self._accepting:
            return
        self._audio_q.put(chunk_int16)

    def stop(self, timeout: float = 30.0) -> list[tuple[float, float, str]]:
        """Ngừng nhận audio, ĐỢI worker xử lý nốt queue (sentinel ở cuối hàng),
        rồi trả snapshot windows cuối.

        Quan trọng: KHÔNG set _running=False ngay — nếu làm vậy worker có thể
        thoát loop trước khi xử lý hết audio đã queue (đặc biệt khi model còn
        đang load lúc feed). Thay vào đó đẩy sentinel None vào CUỐI queue; worker
        drain hết chunk trước sentinel rồi mới dừng.
        """
        if not self._accepting and not self._running:
            return self.get_windows()
        self._accepting = False
        self._audio_q.put(None)          # sentinel — nằm sau mọi chunk đã feed
        if self._thread:
            self._thread.join(timeout=timeout)
        self._running = False
        return self.get_windows()

    # ── Public read API ────────────────────────────────────────────────────
    def get_windows(self) -> list[tuple[float, float, str]]:
        """Snapshot hiện tại của (start, end, label) — match format _speaker_windows."""
        with self._lock:
            return [(w.start, w.end, w.speaker) for w in self._windows]

    def get_segments(self) -> list[SpeakerSegment]:
        with self._lock:
            return [
                SpeakerSegment(speaker=w.speaker, start=w.start, end=w.end)
                for w in self._windows
            ]

    # ── Worker thread ──────────────────────────────────────────────────────
    def _worker(self) -> None:
        print("[DiartStreamer] Khởi động…", flush=True)

        try:
            pipeline, audio_source, infer_thread = self._build_pipeline()
        except Exception as e:
            print(f"[DiartStreamer] ❌ Lỗi load diart: {e}", flush=True)
            self._running = False
            return

        print("[DiartStreamer] ✅ diart pipeline ready — bắt đầu nhận audio", flush=True)

        # Buffer các chunk thành block đủ lớn cho step của diart.
        # 200ms chunks (3200 frames @ 16k) × N → bằng DIART_STEP giây
        step_frames = int(self.sample_rate * DIART_STEP)
        buf = np.empty(0, dtype=np.float32)
        blocks_pushed = 0

        try:
            # Loop tới khi gặp sentinel None — KHÔNG phụ thuộc _running, để
            # drain hết audio đã queue kể cả khi stop() được gọi sớm.
            while True:
                try:
                    item = self._audio_q.get(timeout=1.0)
                except queue.Empty:
                    continue

                if item is None:
                    print("[DiartStreamer] Nhận sentinel — drain xong", flush=True)
                    break

                # int16 → float32 normalized
                chunk_f32 = item.astype(np.float32) / 32768.0
                buf = np.concatenate([buf, chunk_f32])

                # Xử lý theo từng step
                while len(buf) >= step_frames:
                    block = buf[:step_frames]
                    buf = buf[step_frames:]
                    try:
                        audio_source.push(block, self.sample_rate)
                        blocks_pushed += 1
                    except Exception as e:
                        print(f"[DiartStreamer] push error: {e}", flush=True)

            # Đẩy nốt buffer còn lại (đuôi < 1 step) để không mất audio cuối
            if len(buf) > 0:
                try:
                    audio_source.push(buf, self.sample_rate)
                    blocks_pushed += 1
                except Exception:
                    pass

        except Exception as e:
            print(f"[DiartStreamer] worker crash: {e}", flush=True)
        finally:
            # Đóng source → read() loop của diart thoát → inference() trả về.
            try:
                audio_source.close()
            except Exception:
                pass
            # ĐỢI diart xử lý nốt — predictions phát trong infer_thread.
            print(f"[DiartStreamer] Đã push {blocks_pushed} blocks, đợi diart xử lý nốt…",
                  flush=True)
            infer_thread.join(timeout=60.0)
            if infer_thread.is_alive():
                print("[DiartStreamer] ⚠️  diart CHƯA xử lý xong sau 60s — kết quả BỊ CẮT "
                      "(file quá dài cho 1 lần xử lý). Dùng clip ngắn hơn hoặc tăng timeout.",
                      flush=True)

            # Ưu tiên annotation tích luỹ đầy đủ của diart (authoritative).
            # Hook tích luỹ live chỉ là xấp xỉ; accumulator gộp/relabel chuẩn hơn.
            if self._final_annotation is not None:
                try:
                    final = []
                    for segment, _, label in self._final_annotation.itertracks(yield_label=True):
                        final.append(_Window(
                            speaker=self._normalize_label(label),
                            start=round(float(segment.start), 3),
                            end=round(float(segment.end), 3),
                        ))
                    final.sort(key=lambda w: w.start)
                    if final:
                        with self._lock:
                            self._windows = final
                except Exception as e:
                    print(f"[DiartStreamer] parse final annotation lỗi: {e}", flush=True)

            print(f"[DiartStreamer] Đã dừng | hook fired {self._n_predictions}× "
                  f"| tổng {len(self._windows)} windows", flush=True)

    # ── diart pipeline builder ─────────────────────────────────────────────
    def _build_pipeline(self):
        """Tạo diart pipeline + custom audio source bơm-được-từ-ngoài.

        Quan trọng (theo diart 0.9.x source):
          - AudioSource base TỰ tạo self.stream = rx Subject (không override).
          - on_next() nhận numpy raw shape (1, n_samples) float32 — KHÔNG phải
            SlidingWindowFeature. StreamingInference tự rearrange thành chunk.
          - StreamingInference.__call__ gọi source.read() và read() PHẢI BLOCK
            (nếu không, prediction rỗng — xem FIXME trong inference.py).
          - Callback dùng attach_hooks() (nhận Callable), KHÔNG phải
            attach_observers() (nhận rx.Observer).
        """
        try:
            from diart import SpeakerDiarization, SpeakerDiarizationConfig
            from diart.models import SegmentationModel, EmbeddingModel
            from diart.sources import AudioSource
            from diart.inference import StreamingInference
        except ImportError as e:
            raise RuntimeError(
                "diart chưa được cài. Chạy: pip install diart\n"
                "Nếu bị lỗi sklearn/torch, xem requirements.txt."
            ) from e

        # ── Models — ép dùng segmentation-3.0 (mới, không phải 'pyannote/segmentation'
        #    cũ mà diart mặc định) và truyền HF token tường minh ───────────────
        # Nếu không truyền token → diart tải model gated thất bại → trả None →
        # '.to(device)' trên None → lỗi "NoneType has no attribute to".
        hf_token = os.getenv("HF_TOKEN", "") or True   # True = đọc từ huggingface-cli login
        seg_name = os.getenv("DIART_SEGMENTATION", "pyannote/segmentation-3.0")
        emb_name = os.getenv("DIART_EMBEDDING",    "pyannote/embedding")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            segmentation = SegmentationModel.from_pretrained(seg_name, use_hf_token=hf_token)
            embedding    = EmbeddingModel.from_pretrained(emb_name, use_hf_token=hf_token)

        # ── Custom audio source: read() block, pull từ queue nội bộ ───────
        # Mirror chính xác MicrophoneAudioSource: read() là vòng lặp block,
        # poll queue và emit numpy (1, n) qua self.stream.on_next().
        class PushAudioSource(AudioSource):
            def __init__(s, sample_rate: int):
                super().__init__("push-source", sample_rate)   # tạo self.stream Subject
                s._q: queue.Queue = queue.Queue()
                s._closed = False

            def push(s, block: np.ndarray, sr: int):
                """Đẩy 1 block float32 (n,) vào — gọi từ thread ngoài."""
                if s._closed:
                    return
                s._q.put(block.reshape(1, -1).astype(np.float32))   # (1, n)

            def read(s):
                """BLOCKING — diart gọi hàm này để drive stream."""
                while True:
                    item = s._q.get()
                    if item is None:        # sentinel từ close()
                        break
                    try:
                        s.stream.on_next(item)
                    except Exception as exc:
                        s.stream.on_error(exc)
                        return
                s.stream.on_completed()

            def close(s):
                s._closed = True
                s._q.put(None)              # unblock read()

        # ── Pipeline config ─────────────────────────────────────────────
        config_kwargs = dict(
            segmentation = segmentation,
            embedding    = embedding,
            latency      = DIART_LATENCY,
            step         = DIART_STEP,
            duration     = DIART_DURATION,
            tau_active   = DIART_TAU_ACTIVE,
            rho_update   = DIART_RHO_UPDATE,
            delta_new    = DIART_DELTA_NEW,
        )
        if self.num_speakers:
            config_kwargs["max_speakers"] = self.num_speakers

        # Vô hiệu hoá warnings ồn ào của torch/pyannote khi load
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            config = SpeakerDiarizationConfig(**config_kwargs)
            pipeline = SpeakerDiarization(config)

        source = PushAudioSource(self.sample_rate)

        inference = StreamingInference(
            pipeline,
            source,
            do_profile=False,
            do_plot=False,
            show_progress=False,
        )

        def _on_prediction(args):
            # args = (prediction: Annotation, waveform: SlidingWindowFeature)
            try:
                pred, _ = args
            except (TypeError, ValueError):
                pred = args
            self._absorb_annotation(pred)

        # attach_hooks nhận Callable (attach_observers chỉ nhận rx.Observer)
        inference.attach_hooks(_on_prediction)

        # inference() BLOCK trong read() → chạy thread riêng.
        # diart XỬ LÝ và phát prediction TRONG thread này → phải join nó khi stop,
        # nếu không sẽ đọc _windows trước khi diart kịp tính → 0 windows.
        # inference() TRẢ VỀ annotation tích luỹ đầy đủ (accumulator) → lưu lại
        # làm kết quả cuối authoritative.
        def _run_inference():
            try:
                self._final_annotation = inference()
            except Exception as e:
                print(f"[DiartStreamer] inference crash: {e}", flush=True)

        infer_thread = threading.Thread(
            target=_run_inference,
            daemon=True,
            name="diart-inference",
        )
        infer_thread.start()

        return pipeline, source, infer_thread

    # ── Absorb diart predictions → _windows ────────────────────────────────
    def _absorb_annotation(self, annotation) -> None:
        """Convert pyannote Annotation từ diart thành (start, end, SPEAKER_NN)."""
        self._n_predictions += 1     # đếm hook fire (kể cả annotation rỗng)
        if annotation is None:
            return

        try:
            new = []
            for segment, _, label in annotation.itertracks(yield_label=True):
                spk = self._normalize_label(label)
                new.append(_Window(
                    start=round(float(segment.start), 3),
                    end=round(float(segment.end), 3),
                    speaker=spk,
                ))
        except Exception as e:
            print(f"[DiartStreamer] absorb error: {e}", flush=True)
            return

        if not new:
            return

        with self._lock:
            # diart phát prediction INCREMENTAL: mỗi lần chỉ là slice ~0.5s hiện tại,
            # KHÔNG phải toàn bộ lịch sử → phải TÍCH LUỸ rồi gộp đoạn liền kề cùng speaker.
            for w in new:
                if (self._windows
                        and self._windows[-1].speaker == w.speaker
                        and w.start - self._windows[-1].end <= 0.75):
                    # nối tiếp cùng speaker → mở rộng đoạn cuối
                    self._windows[-1].end = max(self._windows[-1].end, w.end)
                elif (self._windows
                        and self._windows[-1].start == w.start
                        and self._windows[-1].speaker == w.speaker):
                    # cùng slice lặp lại → bỏ qua
                    continue
                else:
                    self._windows.append(w)

        if self._on_update:
            try:
                self._on_update(self.get_windows())
            except Exception:
                pass

    @staticmethod
    def _normalize_label(label) -> str:
        """diart trả 'speaker0' / int / 'A' → ép về 'SPEAKER_00'."""
        if isinstance(label, int):
            return f"SPEAKER_{label:02d}"
        s = str(label)
        # 'speaker0' → 0, 'SPEAKER_0' → 0
        digits = "".join(c for c in s if c.isdigit())
        if digits:
            return f"SPEAKER_{int(digits):02d}"
        # fallback: map chữ A/B/C → 0/1/2
        s = s.strip().upper()
        if len(s) == 1 and s.isalpha():
            return f"SPEAKER_{ord(s) - ord('A'):02d}"
        return s


# ────────────────────────────────────────────────────────────────────────────
# OFFLINE wrapper — dùng diart cho file (Batch Mode), thay cho Nexa/pyannote.
# Chữ ký tương thích diarize_file_nexa / diarize_file để Batch Mode gọi thẳng.
# ────────────────────────────────────────────────────────────────────────────
def load_diarizer_diart():
    """Stub cho tương thích load_diarizer() của Batch Mode (diart tạo nội bộ)."""
    return {"backend": "diart"}


def diarize_file_diart(
    pipeline=None,                 # không dùng — giữ chữ ký tương thích
    wav_path: str = "",
    num_speakers: int | None = None,
    min_duration: float = 0.5,
    merge_gap: float = 2.5,
    do_recluster: bool = True,
) -> list:
    """Diarize OFFLINE 1 file bằng diart (feed nhanh, không sleep) + final re-cluster.

    Trả list[SpeakerSegment] — drop-in cho diarize_file_nexa trong Batch Mode.
    Tự downmix stereo → mono. Cần WAV 16kHz 16-bit.
    """
    import wave as _wave

    diar = DiartStreamingDiarizer(sample_rate=16000, num_speakers=num_speakers)
    diar.start()
    try:
        with _wave.open(wav_path, "rb") as wf:
            sr   = wf.getframerate()
            n_ch = wf.getnchannels()
            sw   = wf.getsampwidth()
            if sw != 2 or sr != 16000:
                raise ValueError(
                    f"diart cần WAV 16kHz 16-bit. File: {sr}Hz {sw*8}-bit. "
                    f"Convert: ffmpeg -i in.wav -ar 16000 -ac 1 out.wav"
                )
            while True:
                frames = wf.readframes(3200)   # 200ms
                if not frames:
                    break
                chunk = np.frombuffer(frames, dtype=np.int16)
                if n_ch == 2:
                    chunk = chunk.reshape(-1, 2).mean(axis=1).astype(np.int16)
                diar.feed(chunk)               # feed nhanh — không sleep
    finally:
        windows = diar.stop(timeout=180.0)

    if not windows:
        return []

    if do_recluster:
        try:
            from core.pipeline.postprocess import final_recluster
            return final_recluster(wav_path, windows, num_speakers=num_speakers)
        except Exception as e:
            print(f"[diart-batch] final_recluster lỗi → dùng windows thô: {e}", flush=True)

    return [SpeakerSegment(speaker=spk, start=s, end=e) for s, e, spk in windows]


# ────────────────────────────────────────────────────────────────────────────
# Smoke test khi chạy trực tiếp: python core/diarization/streaming.py
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import wave

    if len(sys.argv) < 2:
        print("Dùng: python core/diarization/streaming.py <file.wav> [num_speakers]")
        print("  num_speakers: số người nói (ghim cứng). Bỏ trống = tự dò.")
        sys.exit(0)

    wav_path = sys.argv[1]

    # arg 2 (tuỳ chọn): ghim số người nói. Vd test 3 người: ... file.wav 3
    num_speakers = int(sys.argv[2]) if len(sys.argv) > 2 else None

    # DIART_TEST_FAST=1 → bỏ sleep, stream nhanh nhất có thể (test nhanh)
    fast = os.getenv("DIART_TEST_FAST", "0") == "1"

    with wave.open(wav_path, "rb") as wf0:
        total_sec = wf0.getnframes() / wf0.getframerate()
    spk_msg = f"{num_speakers} người (ghim)" if num_speakers else "tự dò"
    print(f"Đang stream {wav_path} ({total_sec:.1f}s audio) vào diart... [speakers: {spk_msg}]")
    if not fast:
        print(f"⏱  Stream ở tốc độ THỰC (1×) → sẽ mất ~{total_sec:.0f}s. "
              f"Đặt DIART_TEST_FAST=1 để chạy nhanh.")

    # In live mỗi khi diart phát prediction mới → CHỨNG MINH real-time hoạt động
    _last_print = [0.0]
    def _on_update(windows):
        now = time.perf_counter()
        if now - _last_print[0] < 1.0:    # throttle 1 dòng/giây
            return
        _last_print[0] = now
        if windows:
            s, e, spk = windows[-1]
            speakers = sorted({w[2] for w in windows})
            print(f"  [live] {len(windows):3d} windows | speakers={speakers} "
                  f"| mới nhất: {spk} [{s:.1f}s→{e:.1f}s]", flush=True)

    diarizer = DiartStreamingDiarizer(
        sample_rate=16000,
        num_speakers=num_speakers,
        on_update=_on_update,
    )
    diarizer.start()

    with wave.open(wav_path, "rb") as wf:
        sr        = wf.getframerate()
        n_ch      = wf.getnchannels()
        sampwidth = wf.getsampwidth()

        if sampwidth != 2:
            print(f"❌ Cần PCM 16-bit (sampwidth=2), file đang là {sampwidth*8}-bit.\n"
                  f"   Convert: ffmpeg -i in.wav -ar 16000 -ac 1 -sample_fmt s16 out.wav")
            sys.exit(1)
        if sr != 16000:
            print(f"❌ Cần 16kHz, file đang là {sr}Hz.\n"
                  f"   Convert: ffmpeg -i in.wav -ar 16000 -ac 1 out.wav")
            sys.exit(1)
        if n_ch not in (1, 2):
            print(f"❌ Số kênh lạ: {n_ch}. Cần mono/stereo.")
            sys.exit(1)
        if n_ch == 2:
            print(f"ℹ️  File STEREO → tự downmix sang mono (trung bình 2 kênh).")

        chunk_size = 3200    # 200ms (mỗi kênh)
        while True:
            frames = wf.readframes(chunk_size)
            if not frames:
                break
            chunk = np.frombuffer(frames, dtype=np.int16)
            if n_ch == 2:
                # deinterleave [L,R,L,R,...] → (N,2) → mean → mono int16
                chunk = chunk.reshape(-1, 2).mean(axis=1).astype(np.int16)
            diarizer.feed(chunk)
            if not fast:
                time.sleep(0.2)   # mô phỏng real-time 1×

    print("Sending stop... (đợi diart xử lý nốt)")
    windows = diarizer.stop(timeout=30.0)

    print(f"\n=== KẾT QUẢ: {len(windows)} windows ===")
    for s, e, spk in windows[:30]:
        print(f"  [{s:6.2f}s → {e:6.2f}s] {spk}")
    if len(windows) > 30:
        print(f"  ... và {len(windows)-30} windows nữa")
    print(f"\nTổng speakers phát hiện: {sorted({w[2] for w in windows})}")
