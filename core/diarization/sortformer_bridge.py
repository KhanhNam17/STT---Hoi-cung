# core/diarization/sortformer_bridge.py
#
# Lớp gọi Sortformer cho phần còn lại của app — có 2 đường:
#
# 1) ĐƯỜNG NHANH (mặc định cho demo one-env):
#    Nếu app chạy TRONG sortformer_env (NeMo có sẵn) → gọi TRỰC TIẾP
#    core.diarization.sortformer.diarize_file_sortformer(). KHÔNG subprocess.
#
# 2) ĐƯỜNG BRIDGE (legacy, khi app ở env khác):
#    Nếu NeMo KHÔNG có trong env hiện tại (vd app ở stt_uc2_env, torch 2.2.2) →
#    subprocess sang SORTFORMER_PYTHON, chạy sortformer.py --json, đọc lại kết quả.
#
# Tự động chọn đường — code gọi không cần biết. Cấu hình SORTFORMER_PYTHON CHỈ cần
# cho đường bridge (env riêng).
#
# Interface DROP-IN giống diarize_file_nexa / diarize_file_diart:
#   load_diarizer_sortformer()  + diarize_file_sortformer(pipeline, wav_path, ...)

import json
import os
import subprocess
import sys
import tempfile

from core.diarization.diar_types import SpeakerSegment

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPT = os.path.join(_ROOT, "core", "diarization", "sortformer.py")


def _sortformer_python() -> str:
    """Đường dẫn python của sortformer_env. Đọc từ env SORTFORMER_PYTHON."""
    py = os.getenv("SORTFORMER_PYTHON", "").strip().strip('"')
    if not py:
        raise RuntimeError(
            "Chưa cấu hình SORTFORMER_PYTHON trong .env.\n"
            "Trỏ tới python.exe của env Sortformer, vd:\n"
            r"  SORTFORMER_PYTHON=C:\Users\QC02\miniconda3\envs\sortformer_env\python.exe"
        )
    if not os.path.exists(py):
        raise RuntimeError(f"SORTFORMER_PYTHON không tồn tại: {py}")
    return py


def load_diarizer_sortformer():
    """Stub tương thích load_diarizer() — chỉ kiểm tra cấu hình sẵn sàng."""
    _sortformer_python()
    return {"backend": "sortformer-bridge"}


def _nemo_in_env() -> bool:
    """NeMo có sẵn ngay trong env hiện tại không (vd chạy app trong sortformer_env)."""
    import importlib.util
    return importlib.util.find_spec("nemo") is not None


def diarize_file_sortformer(
    pipeline=None,
    wav_path: str = "",
    num_speakers: int | None = None,
    min_duration: float = 0.5,
    merge_gap: float = 2.5,
    timeout: float = 1800.0,
) -> list:
    """Diarize bằng Sortformer.

    - Nếu NeMo CÓ trong env hiện tại (chạy app trong sortformer_env) → gọi TRỰC TIẾP,
      không cần subprocess/bridge (one-env demo).
    - Nếu KHÔNG (app ở stt_uc2_env) → subprocess sang SORTFORMER_PYTHON.

    wav_path phải là WAV 16kHz mono.
    """
    # ── Đường nhanh: chạy thẳng trong env hiện tại ───────────────────────────
    if _nemo_in_env():
        from core.diarization.sortformer import diarize_file_sortformer as _direct
        print("[sortformer] NeMo có sẵn in-env → chạy trực tiếp (không bridge)", flush=True)
        return _direct(pipeline=None, wav_path=wav_path, num_speakers=num_speakers,
                       min_duration=min_duration, merge_gap=merge_gap)

    # ── Đường bridge: subprocess sang env Sortformer riêng ───────────────────
    py = _sortformer_python()
    out = tempfile.NamedTemporaryFile(suffix="_sortformer.json", delete=False)
    out.close()

    cmd = [py, _SCRIPT, str(wav_path)]
    if num_speakers:
        cmd.append(str(num_speakers))
    cmd += ["--json", out.name]

    print(f"[sortformer-bridge] → {py}", flush=True)
    print(f"[sortformer-bridge] CMD: {' '.join(cmd)}", flush=True)

    # Ép tiến trình con dùng UTF-8 cho stdout/stderr (Windows mặc định cp1252 →
    # emoji/tiếng Việt trong print sẽ crash tiến trình con).
    _env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

    try:
        proc = subprocess.run(
            cmd, cwd=_ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout, env=_env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Sortformer subprocess lỗi (exit {proc.returncode}).\n"
                f"STDERR:\n{proc.stderr[-1500:]}"
            )
        with open(out.name, "r", encoding="utf-8") as f:
            data = json.load(f)
    finally:
        try:
            os.unlink(out.name)
        except OSError:
            pass

    segs = [SpeakerSegment(speaker=d["speaker"], start=float(d["start"]), end=float(d["end"]))
            for d in data]
    print(f"[sortformer-bridge] ✅ {len({s.speaker for s in segs})} speakers | {len(segs)} đoạn",
          flush=True)
    return segs
