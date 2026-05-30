# core/diarization/__init__.py
#
# Public API cho diarization. Mọi code mới import từ đây.
#
# Backends sẵn có:
#   - pyannote   (core.diarization.pyannote_backend) — community-1 / 3.1 / precision-2
#   - nexa       (core.diarization.nexa)             — Nexa SDK
#   - npu        (core.diarization.npu)              — Qualcomm NPU
#   - diart      (core.diarization.streaming)        — real-time streaming
#   - sortformer (core.diarization.sortformer)       — NVIDIA Sortformer (env riêng)
# LƯU Ý: KHÔNG đặt module tên 'pyannote.py' ở đây — sẽ shadow package thật 'pyannote'.
#
# Code cũ vẫn import được từ core.diarizer / core.diarizer_nexa / core.diarizer_npu
# (giữ tương thích ngược trong giai đoạn refactor).

from core.diarization.diar_types import SpeakerSegment
from core.diarization.base  import DiarizerProtocol

__all__ = ["SpeakerSegment", "DiarizerProtocol"]
