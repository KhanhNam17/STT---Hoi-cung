# core/diarization/nexa.py
#
# Nexa SDK backend (tuỳ chọn) — re-export từ core.diarizer_nexa.
# Bật bằng env BATCH_DIARIZATION_BACKEND=nexa (cần Nexa CLI/server chạy ngoài).

from core.diarizer_nexa import (
    load_diarizer_nexa,
    diarize_file_nexa,
)
from core.diarization.diar_types import SpeakerSegment

__all__ = [
    "load_diarizer_nexa",
    "diarize_file_nexa",
    "SpeakerSegment",
]
