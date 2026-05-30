# core/diarization/npu.py
#
# Qualcomm NPU backend (sherpa-onnx segmentation + embedding trên QNN).
# Re-export từ core.diarizer_npu.

from core.diarizer_npu import (
    load_diarizer_hybrid,
    diarize_file_hybrid,
)
from core.diarization.diar_types import SpeakerSegment

__all__ = [
    "load_diarizer_hybrid",
    "diarize_file_hybrid",
    "SpeakerSegment",
]
