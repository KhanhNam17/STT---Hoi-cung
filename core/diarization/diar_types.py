# core/diarization/diar_types.py
#
# Kiểu dữ liệu dùng chung cho mọi diarizer backend.
# Re-export từ core.diarizer (legacy) để không tạo 2 nguồn truth.

from core.diarizer import SpeakerSegment

__all__ = ["SpeakerSegment"]
