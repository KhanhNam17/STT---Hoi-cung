# core/diarization/pyannote_backend.py
# (đổi tên từ pyannote.py → KHÔNG đặt tên 'pyannote.py' trong package này nữa,
#  vì khi chạy script trong core/diarization/ nó shadow package thật 'pyannote'
#  → vỡ 'import pyannote.core' của NeMo/Sortformer.)
#
# Pyannote backend (community-1 / 3.1 / precision-2).
# Re-export từ core.diarizer (legacy) — facade trong giai đoạn refactor.

from core.diarizer import (
    SpeakerSegment,
    load_diarizer,
    diarize_file,
    validate_wav,
    load_audio,
    postprocess_segments,
    merge_for_transcription,
    rename_speakers,
    get_speaker_stats,
    DIARIZATION_MODEL,
)

__all__ = [
    "SpeakerSegment",
    "load_diarizer",
    "diarize_file",
    "validate_wav",
    "load_audio",
    "postprocess_segments",
    "merge_for_transcription",
    "rename_speakers",
    "get_speaker_stats",
    "DIARIZATION_MODEL",
]
