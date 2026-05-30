# core/transcription/whisper.py
#
# Whisper backend (Qualcomm NPU + OpenAI CPU fallback).
# Re-export từ core.transcriber.

from core.transcriber import (
    load_model,
    transcribe_file,
    get_duration,
)

__all__ = ["load_model", "transcribe_file", "get_duration"]
