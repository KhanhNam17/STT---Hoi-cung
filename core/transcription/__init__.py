# core/transcription/__init__.py
#
# Public API cho transcription. Re-export từ core.transcriber (legacy).
#
# Phase 2/3 sẽ thêm StreamingTranscriberProtocol cho Zipformer live.

from core.transcription.base import TranscriberProtocol

__all__ = ["TranscriberProtocol"]
