# core/transcription/base.py
#
# Protocol cho transcriber. Whisper (NPU/CPU) và Zipformer (Phase 3) đều khớp.

from typing import Any, Protocol


class TranscriberProtocol(Protocol):
    """Object có `transcribe(audio_or_path) -> dict` với key tối thiểu:
       - 'text'     : str — full transcript
       - 'segments' : list[dict] — mỗi dict có 'start', 'end', 'text', 'words' (optional)
    """

    def transcribe(self, source: Any, **kwargs) -> dict:
        ...


class StreamingTranscriberProtocol(Protocol):
    """Cho Phase 3 — stream chunks audio, nhận token/text incrementally."""

    def start(self) -> None: ...
    def feed(self, audio_chunk) -> str: ...
    def stop(self) -> str: ...
