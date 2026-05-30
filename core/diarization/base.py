# core/diarization/base.py
#
# Protocol interface cho mọi diarizer backend.
# Pyannote, Nexa, NPU đều thoả mãn protocol này.
#
# Mục đích: code downstream (pipeline, voiceprints) gọi `diarizer.diarize(wav)`
# mà không cần biết backend cụ thể. Phase 3 (live mode) sẽ thêm
# StreamingDiarizerProtocol cho diart.

from typing import Protocol, runtime_checkable

from core.diarization.diar_types import SpeakerSegment


@runtime_checkable
class DiarizerProtocol(Protocol):
    """Bất cứ object nào có method `diarize(wav_path) -> list[SpeakerSegment]`
    đều dùng được làm diarizer trong pipeline."""

    def diarize(
        self,
        wav_path: str,
        num_speakers: int | None = None,
    ) -> list[SpeakerSegment]:
        ...


class StreamingDiarizerProtocol(Protocol):
    """Cho Phase 3 — live mode. Backend phát labels khi audio đang chảy vào."""

    def start(self) -> None: ...
    def feed(self, audio_chunk) -> list[SpeakerSegment]: ...
    def stop(self) -> list[SpeakerSegment]: ...
