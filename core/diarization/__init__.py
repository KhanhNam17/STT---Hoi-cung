# core/diarization/__init__.py
#
# Smart Meeting (Live Mode only) — Sortformer là backend duy nhất.
# Diart / pyannote / NPU đã bị loại; module riêng + protocol đã xoá.
#
# Lưu ý: KHÔNG đặt module tên 'pyannote.py' ở đây — sẽ shadow package thật.

from core.diarization.diar_types import SpeakerSegment

__all__ = ["SpeakerSegment"]
