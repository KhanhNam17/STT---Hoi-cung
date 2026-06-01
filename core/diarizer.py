# core/diarizer.py
#
# Live Mode (Smart Meeting) chỉ cần:
#   - SpeakerSegment dataclass   — đơn vị dữ liệu segment người nói
#   - get_speaker_stats()        — thống kê thời lượng / số turn / phần trăm
#
# Phiên bản trước có pyannote pipeline + diarize_file() đầy đủ cho Batch Mode.
# Đã bỏ vì project chỉ giữ Live Mode + Sortformer.

from dataclasses import dataclass


@dataclass
class SpeakerSegment:
    """1 đoạn nói của 1 người — output của diarizer.

    Attributes:
        speaker : nhãn người nói (vd "SPEAKER_00", hoặc tên thật sau khi rename)
        start   : thời điểm bắt đầu (giây)
        end     : thời điểm kết thúc (giây)
        text    : nội dung (sẽ được điền sau khi alignment với transcript)
    """
    speaker : str
    start   : float
    end     : float
    text    : str = ""


def get_speaker_stats(segments: list[SpeakerSegment]) -> dict:
    """Thống kê cho từng người nói — dùng để hiển thị widget gán tên.

    Returns:
        {"SPEAKER_00": {"duration": 87.4, "turns": 12, "percent": 58.3}, ...}
        đã sắp xếp giảm dần theo duration.
    """
    stats: dict = {}
    for seg in segments:
        dur = seg.end - seg.start
        if seg.speaker not in stats:
            stats[seg.speaker] = {"duration": 0.0, "turns": 0}
        stats[seg.speaker]["duration"] += dur
        stats[seg.speaker]["turns"]    += 1

    total = sum(v["duration"] for v in stats.values())
    for spk in stats:
        stats[spk]["percent"] = round(
            stats[spk]["duration"] / total * 100, 1
        ) if total > 0 else 0.0

    return dict(sorted(stats.items(), key=lambda x: -x[1]["duration"]))
