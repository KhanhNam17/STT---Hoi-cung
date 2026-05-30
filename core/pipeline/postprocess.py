# core/pipeline/postprocess.py
#
# Final pass khi kết thúc recording — pattern của Otter:
#   Live pass (diart) chỉ cho labels TƯƠNG ĐỐI, có thể flicker.
#   Final pass: trích xuất embedding cho từng window, clustering toàn cục → labels NHẤT QUÁN.
#
# Tại sao cần?
#   diart clustering ONLINE — không thấy được tương lai → có thể gán speaker_2 cho
#   một người mà sau này lộ ra là speaker_0. Final pass nhìn toàn bộ recording để
#   sửa lại.
#
# Cách dùng từ Live Mode khi user bấm "Hoàn thiện":
#   from core.pipeline.postprocess import final_recluster
#   segments = final_recluster(
#       wav_path="/tmp/live.wav",
#       windows=diarizer.get_windows(),
#       num_speakers=2,
#   )

import os
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np

from core.diarization.diar_types import SpeakerSegment


# Lazy import — pyannote/torch heavy, không load khi module bị import vì lý do khác
def _load_embedder():
    """Trả về (embedder, device). Cache theo process."""
    global _CACHED_EMBEDDER
    try:
        return _CACHED_EMBEDDER
    except NameError:
        pass

    try:
        import torch
        from pyannote.audio import Model, Inference
    except ImportError as e:
        raise RuntimeError(
            "Cần pyannote.audio cho re-clustering. Đã có trong requirements."
        ) from e

    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN không có trong .env — embedder cần token để pull model"
        )

    # Dùng embedding MẠNH wespeaker-voxceleb-resnet34-LM (chính model mà
    # pyannote/speaker-diarization-3.1 dùng) thay cho 'pyannote/embedding' cũ
    # (trained pyannote 0.0.1 / torch 1.8.1) — model cũ không tách nổi 2 giọng
    # → clustering dồn hết vào 1 speaker (98.7%/1.3%).
    emb_name = os.getenv("RECLUSTER_EMBEDDING", "pyannote/wespeaker-voxceleb-resnet34-LM")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = Model.from_pretrained(emb_name, use_auth_token=hf_token)
            model.to(device)
            embedder = Inference(model, window="whole", device=device)
        except Exception as e:
            print(f"[postprocess] Không load được {emb_name} ({e}) → fallback pyannote/embedding",
                  flush=True)
            model = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_token)
            model.to(device)
            embedder = Inference(model, window="whole", device=device)

    globals()["_CACHED_EMBEDDER"] = (embedder, device)
    return embedder, device


def _extract_embeddings(
    wav_path: str,
    windows: Sequence[tuple[float, float, str]],
    min_duration: float = 0.5,
) -> tuple[np.ndarray, list[int]]:
    """Trả về (embeddings [N, D], indices) — indices map về windows gốc.
    Bỏ qua window quá ngắn (< min_duration) vì embedding sẽ kém chất lượng."""
    from pyannote.audio import Audio
    from pyannote.core import Segment

    embedder, _ = _load_embedder()
    audio = Audio(sample_rate=16000, mono=True)

    embs: list[np.ndarray] = []
    idxs: list[int] = []

    for i, (start, end, _spk) in enumerate(windows):
        if end - start < min_duration:
            continue
        try:
            segment = Segment(start=float(start), end=float(end))
            waveform, sr = audio.crop(wav_path, segment)
            emb = embedder({"waveform": waveform, "sample_rate": sr})
            # Inference window="whole" trả về numpy (D,) hoặc (1, D)
            emb = np.asarray(emb).reshape(-1)
            embs.append(emb)
            idxs.append(i)
        except Exception as e:
            print(f"[postprocess] bỏ qua window {i} ({start}-{end}): {e}", flush=True)
            continue

    if not embs:
        return np.empty((0, 0)), []

    # Normalize → cosine distance = 1 - dot product
    arr = np.stack(embs)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms
    return arr, idxs


def _cluster(
    embeddings: np.ndarray,
    num_speakers: int | None,
    distance_threshold: float = 0.7,
) -> np.ndarray:
    """Agglomerative clustering trên cosine distance. Trả về labels [N]."""
    try:
        from sklearn.cluster import AgglomerativeClustering
    except ImportError as e:
        raise RuntimeError(
            "Cần scikit-learn cho clustering. pip install scikit-learn"
        ) from e

    if len(embeddings) == 0:
        return np.array([], dtype=int)
    if len(embeddings) == 1:
        return np.array([0], dtype=int)

    kwargs: dict = dict(metric="cosine", linkage="average")
    if num_speakers and num_speakers > 0:
        kwargs["n_clusters"] = min(num_speakers, len(embeddings))
    else:
        kwargs["n_clusters"] = None
        kwargs["distance_threshold"] = distance_threshold

    clustering = AgglomerativeClustering(**kwargs)
    return clustering.fit_predict(embeddings)


def final_recluster(
    wav_path: str,
    windows: Sequence[tuple[float, float, str]],
    num_speakers: int | None = None,
    distance_threshold: float = 0.7,
    min_duration: float = 0.5,
) -> list[SpeakerSegment]:
    """Final pass: extract embedding cho mọi window, re-cluster, gán label sạch.

    Parameters
    ----------
    wav_path           : file WAV 16kHz mono đã ghi (toàn bộ recording)
    windows            : output của DiartStreamingDiarizer.get_windows()
    num_speakers       : nếu biết, ép cluster về đúng N. None → auto (dùng threshold)
    distance_threshold : ngưỡng cosine distance để tách cluster (chỉ dùng khi num_speakers=None)
    min_duration       : bỏ window ngắn hơn ngưỡng này (không đủ tin cậy)

    Returns
    -------
    list[SpeakerSegment] với speaker đã re-label SPEAKER_00, SPEAKER_01, ...
    """
    if not windows:
        return []

    if not Path(wav_path).exists():
        raise FileNotFoundError(f"WAV không tồn tại: {wav_path}")

    print(f"[postprocess] Re-cluster {len(windows)} windows...", flush=True)

    # Bước 1: extract embeddings
    embs, idxs = _extract_embeddings(wav_path, windows, min_duration=min_duration)

    if len(embs) == 0:
        # Quá ít data — fallback: giữ nguyên labels từ diart
        print("[postprocess] Không trích được embedding nào — giữ labels diart", flush=True)
        return [
            SpeakerSegment(speaker=spk, start=float(s), end=float(e))
            for s, e, spk in windows
        ]

    # Bước 2: cluster
    labels = _cluster(embs, num_speakers=num_speakers,
                      distance_threshold=distance_threshold)
    n_clusters = len(set(labels.tolist()))
    print(f"[postprocess] Cluster → {n_clusters} speakers từ {len(embs)} segments hợp lệ",
          flush=True)

    # Bước 3: gán label mới cho window đã được clustered
    # Window bị skip (quá ngắn) lấy label của neighbour gần nhất.
    new_labels: list[str | None] = [None] * len(windows)
    for emb_pos, win_idx in enumerate(idxs):
        new_labels[win_idx] = f"SPEAKER_{int(labels[emb_pos]):02d}"

    # Điền chỗ trống bằng neighbour gần nhất
    for i, lbl in enumerate(new_labels):
        if lbl is not None:
            continue
        # Tìm trước
        prev_lbl = next((new_labels[j] for j in range(i - 1, -1, -1)
                         if new_labels[j] is not None), None)
        next_lbl = next((new_labels[j] for j in range(i + 1, len(new_labels))
                         if new_labels[j] is not None), None)
        new_labels[i] = prev_lbl or next_lbl or "SPEAKER_00"

    segments = [
        SpeakerSegment(speaker=new_labels[i], start=float(s), end=float(e))
        for i, (s, e, _) in enumerate(windows)
    ]

    # Sort theo thời gian (diart đôi khi trả không theo thứ tự)
    segments.sort(key=lambda x: x.start)
    return segments
