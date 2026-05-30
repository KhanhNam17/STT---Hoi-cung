import os
import time
import numpy as np
import soundfile as sf
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Cấu hình từ file .env
SEG_MODEL_PATH    = os.getenv("SHERPA_SEG_MODEL",         "")
EMBED_MODEL_PATH  = os.getenv("SHERPA_EMBED_MODEL",       "")
CLUSTER_THRESHOLD = float(os.getenv("SHERPA_CLUSTER_THRESHOLD", "0.5"))

SEG_SAMPLE_RATE   = 16000
SEG_WINDOW_SIZE   = 10      # giây
SEG_STEP_SIZE     = 0.5     # bước trượt

EMBED_SAMPLE_RATE = 16000

# Import từ các file utils có sẵn của bạn
from core.diarizer import SpeakerSegment, postprocess_segments, validate_wav


def _build_npu_session(onnx_path: str):
    """Khởi tạo ONNX Session ép chạy trên QNN (Qualcomm NPU) có lưu Cache"""
    import onnxruntime as ort

    if "QNNExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("QNNExecutionProvider không có sẵn trên máy này!")

    # Bật cache để NPU khởi động nhanh (chỉ mất ~1s cho các lần chạy sau)
    ep_options = {
        "backend_path": "QnnHtp.dll",
        "htp_performance_mode": "burst",
        "qnn_context_cache_enable": "1", 
        "qnn_context_cache_path": f"{onnx_path}.cache" 
    }
    
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        onnx_path,
        providers=[("QNNExecutionProvider", ep_options), "CPUExecutionProvider"],
        sess_options=sess_options,
    )
    return session


def load_diarizer_hybrid():
    """Load Hybrid Pipeline: Segmentation (CPU) + Embedding (NPU)"""
    import onnxruntime as ort

    if not SEG_MODEL_PATH or not Path(SEG_MODEL_PATH).exists():
        raise ValueError(f"SHERPA_SEG_MODEL không hợp lệ: '{SEG_MODEL_PATH}'")
    if not EMBED_MODEL_PATH or not Path(EMBED_MODEL_PATH).exists():
        raise ValueError(f"SHERPA_EMBED_MODEL không hợp lệ: '{EMBED_MODEL_PATH}'")

    print("🚀 Loading Hybrid Diarizer...")

    # 1. Segmentation luôn chạy CPU (Tránh lỗi Crash do toán tử động của NPU)
    print(f"   [CPU] Segmentation : {Path(SEG_MODEL_PATH).name}")
    seg_session = ort.InferenceSession(SEG_MODEL_PATH, providers=["CPUExecutionProvider"])

    # 2. Embedding ép chạy bằng NPU (Phần tính toán nặng nhất)
    print(f"   [NPU] Embedding    : {Path(EMBED_MODEL_PATH).name}")
    embed_session = _build_npu_session(EMBED_MODEL_PATH)

    print(f"   Threshold : {CLUSTER_THRESHOLD}")
    print("✅ Hybrid diarizer loaded successfully!")

    return {"seg_session": seg_session, "embed_session": embed_session}


def _run_segmentation(seg_session, waveform: np.ndarray) -> list:
    """Cắt file âm thanh thành các đoạn có tiếng người (Chạy trên CPU)"""
    inp      = seg_session.get_inputs()[0]
    in_name  = inp.name
    in_shape = inp.shape

    window_samples = int(SEG_WINDOW_SIZE * SEG_SAMPLE_RATE)
    step_samples   = int(SEG_STEP_SIZE   * SEG_SAMPLE_RATE)
    total_samples  = len(waveform)

    raw_events  = []
    offset      = 0

    while offset < total_samples:
        chunk = waveform[offset : offset + window_samples]
        
        # Đệm số 0 (padding) nếu chunk cuối cùng bị ngắn
        if len(chunk) < window_samples:
            chunk = np.pad(chunk, (0, window_samples - len(chunk)))
        else:
            chunk = chunk[:window_samples]

        if len(in_shape) == 3:
            chunk_input = chunk[np.newaxis, np.newaxis, :].astype(np.float32)
        else:
            chunk_input = chunk[np.newaxis, :].astype(np.float32)

        outputs = seg_session.run(None, {in_name: chunk_input})
        raw_out = outputs[0]

        if raw_out.ndim == 3:
            out2d = raw_out[0]  
        else:
            out2d = raw_out      

        dim_a, dim_b = out2d.shape

        if dim_a < dim_b:
            probs = out2d.T
        else:
            probs = out2d

        num_frames, num_spk = probs.shape
        frame_duration = SEG_WINDOW_SIZE / num_frames
        chunk_start_s  = offset / SEG_SAMPLE_RATE

        p_max = float(probs.max())
        p_min = float(probs.min())

        # Xử lý Log-Softmax hoặc Logit
        if p_max <= 0.0:
            probs = np.exp(probs.astype(np.float64)).astype(np.float32)
        elif p_max > 1.01 or p_min < -0.01:
            probs = (1.0 / (1.0 + np.exp(-probs.astype(np.float64)))).astype(np.float32)

        active = probs > 0.5

        for spk in range(num_spk):
            in_seg    = False
            seg_start = 0.0
            for f in range(num_frames):
                t = chunk_start_s + f * frame_duration
                if active[f, spk] and not in_seg:
                    seg_start = t
                    in_seg    = True
                elif not active[f, spk] and in_seg:
                    raw_events.append({"start": round(seg_start, 3),
                                       "end":   round(t, 3),
                                       "speaker_idx": spk})
                    in_seg = False
            if in_seg:
                raw_events.append({"start": round(seg_start, 3),
                                   "end":   round(chunk_start_s + SEG_WINDOW_SIZE, 3),
                                   "speaker_idx": spk})

        offset += step_samples

    return raw_events


def _compute_fbank(waveform: np.ndarray, sample_rate: int = 16000, num_mel_bins: int = 80) -> np.ndarray:
    """Biến đổi âm thanh sang Filterbank (Bắt buộc cho NPU)"""
    frame_length = int(sample_rate * 0.025) 
    frame_shift  = int(sample_rate * 0.010) 
    n_fft        = 512

    wav = np.append(waveform[0], waveform[1:] - 0.97 * waveform[:-1])
    num_frames = 1 + (len(wav) - frame_length) // frame_shift
    if num_frames <= 0:
        return None
        
    idx    = (np.arange(frame_length)[None, :] +
              np.arange(num_frames)[:, None] * frame_shift)
    frames = wav[idx] * np.hamming(frame_length)
    mag = np.abs(np.fft.rfft(frames, n=n_fft))  

    def hz2mel(h): return 2595.0 * np.log10(1.0 + h / 700.0)
    def mel2hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    mel_pts = np.linspace(hz2mel(20.0), hz2mel(sample_rate / 2.0), num_mel_bins + 2)
    hz_pts  = mel2hz(mel_pts)
    bins    = np.floor((n_fft + 1) * hz_pts / sample_rate).astype(int)

    fb = np.zeros((n_fft // 2 + 1, num_mel_bins), dtype=np.float32)
    for m in range(1, num_mel_bins + 1):
        fl, fc, fr = bins[m-1], bins[m], bins[m+1]
        for k in range(fl, fc):
            if fc > fl: fb[k, m-1] += (k - fl) / (fc - fl)
        for k in range(fc, fr):
            if fr > fc: fb[k, m-1] += (fr - k) / (fr - fc)

    fbank = np.log(mag @ fb + 1e-10) 
    fbank = (fbank - fbank.mean(0)) / (fbank.std(0) + 1e-10)
    return fbank.astype(np.float32)


def _extract_embedding(embed_session, waveform: np.ndarray, start_s: float, end_s: float):
    """Trích xuất vector giọng nói bằng NPU (Yêu cầu Static Shape 3.0s)"""
    start_i = int(start_s * EMBED_SAMPLE_RATE)
    end_i   = int(end_s   * EMBED_SAMPLE_RATE)
    chunk   = waveform[start_i:end_i].astype(np.float32)

    if len(chunk) < 400:
        return None

    # ÉP STATIC SHAPE 3.0s ĐỂ NPU KHÔNG BỊ CRASH
    FIXED_SAMPLES = int(3.0 * EMBED_SAMPLE_RATE) 
    
    if len(chunk) < FIXED_SAMPLES:
        chunk = np.pad(chunk, (0, FIXED_SAMPLES - len(chunk)))
    else:
        chunk = chunk[:FIXED_SAMPLES]

    inp      = embed_session.get_inputs()[0]
    in_name  = inp.name
    in_shape = inp.shape

    needs_fbank = (len(in_shape) == 3 and str(in_shape[-1]) == '80')

    if needs_fbank:
        fbank = _compute_fbank(chunk, sample_rate=EMBED_SAMPLE_RATE)
        if fbank is None or len(fbank) < 5:
            return None
        model_input = fbank[np.newaxis, :, :]   
    else:
        model_input = chunk[np.newaxis, :]       

    output = embed_session.run(None, {in_name: model_input})[0]
    emb    = output[0] if output.ndim == 2 else output

    # Chuẩn hóa L2
    norm = np.linalg.norm(emb)
    if norm > 1e-8:
        emb = emb / norm
    return emb


def _cluster_embeddings(embeddings: list, threshold: float) -> list:
    """Phân cụm tự động bằng lõi C++ của scikit-learn (Cực nhanh)"""
    if not embeddings:
        return []
    from sklearn.cluster import AgglomerativeClustering
    
    emb_matrix = np.stack(embeddings)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='cosine', 
        linkage='average',
        distance_threshold=threshold
    )
    labels = clustering.fit_predict(emb_matrix)
    return labels.tolist()


def _cluster_kmeans(embeddings: list, k: int) -> list:
    """Phân cụm khi biết trước số người nói bằng scikit-learn"""
    if not embeddings:
        return []
    from sklearn.cluster import KMeans
    
    emb_matrix = np.stack(embeddings)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(emb_matrix)
    return labels.tolist()


def diarize_file_hybrid(
    pipeline     : dict,
    wav_path     : str,
    num_speakers : int   = None,
    min_duration : float = 0.5,
    merge_gap    : float = 2.5,
) -> list:
    """Hàm chạy tổng hợp quá trình Phân rã người nói"""
    valid, reason = validate_wav(wav_path)
    if not valid:
        raise ValueError(f"File không hợp lệ: {reason}")

    seg_session   = pipeline["seg_session"]
    embed_session = pipeline["embed_session"]

    print(f"🎙️ Diarizing: {Path(wav_path).name}")
    t0 = time.perf_counter()

    waveform, sr = sf.read(wav_path, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    print("   [1/3] Segmentation (CPU - Safety First)...")
    t1         = time.perf_counter()
    raw_events = _run_segmentation(seg_session, waveform)
    print(f"         -> Tìm thấy {len(raw_events)} events ({round(time.perf_counter()-t1,2)}s)")

    if not raw_events:
        print("   Không tìm thấy đoạn hội thoại nào.")
        return []

    print("   [2/3] Embedding (NPU - High Performance)...")
    t2           = time.perf_counter()
    valid_events = []
    embeddings   = []
    for ev in raw_events:
        emb = _extract_embedding(embed_session, waveform, ev["start"], ev["end"])
        if emb is not None:
            valid_events.append(ev)
            embeddings.append(emb)
    print(f"         -> Trích xuất {len(embeddings)} vectors ({round(time.perf_counter()-t2,2)}s)")

    print("   [3/3] Clustering (CPU - Scikit-Learn Fast)...")
    t3 = time.perf_counter()
    if num_speakers and num_speakers > 0:
        labels = _cluster_kmeans(embeddings, num_speakers)
    else:
        labels = _cluster_embeddings(embeddings, threshold=CLUSTER_THRESHOLD)
    print(f"         -> Gom thành {len(set(labels))} người nói ({round(time.perf_counter()-t3,2)}s)")

    # Gắn nhãn
    raw_segments = [
        SpeakerSegment(speaker=f"SPEAKER_{label:02d}", start=ev["start"], end=ev["end"])
        for ev, label in zip(valid_events, labels)
    ]
    raw_segments.sort(key=lambda s: s.start)

    # Post-process dọn dẹp kết quả
    segments = postprocess_segments(raw_segments, min_duration=min_duration, merge_gap=merge_gap)
    
    elapsed = round(time.perf_counter() - t0, 2)
    print(f"   ✅ Hoàn tất: {len({s.speaker for s in segments})} người nói | Tổng thời gian xử lý: {elapsed}s")
    
    return segments


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  TEST: diarizer_hybrid.py (CPU Seg + NPU Embed + Fast Cluster)")
    print("=" * 60)

    try:
        pipeline = load_diarizer_hybrid()
    except Exception as e:
        import traceback; traceback.print_exc()
        sys.exit(1)

    if len(sys.argv) < 2:
        print("\nCần truyền file WAV: python core/diarizer_hybrid.py <file_16k.wav>")
        sys.exit(0)

    wav_path = sys.argv[1]
    
    try:
        t_start  = time.perf_counter()
        segments = diarize_file_hybrid(pipeline, wav_path)
        t_total  = round(time.perf_counter() - t_start, 2)
    except Exception as e:
        import traceback; traceback.print_exc()
        sys.exit(1)

    print(f"\n📊 Kết quả ({len(segments)} đoạn):")
    for i, seg in enumerate(segments[:15]):
        bar = "█" * int((seg.end - seg.start) * 2)
        print(f"    [{i+1:02d}] {seg.speaker:12s} | {seg.start:6.2f}s -> {seg.end:6.2f}s | {bar}")

    import wave as _wave
    with _wave.open(wav_path, "rb") as wf:
        dur = wf.getnframes() / wf.getframerate()
    rtf = t_total / dur if dur > 0 else 0
    
    print(f"\n⚡ Hiệu năng:")
    print(f"    Audio gốc  : {dur:.1f}s")
    print(f"    Xử lý hết  : {t_total}s")
    print(f"    RTF        : {rtf:.3f}x")