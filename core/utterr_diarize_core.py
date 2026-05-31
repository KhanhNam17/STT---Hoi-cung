import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

# ── Hằng số ───────────────────────────────────────────────────────────────────
MAX_SPEAKERS                    = 10
SAMPLE_RATE                     = 16000

# VAD
VAD_THRESH                      = 0.5

# Cosine similarity để phân loại embedding:
#   sim >= EMBEDDING_UPDATE_THRESHOLD : gán speaker + cập nhật mean
#   PENDING_THRESHOLD <= sim < EMBEDDING_UPDATE_THRESHOLD : gán, không cập nhật
#   sim < PENDING_THRESHOLD : đưa vào pending pool

# FIX-3: Nâng lên 0.60 để khắt khe hơn khi cập nhật Sổ cái (chống Hố đen)
EMBEDDING_UPDATE_THRESHOLD      = 0.55   
PENDING_THRESHOLD               = 0.30   

# Clustering trong pending pool
MIN_PENDING_SIZE                = 3      
MIN_CLUSTER_SIZE                = 3      
AUTO_CLUSTER_DISTANCE_THRESHOLD = 0.65   


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 1: SileroVAD
# ══════════════════════════════════════════════════════════════════════════════
class SileroVAD:
    """
    Phát hiện đoạn có tiếng nói bằng Silero VAD.

    Load từ local directory (models/silero-vad) hoặc torch.hub online.
    Luôn chạy CPU — nhẹ (~2MB), nhanh (~5ms/window).
    """

    def __init__(self, threshold: float = VAD_THRESH):
        self.threshold         = threshold
        self.vad_model         = None
        self.get_speech_ts     = None
        self.model_loaded_flag = False

    def load(self, local_dir: str = "models/silero-vad") -> None:
        """
        Load Silero VAD.
        Ưu tiên local_dir; fallback torch.hub online nếu local không có.
        """
        import os
        print("Đang tải model Silero VAD trên CPU...")

        loaded = False

        # Thử load local trước (offline)
        if os.path.isdir(local_dir) and os.path.exists(
            os.path.join(local_dir, "hubconf.py")
        ):
            try:
                model, utils = torch.hub.load(
                    repo_or_dir  = local_dir,
                    model        = "silero_vad",
                    source       = "local",
                    force_reload = False,
                    onnx         = True,
                    trust_repo   = True,
                )
                loaded = True
                print(f"   Silero VAD: load từ {local_dir} (local)")
            except Exception as e:
                print(f"   Load local thất bại ({e}), fallback torch.hub...")

        # Fallback: torch.hub online
        if not loaded:
            model, utils = torch.hub.load(
                "snakers4/silero-vad", "silero_vad",
                force_reload = False,
                onnx         = False,
            )
            print("   Silero VAD: load từ torch.hub (online)")

        self.vad_model         = model
        self.get_speech_ts     = utils[0]
        self.model_loaded_flag = True
        print("✅Load Silero VAD thành công")

    def _detect_speech(self, audio_data: np.ndarray, sr: int = 16000) -> bool:
        """
        Trả về True nếu window audio chứa tiếng nói.

        Args:
            audio_data : float32 numpy shape (N,), range [-1.0, 1.0]
            sr         : sample rate, phải là 16000
        """
        if not self.model_loaded_flag or self.vad_model is None:
            return False
        if len(audio_data) < 1600:   # < 100ms
            return False

        try:
            audio_tensor = torch.from_numpy(audio_data.astype(np.float32))
            with torch.no_grad():
                timestamps = self.get_speech_ts(
                    audio_tensor,
                    self.vad_model,
                    threshold     = self.threshold,
                    sampling_rate = sr,
                    return_seconds= False,
                )
            return len(timestamps) > 0
        except Exception as e:
            print(f"[VAD] Lỗi detect: {e}")
            return False


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 2: WeSpeakerEncoder
# ══════════════════════════════════════════════════════════════════════════════
class WeSpeakerEncoder:
    """
    Trích xuất speaker embedding 256-dim bằng WeSpeaker ResNet34.

    Model: pyannote/wespeaker-voxceleb-resnet34-LM
    """

    def __init__(self, device: str = "cpu", model_path: str = None):
        """
        Args:
            device     : "cpu" hoặc "cuda"
            model_path : local path hoặc HuggingFace repo id
        """
        self.device            = device
        self.model_path        = model_path or "pyannote/wespeaker-voxceleb-resnet34-LM"
        self.model             = None
        self.inference         = None
        self.model_loaded_flag = False

    def load(self) -> None:
        """Load model — gọi 1 lần trước khi dùng."""
        from pyannote.audio import Model, Inference

        print(f"Loading WeSpeaker model on {self.device.upper()}...")
        self.model = Model.from_pretrained(
            self.model_path,
            use_auth_token = False,
        )
        self.model     = self.model.to(torch.device(self.device))
        self.inference = Inference(self.model, window="whole")
        self.model_loaded_flag = True
        print("WeSpeaker model loaded successfully")

    def _compute_emb(self, audio: np.ndarray, sr: int = 16000):
        """
        Tính embedding cho 1 đoạn audio.

        Args:
            audio : float32 numpy shape (N,), range [-1.0, 1.0]
            sr    : sample rate

        Returns:
            numpy array shape (256,) hoặc None nếu thất bại
        """
        if not self.model_loaded_flag or self.inference is None:
            return None

        try:
            waveform = torch.tensor(audio, dtype=torch.float32)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)   # (1, N)

            audio_dict = {"waveform": waveform, "sample_rate": sr}

            with torch.no_grad():
                emb = self.inference(audio_dict)

            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()

            emb = np.array(emb, dtype=np.float32).flatten()

            if emb.shape[0] == 0 or np.isnan(emb).any():
                return None

            return emb

        except Exception as e:
            print(f"[Encoder] Lỗi compute_emb: {e}")
            return None


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 3: SpeakerHandler
# ══════════════════════════════════════════════════════════════════════════════
class SpeakerHandler:

    def __init__(
        self,
        max_spks      : int   = MAX_SPEAKERS,
        change_thresh : float = PENDING_THRESHOLD,
        min_pending   : int   = MIN_PENDING_SIZE,
    ):
        self.max_spks      = max_spks
        self.change_thresh = change_thresh
        self.min_pending   = min_pending

        self.curr_spk    = None
        self.mean_embs   = [None] * max_spks
        self.spk_embs    = [[] for _ in range(max_spks)]
        self.active_spks = set()

        self.pending_embs  = []
        self.pending_times = []

        self.pending_enabled          = True
        self.embedding_update_enabled = True

        # Callbacks (optional)
        self.embedding_updated = None
        self.timeline_manager  = None

    def set_embedding_callback(self, callback) -> None:
        self.embedding_updated = callback

    def set_timeline_manager(self, tm) -> None:
        self.timeline_manager = tm

    # ── Hàm chính ─────────────────────────────────────────────────────────────
    def classify_spk(self, emb: np.ndarray, seg_time: float) -> tuple:
        """
        Phân loại 1 embedding → (speaker_id, similarity).
        """
        # Bootstrap: chưa có speaker
        # if not self.active_spks and self.pending_enabled:
        #     self.pending_embs.append(emb)
        #     self.pending_times.append(seg_time)
        #     self._check_pending_promotion()
        #     return "pending", 0.0

        if not self.active_spks:
            self.spk_embs[0].append(emb)
            self.mean_embs[0] = emb
            self.active_spks.add(0)
            self.curr_spk = 0
            return 0, 1.0

        # Steady: tính cosine sim với tất cả speaker đã biết
        active_means = []
        active_ids   = []
        for spk_id in self.active_spks:
            if self.mean_embs[spk_id] is not None:
                active_means.append(self.mean_embs[spk_id])
                active_ids.append(spk_id)

        if not active_means:
            self.spk_embs[0].append(emb)
            self.mean_embs[0] = emb
            self.active_spks.add(0)
            self.curr_spk = 0
            return 0, 1.0

        emb_norm   = emb / (np.linalg.norm(emb) + 1e-9)
        means_mat  = np.array(active_means, dtype=np.float32)
        norms      = np.linalg.norm(means_mat, axis=1, keepdims=True)
        means_norm = means_mat / (norms + 1e-9)

        sims     = np.dot(means_norm, emb_norm)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_spk = active_ids[best_idx]

        # sim cao: gán + cập nhật mean
        if best_sim >= EMBEDDING_UPDATE_THRESHOLD:
            self.spk_embs[best_spk].append(emb)
            if self.embedding_update_enabled:
                self.mean_embs[best_spk] = np.median(
                    np.array(self.spk_embs[best_spk]), axis=0
                )
            self.curr_spk = best_spk
            return best_spk, best_sim

        # sim trung bình: gán không cập nhật
        if best_sim >= self.change_thresh:
            self.curr_spk = best_spk
            return best_spk, best_sim

        # sim thấp: pending
        if self.pending_enabled and len(self.active_spks) < self.max_spks:
            self.pending_embs.append(emb)
            self.pending_times.append(seg_time)
            self._check_pending_promotion()
            return "pending", best_sim

        self.curr_spk = best_spk
        return best_spk, best_sim

    # ── Promote pending → speaker mới ─────────────────────────────────────────
    def _check_pending_promotion(self) -> bool:
        if len(self.pending_embs) < MIN_CLUSTER_SIZE:
            return False
        if len(self.active_spks) >= self.max_spks:
            self.pending_enabled = False
            return False

        group_indices = self._find_cohesive_group()   # FIX-1: nhận numpy array
        if group_indices is None or len(group_indices) < MIN_CLUSTER_SIZE:
            return False

        new_spk_id = self._get_next_speaker_id()
        if new_spk_id is None:
            return False

        # Lấy đúng embedding của cluster được promote
        group_embs  = [self.pending_embs[i]  for i in group_indices]
        group_times = [self.pending_times[i] for i in group_indices]

        self.spk_embs[new_spk_id]  = list(group_embs)
        self.mean_embs[new_spk_id] = np.median(np.array(group_embs), axis=0)
        self.active_spks.add(new_spk_id)

        # FIX-1: CHỈ xóa các index được promote, giữ lại phần còn lại
        promoted_set  = set(int(i) for i in group_indices)
        remaining_idx = [i for i in range(len(self.pending_embs))
                         if i not in promoted_set]
        self.pending_embs  = [self.pending_embs[i]  for i in remaining_idx]
        self.pending_times = [self.pending_times[i] for i in remaining_idx]

        # Retro-update nếu có TimelineManager
        if self.timeline_manager and group_times:
            self.timeline_manager.update_pending_segments_to_speaker(
                group_times[0], group_times[-1], new_spk_id
            )

        print(
            f"   [SpeakerHandler] Speaker {new_spk_id} promoted "
            f"({len(group_embs)} embs | pending còn lại: {len(self.pending_embs)})"
        )

        if self.embedding_updated:
            self.embedding_updated()

        return True

    def _find_cohesive_group(self):
        """
        AgglomerativeClustering trên pending_embs.
        Trả về numpy array các INDEX thuộc cluster lớn nhất.   ← FIX-1
        Trả về None nếu không có cluster đủ lớn.
        """
        if len(self.pending_embs) < MIN_CLUSTER_SIZE:
            return None

        try:
            X     = np.array(self.pending_embs, dtype=np.float32)
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X_norm = X / (norms + 1e-9)

            clustering = AgglomerativeClustering(
                n_clusters         = None,
                distance_threshold = AUTO_CLUSTER_DISTANCE_THRESHOLD,
                metric             = "cosine",
                linkage            = "average",
            )
            labels = clustering.fit_predict(X_norm)

            unique_labels = np.unique(labels)
            cluster_sizes = {lbl: int(np.sum(labels == lbl)) for lbl in unique_labels}
            target_lbl    = max(cluster_sizes, key=cluster_sizes.get)

            if cluster_sizes[target_lbl] >= MIN_CLUSTER_SIZE:
                return np.where(labels == target_lbl)[0]   # FIX-1: full index array

        except Exception as e:
            print(f"[SpeakerHandler] Clustering error: {e}")

        return None

    def _get_next_speaker_id(self):
        for i in range(self.max_spks):
            if i not in self.active_spks:
                return i
        return None

    # ── Recluster toàn bộ sau khi dừng ghi ───────────────────────────────────
    def recluster_spks(self, target_clusters: int = None) -> bool:
        """
        Recluster toàn bộ embedding đã thu thập.
        Gọi trong diarizer_stop() để cải thiện accuracy sau khi kết thúc session.
        """
        all_embs = []
        for embs in self.spk_embs:
            all_embs.extend(embs)
        all_embs.extend(self.pending_embs)

        if len(all_embs) < 2:
            return False

        n_clusters = min(
            target_clusters if target_clusters else max(len(self.active_spks), 1),
            len(all_embs),
            self.max_spks,
        )
        if n_clusters < 1:
            return False

        # FIX-4: Chuẩn hóa Vector và đồng bộ sử dụng Cosine Metric giống Real-time
        X = np.array(all_embs, dtype=np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_norm = X / (norms + 1e-9)

        try:
            clustering = AgglomerativeClustering(
                n_clusters = n_clusters,
                metric     = "cosine",    # Đổi từ euclidean sang cosine
                linkage    = "average",   # Đổi từ ward sang average
            )
            labels = clustering.fit_predict(X_norm)
        except Exception as e:
            print(f"[SpeakerHandler] Recluster error: {e}")
            return False

        self.spk_embs    = [[] for _ in range(self.max_spks)]
        self.mean_embs   = [None] * self.max_spks
        self.active_spks = set()
        self.pending_embs  = []
        self.pending_times = []

        for emb, lbl in zip(all_embs, labels):
            if lbl < self.max_spks:
                self.spk_embs[lbl].append(emb)
                self.active_spks.add(lbl)

        for i, embs in enumerate(self.spk_embs):
            if embs:
                self.mean_embs[i] = np.median(np.array(embs), axis=0)

        print(
            f"   [SpeakerHandler] Recluster xong | "
            f"active={sorted(self.active_spks)} | n_clusters={n_clusters}"
        )
        if self.embedding_updated:
            self.embedding_updated()

        return True

    # ── Tiện ích ──────────────────────────────────────────────────────────────
    def toggle_pending(self) -> bool:
        self.pending_enabled = not self.pending_enabled
        return self.pending_enabled

    def toggle_embedding_update(self) -> bool:
        self.embedding_update_enabled = not self.embedding_update_enabled
        return self.embedding_update_enabled

    def get_all_embeddings(self) -> tuple:
        """(all_embs_array, labels) — dùng để visualize PCA."""
        all_embs, labels = [], []
        for spk_id in range(self.max_spks):
            if spk_id in self.active_spks:
                for emb in self.spk_embs[spk_id]:
                    all_embs.append(emb); labels.append(spk_id)
        for emb in self.pending_embs:
            all_embs.append(emb); labels.append(-1)
        return (np.array(all_embs) if all_embs else None), labels

    def reset(self) -> None:
        self.curr_spk             = None
        self.mean_embs            = [None] * self.max_spks
        self.spk_embs             = [[] for _ in range(self.max_spks)]
        self.active_spks          = set()
        self.pending_embs         = []
        self.pending_times        = []
        self.pending_enabled      = True
        if self.embedding_updated:
            self.embedding_updated()