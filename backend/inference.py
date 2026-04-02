"""
backend/inference.py
--------------------
Real-time inference pipeline: frame → keypoints → Transformer → word prediction.

Handles:
  - Sliding window buffer (T=32 frames)
  - Temporal voting (require N consecutive same predictions)
  - Confidence thresholding
  - Word accumulation for Gemini
"""

import os
from collections import deque, Counter

import numpy as np
import torch

from models.pose_estimator import RTMPoseEstimator
from models.transformer import load_checkpoint, build_model


class InferencePipeline:
    """
    Stateful per-connection inference pipeline.
    Create one instance per WebSocket connection.

    Args:
        checkpoint_path: path to .pth model file
        num_classes:     number of output classes
        conf_threshold:  minimum confidence to emit a prediction
        vote_count:      consecutive windows needed to confirm a word
        window_size:     number of frames per inference window (T=32)
    """

    NUM_KEYPOINTS = 133
    FEATURE_DIM   = 399   # 133 * 3

    def __init__(
        self,
        checkpoint_path: str = "models/checkpoints/best_model.pth",
        num_classes: int = 100,
        conf_threshold: float = 0.6,
        vote_count: int = 3,
        window_size: int = 32,
    ):
        self.num_classes    = num_classes
        self.conf_threshold = conf_threshold
        self.vote_count     = vote_count
        self.window_size    = window_size

        # ── Device ────────────────────────────────────────────────────────────
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ── Pose estimator ────────────────────────────────────────────────────
        self.pose = RTMPoseEstimator(device=self.device)

        # ── Classifier ────────────────────────────────────────────────────────
        if os.path.exists(checkpoint_path):
            self.model = load_checkpoint(checkpoint_path, num_classes, self.device)
        else:
            print(f"[warn] Checkpoint not found: {checkpoint_path} — using untrained model")
            self.model = build_model(num_classes=num_classes, device=self.device)
        self.model.eval()

        # ── Sliding frame buffer ───────────────────────────────────────────────
        self.frame_buffer: deque = deque(maxlen=window_size)

        # ── Temporal voting ───────────────────────────────────────────────────
        self.vote_buffer: deque = deque(maxlen=vote_count)

        # ── Word accumulation ─────────────────────────────────────────────────
        self.word_buffer: list = []
        self.last_confirmed_word: str = None

    def reset(self):
        """Clear all buffers (called on WebSocket disconnect or user Stop)."""
        self.frame_buffer.clear()
        self.vote_buffer.clear()
        self.word_buffer.clear()
        self.last_confirmed_word = None

    def process_frame(self, frame_bgr: np.ndarray, word_to_idx: dict) -> dict:
        """
        Process a single BGR frame through the full pipeline.

        Args:
            frame_bgr:   np.ndarray (H, W, 3)
            word_to_idx: class index → word mapping

        Returns:
            dict with keys:
              - current_word (str | None)
              - confidence (float | None)
              - confirmed_word (str | None)   — emitted after temporal voting
              - word_buffer (list[str])
              - skeleton_points (list)        — for frontend overlay
        """
        idx_to_word = {v: k for k, v in word_to_idx.items()}

        # ── 1. Pose estimation ────────────────────────────────────────────────
        pose_result = self.pose.predict(frame_bgr)
        keypoints = pose_result.combined  # (133, 3)

        # ── 2. Preprocess frame keypoints ─────────────────────────────────────
        processed = self._preprocess_frame(keypoints)

        # ── 3. Add to sliding window ──────────────────────────────────────────
        self.frame_buffer.append(processed)

        result = {
            "current_word":   None,
            "confidence":     None,
            "confirmed_word": None,
            "word_buffer":    list(self.word_buffer),
            "skeleton_points": self._format_skeleton(pose_result),
        }

        # ── 4. Run classifier when window is full ─────────────────────────────
        if len(self.frame_buffer) < self.window_size:
            return result  # not enough frames yet

        window = np.stack(list(self.frame_buffer), axis=0)  # (32, 399)
        word, confidence = self._classify_window(window, idx_to_word)
        result["current_word"] = word
        result["confidence"]   = confidence

        # ── 5. Temporal voting ────────────────────────────────────────────────
        if word and confidence and confidence >= self.conf_threshold:
            self.vote_buffer.append(word)
        else:
            self.vote_buffer.append(None)

        confirmed = self._check_vote()
        if confirmed and confirmed != self.last_confirmed_word:
            self.word_buffer.append(confirmed)
            self.last_confirmed_word = confirmed
            result["confirmed_word"] = confirmed
            result["word_buffer"]    = list(self.word_buffer)

        return result

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _preprocess_frame(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Single-frame keypoint preprocessing (mirrors data/preprocess.py logic).
        keypoints: (133, 3)  →  returns (399,) flattened
        """
        kp = keypoints.copy()

        # Zero low-confidence
        low_conf = kp[:, 2] < 0.3
        kp[low_conf, :2] = 0

        # Center on torso
        if kp[11, 2] > 0.3 and kp[12, 2] > 0.3:
            torso = (kp[11, :2] + kp[12, :2]) / 2
        else:
            visible = kp[:, 2] >= 0.3
            torso = kp[visible, :2].mean(axis=0) if visible.any() else np.zeros(2)
        kp[:, :2] -= torso

        # Scale
        visible = kp[:, 2] >= 0.3
        if visible.any():
            bbox = kp[visible, :2].max(axis=0) - kp[visible, :2].min(axis=0)
            scale = max(bbox.max(), 1e-6)
            kp[:, :2] /= scale

        return kp.flatten().astype(np.float32)  # (399,)

    def _classify_window(self, window: np.ndarray, idx_to_word: dict) -> tuple:
        """Run Transformer on a (32, 399) window. Returns (word, confidence)."""
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, 32, 399)
        with torch.no_grad():
            logits = self.model(x)
            probs  = torch.softmax(logits, dim=-1)[0]
            top_idx  = probs.argmax().item()
            top_conf = probs[top_idx].item()

        word = idx_to_word.get(top_idx, "UNKNOWN")
        return word, top_conf

    def _check_vote(self) -> str | None:
        """Return the word if it appears consistently in the vote buffer."""
        if len(self.vote_buffer) < self.vote_count:
            return None
        votes = [v for v in self.vote_buffer if v is not None]
        if len(votes) < self.vote_count:
            return None
        most_common, count = Counter(votes).most_common(1)[0]
        return most_common if count >= self.vote_count else None

    def _format_skeleton(self, pose_result) -> list:
        """Format keypoints for JSON transmission to frontend."""
        points = []
        for i, (kp, score) in enumerate(zip(pose_result.keypoints, pose_result.scores)):
            points.append({
                "x": float(kp[0]),
                "y": float(kp[1]),
                "score": float(score),
                "region": _keypoint_region(i),
            })
        return points

    def pop_word_buffer(self) -> list[str]:
        """Return and clear the current word buffer (called after Gemini call)."""
        words = list(self.word_buffer)
        self.word_buffer.clear()
        self.last_confirmed_word = None
        return words


def _keypoint_region(idx: int) -> str:
    if idx < 17:  return "body"
    if idx < 23:  return "feet"
    if idx < 91:  return "face"
    return "hands"
