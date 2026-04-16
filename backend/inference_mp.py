"""
backend/inference_mp.py
-----------------------
Real-time inference pipeline using MediaPipe instead of RTMPose.

Differences from inference.py:
- Uses MediaPipe Hands instead of RTMPoseEstimator
- Input dim: 126 (42 keypoints × 3) instead of 399
- Loads ASLTransformerMP instead of ASLTransformer
- Works on CPU without mmcv
"""

import os
from collections import deque, Counter

import cv2
import mediapipe as mp
import numpy as np
import torch

from models.transformer_mp import load_checkpoint_mp, build_model_mp


class InferencePipelineMP:
    """
    Stateful per-connection MediaPipe inference pipeline.

    Args:
        checkpoint_path: path to best_model_mp.pth
        num_classes:     number of output classes
        conf_threshold:  minimum confidence to emit prediction
        vote_count:      consecutive windows needed to confirm a word
        window_size:     frames per inference window (T=32)
    """

    NUM_HAND_KP  = 21
    NUM_KEYPOINTS = 42     # 21 per hand × 2
    FEATURE_DIM   = 126    # 42 × 3

    def __init__(
        self,
        checkpoint_path: str = "models/checkpoints_mp/best_model_mp.pth",
        num_classes: int = 100,
        conf_threshold: float = 0.6,
        vote_count: int = 3,
        window_size: int = 32,
    ):
        self.num_classes    = num_classes
        self.conf_threshold = conf_threshold
        self.vote_count     = vote_count
        self.window_size    = window_size
        self.device         = "cuda" if torch.cuda.is_available() else "cpu"

        # MediaPipe Hands
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Classifier
        if os.path.exists(checkpoint_path):
            self.model = load_checkpoint_mp(checkpoint_path, num_classes, self.device)
        else:
            print(f"[warn] Checkpoint not found: {checkpoint_path} — using untrained model")
            self.model = build_model_mp(num_classes=num_classes, device=self.device)
        self.model.eval()

        # Buffers
        self.frame_buffer: deque = deque(maxlen=window_size)
        self.vote_buffer:  deque = deque(maxlen=vote_count)
        self.word_buffer:  list  = []
        self.last_confirmed_word: str = None

        print(f"[inference-mp] Ready on {self.device}")

    def reset(self):
        self.frame_buffer.clear()
        self.vote_buffer.clear()
        self.word_buffer.clear()
        self.last_confirmed_word = None

    def process_frame(self, frame_bgr: np.ndarray, word_to_idx: dict) -> dict:
        """
        Process a single BGR frame.

        Returns dict with:
            current_word, confidence, confirmed_word,
            word_buffer, skeleton_points
        """
        idx_to_word = {v: k for k, v in word_to_idx.items()}

        # Extract MediaPipe keypoints
        keypoints, skeleton_points = self._extract_keypoints(frame_bgr)

        # Preprocess
        processed = self._preprocess_frame(keypoints)
        self.frame_buffer.append(processed)

        result = {
            "current_word":   None,
            "confidence":     None,
            "confirmed_word": None,
            "word_buffer":    list(self.word_buffer),
            "skeleton_points": skeleton_points,
        }

        if len(self.frame_buffer) < self.window_size:
            return result

        window = np.stack(list(self.frame_buffer), axis=0)  # (32, 126)
        word, confidence = self._classify_window(window, idx_to_word)
        result["current_word"] = word
        result["confidence"]   = confidence

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

    def _extract_keypoints(self, frame_bgr: np.ndarray):
        """
        Run MediaPipe Hands on frame.
        Returns:
            keypoints: (42, 3)
            skeleton_points: list of dicts for frontend overlay
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w      = frame_rgb.shape[:2]
        results   = self.hands.process(frame_rgb)

        left_kp  = np.zeros((self.NUM_HAND_KP, 3), dtype=np.float32)
        right_kp = np.zeros((self.NUM_HAND_KP, 3), dtype=np.float32)
        skeleton_points = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = handedness.classification[0].label
                kp    = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                    dtype=np.float32,
                )
                if label == "Left":
                    left_kp = kp
                else:
                    right_kp = kp

                for lm in hand_landmarks.landmark:
                    skeleton_points.append({
                        "x":      lm.x * w,
                        "y":      lm.y * h,
                        "score":  1.0,
                        "region": "hands",
                    })

        keypoints = np.concatenate([left_kp, right_kp], axis=0)  # (42, 3)
        return keypoints, skeleton_points

    def _preprocess_frame(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Normalize a single frame's keypoints.
        keypoints: (42, 3) → returns (126,)
        """
        kp = keypoints.copy()

        for hand_start, wrist_idx in [(0, 0), (21, 21)]:
            hand  = kp[hand_start:hand_start + self.NUM_HAND_KP, :2]
            wrist = kp[wrist_idx, :2]
            if np.any(hand != 0):
                hand  -= wrist
                scale  = max(np.abs(hand).max(), 1e-6)
                hand  /= scale
                kp[hand_start:hand_start + self.NUM_HAND_KP, :2] = hand

        return kp.flatten().astype(np.float32)  # (126,)

    def _classify_window(self, window: np.ndarray, idx_to_word: dict):
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits   = self.model(x)
            probs    = torch.softmax(logits, dim=-1)[0]
            top_idx  = probs.argmax().item()
            top_conf = probs[top_idx].item()
        word = idx_to_word.get(top_idx, "UNKNOWN")
        return word, top_conf

    def _check_vote(self):
        if len(self.vote_buffer) < self.vote_count:
            return None
        votes = [v for v in self.vote_buffer if v is not None]
        if len(votes) < self.vote_count:
            return None
        most_common, count = Counter(votes).most_common(1)[0]
        return most_common if count >= self.vote_count else None

    def pop_word_buffer(self) -> list:
        words = list(self.word_buffer)
        self.word_buffer.clear()
        self.last_confirmed_word = None
        return words
