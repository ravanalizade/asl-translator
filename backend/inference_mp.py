"""
backend/inference_mp.py
-----------------------
Real-time inference pipeline using MediaPipe Tasks API (v0.10+).
"""

import os
import urllib.request
from collections import deque, Counter

import cv2
import mediapipe as mp
import numpy as np
import torch

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from models.transformer_mp import load_checkpoint_mp, build_model_mp


HAND_MODEL_PATH = 'models/hand_landmarker.task'
HAND_MODEL_URL  = (
    'https://storage.googleapis.com/mediapipe-models/'
    'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
)


def _download_hand_model():
    if not os.path.exists(HAND_MODEL_PATH):
        os.makedirs(os.path.dirname(HAND_MODEL_PATH), exist_ok=True)
        print('[mediapipe] Downloading hand landmarker model...')
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
        print('[mediapipe] Downloaded.')


class InferencePipelineMP:
    NUM_HAND_KP   = 21
    NUM_KEYPOINTS = 42
    FEATURE_DIM   = 126

    def __init__(
        self,
        checkpoint_path: str = 'models/checkpoints_mp/best_model_mp.pth',
        num_classes: int = 100,
        conf_threshold: float = 0.6,
        vote_count: int = 3,
        window_size: int = 32,
    ):
        self.num_classes    = num_classes
        self.conf_threshold = conf_threshold
        self.vote_count     = vote_count
        self.window_size    = window_size
        self.device         = 'cuda' if torch.cuda.is_available() else 'cpu'

        # MediaPipe Hands (Tasks API)
        _download_hand_model()
        base_options = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.hands = mp_vision.HandLandmarker.create_from_options(options)

        # Classifier
        if os.path.exists(checkpoint_path):
            self.model = load_checkpoint_mp(checkpoint_path, num_classes, self.device)
        else:
            print(f'[warn] Checkpoint not found: {checkpoint_path} — using untrained model')
            self.model = build_model_mp(num_classes=num_classes, device=self.device)
        self.model.eval()

        # Buffers
        self.frame_buffer: deque = deque(maxlen=window_size)
        self.vote_buffer:  deque = deque(maxlen=vote_count)
        self.word_buffer:  list  = []
        self.last_confirmed_word: str = None

        print(f'[inference-mp] Ready on {self.device}')

    def reset(self):
        self.frame_buffer.clear()
        self.vote_buffer.clear()
        self.word_buffer.clear()
        self.last_confirmed_word = None

    def process_frame(self, frame_bgr: np.ndarray, word_to_idx: dict) -> dict:
        idx_to_word = {v: k for k, v in word_to_idx.items()}

        keypoints, skeleton_points = self._extract_keypoints(frame_bgr)
        processed = self._preprocess_frame(keypoints)
        self.frame_buffer.append(processed)

        result = {
            'current_word':    None,
            'confidence':      None,
            'confirmed_word':  None,
            'word_buffer':     list(self.word_buffer),
            'skeleton_points': skeleton_points,
        }

        if len(self.frame_buffer) < self.window_size:
            return result

        window = np.stack(list(self.frame_buffer), axis=0)  # (32, 126)
        word, confidence = self._classify_window(window, idx_to_word)
        result['current_word'] = word
        result['confidence']   = confidence

        if word and confidence and confidence >= self.conf_threshold:
            self.vote_buffer.append(word)
        else:
            self.vote_buffer.append(None)

        confirmed = self._check_vote()
        if confirmed and confirmed != self.last_confirmed_word:
            self.word_buffer.append(confirmed)
            self.last_confirmed_word = confirmed
            result['confirmed_word'] = confirmed
            result['word_buffer']    = list(self.word_buffer)

        return result

    def _extract_keypoints(self, frame_bgr: np.ndarray):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w      = frame_rgb.shape[:2]
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results   = self.hands.detect(mp_image)

        left_kp  = np.zeros((self.NUM_HAND_KP, 3), dtype=np.float32)
        right_kp = np.zeros((self.NUM_HAND_KP, 3), dtype=np.float32)
        skeleton_points = []

        if results.hand_landmarks and results.handedness:
            for hand_landmarks, handedness in zip(
                results.hand_landmarks, results.handedness
            ):
                label = handedness[0].category_name
                kp    = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
                    dtype=np.float32,
                )
                if label == 'Left':
                    left_kp = kp
                else:
                    right_kp = kp

                for lm in hand_landmarks:
                    skeleton_points.append({
                        'x':      lm.x * w,
                        'y':      lm.y * h,
                        'score':  1.0,
                        'region': 'hands',
                    })

        keypoints = np.concatenate([left_kp, right_kp], axis=0)  # (42, 3)
        return keypoints, skeleton_points

    def _preprocess_frame(self, keypoints: np.ndarray) -> np.ndarray:
        kp = keypoints.copy()
        for hand_start, wrist_idx in [(0, 0), (21, 21)]:
            hand  = kp[hand_start:hand_start + self.NUM_HAND_KP, :2]
            wrist = kp[wrist_idx, :2]
            if np.any(hand != 0):
                hand  -= wrist
                scale  = max(np.abs(hand).max(), 1e-6)
                hand  /= scale
                kp[hand_start:hand_start + self.NUM_HAND_KP, :2] = hand
        return kp.flatten().astype(np.float32)

    def _classify_window(self, window: np.ndarray, idx_to_word: dict):
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits   = self.model(x)
            probs    = torch.softmax(logits, dim=-1)[0]
            top_idx  = probs.argmax().item()
            top_conf = probs[top_idx].item()
        word = idx_to_word.get(top_idx, 'UNKNOWN')
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
