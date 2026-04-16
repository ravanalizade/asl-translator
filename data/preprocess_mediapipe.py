"""
data/preprocess_mediapipe.py
-----------------------------
Converts raw WLASL video clips into fixed-length hand keypoint tensors
using MediaPipe instead of RTMPose.

Pipeline per clip:
  video.mp4  →  extract frames at 30fps
             →  MediaPipe Hands: (num_frames, 42, 3)
             →  pad/truncate to T=32 frames
             →  normalize keypoints
             →  save as .npy  →  shape (32, 126)

Key difference from RTMPose version:
  RTMPose: 133 keypoints × 3 = 399 features
  MediaPipe: 42 hand keypoints × 3 = 126 features

Usage:
    python data/preprocess_mediapipe.py
    python data/preprocess_mediapipe.py --video-dir data/raw/videos --output-dir data/keypoints_mp
"""

import argparse
import warnings
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm


# ─── Config ───────────────────────────────────────────────────────────────────
T              = 32
NUM_HANDS      = 2
NUM_HAND_KP    = 21
NUM_KEYPOINTS  = NUM_HANDS * NUM_HAND_KP   # 42
FEATURE_DIM    = NUM_KEYPOINTS * 3          # 126
CONF_THRESHOLD = 0.3
SMOOTH_WINDOW  = 3
TARGET_FPS     = 30


# ─── MediaPipe Setup ──────────────────────────────────────────────────────────

def load_mediapipe():
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return hands


def extract_keypoints_mediapipe(frames: list, hands) -> np.ndarray:
    """
    Run MediaPipe Hands on a list of BGR frames.

    Returns:
        keypoints: (num_frames, 42, 3)  — x, y, z per keypoint
                   Left hand  → indices 0-20
                   Right hand → indices 21-41
                   Missing hand → all zeros
    """
    all_keypoints = []

    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        results = hands.process(frame_rgb)

        left_kp  = np.zeros((NUM_HAND_KP, 3), dtype=np.float32)
        right_kp = np.zeros((NUM_HAND_KP, 3), dtype=np.float32)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = handedness.classification[0].label
                kp = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                    dtype=np.float32,
                )
                if label == "Left":
                    left_kp = kp
                else:
                    right_kp = kp

        frame_kp = np.concatenate([left_kp, right_kp], axis=0)  # (42, 3)
        all_keypoints.append(frame_kp)

    return np.array(all_keypoints, dtype=np.float32)  # (N, 42, 3)


# ─── Video Utilities ──────────────────────────────────────────────────────────

def load_video_frames(video_path: str, target_fps: int = TARGET_FPS) -> list:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / src_fps
    target_count = max(1, int(duration_sec * target_fps))
    sample_indices = set(
        np.round(np.linspace(0, total_frames - 1, target_count)).astype(int)
    )

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in sample_indices:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


# ─── Preprocessing ────────────────────────────────────────────────────────────

def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Normalize hand keypoints per frame:
    - Center on wrist of each hand
    - Scale to unit bounding box

    Left hand wrist  = keypoint 0
    Right hand wrist = keypoint 21
    """
    kp = keypoints.copy()

    for t in range(kp.shape[0]):
        for hand_start, wrist_idx in [(0, 0), (21, 21)]:
            hand = kp[t, hand_start:hand_start + NUM_HAND_KP, :2]
            wrist = kp[t, wrist_idx, :2]

            is_present = np.any(hand != 0)
            if not is_present:
                continue

            hand -= wrist
            scale = max(np.abs(hand).max(), 1e-6)
            hand /= scale
            kp[t, hand_start:hand_start + NUM_HAND_KP, :2] = hand

    return kp


def temporal_smooth(keypoints: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    kp   = keypoints.copy()
    half = window // 2
    T_len = kp.shape[0]
    for t in range(T_len):
        start = max(0, t - half)
        end   = min(T_len, t + half + 1)
        kp[t] = keypoints[start:end].mean(axis=0)
    return kp


def pad_or_truncate(keypoints: np.ndarray, target_T: int = T) -> np.ndarray:
    current_T = keypoints.shape[0]
    if current_T == target_T:
        return keypoints
    if current_T < target_T:
        pad = np.zeros((target_T - current_T, NUM_KEYPOINTS, 3), dtype=np.float32)
        return np.concatenate([keypoints, pad], axis=0)
    indices = np.round(np.linspace(0, current_T - 1, target_T)).astype(int)
    return keypoints[indices]


def preprocess_keypoints(raw_keypoints: np.ndarray) -> np.ndarray:
    """
    Full preprocessing for one clip.
    Input:  (num_frames, 42, 3)
    Output: (32, 126)
    """
    kp = normalize_keypoints(raw_keypoints)
    kp = temporal_smooth(kp)
    kp = pad_or_truncate(kp, target_T=T)
    return kp.reshape(T, FEATURE_DIM).astype(np.float32)


# ─── Main Processing Loop ─────────────────────────────────────────────────────

def process_dataset(video_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    hands      = load_mediapipe()
    word_dirs  = sorted([d for d in video_dir.iterdir() if d.is_dir()])

    print(f"Found {len(word_dirs)} word directories\n")
    stats = {"ok": 0, "failed": 0, "skipped": 0}

    for word_dir in word_dirs:
        gloss        = word_dir.name
        out_word_dir = output_dir / gloss
        out_word_dir.mkdir(exist_ok=True)

        for video_path in tqdm(list(word_dir.glob("*.mp4")), desc=gloss, leave=False):
            out_path = out_word_dir / (video_path.stem + ".npy")
            if out_path.exists():
                stats["skipped"] += 1
                continue
            try:
                frames    = load_video_frames(str(video_path))
                if not frames:
                    raise ValueError("Empty video")
                raw_kp    = extract_keypoints_mediapipe(frames, hands)
                processed = preprocess_keypoints(raw_kp)
                np.save(str(out_path), processed)
                stats["ok"] += 1
            except Exception as e:
                tqdm.write(f"[FAIL] {video_path.name}: {e}")
                stats["failed"] += 1

    hands.close()
    print(f"\nDone — OK: {stats['ok']} | Failed: {stats['failed']} | Skipped: {stats['skipped']}")
    print(f"Output: {output_dir}")


def verify_output(output_dir: Path):
    npy_files = list(output_dir.rglob("*.npy"))
    if not npy_files:
        print("No .npy files found")
        return
    sample = np.load(str(npy_files[0]))
    print(f"Sample shape : {sample.shape}  (expected: ({T}, {FEATURE_DIM}))")
    print(f"dtype        : {sample.dtype}")
    print(f"range        : [{sample.min():.3f}, {sample.max():.3f}]")
    print(f"Total files  : {len(npy_files)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir",  default="data/raw/videos")
    parser.add_argument("--output-dir", default="data/keypoints_mp")
    parser.add_argument("--verify",     action="store_true")
    args = parser.parse_args()

    video_dir  = Path(args.video_dir)
    output_dir = Path(args.output_dir)

    if args.verify:
        verify_output(output_dir)
        return

    print("=" * 50)
    print("  MediaPipe ASL Preprocessor")
    print(f"  Input:   {video_dir}")
    print(f"  Output:  {output_dir}")
    print(f"  Window:  T={T} | features={FEATURE_DIM}")
    print("=" * 50 + "\n")

    process_dataset(video_dir, output_dir)
    verify_output(output_dir)


if __name__ == "__main__":
    main()
