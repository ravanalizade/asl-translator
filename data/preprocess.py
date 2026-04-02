"""
data/preprocess.py
------------------
Converts raw WLASL video clips into fixed-length keypoint tensors.

Pipeline per clip:
  video.mp4  →  extract frames at 30fps
             →  RTMPose: (num_frames, 133, 3)
             →  pad/truncate to T=32 frames
             →  normalize keypoints (center + scale)
             →  save as .npy  →  shape (32, 399)

Usage:
    python data/preprocess.py [--video-dir data/raw/videos] [--output-dir data/keypoints]
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ─── Config ───────────────────────────────────────────────────────────────────
T = 32                      # fixed temporal window length (frames)
NUM_KEYPOINTS = 133         # RTMPose whole-body keypoints
FEATURE_DIM = NUM_KEYPOINTS * 3   # 399 = x, y, confidence per keypoint
CONF_THRESHOLD = 0.3        # zero out keypoints below this confidence
SMOOTH_WINDOW = 3           # temporal smoothing window size
TARGET_FPS = 30             # resample all videos to this FPS


# ─── RTMPose Wrapper ──────────────────────────────────────────────────────────

def load_rtmpose():
    """
    Load RTMPose-L whole-body model via MMPose.
    Falls back to a dummy estimator if MMPose is not installed
    (so the rest of the pipeline can be tested without GPU).
    """
    try:
        from mmpose.apis import init_model, inference_topdown
        from mmpose.utils import adapt_mmdet_pipeline
        import mmdet

        config_file = (
            "https://raw.githubusercontent.com/open-mmlab/mmpose/main/"
            "projects/rtmpose/rtmpose/wholebody_2d_keypoint/"
            "rtmpose-l_8xb32-270e_coco-wholebody-384x288.py"
        )
        checkpoint = (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
            "rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth"
        )

        model = init_model(config_file, checkpoint, device="cuda:0" if _has_cuda() else "cpu")
        print("[ok] RTMPose-L loaded (MMPose)")
        return ("mmpose", model, inference_topdown)

    except ImportError:
        warnings.warn(
            "MMPose not installed — using DUMMY keypoint estimator.\n"
            "Install mmpose for real keypoints: pip install mmpose mmdet mmcv mmengine",
            stacklevel=2,
        )
        return ("dummy", None, None)


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def extract_keypoints_rtmpose(frames: list, model_bundle: tuple) -> np.ndarray:
    """
    Run RTMPose on a list of BGR frames.

    Args:
        frames: list of np.ndarray (H, W, 3), BGR
        model_bundle: return value of load_rtmpose()

    Returns:
        keypoints: np.ndarray of shape (num_frames, 133, 3)  — (x, y, conf)
    """
    mode, model, inference_fn = model_bundle
    num_frames = len(frames)

    if mode == "dummy":
        # Return random keypoints for pipeline testing
        return np.random.rand(num_frames, NUM_KEYPOINTS, 3).astype(np.float32)

    all_keypoints = []
    for frame in frames:
        # inference_topdown expects RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Provide a dummy bounding box covering the whole frame
        h, w = frame_rgb.shape[:2]
        bboxes = np.array([[0, 0, w, h, 1.0]])  # xyxy + score

        results = inference_fn(model, frame_rgb, bboxes)
        if results and len(results) > 0:
            kps = results[0].pred_instances.keypoints[0]       # (133, 2)
            scores = results[0].pred_instances.keypoint_scores[0]  # (133,)
            kp_with_conf = np.concatenate([kps, scores[:, None]], axis=1)  # (133, 3)
        else:
            kp_with_conf = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)

        all_keypoints.append(kp_with_conf)

    return np.array(all_keypoints, dtype=np.float32)  # (N, 133, 3)


# ─── Video Utilities ──────────────────────────────────────────────────────────

def load_video_frames(video_path: str, target_fps: int = TARGET_FPS) -> list:
    """
    Load all frames from a video file, resampled to target_fps.

    Returns:
        list of np.ndarray (H, W, 3) BGR
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Build index list for resampling
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


# ─── Keypoint Preprocessing ───────────────────────────────────────────────────

def zero_low_confidence(keypoints: np.ndarray, threshold: float = CONF_THRESHOLD) -> np.ndarray:
    """Set (x, y) to 0 for keypoints below confidence threshold."""
    kp = keypoints.copy()
    low_conf_mask = kp[:, :, 2] < threshold  # (T, 133)
    kp[low_conf_mask, 0] = 0.0
    kp[low_conf_mask, 1] = 0.0
    return kp


def center_and_scale(keypoints: np.ndarray) -> np.ndarray:
    """
    Normalize each frame's keypoints:
      - Translate so torso midpoint (avg of left/right hip = keypoints 11,12) = (0, 0)
      - Scale so bounding box of all visible points fits in [-1, 1]

    Keypoint indices (COCO body subset within RTMPose 133-pt):
      11 = left hip, 12 = right hip

    Args:
        keypoints: (T, 133, 3)
    Returns:
        normalized: (T, 133, 3)  — conf channel unchanged
    """
    kp = keypoints.copy()
    T_len = kp.shape[0]

    for t in range(T_len):
        frame_kp = kp[t]  # (133, 3)
        visible = frame_kp[:, 2] >= CONF_THRESHOLD
        xy = frame_kp[:, :2]

        # ── Translation: center on torso midpoint ──
        left_hip  = frame_kp[11, :2]
        right_hip = frame_kp[12, :2]
        if frame_kp[11, 2] > CONF_THRESHOLD and frame_kp[12, 2] > CONF_THRESHOLD:
            torso_center = (left_hip + right_hip) / 2.0
        elif visible.any():
            torso_center = xy[visible].mean(axis=0)
        else:
            torso_center = np.array([0.0, 0.0])

        kp[t, :, :2] -= torso_center

        # ── Scale: normalize to unit bounding box ──
        if visible.any():
            visible_xy = kp[t, visible, :2]
            bbox_size = visible_xy.max(axis=0) - visible_xy.min(axis=0)
            scale = max(bbox_size.max(), 1e-6)
            kp[t, :, :2] /= scale

    return kp


def temporal_smooth(keypoints: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Apply moving average along the time axis (x and y only, not confidence)."""
    kp = keypoints.copy()
    half = window // 2
    T_len = kp.shape[0]

    for t in range(T_len):
        start = max(0, t - half)
        end   = min(T_len, t + half + 1)
        kp[t, :, :2] = keypoints[start:end, :, :2].mean(axis=0)

    return kp


def pad_or_truncate(keypoints: np.ndarray, target_T: int = T) -> np.ndarray:
    """
    Resize temporal dimension to exactly target_T frames.
      - If too short: pad with zeros at the end
      - If too long:  uniformly sample target_T frames
    """
    current_T = keypoints.shape[0]

    if current_T == target_T:
        return keypoints

    if current_T < target_T:
        # Zero-pad
        pad = np.zeros((target_T - current_T, NUM_KEYPOINTS, 3), dtype=np.float32)
        return np.concatenate([keypoints, pad], axis=0)
    else:
        # Uniform temporal sampling
        indices = np.round(np.linspace(0, current_T - 1, target_T)).astype(int)
        return keypoints[indices]


def preprocess_keypoints(raw_keypoints: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline for a single clip's keypoints.

    Args:
        raw_keypoints: (num_frames, 133, 3)  — raw RTMPose output
    Returns:
        processed: (32, 399)  — ready for Transformer input
    """
    kp = zero_low_confidence(raw_keypoints)
    kp = center_and_scale(kp)
    kp = temporal_smooth(kp)
    kp = pad_or_truncate(kp, target_T=T)
    # Flatten last two dims: (32, 133, 3) → (32, 399)
    return kp.reshape(T, FEATURE_DIM).astype(np.float32)


# ─── Main Processing Loop ─────────────────────────────────────────────────────

def process_dataset(video_dir: Path, output_dir: Path, model_bundle: tuple):
    """
    Walk video_dir/<GLOSS>/<video_id>.mp4 and produce
    output_dir/<GLOSS>/<video_id>.npy files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    word_dirs = sorted([d for d in video_dir.iterdir() if d.is_dir()])
    print(f"Found {len(word_dirs)} word directories under {video_dir}\n")

    stats = {"ok": 0, "failed": 0, "skipped": 0}

    for word_dir in word_dirs:
        gloss = word_dir.name
        out_word_dir = output_dir / gloss
        out_word_dir.mkdir(exist_ok=True)

        video_files = list(word_dir.glob("*.mp4"))
        for video_path in tqdm(video_files, desc=gloss, leave=False):
            out_path = out_word_dir / (video_path.stem + ".npy")

            if out_path.exists():
                stats["skipped"] += 1
                continue

            try:
                frames = load_video_frames(str(video_path))
                if len(frames) == 0:
                    raise ValueError("Empty video")

                raw_kp = extract_keypoints_rtmpose(frames, model_bundle)
                processed = preprocess_keypoints(raw_kp)

                np.save(str(out_path), processed)
                stats["ok"] += 1

            except Exception as e:
                tqdm.write(f"[FAIL] {video_path.name}: {e}")
                stats["failed"] += 1

    print(f"\n{'='*50}")
    print(f"  Preprocessing complete")
    print(f"  OK:      {stats['ok']}")
    print(f"  Failed:  {stats['failed']}")
    print(f"  Skipped: {stats['skipped']} (already processed)")
    print(f"  Output:  {output_dir}")
    print(f"{'='*50}")


def verify_output(output_dir: Path):
    """Quick sanity check on saved .npy files."""
    npy_files = list(output_dir.rglob("*.npy"))
    if not npy_files:
        print("[warn] No .npy files found!")
        return

    sample = np.load(str(npy_files[0]))
    print(f"\nSample file: {npy_files[0].name}")
    print(f"  Shape:  {sample.shape}  (expected: ({T}, {FEATURE_DIM}))")
    print(f"  dtype:  {sample.dtype}")
    print(f"  min:    {sample.min():.4f}")
    print(f"  max:    {sample.max():.4f}")
    print(f"  mean:   {sample.mean():.4f}")
    print(f"\nTotal .npy files: {len(npy_files)}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess WLASL videos to keypoint tensors")
    parser.add_argument("--video-dir",  default="data/raw/videos",   help="Root dir with <GLOSS>/<id>.mp4")
    parser.add_argument("--output-dir", default="data/keypoints",    help="Where to save .npy files")
    parser.add_argument("--verify",     action="store_true",          help="Only verify existing .npy files")
    args = parser.parse_args()

    video_dir  = Path(args.video_dir)
    output_dir = Path(args.output_dir)

    if args.verify:
        verify_output(output_dir)
        return

    print("=" * 50)
    print("  ASL Keypoint Preprocessor")
    print(f"  Input:  {video_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Window: T={T} frames | {FEATURE_DIM} features")
    print("=" * 50 + "\n")

    model_bundle = load_rtmpose()
    process_dataset(video_dir, output_dir, model_bundle)
    verify_output(output_dir)


if __name__ == "__main__":
    main()
