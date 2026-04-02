"""
models/pose_estimator.py
------------------------
Thin wrapper around RTMPose for real-time keypoint extraction.
Used by the backend inference pipeline (not training — training uses data/preprocess.py).
"""

import warnings
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PoseResult:
    keypoints: np.ndarray   # (133, 2)  — x, y in pixel coords
    scores:    np.ndarray   # (133,)    — confidence per keypoint
    combined:  np.ndarray   # (133, 3)  — x, y, confidence stacked


class RTMPoseEstimator:
    """
    Wraps RTMPose-L Whole-Body for single-frame keypoint extraction.

    Usage:
        estimator = RTMPoseEstimator()
        result = estimator.predict(frame_bgr)
        # result.combined: shape (133, 3)
    """

    # MMPose config + checkpoint (auto-downloaded on first use)
    CONFIG = (
        "https://raw.githubusercontent.com/open-mmlab/mmpose/main/"
        "projects/rtmpose/rtmpose/wholebody_2d_keypoint/"
        "rtmpose-l_8xb32-270e_coco-wholebody-384x288.py"
    )
    CHECKPOINT = (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
        "rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth"
    )
    NUM_KEYPOINTS = 133

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None
        self._inference_fn = None
        self._load_model()

    def _load_model(self):
        try:
            from mmpose.apis import init_model, inference_topdown
            self._model = init_model(self.CONFIG, self.CHECKPOINT, device=self.device)
            self._inference_fn = inference_topdown
            self._mode = "mmpose"
            print(f"[RTMPose] Loaded RTMPose-L on {self.device}")
        except ImportError:
            warnings.warn(
                "[RTMPose] MMPose not installed — using DUMMY estimator.\n"
                "Install: pip install mmpose mmdet mmcv mmengine",
                stacklevel=2,
            )
            self._mode = "dummy"

    def predict(self, frame_bgr: np.ndarray) -> PoseResult:
        """
        Run pose estimation on a single BGR frame.

        Args:
            frame_bgr: np.ndarray (H, W, 3) in BGR format

        Returns:
            PoseResult with keypoints (133,2), scores (133,), combined (133,3)
        """
        if self._mode == "dummy":
            return self._dummy_result(frame_bgr)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        bboxes = np.array([[0, 0, w, h, 1.0]])

        results = self._inference_fn(self._model, frame_rgb, bboxes)

        if results and len(results) > 0 and results[0].pred_instances is not None:
            kps    = results[0].pred_instances.keypoints[0]        # (133, 2)
            scores = results[0].pred_instances.keypoint_scores[0]  # (133,)
        else:
            kps    = np.zeros((self.NUM_KEYPOINTS, 2), dtype=np.float32)
            scores = np.zeros(self.NUM_KEYPOINTS, dtype=np.float32)

        combined = np.concatenate([kps, scores[:, None]], axis=1)  # (133, 3)
        return PoseResult(keypoints=kps, scores=scores, combined=combined)

    def _dummy_result(self, frame_bgr: np.ndarray) -> PoseResult:
        h, w = frame_bgr.shape[:2]
        kps    = np.random.rand(self.NUM_KEYPOINTS, 2).astype(np.float32)
        kps[:, 0] *= w
        kps[:, 1] *= h
        scores = np.random.rand(self.NUM_KEYPOINTS).astype(np.float32)
        combined = np.concatenate([kps, scores[:, None]], axis=1)
        return PoseResult(keypoints=kps, scores=scores, combined=combined)

    def draw_skeleton(self, frame_bgr: np.ndarray, result: PoseResult, conf_thresh: float = 0.3) -> np.ndarray:
        """
        Draw keypoints on frame for visualization.
        Returns a copy of the frame with keypoints drawn.
        """
        vis = frame_bgr.copy()
        for i, (kp, score) in enumerate(zip(result.keypoints, result.scores)):
            if score < conf_thresh:
                continue
            x, y = int(kp[0]), int(kp[1])
            # Color by body region
            if i < 17:        color = (0, 255, 0)    # body: green
            elif i < 23:      color = (0, 255, 255)  # feet: yellow
            elif i < 91:      color = (255, 0, 0)    # face: blue
            else:             color = (0, 0, 255)    # hands: red
            cv2.circle(vis, (x, y), 3, color, -1)
        return vis
