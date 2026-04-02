"""
training/config.py
------------------
All hyperparameters and paths in one place.
Edit this file before training — do NOT scatter magic numbers across scripts.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    # ── Data ──────────────────────────────────────────────────────────────────
    manifest_path:  str = "data/raw/manifest.json"
    keypoints_dir:  str = "data/keypoints"
    word_list_path: str = "data/word_list.json"
    num_classes:    int = 100

    # ── Model ─────────────────────────────────────────────────────────────────
    input_dim:       int   = 399    # 133 keypoints × 3 values
    seq_len:         int   = 32     # frames per window
    d_model:         int   = 256
    nhead:           int   = 8
    num_layers:      int   = 4
    dim_feedforward: int   = 512
    dropout:         float = 0.3

    # ── Training ──────────────────────────────────────────────────────────────
    epochs:           int   = 50
    batch_size:       int   = 64
    learning_rate:    float = 1e-3
    weight_decay:     float = 1e-4
    label_smoothing:  float = 0.1
    early_stop_patience: int = 10

    # ── LR Schedule ───────────────────────────────────────────────────────────
    scheduler: str = "cosine"   # "cosine" | "step" | "none"
    warmup_epochs: int = 3

    # ── Hardware ──────────────────────────────────────────────────────────────
    device:      str = "cuda"   # "cuda" | "cpu"
    num_workers: int = 4
    pin_memory:  bool = True

    # ── Output ────────────────────────────────────────────────────────────────
    checkpoint_dir:  str = "models/checkpoints"
    best_model_name: str = "best_model.pth"
    log_dir:         str = "training/logs"

    # ── Augmentation ──────────────────────────────────────────────────────────
    augment: bool = True


# Default config instance
cfg = TrainConfig()
