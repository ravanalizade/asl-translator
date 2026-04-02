"""
data/dataset.py
---------------
PyTorch Dataset and DataLoader factory for the preprocessed WLASL keypoint data.

Each sample:
  X: torch.Tensor of shape (32, 399)  — T frames × 133 keypoints × 3 values
  y: int label (0–99)
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ─── Dataset ──────────────────────────────────────────────────────────────────

class WLASLDataset(Dataset):
    """
    Loads preprocessed .npy keypoint files produced by data/preprocess.py.

    Args:
        manifest_path: path to data/raw/manifest.json
        split:         "train" | "val" | "test"
        keypoints_dir: root dir containing <GLOSS>/<video_id>.npy
        augment:       apply data augmentation (train split only)
    """

    def __init__(
        self,
        manifest_path: str = "data/raw/manifest.json",
        split: str = "train",
        keypoints_dir: str = "data/keypoints",
        augment: bool = False,
    ):
        self.keypoints_dir = Path(keypoints_dir)
        self.augment = augment and (split == "train")

        with open(manifest_path) as f:
            manifest = json.load(f)

        self.word_to_idx: dict = manifest["word_to_idx"]
        self.idx_to_word: dict = {v: k for k, v in self.word_to_idx.items()}
        self.samples: list = manifest["splits"][split]

        # Filter to only samples whose .npy file exists
        available = []
        for s in self.samples:
            npy_path = self.keypoints_dir / s["gloss"] / f"{s['video_id']}.npy"
            if npy_path.exists():
                available.append({**s, "npy_path": str(npy_path)})

        missing = len(self.samples) - len(available)
        if missing > 0:
            print(f"[warn] {split}: {missing} samples missing .npy files (skipped)")

        self.samples = available
        print(f"[{split}] {len(self.samples)} samples | {len(self.word_to_idx)} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        keypoints = np.load(sample["npy_path"])  # (32, 399)

        if self.augment:
            keypoints = self._augment(keypoints)

        x = torch.tensor(keypoints, dtype=torch.float32)
        y = sample["label"]
        return x, y

    # ── Augmentation ──────────────────────────────────────────────────────────

    def _augment(self, kp: np.ndarray) -> np.ndarray:
        """Apply stochastic augmentations to a (32, 399) keypoint tensor."""
        kp = kp.copy()

        # 1. Temporal jitter: shift window by ±3 frames
        if random.random() < 0.5:
            shift = random.randint(-3, 3)
            kp = np.roll(kp, shift, axis=0)
            if shift > 0:
                kp[:shift] = 0
            elif shift < 0:
                kp[shift:] = 0

        # 2. Gaussian keypoint noise
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.01, kp.shape).astype(np.float32)
            # Only add noise to x,y (every 3rd value starting at 0 and 1, not 2=conf)
            conf_mask = np.zeros(399, dtype=bool)
            conf_mask[2::3] = True          # confidence positions
            noise[:, conf_mask] = 0
            kp += noise

        # 3. Random frame drop: zero out ~10% of frames
        if random.random() < 0.5:
            T = kp.shape[0]
            n_drop = max(1, int(T * 0.1))
            drop_indices = random.sample(range(T), n_drop)
            kp[drop_indices] = 0

        # 4. Horizontal flip: mirror all x coordinates
        #    x values are every 3rd position starting at 0: 0, 3, 6, ...
        if random.random() < 0.5:
            x_indices = list(range(0, 399, 3))
            kp[:, x_indices] = -kp[:, x_indices]

        # 5. Speed perturbation: stretch/compress time by 0.8x–1.2x
        if random.random() < 0.3:
            T = kp.shape[0]
            factor = random.uniform(0.8, 1.2)
            new_T = max(8, int(T * factor))
            indices = np.round(np.linspace(0, T - 1, new_T)).astype(int)
            stretched = kp[indices]
            # Pad or truncate back to T
            if new_T < T:
                pad = np.zeros((T - new_T, 399), dtype=np.float32)
                kp = np.concatenate([stretched, pad], axis=0)
            else:
                kp = stretched[:T]

        return kp

    # ── Utilities ──────────────────────────────────────────────────────────────

    def label_to_word(self, label: int) -> str:
        return self.idx_to_word.get(label, "UNKNOWN")

    def word_to_label(self, word: str) -> int:
        return self.word_to_idx.get(word.upper(), -1)

    def class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for imbalanced training."""
        counts = np.zeros(len(self.word_to_idx))
        for s in self.samples:
            counts[s["label"]] += 1
        counts = np.maximum(counts, 1)
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(counts)
        return torch.tensor(weights, dtype=torch.float32)


# ─── DataLoader Factory ───────────────────────────────────────────────────────

def make_dataloaders(
    manifest_path: str = "data/raw/manifest.json",
    keypoints_dir: str = "data/keypoints",
    batch_size: int = 64,
    num_workers: int = 4,
) -> dict[str, DataLoader]:
    """
    Build train/val/test DataLoaders.

    Returns:
        {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    """
    loaders = {}

    for split in ("train", "val", "test"):
        is_train = split == "train"
        dataset = WLASLDataset(
            manifest_path=manifest_path,
            split=split,
            keypoints_dir=keypoints_dir,
            augment=is_train,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=is_train,
        )

    return loaders


# ─── Quick smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    manifest = "data/raw/manifest.json"
    if not Path(manifest).exists():
        print(f"[error] Manifest not found: {manifest}")
        print("Run: python data/download_wlasl.py first")
        sys.exit(1)

    loaders = make_dataloaders(batch_size=4, num_workers=0)

    for split, loader in loaders.items():
        x, y = next(iter(loader))
        print(f"[{split}] batch x: {x.shape}  y: {y.shape}  y values: {y.tolist()}")

    print("\n[ok] Dataset smoke test passed")
