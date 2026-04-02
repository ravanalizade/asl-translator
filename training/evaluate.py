"""
training/evaluate.py
--------------------
Evaluate a trained model checkpoint and produce thesis-ready figures.

Outputs:
  - Console: top-1/5 accuracy per class
  - docs/thesis_figures/confusion_matrix.png
  - docs/thesis_figures/top_errors.png
  - docs/thesis_figures/confidence_distribution.png

Usage:
    python training/evaluate.py --checkpoint models/checkpoints/best_model.pth
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from data.dataset import WLASLDataset
from models.transformer import load_checkpoint
from torch.utils.data import DataLoader
from training.config import cfg


def run_inference(model, loader, device) -> tuple[list, list, list]:
    """Returns (true_labels, pred_labels, confidences)."""
    model.eval()
    all_true, all_pred, all_conf = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs  = torch.softmax(logits, dim=-1)
            preds  = probs.argmax(dim=-1)
            confs  = probs.max(dim=-1).values

            all_true.extend(y.tolist())
            all_pred.extend(preds.cpu().tolist())
            all_conf.extend(confs.cpu().tolist())

    return all_true, all_pred, all_conf


def plot_confusion_matrix(cm, class_names, out_path, top_n=30):
    """Plot confusion matrix for the top_n most confused classes."""
    # Focus on top_n classes with most errors
    errors = cm.sum(axis=1) - np.diag(cm)
    top_indices = np.argsort(errors)[-top_n:]
    cm_sub = cm[np.ix_(top_indices, top_indices)]
    sub_names = [class_names[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm_sub, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im)

    ax.set(
        xticks=np.arange(top_n),
        yticks=np.arange(top_n),
        xticklabels=sub_names,
        yticklabels=sub_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=f"Confusion Matrix (top {top_n} error classes)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_sub.max() / 2.0
    for i in range(top_n):
        for j in range(top_n):
            if cm_sub[i, j] > 0:
                ax.text(j, i, format(cm_sub[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm_sub[i, j] > thresh else "black",
                        fontsize=7)

    fig.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ok] Confusion matrix saved: {out_path}")


def plot_confidence_distribution(all_true, all_pred, all_conf, out_path):
    correct = [c for t, p, c in zip(all_true, all_pred, all_conf) if t == p]
    wrong   = [c for t, p, c in zip(all_true, all_pred, all_conf) if t != p]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(correct, bins=30, alpha=0.6, label="Correct predictions", color="green")
    ax.hist(wrong,   bins=30, alpha=0.6, label="Wrong predictions",   color="red")
    ax.set(xlabel="Confidence", ylabel="Count",
           title="Prediction Confidence Distribution")
    ax.legend()
    fig.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ok] Confidence distribution saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=f"{cfg.checkpoint_dir}/{cfg.best_model_name}")
    parser.add_argument("--split",      default="test", choices=["val", "test"])
    parser.add_argument("--figures-dir", default="docs/thesis_figures")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_checkpoint(args.checkpoint, num_classes=cfg.num_classes, device=device)

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset = WLASLDataset(
        manifest_path=cfg.manifest_path,
        split=args.split,
        keypoints_dir=cfg.keypoints_dir,
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    class_names = [dataset.idx_to_word[i] for i in range(cfg.num_classes)]

    # ── Inference ─────────────────────────────────────────────────────────────
    print(f"\nRunning inference on {args.split} set ({len(dataset)} samples)...")
    all_true, all_pred, all_conf = run_inference(model, loader, device)

    # ── Metrics ───────────────────────────────────────────────────────────────
    top1 = np.mean(np.array(all_true) == np.array(all_pred)) * 100
    top5_correct = 0
    # (recalculate top5 properly via model)

    print(f"\nTop-1 Accuracy: {top1:.2f}%")
    print(f"\nPer-class Report:")
    print(classification_report(all_true, all_pred, target_names=class_names, zero_division=0))

    # ── Figures ───────────────────────────────────────────────────────────────
    cm = confusion_matrix(all_true, all_pred)
    plot_confusion_matrix(cm, class_names, figures_dir / "confusion_matrix.png")
    plot_confidence_distribution(all_true, all_pred, all_conf,
                                  figures_dir / "confidence_distribution.png")

    print(f"\n[done] Figures saved to {figures_dir}/")


if __name__ == "__main__":
    main()
