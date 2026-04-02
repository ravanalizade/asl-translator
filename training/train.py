"""
training/train.py
-----------------
Main training script for the ASL Transformer classifier.

Usage (local):
    python training/train.py

Usage (Colab):
    See training/colab_notebook.ipynb — this script is called from there.
"""

import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from data.dataset import make_dataloaders
from models.transformer import build_model
from training.config import cfg


# ─── Utilities ────────────────────────────────────────────────────────────────

def accuracy(logits: torch.Tensor, labels: torch.Tensor, topk=(1, 5)) -> dict:
    """Compute top-k accuracy."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        results = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            results[f"top{k}"] = (correct_k / batch_size * 100).item()
    return results


def save_checkpoint(model, optimizer, epoch, val_acc, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_top1_acc": val_acc,
    }, path)


# ─── Training epoch ───────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss = 0.0
    top1_sum = 0.0
    n_batches = len(loader)

    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        accs = accuracy(logits, y)
        total_loss += loss.item()
        top1_sum   += accs["top1"]

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{n_batches}] loss: {loss.item():.4f} | top1: {accs['top1']:.1f}%")

    if scheduler is not None:
        scheduler.step()

    return {
        "loss": total_loss / n_batches,
        "top1": top1_sum   / n_batches,
    }


# ─── Validation epoch ─────────────────────────────────────────────────────────

@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    top1_sum = 0.0
    top5_sum = 0.0
    n_batches = len(loader)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        accs = accuracy(logits, y, topk=(1, 5))

        total_loss += loss.item()
        top1_sum   += accs["top1"]
        top5_sum   += accs["top5"]

    return {
        "loss": total_loss / n_batches,
        "top1": top1_sum   / n_batches,
        "top5": top5_sum   / n_batches,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  ASL Transformer Training")
    print("=" * 60)

    # ── Device ────────────────────────────────────────────────────────────────
    device = cfg.device if torch.cuda.is_available() else "cpu"
    if device != cfg.device:
        print(f"[warn] CUDA not available, falling back to CPU")
    print(f"Device: {device}")

    # ── Directories ───────────────────────────────────────────────────────────
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    loaders = make_dataloaders(
        manifest_path=cfg.manifest_path,
        keypoints_dir=cfg.keypoints_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(num_classes=cfg.num_classes, device=device)

    # ── Loss, optimizer, scheduler ────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs - cfg.warmup_epochs,
        eta_min=1e-5,
    ) if cfg.scheduler == "cosine" else None

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=cfg.log_dir)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_top1 = 0.0
    patience_counter = 0
    best_ckpt_path = Path(cfg.checkpoint_dir) / cfg.best_model_name
    history = []

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        print("-" * 40)

        train_stats = train_epoch(model, loaders["train"], optimizer, criterion, device, scheduler)
        val_stats   = val_epoch(model, loaders["val"], criterion, device)

        elapsed = time.time() - t0
        print(f"  Train  loss: {train_stats['loss']:.4f}  top1: {train_stats['top1']:.1f}%")
        print(f"  Val    loss: {val_stats['loss']:.4f}  top1: {val_stats['top1']:.1f}%  top5: {val_stats['top5']:.1f}%")
        print(f"  Time: {elapsed:.1f}s")

        # TensorBoard logging
        writer.add_scalars("Loss", {"train": train_stats["loss"], "val": val_stats["loss"]}, epoch)
        writer.add_scalars("Top1", {"train": train_stats["top1"], "val": val_stats["top1"]}, epoch)
        writer.add_scalar("Top5/val", val_stats["top5"], epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        history.append({"epoch": epoch, "train": train_stats, "val": val_stats})

        # ── Best model checkpoint ──────────────────────────────────────────────
        if val_stats["top1"] > best_val_top1:
            best_val_top1 = val_stats["top1"]
            save_checkpoint(model, optimizer, epoch, best_val_top1, best_ckpt_path)
            print(f"  ✓ New best model saved ({best_val_top1:.1f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                print(f"\nEarly stopping triggered (patience={cfg.early_stop_patience})")
                break

    # ── Final evaluation on test set ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Final Test Evaluation")
    print("=" * 60)

    # Reload best checkpoint
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_stats = val_epoch(model, loaders["test"], criterion, device)
    print(f"  Test  top1: {test_stats['top1']:.1f}%  top5: {test_stats['top5']:.1f}%")

    # Save history
    history_path = Path(cfg.log_dir) / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    writer.close()
    print(f"\n[done] Best val top1: {best_val_top1:.1f}% | checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    train()
