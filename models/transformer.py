"""
models/transformer.py
---------------------
Custom Transformer classifier for ASL sign recognition.

Architecture:
  Input (B, 32, 399)
    → Linear Embedding (399 → 256) + CLS token + Learnable positional encoding
    → 4× Transformer Encoder Layer (8 heads, ff=512, dropout=0.3)
    → CLS token extraction
    → LayerNorm → Dropout → Linear (256 → num_classes)

~2.5M parameters — lightweight, fast, trainable on Colab in 3-6 hours.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASLTransformer(nn.Module):
    """
    Encoder-only Transformer for ASL sign classification.

    Args:
        num_classes:     number of output classes (default 100)
        input_dim:       flattened keypoint features per frame (133*3 = 399)
        seq_len:         number of frames per window (T=32)
        d_model:         model hidden dimension (256)
        nhead:           number of attention heads (8)
        num_layers:      number of Transformer encoder layers (4)
        dim_feedforward: feedforward hidden size (512)
        dropout:         dropout rate (0.3)
    """

    def __init__(
        self,
        num_classes: int = 100,
        input_dim: int = 399,
        seq_len: int = 32,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len

        # ── 1. Input projection ────────────────────────────────────────────────
        self.input_projection = nn.Linear(input_dim, d_model)

        # ── 2. Learnable [CLS] token ───────────────────────────────────────────
        # Shape: (1, 1, d_model) — broadcasted over batch
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── 3. Learnable positional encoding ──────────────────────────────────
        # seq_len + 1 for the CLS token prepended at position 0
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # ── 4. Transformer Encoder ────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # input shape: (B, S, d_model)
            norm_first=True,    # pre-norm (more stable training)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ── 5. Classification head ────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:            (B, T, 399)  — batch of keypoint sequences
            padding_mask: (B, T) bool tensor — True = padded/ignored frame
                          (generated automatically from all-zero frames)

        Returns:
            logits: (B, num_classes)
        """
        B, T, _ = x.shape

        # ── Auto-detect padding from all-zero frames ──────────────────────────
        if padding_mask is None:
            # Frame is padding if ALL features are 0
            padding_mask = (x.abs().sum(dim=-1) == 0)  # (B, T)

        # ── 1. Project input ──────────────────────────────────────────────────
        x = self.input_projection(x)   # (B, T, d_model)

        # ── 2. Prepend CLS token ──────────────────────────────────────────────
        cls = self.cls_token.expand(B, -1, -1)    # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)            # (B, T+1, d_model)

        # ── 3. Add positional embedding ───────────────────────────────────────
        x = x + self.pos_embedding                # (B, T+1, d_model)

        # ── 4. Extend padding mask for CLS position (never masked) ────────────
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, padding_mask], dim=1)  # (B, T+1)

        # ── 5. Transformer encoder ────────────────────────────────────────────
        x = self.transformer_encoder(x, src_key_padding_mask=full_mask)  # (B, T+1, d_model)

        # ── 6. Extract CLS token ──────────────────────────────────────────────
        cls_out = x[:, 0, :]   # (B, d_model)

        # ── 7. Classify ───────────────────────────────────────────────────────
        logits = self.classifier(cls_out)  # (B, num_classes)
        return logits

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method for inference.

        Returns:
            probs:      (B, num_classes)  — softmax probabilities
            top_class:  (B,)              — argmax prediction
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            top_class = probs.argmax(dim=-1)
        return probs, top_class

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Model factory ────────────────────────────────────────────────────────────

def build_model(num_classes: int = 100, device: str = "cpu") -> ASLTransformer:
    """Build and return a model moved to the target device."""
    model = ASLTransformer(num_classes=num_classes)
    model = model.to(device)
    print(f"[model] ASLTransformer | params: {model.count_parameters():,} | device: {device}")
    return model


def load_checkpoint(
    checkpoint_path: str,
    num_classes: int = 100,
    device: str = "cpu",
) -> ASLTransformer:
    """Load a saved model checkpoint."""
    model = ASLTransformer(num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location=device)
    # Support both raw state_dict and checkpoint dicts
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print(f"[model] Loaded checkpoint: {checkpoint_path}")
    return model


# ─── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(num_classes=100, device=device)

    # Simulate a batch of 4 samples with T=32 frames
    dummy_input = torch.randn(4, 32, 399).to(device)
    logits = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {logits.shape}  (expected: [4, 100])")
    print(f"Parameters:   {model.count_parameters():,}  (target: ~2.5M)")

    probs, preds = model.predict(dummy_input)
    print(f"Predictions:  {preds.tolist()}")
    print(f"Max prob:     {probs.max(dim=-1).values.tolist()}")
    print("\n[ok] Transformer smoke test passed")
