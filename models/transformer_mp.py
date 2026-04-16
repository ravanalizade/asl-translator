"""
models/transformer_mp.py
------------------------
Same Transformer architecture as transformer.py but with input_dim=126
for MediaPipe hand keypoints (42 keypoints × 3 values).

The only difference from transformer.py:
    input_dim: 399 (RTMPose) → 126 (MediaPipe)
    Everything else is identical.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASLTransformerMP(nn.Module):
    """
    Transformer classifier for MediaPipe hand keypoints.

    Input:  (batch, 32, 126)  — 32 frames × 42 keypoints × 3 values
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        num_classes: int = 100,
        input_dim: int = 126,
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

        self.input_projection = nn.Linear(input_dim, d_model)

        self.cls_token    = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))
        nn.init.trunc_normal_(self.cls_token,     std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:            (B, T, 126)
            padding_mask: (B, T) bool — True = padded frame
        Returns:
            logits: (B, num_classes)
        """
        B, T, _ = x.shape

        if padding_mask is None:
            padding_mask = (x.abs().sum(dim=-1) == 0)

        x   = self.input_projection(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1) + self.pos_embedding

        cls_mask  = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, padding_mask], dim=1)

        x = self.transformer_encoder(x, src_key_padding_mask=full_mask)
        return self.classifier(x[:, 0, :])

    def predict(self, x: torch.Tensor):
        with torch.no_grad():
            logits = self.forward(x)
            probs  = F.softmax(logits, dim=-1)
            top_class = probs.argmax(dim=-1)
        return probs, top_class

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model_mp(num_classes: int = 100, device: str = "cpu") -> ASLTransformerMP:
    model = ASLTransformerMP(num_classes=num_classes).to(device)
    print(f"[model-mp] params: {model.count_parameters():,} | device: {device}")
    return model


def load_checkpoint_mp(
    checkpoint_path: str,
    num_classes: int = 100,
    device: str = "cpu",
) -> ASLTransformerMP:
    model = ASLTransformerMP(num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print(f"[model-mp] Loaded: {checkpoint_path}")
    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = build_model_mp(num_classes=100, device=device)

    dummy  = torch.randn(4, 32, 126).to(device)
    logits = model(dummy)

    print(f"Input : {dummy.shape}")
    print(f"Output: {logits.shape}  (expected [4, 100])")
    print(f"Params: {model.count_parameters():,}")
    print("Smoke test passed")
