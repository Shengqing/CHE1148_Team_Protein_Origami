from __future__ import annotations

import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    """
    Baseline model:
      tokens -> embedding (B,L,D)
      masked mean pooling -> (B,D)
      MLP -> scalar ΔG
    """

    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        emb_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L)
        e = self.emb(x)  # (B, L, D)
        mask = (x != self.pad_idx).unsqueeze(-1).float()  # (B, L, 1)

        summed = (e * mask).sum(dim=1)  # (B, D)
        denom = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
        pooled = summed / denom  # (B, D)

        out = self.mlp(pooled).squeeze(-1)  # (B,)
        return out
