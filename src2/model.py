"""Protein VAE model definition."""
from typing import Tuple

import torch
import torch.nn as nn

from .constants import HIDDEN_DIM, INPUT_DIM, LATENT_DIM


class ProteinVAE(nn.Module):
    """Variational Autoencoder for protein sequence embeddings."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        max_seq_len: int = 100,
        dropout: float = 0.2,
    ):
        super(ProteinVAE, self).__init__()
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        self.seq_proj = nn.Linear(input_dim, max_seq_len * 64)
        self.seq_out = nn.Linear(64, 21)

        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        h = self.encoder(x)
        mu, logvar = h[:, : self.latent_dim], h[:, self.latent_dim :]
        z = self.reparameterize(mu, logvar)
        recon_emb = self.decoder(z)

        seq_features = self.seq_proj(recon_emb).view(-1, self.max_seq_len, 64)
        recon_seq_logits = self.seq_out(seq_features)

        pred_y = self.regressor(z)
        return recon_emb, recon_seq_logits, pred_y, mu, logvar
