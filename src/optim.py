"""Latent optimization routines for Protein VAE."""

from typing import Optional

import torch
import torch.nn.functional as F
import torch.optim as optim

from .constants import AA_VOCAB, KL_WEIGHT, OPTIM_LR, OPTIM_STEPS


def optimize_latent_constrained(
    model: torch.nn.Module,
    emb: torch.Tensor,
    base_seq: Optional[str] = None,
    steps: int = 25,
    lr: float = 0.05,
    max_dist: float = 0.1,
    lambda_weight: float = 0.5,
) -> torch.Tensor:
    """Optimizes a latent vector using a squared soft Hamming constraint."""
    model.eval()
    with torch.no_grad():
        h = model.encoder(emb)
        z_start = h[:, : model.latent_dim].detach()

    z_opt = z_start.clone().requires_grad_(True)
    optimizer = optim.Adam([z_opt], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        pred_y = model.regressor(z_opt)

        dist_loss = torch.norm(z_opt - z_start, p=2, dim=1).mean()
        total_loss = pred_y.mean() + (max_dist * dist_loss)

        if base_seq is not None:
            recon_emb = model.decoder(z_opt)
            seq_hidden = model.seq_proj(recon_emb).view(
                -1, model.max_seq_len, 64
            )
            logits = model.seq_out(seq_hidden)

            target_len = len(base_seq)
            sliced_logits = logits[:, :target_len, :]
            target_indices = torch.tensor(
                [AA_VOCAB.get(aa, 20) for aa in base_seq],
                device=z_opt.device,
            ).unsqueeze(0)

            probs = F.softmax(sliced_logits, dim=-1)
            wt_probs = torch.gather(
                probs, 2, target_indices.unsqueeze(-1)
            ).squeeze(-1)

            soft_hamming_dist = (1.0 - wt_probs).sum()
            constraint_penalty = lambda_weight * torch.pow(
                soft_hamming_dist - 1.0, 2
            )
            total_loss += constraint_penalty

        total_loss.backward()
        optimizer.step()

    return z_opt.detach()


def optimize_latent(
    model: torch.nn.Module,
    emb: torch.Tensor,
    steps: int = OPTIM_STEPS,
    lr: float = OPTIM_LR,
) -> torch.Tensor:
    """Optimizes a latent vector for a lower predicted dG without constraints."""
    model.eval()

    with torch.no_grad():
        h = model.encoder(emb)
        z_start = h[:, : model.latent_dim].detach()

    z_opt = z_start.clone().requires_grad_(True)
    optimizer = optim.Adam([z_opt], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        pred_y = model.regressor(z_opt)
        loss = pred_y.mean() + KL_WEIGHT * torch.pow(z_opt, 2).mean()
        loss.backward()
        optimizer.step()

    return z_opt.detach()
