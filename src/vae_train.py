"""Training and evaluation routines for Protein VAE."""

import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import resample
from torch.utils.data import DataLoader

from .constants import (
    BATCH_SIZE,
    HIDDEN_DIM,
    INPUT_DIM,
    LATENT_DIM,
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    WEIGHT_DECAY,
)
from .dataset import ProteinDataset
from .model import ProteinVAE
from .optim import optimize_latent_constrained
from .utils import batch_decode_and_count_mutations, robust_pt_load


def train_and_validate(
    config: Dict[str, Any],
    train_path: str,
    train_mmap: str,
    val_path: str,
    val_mmap: str,
    epochs: int = NUM_EPOCHS,
    patience: int = PATIENCE,
) -> Tuple[ProteinVAE, ProteinDataset, ProteinDataset]:
    """Trains the Protein VAE model and evaluates on a validation set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = ProteinDataset(train_path, train_mmap)
    val_ds = ProteinDataset(val_path, val_mmap, scaler=train_ds.scaler)

    db_max_len = max(
        max(len(s) for s in train_ds.sequences),
        max(len(s) for s in val_ds.sequences),
    )
    print(f"Setting model max_seq_len to database maximum: {db_max_len}")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.get("BATCH_SIZE", BATCH_SIZE),
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.get("BATCH_SIZE", BATCH_SIZE),
        shuffle=False,
        num_workers=0,
    )

    model = ProteinVAE(
        input_dim=INPUT_DIM,
        latent_dim=config.get("LATENT_DIM", LATENT_DIM),
        hidden_dim=config.get("HIDDEN_DIM", HIDDEN_DIM),
        max_seq_len=db_max_len,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get("LEARNING_RATE", LEARNING_RATE),
        weight_decay=config.get("WEIGHT_DECAY", WEIGHT_DECAY),
    )
    mse_loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            embs = batch["emb"].to(device)
            labels = batch["label_scaled"].to(device)

            optimizer.zero_grad()
            recon_emb, recon_seq_logits, pred_y, mu, logvar = model(embs)

            recon_loss = mse_loss_fn(recon_emb, embs)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            pred_loss = mse_loss_fn(pred_y, labels)

            loss = recon_loss + 0.1 * kl_loss + pred_loss
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        all_preds: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        with torch.no_grad():
            for batch in val_loader:
                embs = batch["emb"].to(device)
                labels = batch["label_scaled"].to(device)
                _, _, pred_y, _, _ = model(embs)
                val_loss += mse_loss_fn(pred_y, labels).item()
                all_preds.append(pred_y.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_val = val_loss / len(val_loader)
        preds_final = val_ds.scaler.inverse_transform(
            np.vstack(all_preds)
        ).flatten()
        labels_final = val_ds.scaler.inverse_transform(
            np.vstack(all_labels)
        ).flatten()

        rmse = np.sqrt(mean_squared_error(labels_final, preds_final))
        r2 = r2_score(labels_final, preds_final)
        rho, _ = spearmanr(labels_final, preds_final)

        print(
            f"Epoch {epoch}: Val Loss = {avg_val:.4f} | "
            f"RMSE: {rmse:.4f} | R2: {r2:.4f} | Spearman: {rho:.4f}"
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    return model, train_ds, val_ds


def run_unified_evaluation(
    model: ProteinVAE,
    val_loader: DataLoader,
    val_ds: ProteinDataset,
    train_ds_len: int,
    val_path: str,
    device: torch.device,
    master_lookup: Dict[str, float],
    num_samples: int = 1000,
    n_bootstraps: int = 10,
) -> None:
    """Evaluates the model by generating predictions and conducting a mutation robustness check."""
    model.eval()
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    print("--- Dataset Sizes ---")
    print(f"Total training samples:   {train_ds_len}")
    print(f"Total validation samples: {len(val_ds)}")

    with torch.no_grad():
        for batch in val_loader:
            embs = batch["emb"].to(device)
            labels = batch["label_scaled"].to(device)
            h = model.encoder(embs)
            z = h[:, : model.latent_dim]
            preds_scaled = model.regressor(z)
            all_preds.append(preds_scaled.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    preds_scaled_np = np.vstack(all_preds)
    labels_scaled_np = np.vstack(all_labels)
    preds_final = val_ds.scaler.inverse_transform(preds_scaled_np).flatten()
    labels_final = val_ds.scaler.inverse_transform(labels_scaled_np).flatten()

    metrics_boot: Dict[str, List[float]] = {"rmse": [], "r2": [], "rho": []}

    for i in range(n_bootstraps):
        p_boot, l_boot = resample(
            preds_final, labels_final, replace=True, random_state=i
        )
        metrics_boot["rmse"].append(
            np.sqrt(mean_squared_error(l_boot, p_boot))
        )
        metrics_boot["r2"].append(r2_score(l_boot, p_boot))
        metrics_boot["rho"].append(spearmanr(l_boot, p_boot)[0])

    print(f"\n--- Global Validation Metrics ({n_bootstraps} bootstraps) ---")
    print(
        f"RMSE:          {np.mean(metrics_boot['rmse']):.4f} ± {np.std(metrics_boot['rmse']):.4f}"
    )
    print(
        f"R-squared:     {np.mean(metrics_boot['r2']):.4f} ± {np.std(metrics_boot['r2']):.4f}"
    )
    print(
        f"Spearman's ρ:  {np.mean(metrics_boot['rho']):.4f} ± {np.std(metrics_boot['rho']):.4f}"
    )

    val_indices = random.sample(
        range(len(val_ds)), min(num_samples, len(val_ds))
    )
    report_data = []
    print(
        f"\nProcessing {len(val_indices)} samples for mutation feasibility..."
    )

    data_obj = robust_pt_load(val_path)

    for idx in val_indices:
        input_seq = val_ds.sequences[idx]
        input_dg = data_obj["delta_g"][idx].item()
        input_cluster = val_ds.clusters[idx]
        input_emb = (
            torch.from_numpy(np.array(val_ds.embeddings[idx]))
            .to(device)
            .unsqueeze(0)
        )

        z_opt = optimize_latent_constrained(
            model, input_emb, base_seq=input_seq, steps=25
        )

        with torch.no_grad():
            recon_emb = model.decoder(z_opt)
            logits = model.seq_out(
                model.seq_proj(recon_emb).view(-1, model.max_seq_len, 64)
            )
            mutated_seq, mut_count = batch_decode_and_count_mutations(
                logits, [input_seq]
            )[0]
            pred_dg_scaled = model.regressor(z_opt).item()
            pred_dg = val_ds.scaler.inverse_transform([[pred_dg_scaled]])[0][0]

        if mut_count > 0:
            actual_dg = master_lookup.get(mutated_seq, None)
            is_feasible = actual_dg is not None
            is_truly_better = (
                "Yes"
                if (is_feasible and actual_dg < input_dg)
                else "No" if is_feasible else "N/A"
            )
            mutations = [
                f"{wt_aa}{pos + 1}{mut_aa}"
                for pos, (wt_aa, mut_aa) in enumerate(
                    zip(input_seq, mutated_seq)
                )
                if wt_aa != mut_aa
            ]

            report_data.append(
                {
                    "Input_Cluster": input_cluster,
                    "Mutations": ", ".join(mutations),
                    "Pred_DG": round(float(pred_dg), 3),
                    "Input_DG": round(float(input_dg), 3),
                    "Exists_in_DB": "Yes" if is_feasible else "No",
                    "Truly_Stabilizing": is_truly_better,
                    "Actual_DG": (
                        round(float(actual_dg), 3)
                        if actual_dg is not None
                        else None
                    ),
                }
            )

    results_df = pd.DataFrame(report_data)
    if not results_df.empty:
        feasible_count = (results_df["Exists_in_DB"] == "Yes").sum()
        better_count = (results_df["Truly_Stabilizing"] == "Yes").sum()
        print(
            f"Dreaming complete. Feasible variants: {feasible_count}. "
            f"Truly stabilizing: {better_count}."
        )
        print(results_df.sort_values("Pred_DG").head(15).to_string())
