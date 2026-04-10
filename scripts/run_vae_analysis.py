import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import kendalltau
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

from src.constants import (
    BATCH_SIZE,
    HIDDEN_DIM,
    KL_WEIGHT,
    LEARNING_RATE,
    PATIENCE,
    SEQ_WEIGHT,
    WEIGHT_DECAY,
)
from src.data import VAEProteinDataset as ProteinDataset
from src.vae_train import train_and_validate


def visualize_latent_space(model, dataset, device, num_samples=6000, output_path="latent_space_tsne.svg"):
    """Generates and saves a t-SNE plot of the latent space."""
    model.eval()
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    all_z = []
    clusters = []
    is_wt = []

    with torch.no_grad():
        for idx in indices:
            emb = torch.from_numpy(np.array(dataset.embeddings[idx])).to(device).unsqueeze(0)
            h = model.encoder(emb)
            z = h[:, : model.latent_dim]
            all_z.append(z.cpu().numpy())

            cluster_id = dataset.clusters[idx]
            clusters.append(cluster_id)

            curr_seq = dataset.sequences[idx]
            if cluster_id in dataset.wt_lookup and dataset.wt_lookup[cluster_id]['seq'] == curr_seq:
                is_wt.append(True)
            else:
                is_wt.append(False)

    z_map = np.vstack(all_z)
    print("Running t-SNE... this may take a moment.")
    tsne = TSNE(n_components=2, random_state=42)
    z_2d = tsne.fit_transform(z_map)

    plt.figure(figsize=(12, 8))

    unique_clusters = sorted(list(set(clusters)))
    palette = sns.color_palette("husl", len(unique_clusters))
    cluster_color_map = {c: palette[i] for i, c in enumerate(unique_clusters)}
    point_colors = [cluster_color_map[c] for c in clusters]

    plt.scatter(
        z_2d[:, 0], z_2d[:, 1], alpha=0.5, s=15, c=point_colors, label="Variants (by Cluster)"
    )

    is_wt_mask = np.array(is_wt)
    plt.scatter(
        z_2d[is_wt_mask, 0],
        z_2d[is_wt_mask, 1],
        alpha=1.0,
        s=60,
        c="red",
        edgecolors="black",
        linewidth=1.5,
        label="Wildtypes",
    )

    plt.title(f"t-SNE Visualization of {model.latent_dim}D Latent Space (Color-coded by Cluster)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(loc="upper right", markerscale=1.2)
    
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()
    print(f"Saved t-SNE plot to {output_path}")


def perform_latent_sweep(train_path, train_mmap, val_path, val_mmap, device, output_path="latent_dim_comparison.svg"):
    """Sweeps across different LATENT_DIM values and plots metric correlations."""
    default_config = {
        "LEARNING_RATE": LEARNING_RATE,
        "HIDDEN_DIM": HIDDEN_DIM,
        "BATCH_SIZE": BATCH_SIZE,
        "SEQ_WEIGHT": SEQ_WEIGHT,
        "KL_WEIGHT": KL_WEIGHT,
        "WEIGHT_DECAY": WEIGHT_DECAY,
    }

    latent_dims = [64, 128, 256, 512]
    results = []
    SWEEP_EPOCHS = 100
    NUM_INITIATIONS = 5

    for dim in latent_dims:
        print(f"\n{'='*20}\nTesting Latent Dim: {dim}\n{'='*20}")
        init_taus = []
        init_rmses = []

        for seed in range(NUM_INITIATIONS):
            print(f"--- Initiation {seed+1}/{NUM_INITIATIONS} (Seed: {seed}) ---")
            torch.manual_seed(seed)
            np.random.seed(seed)
            import random
            random.seed(seed)

            sweep_config = default_config.copy()
            sweep_config["LATENT_DIM"] = dim

            sweep_model, _, val_ds = train_and_validate(
                sweep_config,
                train_path,
                train_mmap,
                val_path,
                val_mmap,
                epochs=SWEEP_EPOCHS,
                patience=PATIENCE,
            )

            sweep_model.eval()
            all_preds = []
            all_labels = []

            val_loader_full = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

            with torch.no_grad():
                for batch in val_loader_full:
                    embs = batch["emb"].to(device)
                    labels = batch["label_scaled"].to(device)
                    h = sweep_model.encoder(embs)
                    z = h[:, : sweep_model.latent_dim]
                    preds_scaled = sweep_model.regressor(z)
                    all_preds.append(preds_scaled.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())

            preds_final = val_ds.scaler.inverse_transform(np.vstack(all_preds)).flatten()
            labels_final = val_ds.scaler.inverse_transform(np.vstack(all_labels)).flatten()

            tau, _ = kendalltau(labels_final, preds_final)
            rmse = np.sqrt(mean_squared_error(labels_final, preds_final))

            init_taus.append(tau)
            init_rmses.append(rmse)

        results.append(
            {
                "latent_dim": dim,
                "tau_mean": np.mean(init_taus),
                "tau_std": np.std(init_taus),
                "rmse_mean": np.mean(init_rmses),
                "rmse_std": np.std(init_rmses),
            }
        )

    sweep_df = pd.DataFrame(results)
    print("\nDetailed Performance Metrics across 5 Random Initiations:")
    print(sweep_df.to_string())

    plt.figure(figsize=(10, 6))
    plt.bar(
        sweep_df["latent_dim"].astype(str),
        sweep_df["tau_mean"],
        yerr=sweep_df["tau_std"],
        capsize=10,
        color="skyblue",
        edgecolor="navy",
        alpha=0.8,
    )

    plt.xlabel("Latent Dimension")
    plt.ylabel("Kendall Tau (Mean ± SD)")
    plt.title("Kendall Tau Performance vs Latent Dimension (5 Random Initiations)")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    for i, row in sweep_df.iterrows():
        plt.text(i, row["tau_mean"] + row["tau_std"] + 0.01, f"{row['tau_mean']:.3f}", ha="center")

    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()
    print(f"\nSaved sweep plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Latent space mappings and dimensionality sweep for Protein VAE")
    parser.add_argument("--train_path", type=str, default="data/processed/train_sampled_esm2_650m.pt")
    parser.add_argument("--train_mmap", type=str, default="data/processed/train_sampled_esm2_650m.dat")
    parser.add_argument("--val_path", type=str, default="data/processed/val_full_esm2_650m.pt")
    parser.add_argument("--val_mmap", type=str, default="data/processed/val_full_esm2_650m.dat")
    parser.add_argument("--tsne_only", action="store_true", help="Only run t-SNE plot on default config")
    parser.add_argument("--sweep_only", action="store_true", help="Only perform the latent dimensionality sweep")
    parser.add_argument("--output_dir", type=str, default="results/figures", help="Directory to save figures")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if not args.sweep_only:
        print("Training model for t-SNE visualization...")
        default_config = {
            "LEARNING_RATE": LEARNING_RATE,
            "LATENT_DIM": 256,
            "HIDDEN_DIM": HIDDEN_DIM,
            "BATCH_SIZE": BATCH_SIZE,
            "SEQ_WEIGHT": SEQ_WEIGHT,
            "KL_WEIGHT": KL_WEIGHT,
            "WEIGHT_DECAY": WEIGHT_DECAY,
        }
        
        model, _, val_ds = train_and_validate(
            default_config, 
            args.train_path, 
            args.train_mmap, 
            args.val_path, 
            args.val_mmap, 
            epochs=100
        )
        visualize_latent_space(model, val_ds, device, output_path=f"{args.output_dir}/latent_space_tsne.svg")

    if not args.tsne_only:
        print("\nStarting Latent Dimension Sweep...")
        perform_latent_sweep(
            args.train_path, 
            args.train_mmap, 
            args.val_path, 
            args.val_mmap, 
            device,
            output_path=f"{args.output_dir}/latent_dim_comparison.svg"
        )


if __name__ == "__main__":
    main()