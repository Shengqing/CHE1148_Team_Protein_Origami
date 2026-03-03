# =========================
# FILE: src/run_baseline_mlp.py
# =========================
from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from .data import (
    PAD_IDX,
    VOCAB,
    ProteinDataset,
    compute_max_len,
    load_and_align,
)
from .eda import run_eda
from .model import MLPRegressor
from .train import train_model
from .utils import DeviceConfig, ensure_dir, load_yaml, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train baseline MLP for ΔG prediction.")
    p.add_argument("--config", type=str, default="configs/baseline_mlp.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    set_seed(int(cfg["train"]["seed"]))
    device = DeviceConfig.auto().device
    print("Device:", device)

    out_root = cfg["data"]["output_dir"]
    fig_dir = f"{out_root}/figures"
    metrics_dir = f"{out_root}/metrics"
    ensure_dir(fig_dir)
    ensure_dir(metrics_dir)

    train_df = load_and_align(cfg["data"]["train_path"])
    val_df = load_and_align(cfg["data"]["val_path"])

    # MAX_LEN (matches your Colab code)
    max_len = compute_max_len(
        train_df=train_df,
        val_df=val_df,
        q=float(cfg["data"]["max_len_quantile"]),
        cap=int(cfg["data"]["max_len_cap"]),
    )
    print("Using MAX_LEN:", max_len)

    # EDA figures
    run_eda(train_df, val_df, out_dir=fig_dir)
    print(f"Saved EDA figures to: {fig_dir}")

    # Datasets + loaders (matches your Colab settings)
    train_ds = ProteinDataset(train_df, max_len)
    val_ds = ProteinDataset(val_df, max_len)

    pin = torch.cuda.is_available()
    bs = 512 if torch.cuda.is_available() else 256

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=2,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=2,
        pin_memory=pin,
    )

    # Model
    model = MLPRegressor(
        vocab_size=len(VOCAB),
        pad_idx=PAD_IDX,
        emb_dim=int(cfg["model"]["emb_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=int(cfg["train"]["epochs"]),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        clip_grad_norm=float(cfg["train"]["clip_grad_norm"]),
        patience=int(cfg["train"]["patience"]),
        factor=float(cfg["train"]["factor"]),
    )

    # Save best model
    model.load_state_dict(result.best_state_dict)
    model_path = f"{out_root}/baseline_mlp.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "max_len": max_len,
            "vocab": VOCAB,
            "config": cfg,
        },
        model_path,
    )
    print("Saved model:", model_path)

    # Save summary metrics
    summary = {"best_val_mae": float(result.best_val_mae)}
    with open(f"{metrics_dir}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved metrics:", f"{metrics_dir}/summary.json")


if __name__ == "__main__":
    main()
