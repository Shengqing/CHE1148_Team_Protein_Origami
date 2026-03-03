from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dir


def plot_hist(train: pd.Series, val: pd.Series, title: str, xlabel: str, outpath: str) -> None:
    plt.figure()
    plt.hist(train.dropna().values, bins=60, alpha=0.5, label="train")
    plt.hist(val.dropna().values, bins=60, alpha=0.5, label="val")
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_top_counts(
    series: pd.Series, title: str, xlabel: str, outpath: str, topn: int = 30
) -> None:
    counts = series.value_counts().head(topn)
    plt.figure(figsize=(12, 4))
    counts.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def run_eda(train_df: pd.DataFrame, val_df: pd.DataFrame, out_dir: str) -> None:
    ensure_dir(out_dir)

    plot_hist(
        train_df["deltaG"],
        val_df["deltaG"],
        title="ΔG distribution (train vs val)",
        xlabel="ΔG",
        outpath=f"{out_dir}/dg_hist.png",
    )

    plot_hist(
        train_df["seq_len"],
        val_df["seq_len"],
        title="Sequence length distribution (train vs val)",
        xlabel="Sequence length",
        outpath=f"{out_dir}/len_hist.png",
    )

    if "deltaG_95CI_high" in train_df.columns and "deltaG_95CI_low" in train_df.columns:
        train_ci = train_df["deltaG_95CI_high"] - train_df["deltaG_95CI_low"]
        val_ci = val_df["deltaG_95CI_high"] - val_df["deltaG_95CI_low"]
        plot_hist(
            train_ci,
            val_ci,
            title="ΔG 95% CI width (train vs val)",
            xlabel="CI width",
            outpath=f"{out_dir}/ci_width_hist.png",
        )

    plot_top_counts(
        train_df["WT_cluster"],
        title="Top WT_cluster counts (train)",
        xlabel="WT_cluster",
        outpath=f"{out_dir}/train_top_clusters.png",
    )

    plot_top_counts(
        val_df["WT_cluster"],
        title="Top WT_cluster counts (val)",
        xlabel="WT_cluster",
        outpath=f"{out_dir}/val_top_clusters.png",
    )

    # Split sanity: cluster overlap
    overlap = set(train_df["WT_cluster"]) & set(val_df["WT_cluster"])
    with open(f"{out_dir}/split_sanity.txt", "w", encoding="utf-8") as f:
        f.write(f"Train rows: {len(train_df)}\n")
        f.write(f"Val rows: {len(val_df)}\n")
        f.write(f"Unique train clusters: {train_df['WT_cluster'].nunique()}\n")
        f.write(f"Unique val clusters: {val_df['WT_cluster'].nunique()}\n")
        f.write(f"Cluster overlap (count): {len(overlap)}\n")
