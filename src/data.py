# =========================
# FILE: src/data.py
# =========================
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

SCHEMA = [
    "deltaG",
    "deltaG_95CI_high",
    "deltaG_95CI_low",
    "aa_seq",
    "mut_type",
    "WT_cluster",
    "Stabilizing_mut",
]

AA_STANDARD = list("ACDEFGHIKLMNPQRSTVWY")
SPECIALS = ["<PAD>", "<UNK>"]
VOCAB = SPECIALS + AA_STANDARD

STOI: Dict[str, int] = {ch: i for i, ch in enumerate(VOCAB)}
PAD_IDX = STOI["<PAD>"]
UNK_IDX = STOI["<UNK>"]


def load_and_align(path: str) -> pd.DataFrame:
    """
    Match your Colab cleaning:
    - keep WT_cluster as string
    - drop rows missing aa_seq/deltaG/WT_cluster
    - add seq_len
    """
    df = pd.read_csv(path, low_memory=False)

    missing = [c for c in SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df = df[SCHEMA].copy()
    df = df.dropna(subset=["aa_seq", "deltaG", "WT_cluster"])

    df["aa_seq"] = df["aa_seq"].astype(str)
    df["WT_cluster"] = df["WT_cluster"].astype(str)

    if "mut_type" in df.columns:
        df["mut_type"] = df["mut_type"].astype(str)

    df["seq_len"] = df["aa_seq"].str.len()
    return df


def compute_max_len(train_df: pd.DataFrame, val_df: pd.DataFrame, q: float, cap: int) -> int:
    """
    Match your Colab behavior:
      MAX_LEN = int(max(train_q, val_q))
      MAX_LEN = max(10, min(MAX_LEN, cap))
    """
    train_q = train_df["seq_len"].quantile(q)
    val_q = val_df["seq_len"].quantile(q)

    max_len = int(max(train_q, val_q))
    max_len = max(10, min(max_len, cap))
    return max_len


def encode_seq(seq: str) -> List[int]:
    return [STOI.get(ch, UNK_IDX) for ch in seq]


def pad_trunc(ids: List[int], max_len: int) -> List[int]:
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [PAD_IDX] * (max_len - len(ids))


class ProteinDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_len: int):
        self.seqs = df["aa_seq"].tolist()
        self.targets = df["deltaG"].astype(float).values
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = pad_trunc(encode_seq(self.seqs[idx]), self.max_len)
        x = torch.tensor(ids, dtype=torch.long)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y
