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
    Loads data from a CSV file, enforces the required schema, and cleans the data.
    Drops rows missing essential fields ('aa_seq', 'deltaG', 'WT_cluster') and
    computes the sequence length for each row.
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


def compute_max_len(
    train_df: pd.DataFrame, val_df: pd.DataFrame, q: float, cap: int
) -> int:
    """
    Computes the maximum sequence length to use for padding/truncating based on
    the given quantile of sequence lengths in the training and validation datasets.
    The computed length is bounded between 10 and the specified cap.
    """
    train_q = train_df["seq_len"].quantile(q)
    val_q = val_df["seq_len"].quantile(q)

    max_len = int(max(train_q, val_q))
    max_len = max(10, min(max_len, cap))
    return max_len


def encode_seq(seq: str) -> List[int]:
    """
    Encodes an amino acid sequence into a list of integer indices based on the vocabulary.
    Unknown characters are mapped to the <UNK> token index.
    """
    return [STOI.get(ch, UNK_IDX) for ch in seq]


def pad_trunc(ids: List[int], max_len: int) -> List[int]:
    """
    Pads or truncates a list of token indices to match the specified maximum length.
    Sequences longer than max_len are truncated; shorter sequences are right-padded with <PAD>.
    """
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [PAD_IDX] * (max_len - len(ids))


class ProteinDataset(Dataset):
    """
    Dataset class for loading amino acid sequences and their corresponding deltaG values.
    Handles encoding, padding, and truncating sequences to a fixed maximum length.
    """

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


"""Dataset class for Protein VAE."""
from typing import Any, Dict, Optional

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from .utils import robust_pt_load


class ProteinDataset(Dataset):
    """Dataset for loading memory-mapped embeddings and scaled labels."""

    def __init__(
        self,
        pt_path: str,
        mmap_path: str,
        scaler: Optional[StandardScaler] = None,
    ):
        print(f"Loading Dataset: {pt_path}")
        data_obj = robust_pt_load(pt_path)

        shape = tuple(np.load(f"{mmap_path}_shape.npy"))
        self.embeddings = np.memmap(
            mmap_path, dtype="float32", mode="r", shape=shape
        )

        labels_np = np.array(data_obj["delta_g"]).reshape(-1, 1)
        self.clusters = np.array(data_obj["wt_cluster"])
        self.mut_types = [str(m).lower().strip() for m in data_obj["mut_type"]]

        if "aa_seq" in data_obj:
            self.sequences = data_obj["aa_seq"]
        elif "aa_seq_full" in data_obj:
            self.sequences = data_obj["aa_seq_full"]
        else:
            self.sequences = ["" for _ in range(len(self.embeddings))]

        self.wt_lookup: Dict[str, Dict[str, Any]] = {}
        for i, mtype in enumerate(self.mut_types):
            if mtype in ["wt", "wildtype", "wild-type"]:
                self.wt_lookup[self.clusters[i]] = {
                    "dg": labels_np[i][0],
                    "emb": self.embeddings[i],
                    "seq": self.sequences[i],
                }

        all_unique_clusters = np.unique(self.clusters)
        for c in all_unique_clusters:
            if c not in self.wt_lookup:
                idx = np.where(self.clusters == c)[0][0]
                self.wt_lookup[c] = {
                    "dg": labels_np[idx][0],
                    "emb": self.embeddings[idx],
                    "seq": self.sequences[idx],
                }

        if scaler is None:
            self.scaler = StandardScaler()
            self.labels_scaled = self.scaler.fit_transform(labels_np)
        else:
            self.scaler = scaler
            self.labels_scaled = self.scaler.transform(labels_np)

        self.labels_scaled = torch.from_numpy(self.labels_scaled).float()

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "emb": torch.from_numpy(np.array(self.embeddings[idx])).float(),
            "label_scaled": self.labels_scaled[idx],
        }
