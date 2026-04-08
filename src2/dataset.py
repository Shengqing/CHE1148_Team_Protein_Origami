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
