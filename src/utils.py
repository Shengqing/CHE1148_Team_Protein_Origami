from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class DeviceConfig:
    device: torch.device

    @staticmethod
    def auto() -> "DeviceConfig":
        return DeviceConfig(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )


"""Utility functions for memory management, loading, and sequence processing."""
import io
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .constants import AA_VOCAB


def robust_pt_load(path: str) -> Dict[str, Any]:
    """Safety utility for loading PyTorch files."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:

        class CPUUnpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str) -> Any:
                if module == "torch.storage" and name == "_load_from_bytes":
                    return lambda b: torch.load(
                        io.BytesIO(b), map_location="cpu"
                    )
                return super().find_class(module, name)

        with open(path, "rb") as f:
            return CPUUnpickler(f).load()


def convert_to_memmap(pt_path: str, out_mmap_path: str) -> Dict[str, Any]:
    """Converts a standard .pt embedding file to a memmap."""
    data_obj = robust_pt_load(pt_path)
    emb_array = np.array(data_obj["seq_embeddings"], dtype=np.float32)
    shape = emb_array.shape

    np.save(f"{out_mmap_path}_shape.npy", np.array(shape))

    fp = np.memmap(out_mmap_path, dtype="float32", mode="w+", shape=shape)
    fp[:] = emb_array[:]
    fp.flush()

    del fp
    check = np.memmap(out_mmap_path, dtype="float32", mode="r", shape=shape)
    assert check.shape == shape, f"Memmap shape mismatch for {out_mmap_path}"
    print(
        f"Successfully converted {pt_path} to memmap "
        f"at {out_mmap_path} with shape {shape}"
    )
    return data_obj


def seq_to_tensor(
    seqs: List[str], max_len: int, device: torch.device
) -> torch.Tensor:
    """Converts amino acid sequences to padded tensors of indices."""
    tensor = torch.full(
        (len(seqs), max_len), 20, dtype=torch.long, device=device
    )
    for i, seq in enumerate(seqs):
        indices = [AA_VOCAB.get(aa.upper(), 20) for aa in seq[:max_len]]
        tensor[i, : len(indices)] = torch.tensor(indices, device=device)
    return tensor


def batch_decode_and_count_mutations(
    logits: torch.Tensor, wt_seqs: List[str]
) -> List[Tuple[str, int]]:
    """Vectorized decoding and mutation counting for a batch of sequences."""
    inv_vocab = {v: k for k, v in AA_VOCAB.items()}
    indices = torch.argmax(logits, dim=-1)

    batch_results = []
    for i, wt_seq in enumerate(wt_seqs):
        length = len(wt_seq)
        seq_indices = indices[i, :length]

        generated_seq = ""
        mutations = 0
        for j in range(length):
            char = inv_vocab.get(seq_indices[j].item(), "X")
            generated_seq += char
            if char != wt_seq[j]:
                mutations += 1
        batch_results.append((generated_seq, mutations))

    return batch_results
