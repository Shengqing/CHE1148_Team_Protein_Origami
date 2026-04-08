#!/usr/bin/env python3
import argparse
import csv
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

MUT_PAT = __import__("re").compile(r"^([A-Z\*])(\d+)([A-Z\*])$")


@dataclass
class Record:
    aa_seq: str
    delta_g: float
    wt_cluster: str
    mut_type: str


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, y: torch.Tensor):
        self.embeddings = embeddings
        self.y = y

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, idx):
        return {
            "embedding": self.embeddings[idx],
            "y": self.y[idx],
        }


class MLPRegressor(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: int = 1024, dropout: float = 0.15
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, embedding: torch.Tensor):
        return self.net(embedding).squeeze(-1)


class ESM2Embedder:
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        cache_dir: Optional[Path],
        logger: logging.Logger,
    ):
        try:
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "transformers is required for ESM2 embeddings. Install with: pip install transformers"
            ) from exc

        self.device = device
        self.logger = logger
        cache_dir_str = str(cache_dir) if cache_dir is not None else None
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir_str
        )
        self.model = AutoModel.from_pretrained(
            model_name, cache_dir=cache_dir_str
        )
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            raise RuntimeError("Could not determine hidden_size from ESM2 model config")
        self.embedding_dim = int(hidden_size)

    def embed_sequences_pooled(
        self, seqs: Sequence[str], batch_size: int = 16
    ) -> torch.Tensor:
        pooled_out = []
        with torch.no_grad():
            for i in range(0, len(seqs), batch_size):
                batch = list(seqs[i : i + batch_size])
                tok = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                    add_special_tokens=True,
                )
                tok = {k: v.to(self.device) for k, v in tok.items()}
                out = self.model(**tok)
                h = out.last_hidden_state
                attn = tok["attention_mask"]

                batch_pooled = []
                for row_idx, seq in enumerate(batch):
                    seq_len = len(seq)
                    token_start = 1
                    token_end = token_start + seq_len
                    h_seq = h[row_idx, token_start:token_end, :]
                    if h_seq.shape[0] == 0:
                        valid = attn[row_idx].nonzero(as_tuple=False).flatten()
                        h_seq = h[row_idx, valid, :]
                    pooled = h_seq.mean(dim=0)
                    batch_pooled.append(pooled)
                pooled_out.append(torch.stack(batch_pooled, dim=0).detach().cpu())
        return torch.cat(pooled_out, dim=0)


def setup_logging(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_eval_esm2.log"
    logger = logging.getLogger("esm2_regressor")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def read_records(path: Path, max_rows: Optional[int] = None) -> List[Record]:
    out = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(
                Record(
                    aa_seq=row["aa_seq"],
                    delta_g=float(row["deltaG"]),
                    wt_cluster=row["WT_cluster"],
                    mut_type=row["mut_type"],
                )
            )
            if max_rows is not None and len(out) >= max_rows:
                break
    return out


def split_train_dev_indices(
    records: List[Record], dev_frac: float, seed: int
) -> Tuple[List[int], List[int]]:
    rnd = random.Random(seed)
    by_cluster: Dict[str, List[int]] = {}
    for idx, rec in enumerate(records):
        by_cluster.setdefault(rec.wt_cluster, []).append(idx)

    train_idx, dev_idx = [], []
    for _cluster, idxs in by_cluster.items():
        idxs_copy = idxs[:]
        rnd.shuffle(idxs_copy)
        n_dev = int(round(len(idxs_copy) * dev_frac))
        if len(idxs_copy) >= 10 and n_dev < 1:
            n_dev = 1
        if n_dev >= len(idxs_copy):
            n_dev = max(0, len(idxs_copy) - 1)
        dev_idx.extend(idxs_copy[:n_dev])
        train_idx.extend(idxs_copy[n_dev:])
    return train_idx, dev_idx


def rmse(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - y) ** 2)).item())


def mae(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred - y)).item())


def r2_score(pred: torch.Tensor, y: torch.Tensor) -> float:
    y_mean = torch.mean(y)
    ss_res = torch.sum((y - pred) ** 2)
    ss_tot = torch.sum((y - y_mean) ** 2)
    if float(ss_tot.item()) == 0.0:
        return float("nan")
    return float((1.0 - ss_res / ss_tot).item())


def pearson(pred: torch.Tensor, y: torch.Tensor) -> float:
    x = pred - pred.mean()
    z = y - y.mean()
    denom = torch.sqrt(torch.sum(x * x) * torch.sum(z * z))
    if float(denom.item()) == 0.0:
        return float("nan")
    return float((torch.sum(x * z) / denom).item())


def rankdata(values: List[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman(pred: torch.Tensor, y: torch.Tensor) -> float:
    px = pred.detach().cpu().tolist()
    py = y.detach().cpu().tolist()
    rx = torch.tensor(rankdata(px), dtype=torch.float32)
    ry = torch.tensor(rankdata(py), dtype=torch.float32)
    return pearson(rx, ry)


def regression_metrics_from_lists(
    y_true_list: List[float], y_pred_list: List[float]
) -> dict:
    if len(y_true_list) == 0:
        return {
            "n": 0,
            "mae": None,
            "rmse": None,
            "r2": None,
            "pearson": None,
            "spearman": None,
        }

    y_true = torch.tensor(y_true_list, dtype=torch.float32)
    y_pred = torch.tensor(y_pred_list, dtype=torch.float32)
    return {
        "n": int(y_true.shape[0]),
        "mae": mae(y_pred, y_true),
        "rmse": rmse(y_pred, y_true),
        "r2": r2_score(y_pred, y_true),
        "pearson": pearson(y_pred, y_true),
        "spearman": spearman(y_pred, y_true),
    }


def parse_mut(mut_type: str):
    m = MUT_PAT.match(mut_type)
    if not m:
        return None
    wt_aa, pos, mut_aa = m.group(1), int(m.group(2)), m.group(3)
    return wt_aa, pos, mut_aa


def reconstruct_wt_from_mutant(mutant_seq: str, mut_type: str):
    parsed = parse_mut(mut_type)
    if parsed is None:
        return None
    wt_aa, pos, _ = parsed
    idx = pos - 1
    if idx < 0 or idx >= len(mutant_seq):
        return None
    arr = list(mutant_seq)
    arr[idx] = wt_aa
    return "".join(arr)


AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def generate_all_single_mutants(wt_seq: str) -> List[str]:
    muts = []
    arr = list(wt_seq)
    for i, wt_aa in enumerate(arr):
        for aa in AA_ALPHABET:
            if aa == wt_aa:
                continue
            b = arr.copy()
            b[i] = aa
            muts.append("".join(b))
    return muts


def precompute_embeddings(
    embedder: ESM2Embedder,
    records: Sequence[Record],
    out_path: Path,
    batch_size: int,
    logger: logging.Logger,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seqs = [r.aa_seq for r in records]
    logger.info("Precomputing embeddings: %s n=%d", out_path, len(seqs))
    emb = embedder.embed_sequences_pooled(seqs, batch_size=batch_size).to(torch.float32)
    payload = {
        "seq_embeddings": emb,
        "delta_g": torch.tensor([r.delta_g for r in records], dtype=torch.float32),
        "wt_cluster": [r.wt_cluster for r in records],
        "mut_type": [r.mut_type for r in records],
        "aa_seq": seqs,
        "embedding_model": "facebook/esm2_t33_650M_UR50D",
        "embedding_dim": int(emb.shape[1]),
    }
    torch.save(payload, out_path)
    logger.info("Saved embedding dataset: %s", out_path)


def load_embedding_payload(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")
    return torch.load(path, map_location="cpu")


def eval_regression(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    return_arrays: bool = False,
):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["embedding"].to(device)
            y = batch["y"].to(device)
            p = model(x)
            ys.append(y)
            preds.append(p)
    y_all = torch.cat(ys, dim=0)
    p_all = torch.cat(preds, dim=0)

    result = {
        "mae": mae(p_all, y_all),
        "rmse": rmse(p_all, y_all),
        "r2": r2_score(p_all, y_all),
        "pearson": pearson(p_all, y_all),
        "spearman": spearman(p_all, y_all),
        "n": int(y_all.shape[0]),
    }
    if return_arrays:
        result["y_true"] = y_all.detach().cpu().tolist()
        result["y_pred"] = p_all.detach().cpu().tolist()
    return result


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_items = 0
    loss_fn = nn.MSELoss()
    for batch in loader:
        x = batch["embedding"].to(device)
        y = batch["y"].to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * y.shape[0]
        n_items += int(y.shape[0])
    return total_loss / max(n_items, 1)


def make_predict_fn(
    model: nn.Module,
    embedder: ESM2Embedder,
    device: torch.device,
    batch_size: int,
):
    def _predict(seqs: Sequence[str]) -> List[float]:
        if len(seqs) == 0:
            return []
        model.eval()
        out_all = []
        with torch.no_grad():
            for i in range(0, len(seqs), batch_size):
                batch = seqs[i : i + batch_size]
                x = embedder.embed_sequences_pooled(batch, batch_size=min(batch_size, 32)).to(
                    device
                )
                pred = model(x).detach().cpu().tolist()
                out_all.extend(pred)
        return out_all

    return _predict


def evaluate_generated_vs_observed(
    predict_fn,
    val_records: Sequence[Record],
    top_k: int,
    k_list: Sequence[int],
):
    proteins: Dict[Tuple[str, str], List[Record]] = {}
    cluster_row_counts: Dict[str, int] = {}
    unparsed_rows = 0

    for rec in val_records:
        cluster_row_counts[rec.wt_cluster] = cluster_row_counts.get(rec.wt_cluster, 0) + 1
        wt_seq = reconstruct_wt_from_mutant(rec.aa_seq, rec.mut_type)
        if wt_seq is None:
            unparsed_rows += 1
            continue
        proteins.setdefault((rec.wt_cluster, wt_seq), []).append(rec)

    cluster_wt_counts: Dict[str, int] = {}
    for cluster, _wt in proteins.keys():
        cluster_wt_counts[cluster] = cluster_wt_counts.get(cluster, 0) + 1

    all_k = sorted({k for k in list(k_list) + [top_k] if k > 0})
    hit_stats = {k: {"matched": 0, "total": 0, "wt_with_any_match": 0} for k in all_k}

    top3_wt_improve_total = 0
    top3_wt_improve_by_cluster: Dict[str, Dict[str, float]] = {}

    exhaustive_matched_true = []
    exhaustive_matched_pred = []
    exhaustive_unmatched_total = 0
    exhaustive_unmatched_above_obs_max = 0
    exhaustive_unmatched_below_obs_min = 0
    exhaustive_by_cluster: Dict[str, Dict[str, float]] = {}

    detailed_rows = []
    matched_true_delta_g = []
    matched_true_rank = []
    matched_true_rank_pct = []

    for (cluster, wt_seq), obs_rows in proteins.items():
        if cluster not in top3_wt_improve_by_cluster:
            top3_wt_improve_by_cluster[cluster] = {
                "n_proteins": 0,
                "n_with_top3_lower_than_wt": 0,
            }
        top3_wt_improve_by_cluster[cluster]["n_proteins"] += 1
        if cluster not in exhaustive_by_cluster:
            exhaustive_by_cluster[cluster] = {
                "matched_n": 0,
                "unmatched_n": 0,
                "unmatched_above_obs_max_n": 0,
                "unmatched_below_obs_min_n": 0,
            }

        obs_map: Dict[str, float] = {}
        for rec in obs_rows:
            if rec.aa_seq not in obs_map:
                obs_map[rec.aa_seq] = rec.delta_g
            else:
                obs_map[rec.aa_seq] = min(obs_map[rec.aa_seq], rec.delta_g)

        ranked_obs = sorted(obs_map.items(), key=lambda x: x[1])
        rank_map = {seq: i + 1 for i, (seq, _dg) in enumerate(ranked_obs)}
        n_obs = len(ranked_obs)
        obs_min = ranked_obs[0][1]
        obs_max = ranked_obs[-1][1]

        candidates = generate_all_single_mutants(wt_seq)
        cand_pred = predict_fn(candidates)
        idx_sorted = sorted(range(len(candidates)), key=lambda i: cand_pred[i])

        wt_pred = predict_fn([wt_seq])[0]
        top3_idx = idx_sorted[:3]
        top3_has_lower_than_wt = any(cand_pred[i] < wt_pred for i in top3_idx)
        if top3_has_lower_than_wt:
            top3_wt_improve_total += 1
            top3_wt_improve_by_cluster[cluster]["n_with_top3_lower_than_wt"] += 1

        for i, seq in enumerate(candidates):
            pred_val = cand_pred[i]
            if seq in obs_map:
                exhaustive_matched_true.append(obs_map[seq])
                exhaustive_matched_pred.append(pred_val)
                exhaustive_by_cluster[cluster]["matched_n"] += 1
            else:
                exhaustive_unmatched_total += 1
                exhaustive_by_cluster[cluster]["unmatched_n"] += 1
                if pred_val > obs_max:
                    exhaustive_unmatched_above_obs_max += 1
                    exhaustive_by_cluster[cluster]["unmatched_above_obs_max_n"] += 1
                if pred_val < obs_min:
                    exhaustive_unmatched_below_obs_min += 1
                    exhaustive_by_cluster[cluster]["unmatched_below_obs_min_n"] += 1

        for k in all_k:
            top_idx = idx_sorted[:k]
            matches = [i for i in top_idx if candidates[i] in obs_map]
            hit_stats[k]["matched"] += len(matches)
            hit_stats[k]["total"] += k
            if matches:
                hit_stats[k]["wt_with_any_match"] += 1

        for gen_rank, idx in enumerate(idx_sorted[:top_k], start=1):
            seq = candidates[idx]
            pred_dg = cand_pred[idx]
            is_matched = seq in obs_map
            true_dg = obs_map.get(seq)
            true_rank = rank_map.get(seq)
            rank_pct = (100.0 * true_rank / n_obs) if true_rank is not None and n_obs > 0 else None

            if is_matched and true_dg is not None and true_rank is not None:
                matched_true_delta_g.append(true_dg)
                matched_true_rank.append(true_rank)
                matched_true_rank_pct.append(rank_pct)

            detailed_rows.append(
                {
                    "WT_cluster": cluster,
                    "wt_seq": wt_seq,
                    "generated_rank_by_pred": gen_rank,
                    "generated_seq": seq,
                    "pred_deltaG": pred_dg,
                    "is_match_in_validation_variants": is_matched,
                    "true_deltaG_if_matched": true_dg,
                    "true_rank_among_observed_variants_if_matched": true_rank,
                    "true_rank_percentile_if_matched": rank_pct,
                    "n_observed_variants_for_wt": n_obs,
                }
            )

    n_wt = len(proteins)
    topk_hits = hit_stats[top_k]
    topk_total = topk_hits["total"]
    topk_match = topk_hits["matched"]

    hit_curve = []
    for k in all_k:
        total = max(hit_stats[k]["total"], 1)
        wt_total = max(n_wt, 1)
        hit_curve.append(
            {
                "k": k,
                "match_rate_over_generated_pct": 100.0 * hit_stats[k]["matched"] / total,
                "wt_with_any_match": hit_stats[k]["wt_with_any_match"],
                "wt_with_any_match_pct": 100.0 * hit_stats[k]["wt_with_any_match"] / wt_total,
            }
        )

    max_wt_hit = max(x["wt_with_any_match_pct"] for x in hit_curve) if hit_curve else 0.0
    suggested_k = top_k
    for item in hit_curve:
        if item["wt_with_any_match_pct"] >= 0.9 * max_wt_hit:
            suggested_k = item["k"]
            break

    def cluster_sort_key(item):
        key = str(item[0])
        if key.isdigit():
            return (0, int(key), key)
        return (1, 0, key)

    top3_by_cluster_pct = {}
    for cluster, vals in top3_wt_improve_by_cluster.items():
        n_proteins_cluster = int(vals["n_proteins"])
        n_improved_cluster = int(vals["n_with_top3_lower_than_wt"])
        pct_cluster = 100.0 * n_improved_cluster / max(n_proteins_cluster, 1)
        top3_by_cluster_pct[cluster] = {
            "n_proteins": n_proteins_cluster,
            "n_with_top3_lower_than_wt": n_improved_cluster,
            "pct_with_top3_lower_than_wt": pct_cluster,
        }

    exhaustive_by_cluster_summary = {}
    for cluster, vals in exhaustive_by_cluster.items():
        unmatched_n = int(vals["unmatched_n"])
        exhaustive_by_cluster_summary[cluster] = {
            "matched_n": int(vals["matched_n"]),
            "unmatched_n": unmatched_n,
            "unmatched_above_obs_max_n": int(vals["unmatched_above_obs_max_n"]),
            "unmatched_below_obs_min_n": int(vals["unmatched_below_obs_min_n"]),
            "pct_unmatched_above_obs_max": 100.0
            * vals["unmatched_above_obs_max_n"]
            / max(unmatched_n, 1),
            "pct_unmatched_below_obs_min": 100.0
            * vals["unmatched_below_obs_min_n"]
            / max(unmatched_n, 1),
        }

    exhaustive_matched_metrics = regression_metrics_from_lists(
        exhaustive_matched_true, exhaustive_matched_pred
    )

    summary = {
        "n_validation_rows": len(val_records),
        "n_unparsed_rows_for_wt_reconstruction": unparsed_rows,
        "n_unique_wt_proteins": n_wt,
        "cluster_row_counts": dict(sorted(cluster_row_counts.items(), key=cluster_sort_key)),
        "cluster_wt_counts": dict(sorted(cluster_wt_counts.items(), key=cluster_sort_key)),
        "ood_clusters": {
            "71": {
                "row_count": cluster_row_counts.get("71", 0),
                "wt_count": cluster_wt_counts.get("71", 0),
            },
            "213": {
                "row_count": cluster_row_counts.get("213", 0),
                "wt_count": cluster_wt_counts.get("213", 0),
            },
        },
        "top3_vs_wt_predicted_deltaG": {
            "n_proteins": n_wt,
            "n_with_top3_lower_than_wt": top3_wt_improve_total,
            "pct_with_top3_lower_than_wt": 100.0 * top3_wt_improve_total / max(n_wt, 1),
            "by_cluster": dict(sorted(top3_by_cluster_pct.items(), key=cluster_sort_key)),
            "ood_cluster_breakdown": {
                "71": top3_by_cluster_pct.get(
                    "71",
                    {
                        "n_proteins": 0,
                        "n_with_top3_lower_than_wt": 0,
                        "pct_with_top3_lower_than_wt": 0.0,
                    },
                ),
                "213": top3_by_cluster_pct.get(
                    "213",
                    {
                        "n_proteins": 0,
                        "n_with_top3_lower_than_wt": 0,
                        "pct_with_top3_lower_than_wt": 0.0,
                    },
                ),
            },
        },
        "exhaustive_generated_vs_observed": {
            "matched_prediction_vs_actual": exhaustive_matched_metrics,
            "unmatched_distribution_vs_observed_range": {
                "unmatched_n": exhaustive_unmatched_total,
                "unmatched_above_observed_max_n": exhaustive_unmatched_above_obs_max,
                "unmatched_below_observed_min_n": exhaustive_unmatched_below_obs_min,
                "pct_unmatched_above_observed_max": 100.0
                * exhaustive_unmatched_above_obs_max
                / max(exhaustive_unmatched_total, 1),
                "pct_unmatched_below_observed_min": 100.0
                * exhaustive_unmatched_below_obs_min
                / max(exhaustive_unmatched_total, 1),
            },
            "by_cluster": dict(
                sorted(exhaustive_by_cluster_summary.items(), key=cluster_sort_key)
            ),
            "ood_cluster_breakdown": {
                "71": exhaustive_by_cluster_summary.get(
                    "71",
                    {
                        "matched_n": 0,
                        "unmatched_n": 0,
                        "unmatched_above_obs_max_n": 0,
                        "unmatched_below_obs_min_n": 0,
                        "pct_unmatched_above_obs_max": 0.0,
                        "pct_unmatched_below_obs_min": 0.0,
                    },
                ),
                "213": exhaustive_by_cluster_summary.get(
                    "213",
                    {
                        "matched_n": 0,
                        "unmatched_n": 0,
                        "unmatched_above_obs_max_n": 0,
                        "unmatched_below_obs_min_n": 0,
                        "pct_unmatched_above_obs_max": 0.0,
                        "pct_unmatched_below_obs_min": 0.0,
                    },
                ),
            },
        },
        "topk_evaluation": {
            "k": top_k,
            "n_generated": topk_total,
            "n_matched": topk_match,
            "n_unmatched": topk_total - topk_match,
            "match_rate_over_generated_pct": 100.0 * topk_match / max(topk_total, 1),
            "n_wt_with_any_match": topk_hits["wt_with_any_match"],
            "pct_wt_with_any_match": 100.0 * topk_hits["wt_with_any_match"] / max(n_wt, 1),
        },
        "matched_variant_quality": {
            "n_matched": len(matched_true_delta_g),
            "mean_true_deltaG": (sum(matched_true_delta_g) / len(matched_true_delta_g))
            if matched_true_delta_g
            else None,
            "median_true_rank": (
                float(np.median(np.array(matched_true_rank)))
                if matched_true_rank and HAS_NUMPY
                else None
            ),
            "median_true_rank_percentile": (
                float(np.median(np.array(matched_true_rank_pct)))
                if matched_true_rank_pct and HAS_NUMPY
                else None
            ),
        },
        "hit_curve": hit_curve,
        "suggested_k_from_hit_curve": suggested_k,
    }
    return summary, detailed_rows


def write_generated_details(path: Path, rows: Sequence[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "WT_cluster",
        "wt_seq",
        "generated_rank_by_pred",
        "generated_seq",
        "pred_deltaG",
        "is_match_in_validation_variants",
        "true_deltaG_if_matched",
        "true_rank_among_observed_variants_if_matched",
        "true_rank_percentile_if_matched",
        "n_observed_variants_for_wt",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def compare_with_baseline(
    baseline_metrics_path: Path, current_val_metrics: dict, out_path: Path
):
    baseline_val = None
    if baseline_metrics_path.exists():
        with baseline_metrics_path.open() as f:
            b = json.load(f)
            baseline_val = b.get("val_regression_metrics")

    comp = {
        "baseline_metrics_path": str(baseline_metrics_path),
        "baseline_val_regression_metrics": baseline_val,
        "esm2_val_regression_metrics": current_val_metrics,
        "delta": None,
    }

    if baseline_val is not None:
        keys = ["mae", "rmse", "r2", "pearson", "spearman"]
        delta = {}
        for k in keys:
            if k in baseline_val and k in current_val_metrics:
                delta[k] = current_val_metrics[k] - baseline_val[k]
        comp["delta"] = delta

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(comp, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Train ESM2-650M regressor for Protein Origami"
    )
    parser.add_argument(
        "--train_csv",
        type=Path,
        default=Path(
            "/home/uoftshen/scratch/CHE1148_Team_Protein_Origami/data/processed/tsuboyama_processed_train_sampled.csv"
        ),
    )
    parser.add_argument(
        "--val_csv",
        type=Path,
        default=Path(
            "/home/uoftshen/scratch/CHE1148_Team_Protein_Origami/data/processed/tsuboyama_processed_val_full.csv"
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/home/uoftshen/scratch/CHE1148_Team_Protein_Origami/results/esm2_650m"),
    )
    parser.add_argument(
        "--embedding_dir",
        type=Path,
        default=Path(
            "/home/uoftshen/scratch/CHE1148_Team_Protein_Origami/results/esm2_650m/embeddings"
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train_eval", "eval_only", "precompute_only"],
        default="train_eval",
    )
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mlp_hidden_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--dev_frac", type=float, default=0.1)
    parser.add_argument("--gen_top_k", type=int, default=10)
    parser.add_argument("--eval_k_list", type=str, default="1,3,5,10,20,50")
    parser.add_argument(
        "--esm_model_name", type=str, default="facebook/esm2_t33_650M_UR50D"
    )
    parser.add_argument("--esm_cache_dir", type=Path, default=None)
    parser.add_argument("--embed_batch_size", type=int, default=16)
    parser.add_argument("--predict_batch_size", type=int, default=256)
    parser.add_argument("--train_embed_path", type=Path, default=None)
    parser.add_argument("--val_embed_path", type=Path, default=None)
    parser.add_argument("--prepare_embeddings", action="store_true")
    parser.add_argument("--skip_generative_eval", action="store_true")
    parser.add_argument("--max_rows_train", type=int, default=None)
    parser.add_argument("--max_rows_val", type=int, default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    logger = setup_logging(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("mode=%s device=%s", args.mode, device)

    train_embed_path = args.train_embed_path or (
        args.embedding_dir / "train_sampled_esm2_650m.pt"
    )
    val_embed_path = args.val_embed_path or (
        args.embedding_dir / "val_full_esm2_650m.pt"
    )

    need_precompute = (
        args.prepare_embeddings
        or not train_embed_path.exists()
        or not val_embed_path.exists()
        or args.mode == "precompute_only"
    )
    train_records = None
    val_records = None
    embedder = None

    if need_precompute:
        train_records = read_records(args.train_csv, max_rows=args.max_rows_train)
        val_records = read_records(args.val_csv, max_rows=args.max_rows_val)
        embedder = ESM2Embedder(
            args.esm_model_name, device=device, cache_dir=args.esm_cache_dir, logger=logger
        )
        precompute_embeddings(
            embedder, train_records, train_embed_path, args.embed_batch_size, logger
        )
        precompute_embeddings(embedder, val_records, val_embed_path, args.embed_batch_size, logger)

    if args.mode == "precompute_only":
        logger.info("Precompute-only mode complete.")
        return

    train_payload = load_embedding_payload(train_embed_path)
    val_payload = load_embedding_payload(val_embed_path)
    train_embeddings = train_payload["seq_embeddings"].to(torch.float32)
    train_targets = train_payload["delta_g"].to(torch.float32)
    val_embeddings = val_payload["seq_embeddings"].to(torch.float32)
    val_targets = val_payload["delta_g"].to(torch.float32)

    if train_records is None:
        train_records = [
            Record(
                aa_seq=s,
                delta_g=float(d),
                wt_cluster=c,
                mut_type=m,
            )
            for s, d, c, m in zip(
                train_payload["aa_seq"],
                train_payload["delta_g"].tolist(),
                train_payload["wt_cluster"],
                train_payload["mut_type"],
            )
        ]

    if val_records is None:
        val_records = [
            Record(
                aa_seq=s,
                delta_g=float(d),
                wt_cluster=c,
                mut_type=m,
            )
            for s, d, c, m in zip(
                val_payload["aa_seq"],
                val_payload["delta_g"].tolist(),
                val_payload["wt_cluster"],
                val_payload["mut_type"],
            )
        ]

    train_idx, dev_idx = split_train_dev_indices(
        train_records, dev_frac=args.dev_frac, seed=args.seed
    )

    train_loader = DataLoader(
        EmbeddingDataset(train_embeddings[train_idx], train_targets[train_idx]),
        batch_size=args.batch_size,
        shuffle=True,
    )
    dev_loader = DataLoader(
        EmbeddingDataset(train_embeddings[dev_idx], train_targets[dev_idx]),
        batch_size=args.batch_size,
        shuffle=False,
    )
    val_loader = DataLoader(
        EmbeddingDataset(val_embeddings, val_targets),
        batch_size=args.batch_size,
        shuffle=False,
    )

    in_dim = int(train_embeddings.shape[1])
    model = MLPRegressor(
        in_dim=in_dim, hidden_dim=args.mlp_hidden_dim, dropout=args.dropout
    ).to(device)

    history = []
    if args.mode == "train_eval":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_dev_rmse = float("inf")
        best_state = None
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            dev_metrics = eval_regression(model, dev_loader, device)
            history.append(
                {
                    "epoch": epoch,
                    "train_mse": train_loss,
                    "dev_rmse": dev_metrics["rmse"],
                    "dev_mae": dev_metrics["mae"],
                    "dev_r2": dev_metrics["r2"],
                    "dev_pearson": dev_metrics["pearson"],
                    "dev_spearman": dev_metrics["spearman"],
                }
            )
            logger.info(
                "epoch=%d train_mse=%.4f dev_rmse=%.4f dev_mae=%.4f dev_r2=%.4f dev_pearson=%.4f dev_spearman=%.4f",
                epoch,
                train_loss,
                dev_metrics["rmse"],
                dev_metrics["mae"],
                dev_metrics["r2"],
                dev_metrics["pearson"],
                dev_metrics["spearman"],
            )
            if dev_metrics["rmse"] < best_dev_rmse:
                best_dev_rmse = dev_metrics["rmse"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
            save_path = args.out_dir / "esm2_regressor_model.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "in_dim": in_dim,
                    "mlp_hidden_dim": args.mlp_hidden_dim,
                    "dropout": args.dropout,
                    "esm_model_name": args.esm_model_name,
                },
                save_path,
            )
            logger.info("Saved model checkpoint: %s", save_path)
    else:
        model_path = args.model_path or (args.out_dir / "esm2_regressor_model.pt")
        ckpt = torch.load(model_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        logger.info("Loaded model checkpoint: %s", model_path)

    val_metrics_full = eval_regression(
        model, val_loader, device, return_arrays=True
    )
    val_metrics = {
        k: v
        for k, v in val_metrics_full.items()
        if k not in {"y_true", "y_pred"}
    }

    if embedder is None:
        embedder = ESM2Embedder(
            args.esm_model_name, device=device, cache_dir=args.esm_cache_dir, logger=logger
        )

    if args.skip_generative_eval:
        gen_metrics = {"skipped": True}
        gen_rows = []
    else:
        eval_k_list = [int(x.strip()) for x in args.eval_k_list.split(",") if x.strip()]
        predict_fn = make_predict_fn(
            model, embedder, device=device, batch_size=args.predict_batch_size
        )
        gen_metrics, gen_rows = evaluate_generated_vs_observed(
            predict_fn,
            val_records,
            top_k=args.gen_top_k,
            k_list=eval_k_list,
        )
        write_generated_details(
            args.out_dir / f"generated_top{args.gen_top_k}_vs_val_details.csv", gen_rows
        )

    report = {
        "config": {
            "mode": args.mode,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "mlp_hidden_dim": args.mlp_hidden_dim,
            "dropout": args.dropout,
            "dev_frac": args.dev_frac,
            "device": str(device),
            "gen_top_k": args.gen_top_k,
            "eval_k_list": [
            int(x.strip()) for x in args.eval_k_list.split(",") if x.strip()
        ],
            "embedding_model": args.esm_model_name,
            "embedding_train_path": str(train_embed_path),
            "embedding_val_path": str(val_embed_path),
        },
        "data": {
            "train_sampled_n": len(train_records),
            "train_inner_n": len(train_idx),
            "dev_inner_n": len(dev_idx),
            "val_n": len(val_records),
        },
        "val_regression_metrics": val_metrics,
        "generative_eval_against_observed_val": gen_metrics,
        "training_history": history,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "metrics.json").open("w") as f:
        json.dump(report, f, indent=2)

    baseline_path = Path(
        "/home/uoftshen/scratch/CHE1148_Team_Protein_Origami/results/graphnet/metrics.json"
    )
    compare_with_baseline(
        baseline_metrics_path=baseline_path,
        current_val_metrics=val_metrics,
        out_path=args.out_dir / "comparison_vs_graphnet.json",
    )

    logger.info("Final validation metrics: %s", val_metrics)
    logger.info("Saved artifacts under: %s", args.out_dir)


if __name__ == "__main__":
    main()
