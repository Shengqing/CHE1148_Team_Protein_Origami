#!/usr/bin/env python3
import argparse
import csv
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import random
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    import matplotlib.pyplot as plt
    import numpy as np

    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ALPHABET)}
UNK_IDX = len(AA_ALPHABET)

MUT_PAT = __import__("re").compile(r"^([A-Z\*])(\d+)([A-Z\*])$")


@dataclass
class Record:
    aa_seq: str
    delta_g: float
    wt_cluster: str
    mut_type: str


class SequenceDataset(Dataset):
    def __init__(self, records: Sequence[Record]):
        self.records = list(records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def setup_logging(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_eval.log"
    logger = logging.getLogger("graphnet")
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


def write_history_csv(path: Path, history: Sequence[dict]):
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(history[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in history:
            w.writerow(row)


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def aa_idx(aa: str) -> int:
    return AA_TO_IDX.get(aa, UNK_IDX)


def encode_sequence(seq: str) -> List[int]:
    return [aa_idx(ch) for ch in seq]


def collate_fn(batch: Sequence[Record]):
    lengths = [len(x.aa_seq) for x in batch]
    max_len = max(lengths)
    token_ids = torch.full((len(batch), max_len), UNK_IDX, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.float32)
    y = torch.zeros((len(batch),), dtype=torch.float32)

    for i, rec in enumerate(batch):
        ids = encode_sequence(rec.aa_seq)
        n = len(ids)
        token_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
        mask[i, :n] = 1.0
        y[i] = rec.delta_g

    return {
        "token_ids": token_ids,
        "mask": mask,
        "y": y,
    }


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GraphNetLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.edge_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_dim, dropout)
        self.node_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_dim, dropout)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden_dim = h.shape

        if seq_len < 2:
            return h

        left = h[:, :-1, :]
        right = h[:, 1:, :]

        m_lr = self.edge_mlp(torch.cat([left, right], dim=-1))
        m_rl = self.edge_mlp(torch.cat([right, left], dim=-1))

        agg = torch.zeros_like(h)
        agg[:, 1:, :] += m_lr
        agg[:, :-1, :] += m_rl

        updated = self.node_mlp(torch.cat([h, agg], dim=-1))
        updated = updated * mask.unsqueeze(-1)
        return h + updated


class GraphNetRegressor(nn.Module):
    def __init__(
        self, hidden_dim: int = 128, n_layers: int = 4, dropout: float = 0.1, max_len: int = 512
    ):
        super().__init__()
        self.aa_embed = nn.Embedding(len(AA_ALPHABET) + 1, hidden_dim)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)
        self.layers = nn.ModuleList([GraphNetLayer(hidden_dim, dropout) for _ in range(n_layers)])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor):
        bsz, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(bsz, -1)
        h = self.aa_embed(token_ids) + self.pos_embed(positions)
        h = h * mask.unsqueeze(-1)

        for layer in self.layers:
            h = layer(h, mask)

        denom = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)
        pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / denom
        out = self.readout(pooled).squeeze(-1)
        return out


def read_records(path: Path) -> List[Record]:
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
    return out


def split_train_dev(
    records: List[Record], dev_frac: float, seed: int
) -> Tuple[List[Record], List[Record]]:
    rnd = random.Random(seed)
    by_cluster: Dict[str, List[Record]] = {}
    for rec in records:
        by_cluster.setdefault(rec.wt_cluster, []).append(rec)

    train, dev = [], []
    for cluster, vals in by_cluster.items():
        vals_copy = vals[:]
        rnd.shuffle(vals_copy)
        n_dev = int(round(len(vals_copy) * dev_frac))
        if len(vals_copy) >= 10 and n_dev < 1:
            n_dev = 1
        if n_dev >= len(vals_copy):
            n_dev = max(0, len(vals_copy) - 1)
        dev.extend(vals_copy[:n_dev])
        train.extend(vals_copy[n_dev:])
    return train, dev


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


def regression_metrics_from_lists(y_true_list: List[float], y_pred_list: List[float]) -> dict:
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
            token_ids = batch["token_ids"].to(device)
            mask = batch["mask"].to(device)
            y = batch["y"].to(device)
            p = model(token_ids, mask)
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


def predict_sequences(
    model: nn.Module, seqs: Sequence[str], device: torch.device, batch_size: int = 1024
) -> List[float]:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i : i + batch_size]
            lengths = [len(s) for s in batch]
            max_len = max(lengths)
            token_ids = torch.full((len(batch), max_len), UNK_IDX, dtype=torch.long, device=device)
            mask = torch.zeros((len(batch), max_len), dtype=torch.float32, device=device)
            for j, seq in enumerate(batch):
                ids = encode_sequence(seq)
                token_ids[j, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
                mask[j, : len(ids)] = 1.0
            pred = model(token_ids, mask).detach().cpu().tolist()
            out.extend(pred)
    return out


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


def evaluate_generated_vs_observed(
    model: nn.Module,
    val_records: Sequence[Record],
    device: torch.device,
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
        cand_pred = predict_sequences(model, candidates, device=device, batch_size=4096)
        idx_sorted = sorted(range(len(candidates)), key=lambda i: cand_pred[i])

        wt_pred = predict_sequences(model, [wt_seq], device=device, batch_size=1)[0]
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
                if matched_true_rank and HAS_PLOTTING
                else None
            ),
            "median_true_rank_percentile": (
                float(np.median(np.array(matched_true_rank_pct)))
                if matched_true_rank_pct and HAS_PLOTTING
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


def plot_diagnostics(
    out_dir: Path,
    history: Sequence[dict],
    y_true: List[float],
    y_pred: List[float],
    gen_summary: dict,
    gen_rows: Sequence[dict],
):
    if not HAS_PLOTTING:
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if history:
        epochs = [h["epoch"] for h in history]
        train_mse = [h["train_mse"] for h in history]
        dev_rmse = [h["dev_rmse"] for h in history]
        dev_mae = [h["dev_mae"] for h in history]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_mse, label="Train MSE")
        plt.plot(epochs, dev_rmse, label="Dev RMSE")
        plt.plot(epochs, dev_mae, label="Dev MAE")
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.title("Training Curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "training_curves.png", dpi=180)
        plt.close()

    if y_true and y_pred:
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        lo = min(float(y_true_np.min()), float(y_pred_np.min()))
        hi = max(float(y_true_np.max()), float(y_pred_np.max()))

        plt.figure(figsize=(6, 6))
        plt.scatter(y_true_np, y_pred_np, s=6, alpha=0.25)
        plt.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        plt.xlabel("True ΔG")
        plt.ylabel("Predicted ΔG")
        plt.title("Validation Parity Plot")
        plt.tight_layout()
        plt.savefig(fig_dir / "val_parity_plot.png", dpi=180)
        plt.close()

        residuals = y_pred_np - y_true_np
        plt.figure(figsize=(7, 4.5))
        plt.hist(residuals, bins=60)
        plt.xlabel("Residual (Pred - True)")
        plt.ylabel("Count")
        plt.title("Validation Residual Histogram")
        plt.tight_layout()
        plt.savefig(fig_dir / "val_residual_hist.png", dpi=180)
        plt.close()

    hit_curve = gen_summary.get("hit_curve", [])
    if hit_curve:
        ks = [x["k"] for x in hit_curve]
        match_rate = [x["match_rate_over_generated_pct"] for x in hit_curve]
        wt_any = [x["wt_with_any_match_pct"] for x in hit_curve]

        plt.figure(figsize=(7, 4.5))
        plt.plot(ks, match_rate, marker="o", label="Generated match rate (%)")
        plt.plot(ks, wt_any, marker="o", label="WT with any match (%)")
        plt.xlabel("Top-k generated per WT")
        plt.ylabel("Percent")
        plt.title("Generative Retrieval Hit Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "generative_hit_curve.png", dpi=180)
        plt.close()

    matched_rank_pct = [
        float(r["true_rank_percentile_if_matched"])
        for r in gen_rows
        if r.get("is_match_in_validation_variants")
        and r.get("true_rank_percentile_if_matched") is not None
    ]
    if matched_rank_pct:
        plt.figure(figsize=(7, 4.5))
        plt.hist(matched_rank_pct, bins=40)
        plt.xlabel("True rank percentile among observed variants (lower is better)")
        plt.ylabel("Count")
        plt.title("Quality of Matched Generated Variants")
        plt.tight_layout()
        plt.savefig(fig_dir / "matched_variant_rank_percentile_hist.png", dpi=180)
        plt.close()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_items = 0
    loss_fn = nn.MSELoss()

    for batch in loader:
        token_ids = batch["token_ids"].to(device)
        mask = batch["mask"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()
        pred = model(token_ids, mask)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * y.shape[0]
        n_items += int(y.shape[0])

    return total_loss / max(n_items, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Train GraphNet generative model for Protein Origami"
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
        default=Path("/home/uoftshen/scratch/CHE1148_Team_Protein_Origami/results/graphnet"),
    )
    parser.add_argument(
        "--mode", type=str, choices=["train_eval", "eval_only"], default="train_eval"
    )
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--dev_frac", type=float, default=0.1)
    parser.add_argument("--gen_top_k", type=int, default=10)
    parser.add_argument("--eval_k_list", type=str, default="1,3,5,10,20")
    args = parser.parse_args()

    seed_everything(args.seed)
    logger = setup_logging(args.out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"mode={args.mode} device={device}")

    train_records_all = read_records(args.train_csv)
    val_records = read_records(args.val_csv)
    max_len = max(len(r.aa_seq) for r in (train_records_all + val_records))
    val_loader = DataLoader(
        SequenceDataset(val_records),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    train_records, dev_records = split_train_dev(
        train_records_all, dev_frac=args.dev_frac, seed=args.seed
    )
    train_loader = DataLoader(
        SequenceDataset(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        SequenceDataset(dev_records),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    checkpoint_state = None
    checkpoint_cfg = None
    if args.mode == "eval_only":
        model_path = args.model_path or (args.out_dir / "graphnet_model.pt")
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found for eval_only mode: {model_path}")
        checkpoint_state = torch.load(model_path, map_location="cpu")
        checkpoint_hidden_dim = int(checkpoint_state["aa_embed.weight"].shape[1])
        checkpoint_max_len = int(checkpoint_state["pos_embed.weight"].shape[0])
        layer_ids = {
            int(k.split(".")[1])
            for k in checkpoint_state.keys()
            if k.startswith("layers.") and len(k.split(".")) > 2 and k.split(".")[1].isdigit()
        }
        checkpoint_layers = (max(layer_ids) + 1) if layer_ids else args.layers
        checkpoint_cfg = {
            "hidden_dim": checkpoint_hidden_dim,
            "layers": checkpoint_layers,
            "max_len": checkpoint_max_len,
        }

    model_hidden_dim = checkpoint_cfg["hidden_dim"] if checkpoint_cfg else args.hidden_dim
    model_layers = checkpoint_cfg["layers"] if checkpoint_cfg else args.layers
    model_max_len = checkpoint_cfg["max_len"] if checkpoint_cfg else (max_len + 2)

    model = GraphNetRegressor(
        hidden_dim=model_hidden_dim,
        n_layers=model_layers,
        dropout=args.dropout,
        max_len=model_max_len,
    ).to(device)

    history = []

    if args.mode == "train_eval":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_dev_rmse = float("inf")
        best_state = None

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            dev_metrics = eval_regression(model, dev_loader, device)
            epoch_row = {
                "epoch": epoch,
                "train_mse": train_loss,
                "dev_rmse": dev_metrics["rmse"],
                "dev_mae": dev_metrics["mae"],
                "dev_r2": dev_metrics["r2"],
                "dev_pearson": dev_metrics["pearson"],
                "dev_spearman": dev_metrics["spearman"],
            }
            history.append(epoch_row)
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
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
            torch.save(model.state_dict(), args.out_dir / "graphnet_model.pt")
    else:
        model_path = args.model_path or (args.out_dir / "graphnet_model.pt")
        state = (
            checkpoint_state
            if checkpoint_state is not None
            else torch.load(model_path, map_location="cpu")
        )
        model.load_state_dict(state)
        logger.info(f"Loaded model checkpoint: {model_path}")

    val_metrics_full = eval_regression(model, val_loader, device, return_arrays=True)
    y_true = val_metrics_full.pop("y_true")
    y_pred = val_metrics_full.pop("y_pred")

    eval_k_list = [int(x.strip()) for x in args.eval_k_list.split(",") if x.strip()]
    gen_metrics, gen_rows = evaluate_generated_vs_observed(
        model,
        val_records,
        device,
        top_k=args.gen_top_k,
        k_list=eval_k_list,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_generated_details(
        args.out_dir / f"generated_top{args.gen_top_k}_vs_val_details.csv", gen_rows
    )
    write_history_csv(args.out_dir / "training_history.csv", history)
    plot_diagnostics(args.out_dir, history, y_true, y_pred, gen_metrics, gen_rows)

    report = {
        "config": {
            "mode": args.mode,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "dropout": args.dropout,
            "dev_frac": args.dev_frac,
            "device": str(device),
            "gen_top_k": args.gen_top_k,
            "eval_k_list": eval_k_list,
            "plotting_enabled": HAS_PLOTTING,
        },
        "data": {
            "train_sampled_n": len(train_records_all),
            "train_inner_n": len(train_records),
            "dev_inner_n": len(dev_records),
            "val_n": len(val_records),
        },
        "val_regression_metrics": val_metrics_full,
        "generative_eval_against_observed_val": gen_metrics,
        "training_history": history,
    }

    with (args.out_dir / "metrics.json").open("w") as f:
        json.dump(report, f, indent=2)

    logger.info("Final validation metrics:")
    for k, v in val_metrics_full.items():
        logger.info(f"  {k}: {v}")

    logger.info("Generative evaluation against observed validation variants:")
    logger.info(f"  n_unique_wt_proteins: {gen_metrics['n_unique_wt_proteins']}")
    logger.info(f"  ood_cluster_71_rows: {gen_metrics['ood_clusters']['71']['row_count']}")
    logger.info(f"  ood_cluster_213_rows: {gen_metrics['ood_clusters']['213']['row_count']}")
    logger.info(f"  top3_vs_wt_predicted_deltaG: {gen_metrics['top3_vs_wt_predicted_deltaG']}")
    logger.info(
        f"  exhaustive_generated_vs_observed.matched_prediction_vs_actual: {gen_metrics['exhaustive_generated_vs_observed']['matched_prediction_vs_actual']}"
    )
    logger.info(
        f"  exhaustive_generated_vs_observed.unmatched_distribution_vs_observed_range: {gen_metrics['exhaustive_generated_vs_observed']['unmatched_distribution_vs_observed_range']}"
    )
    logger.info(f"  topk_evaluation: {gen_metrics['topk_evaluation']}")
    logger.info(f"  suggested_k_from_hit_curve: {gen_metrics['suggested_k_from_hit_curve']}")

    logger.info(f"Saved model/metrics/artifacts under: {args.out_dir}")


if __name__ == "__main__":
    main()
