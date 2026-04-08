#!/usr/bin/env python3
from collections import Counter, defaultdict
import csv
import math
from pathlib import Path
import random
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = PROJECT_ROOT / "data" / "raw" / "Tsuboyama2023_DS2and3_20230416_ColFiltered.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"

SEED = 42
VAL_FRAC_NON_FORCED = 0.10
FORCED_VAL_CLUSTERS = {"71", "213"}

TRAIN_SAMPLE_MAX = 60_000

FINAL_DROP_COLUMNS = {
    "source_row_index",
    "parent_WT_id",
    "parent_mapping_status",
    "collapsed_n_replicates",
    "collapsed_deltaG_ivw",
    "collapsed_deltaG_ivw_95CI_low",
    "collapsed_deltaG_ivw_95CI_high",
    "collapsed_deltaG_ivw_var",
    "collapsed_deltaG_mean_unweighted",
}

SUB_PAT = re.compile(r"^([A-Z\*])(\d+)([A-Z\*])$")
DEL_PAT = re.compile(r"^del([A-Z\*])(\d+)$", re.IGNORECASE)
INS_PAT = re.compile(r"^ins([A-Z\*])(\d+)$", re.IGNORECASE)


def parse_substitutions(mut_type: str):
    tokens = [t for t in re.split(r"[:,; ]+", mut_type.strip()) if t]
    if not tokens:
        return None
    parsed = []
    for tok in tokens:
        m = SUB_PAT.match(tok)
        if not m:
            return None
        wt_aa, pos, mut_aa = m.group(1), int(m.group(2)), m.group(3)
        parsed.append((wt_aa, pos, mut_aa))
    return parsed


def apply_substitutions(wt_seq: str, muts):
    arr = list(wt_seq)
    seq_len = len(arr)
    for wt_aa, pos, mut_aa in muts:
        idx = pos - 1
        if idx < 0 or idx >= seq_len:
            return None
        if arr[idx] != wt_aa:
            return None
        arr[idx] = mut_aa
    return "".join(arr)


def ci_width(row):
    try:
        return float(row["deltaG_95CI_high"]) - float(row["deltaG_95CI_low"])
    except (KeyError, TypeError, ValueError):
        return float("inf")


def row_variance_from_ci95(row):
    try:
        high = float(row["deltaG_95CI_high"])
        low = float(row["deltaG_95CI_low"])
        width = max(high - low, 1e-9)
        sigma = width / (2.0 * 1.96)
        return max(sigma * sigma, 1e-12)
    except (KeyError, TypeError, ValueError):
        return 1.0


def detect_single_substitution(mut_type: str):
    muts = parse_substitutions(mut_type)
    return muts is not None and len(muts) == 1


def map_parent_wt_id(row, wt_candidates_by_cluster):
    cluster = row["WT_cluster"]
    mut_type = row["mut_type"]
    aa_seq = row["aa_seq"]
    candidates = wt_candidates_by_cluster.get(cluster, [])

    if mut_type.lower() == "wt":
        for wt_id, wt_seq in candidates:
            if wt_seq == aa_seq:
                return wt_id, "wt_exact"
        return f"UNRESOLVED::{cluster}", "wt_unresolved"

    matched = []

    muts = parse_substitutions(mut_type)
    if muts is not None:
        for wt_id, wt_seq in candidates:
            out = apply_substitutions(wt_seq, muts)
            if out == aa_seq:
                matched.append(wt_id)
        if len(matched) == 1:
            return matched[0], "substitution_unique"
        if len(matched) > 1:
            return matched[0], "substitution_multi"

    m_del = DEL_PAT.match(mut_type)
    if m_del:
        del_aa = m_del.group(1)
        pos = int(m_del.group(2)) - 1
        for wt_id, wt_seq in candidates:
            if pos < 0 or pos >= len(wt_seq):
                continue
            if wt_seq[pos] != del_aa:
                continue
            if wt_seq[:pos] + wt_seq[pos + 1 :] == aa_seq:
                matched.append(wt_id)
        if len(matched) == 1:
            return matched[0], "deletion_unique"
        if len(matched) > 1:
            return matched[0], "deletion_multi"

    m_ins = INS_PAT.match(mut_type)
    if m_ins:
        ins_aa = m_ins.group(1)
        pos = int(m_ins.group(2)) - 1
        for wt_id, wt_seq in candidates:
            ok = False
            if 0 <= pos <= len(wt_seq):
                if wt_seq[:pos] + ins_aa + wt_seq[pos:] == aa_seq:
                    ok = True
            if not ok and 0 <= pos + 1 <= len(wt_seq):
                if wt_seq[: pos + 1] + ins_aa + wt_seq[pos + 1 :] == aa_seq:
                    ok = True
            if ok:
                matched.append(wt_id)
        if len(matched) == 1:
            return matched[0], "insertion_unique"
        if len(matched) > 1:
            return matched[0], "insertion_multi"

    return f"UNRESOLVED::{cluster}", "unresolved"


def stratified_split_indices(rows, forced_val_clusters, val_frac, seed):
    rnd = random.Random(seed)
    by_cluster = defaultdict(list)
    for i, row in enumerate(rows):
        by_cluster[row["WT_cluster"]].append(i)

    val_idx = set()
    train_idx = set()

    for cluster, idxs in by_cluster.items():
        if cluster in forced_val_clusters:
            val_idx.update(idxs)
            continue

        idxs_copy = idxs[:]
        rnd.shuffle(idxs_copy)
        n = len(idxs_copy)
        n_val = int(round(n * val_frac))
        if n >= 10 and n_val < 1:
            n_val = 1
        if n_val >= n:
            n_val = max(0, n - 1)

        val_here = idxs_copy[:n_val]
        train_here = idxs_copy[n_val:]
        val_idx.update(val_here)
        train_idx.update(train_here)

    return sorted(train_idx), sorted(val_idx)


def stratified_downsample(rows, max_n, seed):
    if len(rows) <= max_n:
        return rows
    rnd = random.Random(seed)
    by_cluster = defaultdict(list)
    for row in rows:
        by_cluster[row["WT_cluster"]].append(row)

    total = len(rows)
    targets = {}
    assigned = 0
    clusters = sorted(by_cluster.keys())
    for c in clusters:
        n = len(by_cluster[c])
        t = int(math.floor(max_n * (n / total)))
        if t < 1:
            t = 1
        t = min(t, n)
        targets[c] = t
        assigned += t

    while assigned > max_n:
        for c in sorted(clusters, key=lambda x: targets[x], reverse=True):
            if assigned <= max_n:
                break
            if targets[c] > 1:
                targets[c] -= 1
                assigned -= 1

    while assigned < max_n:
        for c in clusters:
            if assigned >= max_n:
                break
            if targets[c] < len(by_cluster[c]):
                targets[c] += 1
                assigned += 1

    out = []
    for c in clusters:
        vals = by_cluster[c][:]
        rnd.shuffle(vals)
        out.extend(vals[: targets[c]])
    rnd.shuffle(out)
    return out


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def drop_unneeded_columns(rows, cols_to_remove):
    out = []
    for row in rows:
        r = dict(row)
        for col in cols_to_remove:
            r.pop(col, None)
        out.append(r)
    return out


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    wt_candidates_by_cluster = defaultdict(list)
    wt_seq_to_id = {}
    cluster_wt_counter = Counter()

    with INPUT_CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["mut_type"].lower() != "wt":
                continue
            cluster = row["WT_cluster"]
            wt_seq = row["aa_seq"]
            key = (cluster, wt_seq)
            if key in wt_seq_to_id:
                continue
            cluster_wt_counter[cluster] += 1
            wt_id = f"WT::{cluster}::{cluster_wt_counter[cluster]:03d}"
            wt_seq_to_id[key] = wt_id
            wt_candidates_by_cluster[cluster].append((wt_id, wt_seq))

    grouped = {}
    mapping_status_counter = Counter()
    total_rows = 0

    with INPUT_CSV.open() as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            total_rows += 1

            parent_wt_id, mapping_status = map_parent_wt_id(
                row, wt_candidates_by_cluster
            )
            mapping_status_counter[mapping_status] += 1

            key = (parent_wt_id, row["aa_seq"])
            var = row_variance_from_ci95(row)
            w = 1.0 / var
            x = float(row["deltaG"])
            cw = ci_width(row)

            if key not in grouped:
                rep = dict(row)
                rep["source_row_index"] = str(row_idx)
                rep["parent_WT_id"] = parent_wt_id
                rep["parent_mapping_status"] = mapping_status
                grouped[key] = {
                    "rep": rep,
                    "rep_ci_width": cw,
                    "sum_w": w,
                    "sum_wx": w * x,
                    "n": 1,
                    "deltaG_values": [x],
                }
            else:
                g = grouped[key]
                g["sum_w"] += w
                g["sum_wx"] += w * x
                g["n"] += 1
                g["deltaG_values"].append(x)
                if cw < g["rep_ci_width"]:
                    rep = dict(row)
                    rep["source_row_index"] = str(row_idx)
                    rep["parent_WT_id"] = parent_wt_id
                    rep["parent_mapping_status"] = mapping_status
                    g["rep"] = rep
                    g["rep_ci_width"] = cw

    collapsed_rows = []
    for (_parent, _seq), g in grouped.items():
        rep = g["rep"]
        sum_w = g["sum_w"]
        mean_ivw = g["sum_wx"] / sum_w if sum_w > 0 else float(rep["deltaG"])
        var_ivw = 1.0 / sum_w if sum_w > 0 else 1.0
        sd_ivw = math.sqrt(var_ivw)
        ci95 = 1.96 * sd_ivw

        rep_out = dict(rep)
        rep_out["collapsed_n_replicates"] = str(g["n"])
        rep_out["collapsed_deltaG_ivw"] = f"{mean_ivw:.9f}"
        rep_out["collapsed_deltaG_ivw_95CI_low"] = f"{(mean_ivw - ci95):.9f}"
        rep_out["collapsed_deltaG_ivw_95CI_high"] = f"{(mean_ivw + ci95):.9f}"
        rep_out["collapsed_deltaG_ivw_var"] = f"{var_ivw:.12f}"
        rep_out["collapsed_deltaG_mean_unweighted"] = (
            f"{(sum(g['deltaG_values']) / len(g['deltaG_values'])):.9f}"
        )
        collapsed_rows.append(rep_out)

    collapsed_rows = drop_unneeded_columns(collapsed_rows, cols_to_remove={"dna_seq"})

    collapsed_fieldnames = list(collapsed_rows[0].keys())

    phase1_rows = [
        r
        for r in collapsed_rows
        if detect_single_substitution(r["mut_type"])
        and not r["parent_WT_id"].startswith("UNRESOLVED::")
    ]

    train_idx, val_idx = stratified_split_indices(
        phase1_rows,
        forced_val_clusters=FORCED_VAL_CLUSTERS,
        val_frac=VAL_FRAC_NON_FORCED,
        seed=SEED,
    )

    train_full = [phase1_rows[i] for i in train_idx]
    val_full = [phase1_rows[i] for i in val_idx]

    train_sampled = stratified_downsample(train_full, TRAIN_SAMPLE_MAX, seed=SEED)

    train_full_out = drop_unneeded_columns(
        train_full, cols_to_remove=FINAL_DROP_COLUMNS
    )
    val_full_out = drop_unneeded_columns(
        val_full, cols_to_remove=FINAL_DROP_COLUMNS
    )
    train_sampled_out = drop_unneeded_columns(
        train_sampled, cols_to_remove=FINAL_DROP_COLUMNS
    )

    phase1_fieldnames = [c for c in collapsed_fieldnames if c not in FINAL_DROP_COLUMNS]
    write_csv(
        OUT_DIR / "tsuboyama_processed_train_full.csv",
        phase1_fieldnames,
        train_full_out,
    )
    write_csv(
        OUT_DIR / "tsuboyama_processed_val_full.csv",
        phase1_fieldnames,
        val_full_out,
    )
    write_csv(
        OUT_DIR / "tsuboyama_processed_train_sampled.csv",
        phase1_fieldnames,
        train_sampled_out,
    )

    print("Processing complete.")
    print(f"Train full: {OUT_DIR / 'tsuboyama_processed_train_full.csv'}")
    print(f"Val full: {OUT_DIR / 'tsuboyama_processed_val_full.csv'}")
    print(f"Train sampled: {OUT_DIR / 'tsuboyama_processed_train_sampled.csv'}")
    print(f"Rows (phase1 resolved single substitutions): {len(phase1_rows)}")
    print(
        f"Train full rows: {len(train_full)} | "
        f"Val full rows: {len(val_full)} | "
        f"Train sampled rows: {len(train_sampled)}"
    )


if __name__ == "__main__":
    main()
