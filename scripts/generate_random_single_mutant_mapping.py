#!/usr/bin/env python3
import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
MUT_PAT = re.compile(r"^([A-Z\*])(\d+)([A-Z\*])$")


def parse_mut(mut_type: str) -> Optional[Tuple[str, int, str]]:
    m = MUT_PAT.match(mut_type)
    if not m:
        return None
    wt_aa, pos, mut_aa = m.group(1), int(m.group(2)), m.group(3)
    return wt_aa, pos, mut_aa


def reconstruct_wt_from_mutant(mutant_seq: str, mut_type: str) -> Optional[str]:
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


def derive_wt_entries(source_csv: Path) -> Tuple[List[Tuple[str, str, int]], dict]:
    wt_pair_counter: Counter = Counter()
    cluster_to_wt_set: Dict[str, set] = defaultdict(set)
    total_rows = 0
    parsed_rows = 0
    skipped_rows = 0

    with source_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            cluster = str(row["WT_cluster"])
            mutant_seq = str(row["aa_seq"]).strip().upper()
            mut_type = str(row["mut_type"]).strip().upper()
            wt_seq = reconstruct_wt_from_mutant(mutant_seq, mut_type)
            if wt_seq is None:
                skipped_rows += 1
                continue
            parsed_rows += 1
            wt_pair_counter[(cluster, wt_seq)] += 1
            cluster_to_wt_set[cluster].add(wt_seq)

    wt_entries = []
    for (cluster, wt_seq), count in sorted(wt_pair_counter.items()):
        wt_entries.append((cluster, wt_seq, count))

    inconsistent_clusters = sum(
        1 for wt_set in cluster_to_wt_set.values() if len(wt_set) > 1
    )

    summary = {
        "source_rows_total": total_rows,
        "source_rows_parsed": parsed_rows,
        "source_rows_skipped": skipped_rows,
        "n_clusters_with_wt": len(cluster_to_wt_set),
        "n_unique_wt_sequences": len(wt_entries),
        "n_clusters_with_multiple_wt_sequences": inconsistent_clusters,
    }
    return wt_entries, summary


def build_lookup(lookup_csv: Path) -> Dict[Tuple[str, str], List[float]]:
    out: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    with lookup_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cluster = str(row["WT_cluster"])
            seq = str(row["aa_seq"]).strip().upper()
            delta_g = float(row["deltaG"])
            out[(cluster, seq)].append(delta_g)
    return out


def make_random_single_mutant(wt_seq: str, rng: random.Random):
    idx = rng.randrange(len(wt_seq))
    wt_aa = wt_seq[idx]
    choices = [aa for aa in AA_ALPHABET if aa != wt_aa]
    mut_aa = rng.choice(choices)

    var_seq = wt_seq[:idx] + mut_aa + wt_seq[idx + 1 :]
    mut_pos = idx + 1
    mut_label = f"{wt_aa}{mut_pos}{mut_aa}"
    return var_seq, mut_pos, wt_aa, mut_aa, mut_label


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate one random single mutant per unique reconstructed WT sequence, "
            "then map generated variants to a lookup CSV and collect deltaG if found."
        )
    )
    parser.add_argument(
        "--wt_source_csv",
        type=Path,
        default=Path("data/processed/tsuboyama_processed_val_full.csv"),
        help="CSV used to derive WT sequences by cluster",
    )
    parser.add_argument(
        "--lookup_csv",
        type=Path,
        default=Path("data/processed/tsuboyama_processed_train_full.csv"),
        help="CSV used for mapping generated variants",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("results/generative_eval/random_single_mutants_val_wt_mapped_to_train_full.csv"),
    )
    parser.add_argument(
        "--summary_json",
        type=Path,
        default=Path("results/generative_eval/random_single_mutants_val_wt_mapped_to_train_full.summary.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    wt_entries, wt_summary = derive_wt_entries(args.wt_source_csv)
    lookup = build_lookup(args.lookup_csv)

    rows_out = []
    mapped_n = 0
    unmapped_n = 0

    for cluster, wt_seq, wt_reconstruction_count in wt_entries:
        gen_seq, pos, wt_aa, mut_aa, mut_label = make_random_single_mutant(wt_seq, rng)

        hits = lookup.get((cluster, gen_seq), [])
        mapped = len(hits) > 0
        if mapped:
            mapped_n += 1
            mapped_delta_g = min(hits)
        else:
            unmapped_n += 1
            mapped_delta_g = ""

        rows_out.append(
            {
                "WT_cluster": cluster,
                "wt_seq": wt_seq,
                "wt_reconstruction_count": wt_reconstruction_count,
                "generated_seq": gen_seq,
                "mut_pos_1based": pos,
                "wt_aa": wt_aa,
                "mut_aa": mut_aa,
                "generated_mut_type": mut_label,
                "mapped_to_train_full": int(mapped),
                "mapped_match_count": len(hits),
                "mapped_deltaG": mapped_delta_g,
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "WT_cluster",
                "wt_seq",
                "wt_reconstruction_count",
                "generated_seq",
                "mut_pos_1based",
                "wt_aa",
                "mut_aa",
                "generated_mut_type",
                "mapped_to_train_full",
                "mapped_match_count",
                "mapped_deltaG",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    summary = {
        "seed": args.seed,
        "wt_source_csv": str(args.wt_source_csv),
        "lookup_csv": str(args.lookup_csv),
        **wt_summary,
        "n_generated_variants": len(rows_out),
        "n_mapped": mapped_n,
        "n_unmapped": unmapped_n,
        "mapped_fraction": (mapped_n / len(rows_out)) if rows_out else 0.0,
        "output_csv": str(args.output_csv),
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
