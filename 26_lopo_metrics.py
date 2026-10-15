"""Exploratory LOPO-sensitivity metrics -- no a priori reference to canonical panels.

Tiers (each tier descriptive only; no canonical reference is used to compute it):

  Tier 1.  Per-protein appearance frequency across 16 LOPO runs.
           For each of 449 proteins: how often does it appear in
           rank-1 / top-3 / top-5 / any zero-CV panel?
           -> Table S16: top-20 proteins by top-3 frequency.

  Tier 2.  Pair-level stability across 16 LOPO runs.
           For each protein pair: median, max, mean train_error and
           the count of LOPO runs in which the pair qualifies as seed
           (train_error <= 5).  Plus protein hub centrality (= number
           of robust pairs containing each protein).
           -> Table S17: robust seed pairs (qualify in >=14/16 runs).
           -> Table S18: per-error-level pair landscape (how many pairs
              hit each error level in N of 16 runs).
           -> Table S19: protein hub centrality.

  Tier 3.  Panel composition clustering via Jaccard similarity.
           Pool all top-5 panels across 16 LOPO runs, cluster by
           Jaccard, identify emergent panel families.
           -> Table S20: emergent panel families with representatives.

  Tier 4 (post hoc).  Transparent comparison of emergent patterns against
           canonical panels (Core-3, 4-protein backbone, Complement-5,
           Metabolic-4). This is NOT used to select metrics in Tiers 1-3.
           -> Reported in Supplementary Note S1.
"""

from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
import pandas as pd
import numpy as np

from lib.panels import PANELS
from lib.data_utils import load_longitudinal

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)

SEED_CUTOFF_ERROR = 5
ROBUST_THRESHOLD = 14   # >= 14/16 runs to be called "robust"


def proteins_of(row):
    return set(row["proteins"].split("|"))


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def main():
    unique = pd.read_csv(OUT / "lopo_all_unique_sets.csv")
    pairs = pd.read_csv(OUT / "lopo_all_pair_scans.csv")
    patients = sorted(unique["excluded_patient"].unique())
    N_RUNS = len(patients)
    print(f"LOPO runs analysed: {N_RUNS}")

    # =================== Tier 1: protein frequency ===================
    print("\n=== Tier 1: per-protein appearance frequency (no canonical ref) ===")
    all_proteins = set()
    for _, row in unique.iterrows():
        all_proteins.update(proteins_of(row))
    print(f"  Proteins appearing in any LOPO panel: {len(all_proteins)}")

    tier1_rows = []
    for p in sorted(all_proteins):
        n1 = n3 = n5 = nz = 0
        for pid in patients:
            run = unique[unique["excluded_patient"] == pid].sort_values("rank")
            top1 = run.head(1); top3 = run.head(3); top5 = run.head(5)
            zero = run[run["cv_error"] == 0]
            if any(p in proteins_of(r) for _, r in top1.iterrows()): n1 += 1
            if any(p in proteins_of(r) for _, r in top3.iterrows()): n3 += 1
            if any(p in proteins_of(r) for _, r in top5.iterrows()): n5 += 1
            if any(p in proteins_of(r) for _, r in zero.iterrows()): nz += 1
        tier1_rows.append({"protein": p, "n_rank1": n1, "n_top3": n3,
                           "n_top5": n5, "n_zero_CV": nz})
    tier1 = pd.DataFrame(tier1_rows).sort_values(
        ["n_top3", "n_top5", "n_rank1", "protein"],
        ascending=[False, False, False, True])
    tier1["robust"] = tier1["n_top3"] >= ROBUST_THRESHOLD
    tier1.to_csv(OUT / "tableS16_lopo_protein_frequency.csv", index=False)
    print(tier1.head(20).to_string(index=False))

    # =================== Tier 2: pair stability ===================
    print("\n=== Tier 2: pair-level stability ===")
    # Normalize pair order
    pairs["pair"] = pairs.apply(
        lambda r: "|".join(sorted([r["protein_1"], r["protein_2"]])), axis=1)
    grp = pairs.groupby("pair")["train_error"]
    pair_stats = grp.agg(["median", "mean", "min", "max", "std"]).reset_index()
    pair_stats["n_seed_qualifying"] = grp.apply(
        lambda s: int((s <= SEED_CUTOFF_ERROR).sum())).values
    pair_stats["n_strong"] = grp.apply(
        lambda s: int((s <= 3).sum())).values
    pair_stats = pair_stats.sort_values(
        ["n_seed_qualifying", "median", "max"],
        ascending=[False, True, True]).reset_index(drop=True)
    print(f"  Pairs scanned per LOPO run: ~{len(pair_stats):,}")

    robust_seeds = pair_stats[pair_stats["n_seed_qualifying"] >= ROBUST_THRESHOLD]
    robust_seeds.to_csv(OUT / "tableS17_lopo_robust_seed_pairs.csv", index=False)
    print(f"  Robust seed pairs (qualifying as seed in >= {ROBUST_THRESHOLD}/{N_RUNS} runs): "
          f"{len(robust_seeds)}")
    if len(robust_seeds) > 0:
        print(robust_seeds.head(20).to_string(index=False))

    # Hub centrality: per protein, how many robust pairs contain it
    print("\n  Protein hub centrality (in robust seed pairs):")
    hub_counts = Counter()
    for p in robust_seeds["pair"]:
        for prot in p.split("|"):
            hub_counts[prot] += 1
    hubs = pd.DataFrame([{"protein": p, "n_robust_pairs": c}
                        for p, c in hub_counts.most_common()])
    hubs.to_csv(OUT / "tableS18_lopo_protein_hub_centrality.csv", index=False)
    print(hubs.head(20).to_string(index=False))

    # =================== Tier 3: frequent protein subsets ===================
    print("\n=== Tier 3: frequent protein subsets across 16 LOPO runs ===")
    runs_top5 = defaultdict(list)
    for _, row in unique[unique["rank"] <= 5].iterrows():
        runs_top5[int(row["excluded_patient"])].append(
            frozenset(proteins_of(row)))

    subset_rows = []
    for k in (3, 4, 5):
        subset_to_runs = defaultdict(set)
        for run_id, panels in runs_top5.items():
            seen_in_run = set()
            for panel in panels:
                if len(panel) >= k:
                    for combo in combinations(sorted(panel), k):
                        seen_in_run.add(combo)
            for combo in seen_in_run:
                subset_to_runs[combo].add(run_id)
        # ties broken alphabetically so that the table is reproducible
        sorted_subsets = sorted(subset_to_runs.items(),
                                key=lambda x: (-len(x[1]), x[0]))
        for combo, run_set in sorted_subsets[:20]:
            subset_rows.append({
                "size": k,
                "subset": " + ".join(combo),
                "n_lopo_runs": len(run_set),
                "fraction": f"{len(run_set)}/{N_RUNS}",
            })
    freq_subsets = pd.DataFrame(subset_rows)
    freq_subsets.to_csv(OUT / "tableS19_lopo_frequent_subsets.csv", index=False)
    print(freq_subsets.to_string(index=False))


if __name__ == "__main__":
    main()
