"""Table 2, Tables S4 and S5 -- panel search under all-68-LOPPO.

Reproduces the primary panel-search pipeline:

  1. Pair scan: LDA training error for every pair in the 449-protein pool.
  2. The 45 seed pairs at train_error <= 5 enter step 2.
  3. Greedy forward branching with patient-pair LOPPO cross-validation
     (top_k = 5 candidates by training error per step; CV-rerank then
     branch on (cv_error, train_error) ties; stop at zero error,
     no improvement, or 20 iterations).
  4. Aggregate to order-invariant unique sets.
  5. Pick the top 10 train_error = 0 sets, ranked by CV error.

Outputs:
  output/tableS4_all_branches_summary.csv  -- every branch from every seed
  output/tableS5_unique_sets_LOPPO.csv     -- order-invariant deduplication
  output/table2_top10_panels_LOPPO.csv     -- main-text Table 2

This script is the slowest of the pipeline (~5-10 minutes on a laptop).
"""

from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

from lib.data_utils import load_longitudinal
from lib.classifier_search import scan_all_pairs
from lib.branching import branching_search
from lib.search_log import SearchLog

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)

SEED_CUTOFF_ERROR = 5   # strict threshold: pairs with train_error <= 5 are seeds.
                        # Corresponds to a natural break in the error distribution
                        # (Figure S4); yields 45 seed pairs in this cohort.
TOP_K_FOR_CV = 5
MAX_ITERATIONS = 20


def main():
    m, y, g, vis, wk, names, info = load_longitudinal(
        apply_per_protein_qc=True, apply_per_sample_qc=True)
    print(f"[LOPPO] {info['n_samples_after']} samples, "
          f"{info['n_proteins_after']} proteins.")

    # ---- pair scan ----
    print("[LOPPO] pair scan ...")
    pair_df = scan_all_pairs(m, y, names, method="lda",
                             cv_for_top=200, groups=g)
    pair_df.to_csv(OUT / "loppo_pair_scan.csv", index=False)
    seeds = pair_df[pair_df["train_error"] <= SEED_CUTOFF_ERROR].reset_index(drop=True)
    print(f"[LOPPO] {len(seeds)} seed pairs (train_error <= {SEED_CUTOFF_ERROR})")

    # ---- branching from each seed ----
    name_to_idx = {n: i for i, n in enumerate(names)}
    summary_rows = []
    for rank, row in seeds.iterrows():
        p1, p2 = row["protein_1"], row["protein_2"]
        i1, i2 = name_to_idx[p1], name_to_idx[p2]
        log = SearchLog()
        branches = branching_search(m, y, names, method="lda",
                                    top_k=TOP_K_FOR_CV,
                                    max_iterations=MAX_ITERATIONS, log=log,
                                    seed_pairs=[(i1, i2)], groups=g)
        branches.sort(key=lambda b: (b["cv_error"], b["train_error"],
                                     len(b["names"])))
        for i, b in enumerate(branches):
            summary_rows.append({
                "seed_rank": rank + 1,
                "seed_pair": f"{p1}_{p2}",
                "branch": i + 1,
                "size": len(b["names"]),
                "train_error": b["train_error"],
                "cv_error": b["cv_error"],
                "stop_reason": b["stop_reason"],
                "proteins": "|".join(b["names"]),
            })
        print(f"[LOPPO]   seed {rank+1:>2d} {p1}+{p2}: "
              f"{len(branches)} branches, "
              f"best cv = {min(b['cv_error'] for b in branches)}")

    all_df = pd.DataFrame(summary_rows).sort_values(
        ["cv_error", "train_error", "size", "seed_rank"]).reset_index(drop=True)
    all_df.to_csv(OUT / "tableS4_all_branches_summary.csv", index=False)
    print(f"[LOPPO] saved {OUT/'tableS4_all_branches_summary.csv'}")

    # ---- unique order-invariant sets ----
    all_df["protein_key"] = all_df["proteins"].apply(
        lambda s: tuple(sorted(s.split("|"))))
    unique = all_df.groupby("protein_key").agg(
        size=("size", "first"),
        train_error=("train_error", "min"),
        cv_error=("cv_error", "min"),
        n_times_found=("seed_rank", "count"),
        seed_ranks=("seed_rank", lambda s: ";".join(
            str(x) for x in sorted(set(s)))),
    ).reset_index()
    unique["proteins"] = unique["protein_key"].apply(lambda t: "|".join(t))
    unique = unique.drop(columns=["protein_key"]).sort_values(
        ["cv_error", "train_error", "size", "n_times_found"],
        ascending=[True, True, True, False]).reset_index(drop=True)
    unique.to_csv(OUT / "tableS5_unique_sets_LOPPO.csv", index=False)
    print(f"[LOPPO] {len(unique)} unique sets -> {OUT/'tableS5_unique_sets_LOPPO.csv'}")

    # ---- top-10 train=0 ----
    zero = unique[unique["train_error"] == 0].copy()
    top10 = zero.head(10).reset_index(drop=True)
    top10.insert(0, "set_id", range(1, len(top10) + 1))
    top10.to_csv(OUT / "table2_top10_panels_LOPPO.csv", index=False)
    print(f"[LOPPO] saved {OUT/'table2_top10_panels_LOPPO.csv'}")
    print()
    print(top10[["set_id", "size", "cv_error", "proteins"]].to_string(index=False))


if __name__ == "__main__":
    main()
