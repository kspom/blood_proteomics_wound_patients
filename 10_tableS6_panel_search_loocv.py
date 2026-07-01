"""Table S6 -- panel search under all-68-LOOCV.

Same pipeline as 09_table2_panel_search_loppo.py but with sample-level
leave-one-out cross-validation (groups = None) instead of patient-pair
LOPPO. Produces Table S6 (unique panel sets ranked by LOOCV error).
"""

from pathlib import Path
import numpy as np
import pandas as pd

from lib.data_utils import load_longitudinal
from lib.classifier_search import scan_all_pairs
from lib.branching import branching_search
from lib.search_log import SearchLog

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)

SEED_CUTOFF_ERROR = 5
TOP_K_FOR_CV = 5
MAX_ITERATIONS = 20


def main():
    m, y, g, vis, wk, names, info = load_longitudinal(
        apply_per_protein_qc=True, apply_per_sample_qc=True)
    print(f"[LOOCV] {info['n_samples_after']} samples, "
          f"{info['n_proteins_after']} proteins.")

    print("[LOOCV] pair scan ...")
    # groups=None => LOOCV; same train-error ranking as LOPPO scan
    pair_df = scan_all_pairs(m, y, names, method="lda",
                             cv_for_top=200, groups=None)
    pair_df.to_csv(OUT / "loocv_pair_scan.csv", index=False)
    seeds = pair_df[pair_df["train_error"] <= SEED_CUTOFF_ERROR].reset_index(drop=True)
    print(f"[LOOCV] {len(seeds)} seed pairs (train_error <= {SEED_CUTOFF_ERROR})")

    name_to_idx = {n: i for i, n in enumerate(names)}
    summary_rows = []
    for rank, row in seeds.iterrows():
        p1, p2 = row["protein_1"], row["protein_2"]
        i1, i2 = name_to_idx[p1], name_to_idx[p2]
        log = SearchLog()
        branches = branching_search(m, y, names, method="lda",
                                    top_k=TOP_K_FOR_CV,
                                    max_iterations=MAX_ITERATIONS, log=log,
                                    seed_pairs=[(i1, i2)], groups=None)
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
        print(f"[LOOCV]   seed {rank+1:>2d} {p1}+{p2}: "
              f"{len(branches)} branches, "
              f"best cv = {min(b['cv_error'] for b in branches)}")

    all_df = pd.DataFrame(summary_rows)
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
    unique.to_csv(OUT / "tableS6_unique_sets_LOOCV.csv", index=False)
    print(f"[LOOCV] {len(unique)} unique sets -> "
          f"{OUT/'tableS6_unique_sets_LOOCV.csv'}")
    zero = unique[(unique["train_error"] == 0) & (unique["cv_error"] == 0)]
    print(f"[LOOCV] zero-error panels: {len(zero)}")
    for _, r in zero.iterrows():
        print(f"   size={int(r['size'])}  {r['proteins']}")


if __name__ == "__main__":
    main()
