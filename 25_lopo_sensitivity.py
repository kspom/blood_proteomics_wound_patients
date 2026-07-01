"""Leave-One-Patient-Out (LOPO) sensitivity analysis -- exploratory.

For each of the 16 analytical-cohort patients, re-run the full pair scan
and the greedy forward-branching search (threshold <= 5) with that
patient excluded. Save BOTH the per-LOPO pair-scan results AND the
per-LOPO unique panel sets, so that downstream metrics can be computed
without any reference to canonical panels (Tier 1: protein frequency,
Tier 2: pair stability, Tier 3: panel-composition clustering).

Outputs:
  output/lopo_all_pair_scans.csv   -- per-LOPO pair train_error and cv_error.
                                      Long format: (excluded_patient, protein_1, protein_2, train_error, cv_error).
  output/lopo_all_unique_sets.csv  -- per-LOPO unique panel sets, all ranks.
  output/lopo_top5_per_run.csv     -- per-LOPO top-5 panels (convenience subset).
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
TOP_PANELS_TO_RECORD = 5


def run_search_excluding(excluded_patient, m_all, y_all, g_all, names):
    keep = g_all != excluded_patient
    m, y, g = m_all[:, keep], y_all[keep], g_all[keep]

    pair_df = scan_all_pairs(m, y, names, method="lda",
                             cv_for_top=200, groups=g)
    seeds = pair_df[pair_df["train_error"] <= SEED_CUTOFF_ERROR].reset_index(drop=True)

    name_to_idx = {n: i for i, n in enumerate(names)}
    rows = []
    for _, row in seeds.iterrows():
        i1, i2 = name_to_idx[row["protein_1"]], name_to_idx[row["protein_2"]]
        log = SearchLog()
        branches = branching_search(m, y, names, method="lda",
                                    top_k=TOP_K_FOR_CV,
                                    max_iterations=MAX_ITERATIONS, log=log,
                                    seed_pairs=[(i1, i2)], groups=g)
        for b in branches:
            rows.append({
                "size": len(b["names"]),
                "train_error": b["train_error"],
                "cv_error": b["cv_error"],
                "proteins": "|".join(sorted(b["names"])),
            })

    branches_df = pd.DataFrame(rows)
    unique = (branches_df
              .groupby("proteins", as_index=False)
              .agg(size=("size", "first"),
                   train_error=("train_error", "min"),
                   cv_error=("cv_error", "min"),
                   n_times_found=("proteins", "size"))
              .sort_values(["cv_error", "train_error", "size", "n_times_found"],
                           ascending=[True, True, True, False])
              .reset_index(drop=True))
    n_zero = int((unique["cv_error"] == 0).sum())
    print(f"[LOPO p{excluded_patient}] {keep.sum()} samples, "
          f"{len(seeds)} seeds, {len(unique)} unique sets, "
          f"{n_zero} zero-error.")
    return pair_df, unique


def main():
    m_all, y_all, g_all, vis, wk, names, info = load_longitudinal(
        apply_per_protein_qc=True, apply_per_sample_qc=True)
    patients = sorted(np.unique(g_all).tolist())
    print(f"Full cohort: {info['n_samples_after']} samples, "
          f"{info['n_proteins_after']} proteins, "
          f"{len(patients)} patients.\n")

    pair_rows = []
    all_unique_rows = []
    top5_rows = []
    for pid in patients:
        pair_df, unique = run_search_excluding(pid, m_all, y_all, g_all, names)

        pair_df_to_save = pair_df[["protein_1", "protein_2", "train_error"]].copy()
        pair_df_to_save["excluded_patient"] = pid
        pair_rows.append(pair_df_to_save)

        for rank, row in unique.iterrows():
            all_unique_rows.append({
                "excluded_patient": pid,
                "rank": rank + 1,
                "size": row["size"],
                "train_error": row["train_error"],
                "cv_error": row["cv_error"],
                "n_times_found": row["n_times_found"],
                "proteins": row["proteins"],
            })
        for rank, row in unique.head(TOP_PANELS_TO_RECORD).iterrows():
            top5_rows.append({
                "excluded_patient": pid,
                "rank": rank + 1,
                "size": row["size"],
                "train_error": row["train_error"],
                "cv_error": row["cv_error"],
                "proteins": row["proteins"],
            })

    pd.concat(pair_rows, ignore_index=True).to_csv(
        OUT / "lopo_all_pair_scans.csv", index=False)
    pd.DataFrame(all_unique_rows).to_csv(OUT / "lopo_all_unique_sets.csv", index=False)
    pd.DataFrame(top5_rows).to_csv(OUT / "lopo_top5_per_run.csv", index=False)
    print(f"\nSaved {OUT}/lopo_all_pair_scans.csv")
    print(f"Saved {OUT}/lopo_all_unique_sets.csv")
    print(f"Saved {OUT}/lopo_top5_per_run.csv")


if __name__ == "__main__":
    main()
