"""Re-run the LOPPO greedy-forward branching and save the actual step-by-step
addition history for each of the top-5 unique panels (by cv error).

Output: output/top5_branch_histories.csv
Columns: panel_label, step_index, size, proteins, train_error, cv_error
"""

from pathlib import Path
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
N_TOP_PANELS = 5


def main():
    m, y, g, vis, wk, names, info = load_longitudinal(
        apply_per_protein_qc=True, apply_per_sample_qc=True)
    print(f"[hist] {info['n_samples_after']} samples, "
          f"{info['n_proteins_after']} proteins.")

    pair_df = scan_all_pairs(m, y, names, method="lda",
                             cv_for_top=200, groups=g)
    seeds = pair_df[pair_df["train_error"] <= SEED_CUTOFF_ERROR].reset_index(drop=True)
    print(f"[hist] {len(seeds)} seed pairs.")

    name_to_idx = {n: i for i, n in enumerate(names)}
    all_branches = []
    for _, row in seeds.iterrows():
        p1, p2 = row["protein_1"], row["protein_2"]
        i1, i2 = name_to_idx[p1], name_to_idx[p2]
        log = SearchLog()
        branches = branching_search(m, y, names, method="lda",
                                    top_k=TOP_K_FOR_CV,
                                    max_iterations=MAX_ITERATIONS, log=log,
                                    seed_pairs=[(i1, i2)], groups=g)
        for b in branches:
            all_branches.append(b)

    by_panel = {}
    for b in all_branches:
        key = tuple(sorted(b["names"]))
        cv = b["cv_error"]
        if key not in by_panel or cv < by_panel[key]["cv_error"]:
            by_panel[key] = b

    # Variant A: keep only size-5 panels, ranked by cv_error
    size5_panels = [b for b in by_panel.values() if len(b["names"]) == 5]
    ranked = sorted(size5_panels,
                    key=lambda b: (b["cv_error"], b["train_error"]))
    canonical = {
        ("CFB", "IGHV3-64", "MINPP1", "PI16", "SELL"): "Complement-5",
    }

    rows = []
    for rank, b in enumerate(ranked[:N_TOP_PANELS], start=1):
        key = tuple(sorted(b["names"]))
        label = canonical.get(key, f"Set {rank}")
        for step_i, h in enumerate(b["history"]):
            rows.append({
                "panel_label": label,
                "rank": rank,
                "step_index": step_i,
                "size": len(h["names"]),
                "proteins": "|".join(h["names"]),
                "train_error": h["train_error"],
                "cv_error": h["cv_error"],
                "stop_reason": b["stop_reason"],
            })
        print(f"  rank {rank} ({label}): seed={b['history'][0]['names']}, "
              f"final={b['names']}, cv={b['cv_error']}")

    pd.DataFrame(rows).to_csv(OUT / "top5_branch_histories.csv", index=False)
    print(f"Saved {OUT / 'top5_branch_histories.csv'}")


if __name__ == "__main__":
    main()
