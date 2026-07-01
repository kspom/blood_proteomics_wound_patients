"""Table 5 -- proteins that recur in the seed-pair scan or the top-10 panels.

Lists every protein that appears more than once in either the 45 seed
pairs (train_error <= 5) or the top-10 all-68-LOPPO panels, together
with:
  - count of seed pairs and top-10 panels containing it,
  - direction in the Protein Healing Score (+, -, or "mixed"),
  - the source from which the sign was derived (top-10 panels or
    2-protein seed pairs),
  - longitudinal and visit-1 log2 fold-changes,
  - longitudinal and visit-1 q-values from the DE analysis.

Reads from output/tableS5_unique_sets_LOPPO.csv,
output/loppo_pair_scan.csv (both produced by 09_*),
output/tableS1_de_longitudinal_all.csv,
output/tableS2_de_baseline_all.csv (both produced by 04_*).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from lib.data_utils import load_longitudinal

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)

SEED_CUTOFF_ERROR = 5


def collect_signs(panels_with_protein, m, y, name_to_idx, target):
    ws = []
    for proteins in panels_with_protein:
        idxs = [name_to_idx[p] for p in proteins]
        X = m[idxs].T
        clf = LinearDiscriminantAnalysis().fit(X, y)
        for p, w in zip(proteins, clf.coef_.ravel()):
            if p == target:
                ws.append(float(w)); break
    return ws


def main():
    pair_scan = pd.read_csv(OUT / "loppo_pair_scan.csv")
    seed_pairs = pair_scan[pair_scan["train_error"] <= SEED_CUTOFF_ERROR]
    seed_lists = [[r["protein_1"], r["protein_2"]]
                  for _, r in seed_pairs.iterrows()]

    top10 = pd.read_csv(OUT / "table2_top10_panels_LOPPO.csv")
    top10_lists = [r["proteins"].split("|") for _, r in top10.iterrows()]

    seed_count = {}
    for lst in seed_lists:
        for p in lst:
            seed_count[p] = seed_count.get(p, 0) + 1
    panel_count = {}
    for lst in top10_lists:
        for p in lst:
            panel_count[p] = panel_count.get(p, 0) + 1
    n_seeds = len(seed_lists); n_panels = len(top10_lists)

    recurrent = sorted({p for p, c in seed_count.items() if c >= 2}
                       | {p for p, c in panel_count.items() if c >= 2})

    de_long = pd.read_csv(OUT / "tableS1_de_longitudinal_all.csv").set_index("protein")
    de_base = pd.read_csv(OUT / "tableS2_de_baseline_all.csv").set_index("protein")

    m, y, g, vis, wk, names, info = load_longitudinal(
        apply_per_protein_qc=False, apply_per_sample_qc=True)
    name_to_idx = {n: i for i, n in enumerate(names)}

    rows = []
    for p in recurrent:
        n_seed = seed_count.get(p, 0); n_pan = panel_count.get(p, 0)
        if n_pan > 0:
            hosts = [pl for pl in top10_lists if p in pl]
            source = f"top-10 panels (n={len(hosts)})"
        else:
            hosts = [pl for pl in seed_lists if p in pl]
            source = f"2-protein seed pairs (n={len(hosts)})"
        ws = collect_signs(hosts, m, y, name_to_idx, p)
        signs = {"+" if w > 0 else "-" for w in ws}
        sign = signs.pop() if len(signs) == 1 else "mixed"

        lfc_l = (float(de_long.at[p, "log2FC_healer_vs_nonhealer"])
                 if p in de_long.index else np.nan)
        lfc_b = (float(de_base.at[p, "log2FC_healer_vs_nonhealer"])
                 if p in de_base.index else np.nan)
        q_l = float(de_long.at[p, "mw_q_patient_mean"]) if p in de_long.index else np.nan
        q_b = float(de_base.at[p, "welch_q"]) if p in de_base.index else np.nan
        rows.append({
            "Protein": p,
            f"Seed pairs (of {n_seeds})": n_seed,
            f"Top-10 panels (of {n_panels})": n_pan,
            "Direction in PHS": sign,
            "Sign derived from": source,
            "log2 fold-change, longitudinal":
                f"{lfc_l:+.2f}" if pd.notna(lfc_l) else "-",
            "log2 fold-change, visit-1 only":
                f"{lfc_b:+.2f}" if pd.notna(lfc_b) else "-",
            "q-value, longitudinal":
                f"{q_l:.3g}" if pd.notna(q_l) else "-",
            "q-value, visit-1 only":
                f"{q_b:.3g}" if pd.notna(q_b) else "-",
        })
    df = pd.DataFrame(rows).sort_values(
        by=[f"Top-10 panels (of {n_panels})", f"Seed pairs (of {n_seeds})"],
        ascending=[False, False]).reset_index(drop=True)
    out = OUT / "table5_recurrent_proteins.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
