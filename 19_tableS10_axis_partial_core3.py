"""Table S10 -- top-10 panels that contain only one of {CFB, IGHV3-64, PI16}.

From the unique LOPPO panels (Table S5), filters to panels with
core3_overlap == 1 (exactly one of the three Core-3 proteins present)
and reports the top 10 by CV error. For each panel, fits LDA on all 71
samples and reports the per-protein PHS sign.
"""

from pathlib import Path
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from lib.data_utils import load_longitudinal

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)

CORE3 = {"CFB", "IGHV3-64", "PI16"}


def main():
    unique = pd.read_csv(OUT / "tableS5_unique_sets_LOPPO.csv")
    unique["core3_overlap"] = unique["proteins"].apply(
        lambda s: len(CORE3 & set(s.split("|"))))
    ranked = unique.sort_values(
        ["cv_error", "train_error", "size", "n_times_found"],
        ascending=[True, True, True, False]).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    picked = ranked[ranked["core3_overlap"] == 1].head(10).copy()

    m, y, g, vis, wk, names, info = load_longitudinal(
        apply_per_protein_qc=True, apply_per_sample_qc=True)
    name_to_idx = {n: i for i, n in enumerate(names)}

    rows = []
    for _, r in picked.iterrows():
        ps = r["proteins"].split("|")
        idxs = [name_to_idx[p] for p in ps]
        X = m[idxs].T
        clf = LinearDiscriminantAnalysis().fit(X, y)
        signed = [f"{p}({'+' if w > 0 else '-'})"
                  for p, w in zip(ps, clf.coef_.ravel())]
        rows.append({
            "Rank": int(r["rank"]),
            "Size": int(r["size"]),
            "Proteins": " | ".join(ps),
            "Proteins with PHS sign": " | ".join(signed),
            "Train error": int(r["train_error"]),
            "CV error (LOPPO, /528)": int(r["cv_error"]),
        })
    df = pd.DataFrame(rows)
    out = OUT / "tableS10_axis_only_one_core3_panels.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
