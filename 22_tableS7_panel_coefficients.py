"""Table S7 -- LDA coefficients for the five comparison panels.

For each panel, fits LDA on all 71 QC-passed samples and writes the
per-protein coefficient and intercept in a long-format CSV. Sign
convention: classes_ = [0, 1] with 0 = non-healer, 1 = healer.
A positive coefficient means higher intensity raises PHS toward healer.
"""

from pathlib import Path
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from lib.data_utils import load_longitudinal
from lib.panels import PANELS

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    m, y, g, vis, wk, names, info = load_longitudinal(
        apply_per_protein_qc=False, apply_per_sample_qc=True)
    name_to_idx = {n: i for i, n in enumerate(names)}

    rows = []
    for pname, proteins in PANELS.items():
        idxs = [name_to_idx[p] for p in proteins]
        X = m[idxs].T
        clf = LinearDiscriminantAnalysis().fit(X, y)
        assert list(clf.classes_) == [0, 1]
        coefs = clf.coef_.ravel()
        b = float(clf.intercept_.ravel()[0])
        for p, w in zip(proteins, coefs):
            rows.append({
                "panel": pname, "protein": p,
                "coefficient": float(w), "intercept": b,
            })
    df = pd.DataFrame(rows)
    out = OUT / "tableS7_panel_coefficients_long.csv"
    df.to_csv(out, index=False, float_format="%.6g")
    print(f"Saved {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
