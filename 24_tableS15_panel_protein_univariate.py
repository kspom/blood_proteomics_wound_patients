"""Table S15 -- univariate-vs-multivariate role of each panel protein.

For each of the 10 unique proteins across the 4 comparison panels, report:
  - which panel(s) contain it
  - longitudinal log2 fold-change and q-value
  - visit-1 log2 fold-change and q-value
  - direction of contribution in PHS (+ or -)
  - classification of univariate signal strength

The classification illuminates which panel proteins are "multivariate-only"
contributors -- selected by the search despite no univariate signal -- and
which carry a univariate trend that becomes significant in multivariate
combination.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from lib.data_utils import load_longitudinal
from lib.panels import PANELS

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)


def classify(q_long):
    if q_long <= 0.05:
        return "DE-significant (univariate)"
    if q_long <= 0.10:
        return "univariate near-miss (q<=0.10)"
    if q_long <= 0.20:
        return "univariate trend (q<=0.20)"
    return "no univariate signal (multivariate-only contributor)"


def panels_containing(protein):
    return [name for name, prots in PANELS.items() if protein in prots]


def main():
    de_long = pd.read_csv(OUT / "tableS1_de_longitudinal_all.csv").set_index("protein")
    de_base = pd.read_csv(OUT / "tableS2_de_baseline_all.csv").set_index("protein")

    panel_union = sorted({p for prots in PANELS.values() for p in prots})

    # Fit each panel's LDA to get PHS sign per protein-in-panel
    m, y, g, vis, wk, names, info = load_longitudinal(
        apply_per_protein_qc=False, apply_per_sample_qc=True)
    name_to_idx = {n: i for i, n in enumerate(names)}
    sign_by_panel_protein = {}
    for pname, proteins in PANELS.items():
        idxs = [name_to_idx[p] for p in proteins]
        X = m[idxs].T
        clf = LinearDiscriminantAnalysis().fit(X, y)
        for p, w in zip(proteins, clf.coef_.ravel()):
            sign_by_panel_protein[(pname, p)] = "+" if w > 0 else "-"

    rows = []
    for p in panel_union:
        panels = panels_containing(p)
        signs = sorted({sign_by_panel_protein[(pn, p)] for pn in panels})
        sign_str = signs[0] if len(signs) == 1 else "mixed (" + "/".join(signs) + ")"
        lfc_long = float(de_long.at[p, "log2FC_healer_vs_nonhealer"])
        q_long = float(de_long.at[p, "mw_q_patient_mean"])
        lfc_v1 = float(de_base.at[p, "log2FC_healer_vs_nonhealer"]) \
            if p in de_base.index else float("nan")
        q_v1 = float(de_base.at[p, "welch_q"]) if p in de_base.index else float("nan")
        rows.append({
            "Protein": p,
            "Appears in panel(s)": ", ".join(panels),
            "log2 FC (longitudinal)": f"{lfc_long:+.2f}",
            "q (longitudinal)": f"{q_long:.3g}",
            "log2 FC (visit-1)":
                f"{lfc_v1:+.2f}" if not np.isnan(lfc_v1) else "-",
            "q (visit-1)":
                f"{q_v1:.3g}" if not np.isnan(q_v1) else "-",
            "Direction in PHS": sign_str,
            "Univariate signal classification": classify(q_long),
        })

    # sort by q ascending
    rows.sort(key=lambda r: float(r["q (longitudinal)"]))
    df = pd.DataFrame(rows)
    out = OUT / "tableS15_panel_protein_univariate_classification.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
