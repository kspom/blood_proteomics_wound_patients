"""Tables S12, S13 -- Clinical-3 fitted coefficients and CV errors.

Fits LDA on all 68 QC-passed samples for the Clinical-3 panel
(F2, CFB, TTR) and reports:

  S12: per-protein coefficient, sign, direction, plus the LDA intercept,
       in a compact main-text-friendly layout.
  S13: misclassification count and rate under each of the seven CV
       schemes from the glossary.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from lib.data_utils import load_longitudinal
from lib.panels import PANELS

# Re-use the helper functions defined in 14_*.
import importlib.util
spec = importlib.util.spec_from_file_location(
    "cv14", str(Path(__file__).resolve().parent /
                "14_table4_panel_cv_evaluation.py"))
cv14 = importlib.util.module_from_spec(spec); spec.loader.exec_module(cv14)

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    proteins = PANELS["Clinical-3"]
    m, y, g, vis, wk, names, info = load_longitudinal(
        apply_per_protein_qc=False, apply_per_sample_qc=True)
    name_to_idx = {n: i for i, n in enumerate(names)}
    idxs = [name_to_idx[p] for p in proteins]
    X = m[idxs].T
    clf = LinearDiscriminantAnalysis().fit(X, y)
    coefs = clf.coef_.ravel()
    intercept = float(clf.intercept_.ravel()[0])

    rows = []
    for p, w in zip(proteins, coefs):
        rows.append({
            "Protein": p,
            "Coefficient": f"{float(w):+.5g}",
            "Sign": "+" if w > 0 else "-",
            "Direction": "healer" if w > 0 else "non-healer",
        })
    rows.append({"Protein": "(intercept)",
                 "Coefficient": f"{intercept:+.5g}",
                 "Sign": "", "Direction": ""})
    coef_df = pd.DataFrame(rows)
    coef_path = OUT / "tableS12_clinical3_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)
    print(f"Saved {coef_path}")
    print(coef_df.to_string(index=False))
    print()

    schemes = {
        "baseline-LOOCV":      cv14.baseline_loocv(X, y, vis),
        "baseline-LOPPO":      cv14.baseline_loppo(X, y, g, vis),
        "all-68-LOOCV":        cv14.all68_loocv(X, y),
        "all-68-LOPPO":        cv14.all68_loppo(X, y, g),
        "LOPO":                cv14.lopo(X, y, g),
        "LOVO":                cv14.lovo(X, y, vis),
        "baseline->follow-up": cv14.baseline_to_followup(X, y, vis),
    }
    rows = []
    for name, (err, n) in schemes.items():
        rows.append({
            "Cross-validation scheme": name,
            "Misclassifications": err,
            "Total predictions": n,
            "Error rate": f"{err/n:.1%}" if n else "-",
        })
    cv_df = pd.DataFrame(rows)
    cv_path = OUT / "tableS13_clinical3_cv_errors.csv"
    cv_df.to_csv(cv_path, index=False)
    print(f"Saved {cv_path}")
    print(cv_df.to_string(index=False))


if __name__ == "__main__":
    main()
