"""Table S21 -- Table 4 panels re-evaluated with shrinkage LDA.

Re-evaluates the four comparison panels under the same seven
cross-validation schemes as Table 4 (identical folds, re-used from
14_table4_panel_cv_evaluation.py), but fits each fold with
shrinkage LDA (Ledoit-Wolf, `solver="lsqr", shrinkage="auto"`) instead
of the unpenalised LDA used in the paper. Panel composition is fixed;
only the coefficients are re-estimated in each fold.

Shrinkage pulls the within-class covariance towards a scaled identity
(sklearn standardises each protein internally for shrinkage="auto").
The shrinkage intensity alpha is chosen from the data: close to 0 when
there are many samples per protein, larger when there are few (the
baseline-only schemes train on 10-12 samples).

Writes output/tableS21_shrinkage_lda.csv: for every panel and scheme, the
error of unpenalised LDA (as in Table 4), the error of shrinkage LDA,
and the mean Ledoit-Wolf alpha across folds.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.covariance import ledoit_wolf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from lib.data_utils import load_longitudinal
from lib.panels import PANELS

# Re-use the cross-validation schemes defined in 14_*.
import importlib.util
spec = importlib.util.spec_from_file_location(
    "cv14", str(Path(__file__).resolve().parent /
                "14_table4_panel_cv_evaluation.py"))
cv14 = importlib.util.module_from_spec(spec); spec.loader.exec_module(cv14)

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)

unpenalised_fit_predict = cv14.fit_predict
alphas = []   # Ledoit-Wolf alpha of every shrinkage fit in the current scheme


def lw_alpha(X_train, y_train):
    """Prior-weighted Ledoit-Wolf alpha, as used by sklearn's shrinkage='auto'."""
    a, w = [], []
    for c in np.unique(y_train):
        Xc = StandardScaler().fit_transform(X_train[y_train == c])
        a.append(ledoit_wolf(Xc)[1]); w.append(np.mean(y_train == c))
    return float(np.average(a, weights=w))


def shrinkage_fit_predict(X_train, y_train, X_test):
    alphas.append(lw_alpha(X_train, y_train))
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    return clf.fit(X_train, y_train).predict(X_test)


def run_schemes(X, y, g, vis):
    return {
        "baseline-LOOCV":      lambda: cv14.baseline_loocv(X, y, vis),
        "baseline-LOPPO":      lambda: cv14.baseline_loppo(X, y, g, vis),
        "all-68-LOOCV":        lambda: cv14.all68_loocv(X, y),
        "all-68-LOPPO":        lambda: cv14.all68_loppo(X, y, g),
        "LOPO":                lambda: cv14.lopo(X, y, g),
        "LOVO":                lambda: cv14.lovo(X, y, vis),
        "baseline->follow-up": lambda: cv14.baseline_to_followup(X, y, vis),
    }


def fmt(err, n):
    return f"{err}/{n}  ({err/n*100:.0f}%)" if n else "-"


def main():
    m, y, g, vis, wk, names, info = load_longitudinal(
        apply_per_protein_qc=False, apply_per_sample_qc=True)
    name_to_idx = {n: i for i, n in enumerate(names)}

    rows = []
    for pname, proteins in PANELS.items():
        X = m[[name_to_idx[p] for p in proteins]].T
        for scheme, run in run_schemes(X, y, g, vis).items():
            cv14.fit_predict = unpenalised_fit_predict
            err_u, n_u = run()
            cv14.fit_predict = shrinkage_fit_predict
            alphas.clear()
            err_s, n_s = run()
            rows.append({
                "panel": pname,
                "n_proteins": len(proteins),
                "scheme": scheme,
                "unpenalised_LDA": fmt(err_u, n_u),
                "shrinkage_LDA": fmt(err_s, n_s),
                "same_errors": err_u == err_s,
                "mean_alpha": round(float(np.mean(alphas)), 3),
            })
    cv14.fit_predict = unpenalised_fit_predict

    df = pd.DataFrame(rows)
    out = OUT / "tableS21_shrinkage_lda.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
