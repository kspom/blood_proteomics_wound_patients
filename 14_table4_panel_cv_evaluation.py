"""Table 4 -- panel performance under all seven cross-validation schemes.

For each of the five comparison panels (Pairwise-1, LOOCV-1, LOOCV-2,
Core-3, Clinical-3) and each of the seven schemes from the glossary
(Table S3), compute the misclassification count and rate.

Schemes:
  baseline-LOOCV       leave one of the 12 baseline samples out
  baseline-LOPPO       leave one (healer, non-healer) patient pair out,
                       baseline-only subset
  all-68-LOOCV         leave one of the 68 samples out
  all-68-LOPPO         leave one (healer, non-healer) patient pair out
                       across all 68 samples
  LOPO                 leave all visits of one patient out
  LOVO                 leave all samples of one visit out
  baseline -> follow-up  fit on visit 1 only, test on visits 2..6
"""

from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut

from lib.data_utils import load_longitudinal
from lib.panels import PANELS

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)


def fit_predict(X_train, y_train, X_test):
    return LinearDiscriminantAnalysis().fit(X_train, y_train).predict(X_test)


def baseline_loocv(X, y, vis):
    base = vis == 1
    Xb, yb = X[base], y[base]
    err = 0; n = len(yb)
    for tr, te in LeaveOneOut().split(Xb):
        err += int((fit_predict(Xb[tr], yb[tr], Xb[te]) != yb[te]).sum())
    return err, n


def patient_pair_splits(X, y, g):
    healthy = sorted({int(p) for p in g[y == 0]})
    diseased = sorted({int(p) for p in g[y == 1]})
    for hp, dp in product(healthy, diseased):
        test = (g == hp) | (g == dp)
        if test.sum() == 0:
            continue
        yield ~test, test


def baseline_loppo(X, y, g, vis):
    base = vis == 1
    Xb, yb, gb = X[base], y[base], g[base]
    err = 0; n = 0
    for tr, te in patient_pair_splits(Xb, yb, gb):
        if te.sum() == 0 or tr.sum() == 0:
            continue
        if len(set(yb[tr])) < 2:
            continue
        pred = fit_predict(Xb[tr], yb[tr], Xb[te])
        err += int((pred != yb[te]).sum()); n += int(te.sum())
    return err, n


def all68_loocv(X, y):
    err = 0
    for tr, te in LeaveOneOut().split(X):
        err += int((fit_predict(X[tr], y[tr], X[te]) != y[te]).sum())
    return err, len(y)


def all68_loppo(X, y, g):
    err = 0; n = 0
    for tr, te in patient_pair_splits(X, y, g):
        if len(set(y[tr])) < 2:
            continue
        pred = fit_predict(X[tr], y[tr], X[te])
        err += int((pred != y[te]).sum()); n += int(te.sum())
    return err, n


def lopo(X, y, g, vis=None, baseline_log=None):
    err = 0
    for p in sorted({int(p) for p in g}):
        te = g == p; tr = ~te
        if len(set(y[tr])) < 2:
            continue
        pred = fit_predict(X[tr], y[tr], X[te])
        err += int((pred != y[te]).sum())
        if vis is not None and baseline_log is not None:
            for v, true_y, p_pred in zip(vis[te], y[te], pred):
                if int(v) == 1:
                    baseline_log.append({
                        "patient_id": p,
                        "actual_class": int(true_y),
                        "predicted_class": int(p_pred),
                        "correct": bool(p_pred == true_y),
                    })
    return err, len(y)


def lovo(X, y, vis):
    err = 0
    for v in sorted({int(x) for x in vis}):
        te = vis == v; tr = ~te
        if len(set(y[tr])) < 2:
            continue
        err += int((fit_predict(X[tr], y[tr], X[te]) != y[te]).sum())
    return err, len(y)


def baseline_to_followup(X, y, vis):
    base = vis == 1; foll = vis > 1
    if base.sum() == 0 or foll.sum() == 0 or len(set(y[base])) < 2:
        return 0, 0
    pred = fit_predict(X[base], y[base], X[foll])
    return int((pred != y[foll]).sum()), int(foll.sum())


def main():
    m, y, g, vis, wk, names, info = load_longitudinal(
        apply_per_protein_qc=False, apply_per_sample_qc=True)
    name_to_idx = {n: i for i, n in enumerate(names)}

    schemes = ["baseline-LOOCV", "baseline-LOPPO", "all-68-LOOCV",
               "all-68-LOPPO", "LOPO", "LOVO", "baseline->follow-up"]
    rows = []
    baseline_rows = []
    for pname, proteins in PANELS.items():
        idxs = [name_to_idx[p] for p in proteins]
        X = m[idxs].T
        baseline_log = []
        results = {
            "baseline-LOOCV":      baseline_loocv(X, y, vis),
            "baseline-LOPPO":      baseline_loppo(X, y, g, vis),
            "all-68-LOOCV":        all68_loocv(X, y),
            "all-68-LOPPO":        all68_loppo(X, y, g),
            "LOPO":                lopo(X, y, g, vis=vis, baseline_log=baseline_log),
            "LOVO":                lovo(X, y, vis),
            "baseline->follow-up": baseline_to_followup(X, y, vis),
        }
        row = {"panel": pname}
        for sch in schemes:
            err, n = results[sch]
            row[sch] = f"{err}/{n}  ({err/n*100:.0f}%)" if n else "-"
        rows.append(row)
        for r in baseline_log:
            r["panel"] = pname
            baseline_rows.append(r)

    df = pd.DataFrame(rows)
    out = OUT / "table4_panel_cv_evaluation.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out}")
    print(df.to_string(index=False))

    # LOPO-baseline predictions per panel (Table S20)
    cols = ["panel", "patient_id", "actual_class", "predicted_class", "correct"]
    df_base = pd.DataFrame(baseline_rows)[cols].sort_values(
        ["panel", "patient_id"]).reset_index(drop=True)
    out_b = OUT / "tableS20_lopo_baseline_predictions.csv"
    df_base.to_csv(out_b, index=False)
    print(f"\nSaved {out_b}")
    # summary per panel
    summary = df_base.groupby("panel").agg(
        n_baseline=("correct", "size"),
        n_correct=("correct", "sum"),
        n_misclassified=("correct", lambda s: int((~s).sum())),
    ).reset_index()
    summary["error"] = summary.apply(
        lambda r: f"{r['n_misclassified']}/{r['n_baseline']} "
                  f"({r['n_misclassified']/r['n_baseline']*100:.0f}%)", axis=1)
    print("\nLOPO-baseline summary per panel:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
