"""Table S24 -- standard methods compared with the panel search.

All methods are evaluated on the same 16 outer leave-one-patient-out folds
as the nested validation of the panel search (Table S22): everything is
fitted on 15 patients and the held-out patient is predicted. Same 449
proteins, same three levels of evaluation (samples; patients by the sign of
the mean score over their samples; visit-1 sample only).

Standard methods (fitted here):
  shrinkage LDA (Ledoit-Wolf) on all 449 proteins;
  linear SVM on all 449 proteins (proteins standardised);
  random forest on all 449 proteins;
  LASSO: L1-penalised logistic regression (proteins standardised) with
         the penalty chosen by inner leave-one-patient-out cross-validation
         over the 15 training patients, so LASSO decides itself how many
         proteins to keep.
Panel search of this paper (read from earlier scripts):
  with LDA (output of 28_tableS22_nested_lopo.py);
  with a linear SVM inside the search (output of 31_nested_lopo_svm.py).
Run 28 and 31 first.

Output: output/tableS24_benchmark_methods.csv, one row per method.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, l1_min_c

from lib.data_utils import load_longitudinal

import importlib.util
spec = importlib.util.spec_from_file_location(
    "nested28", str(Path(__file__).resolve().parent /
                    "28_tableS22_nested_lopo.py"))
nested28 = importlib.util.module_from_spec(spec); spec.loader.exec_module(nested28)

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)

RNG_SEED = 0


def lasso_cv(X, y, g):
    """LASSO with the penalty chosen by inner leave-one-patient-out CV."""
    c0 = l1_min_c(StandardScaler().fit_transform(X), y, loss="log")
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(l1_ratio=1.0, solver="liblinear",
                                            max_iter=5000,
                                            random_state=RNG_SEED))
    grid = GridSearchCV(pipe, {"logisticregression__C": c0 * np.logspace(0, 3, 20)},
                        cv=LeaveOneGroupOut(), scoring="accuracy")
    grid.fit(X, y, groups=g)
    model = grid.best_estimator_
    return model, list(np.flatnonzero(model[-1].coef_.ravel()))


def all_protein_models():
    return {
        "shrinkage LDA": lambda: LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto"),
        "linear SVM": lambda: make_pipeline(
            StandardScaler(), LinearSVC(dual="auto", max_iter=20000)),
        "random forest": lambda: RandomForestClassifier(
            n_estimators=500, random_state=RNG_SEED),
    }


def scores_of(model, X):
    """Signed score: > 0 means healer (class 1)."""
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1] - 0.5


def fold_row(method, pid, n_proteins, proteins, scores, y_test, vis_test):
    true = int(y_test[0]); base = vis_test == 1
    return {"method": method, "patient": pid, "n_proteins": n_proteins,
            "proteins": proteins, "true_class": true,
            "sample_errors": int(((scores > 0).astype(int) != y_test).sum()),
            "n_samples": int(len(y_test)),
            "patient_pred": int(scores.mean() > 0),
            "visit1_pred": int(scores[base].mean() > 0) if base.any() else np.nan}


def search_rows(method, path):
    """Per-patient rows of the panel search (outputs of scripts 28 / 31)."""
    d = pd.read_csv(path)
    rows = []
    for r in d.itertuples():
        panels = r.panels.split("; ")
        rows.append({"method": method, "patient": r.excluded_patient,
                     "n_proteins": float(np.mean([len(p.split("|")) for p in panels])),
                     "proteins": "|".join(sorted(set("|".join(panels).split("|")))),
                     "true_class": r.true_class, "sample_errors": r.sample_errors,
                     "n_samples": r.n_samples, "patient_pred": r.patient_pred,
                     "visit1_pred": r.visit1_pred})
    return rows


def main():
    m, y, g, vis, wk, names, info = load_longitudinal(
        apply_per_protein_qc=True, apply_per_sample_qc=True)
    X = m.T
    rows = []
    for pid in sorted(np.unique(g).tolist()):
        t0 = time.time()
        test = g == pid; train = ~test
        for name, make in all_protein_models().items():
            mdl = make().fit(X[train], y[train])
            rows.append(fold_row(f"{name} (all proteins)", pid, 449, "",
                                 scores_of(mdl, X[test]), y[test], vis[test]))
        mdl, idx = lasso_cv(X[train], y[train], g[train])
        rows.append(fold_row("LASSO (penalty by inner CV)", pid, len(idx),
                             "|".join(sorted(names[i] for i in idx)),
                             scores_of(mdl, X[test]), y[test], vis[test]))
        print(f"patient {pid}: {time.time() - t0:.1f} s", flush=True)
    rows += search_rows("panel search with LDA (this paper)",
                        OUT / "tableS22_nested_lopo.csv")
    rows += search_rows("panel search with linear SVM",
                        OUT / "nested_lopo_svm.csv")
    folds = pd.DataFrame(rows)

    summary = []
    for method, d in folds.groupby("method", sort=False):
        v1 = d.dropna(subset=["visit1_pred"])
        prot = "|".join(d["proteins"])
        top = (pd.Series(prot.split("|")).value_counts().head(5)
               if prot.strip("|") else pd.Series(dtype=int))
        n = d["n_proteins"]
        summary.append({
            "method": method,
            "n_proteins": (f"{n.iloc[0]:.0f}" if n.nunique() == 1 else
                           f"{n.mean():.1f} ({n.min():.0f}-{n.max():.0f})"),
            "samples": nested28.ci(int(d["sample_errors"].sum()), int(d["n_samples"].sum())),
            "patients": nested28.ci(int((d["patient_pred"] != d["true_class"]).sum()), len(d)),
            "visit1": nested28.ci(int((v1["visit1_pred"] != v1["true_class"]).sum()), len(v1)),
            "most_selected_proteins":
                "; ".join(f"{p} ({c})" for p, c in top.items()),
        })
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "tableS24_benchmark_methods.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 70):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
