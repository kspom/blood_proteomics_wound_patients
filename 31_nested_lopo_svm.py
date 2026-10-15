"""Nested leave-one-patient-out validation of the panel search with a linear
SVM instead of LDA inside the search (one row of Table S24).

Same procedure as 25_lopo_sensitivity.py + 28_tableS22_nested_lopo.py, but
every LDA fit inside the search (pair scan, greedy branching and its
patient-pair cross-validation) and the final prediction use a linear SVM.
Unlike LDA, a linear SVM depends on the units of the proteins, so in each
fold the proteins are standardised (z-score) with the mean and standard
deviation of the 15 training patients; the held-out patient is scaled with
the same values.

Rules as in scripts 25 and 28: seed pairs with training error <= 5 (if no
pair qualifies, the pairs with the minimum training error); panels tied at
rank 1 vote; patient = sign of the mean SVM score over the patient's
samples.

Outputs:
  output/nested_lopo_svm.csv          -- one row per held-out patient.
  output/nested_lopo_svm_summary.csv  -- errors with exact 95% intervals.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC

from lib.data_utils import load_longitudinal
from lib.classifier_search import scan_all_pairs
from lib.branching import branching_search

import importlib.util
spec = importlib.util.spec_from_file_location(
    "nested28", str(Path(__file__).resolve().parent /
                    "28_tableS22_nested_lopo.py"))
nested28 = importlib.util.module_from_spec(spec); spec.loader.exec_module(nested28)

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)

SEED_CUTOFF_ERROR = 5
TOP_K_FOR_CV = 5
MAX_ITERATIONS = 20


def svm():
    return LinearSVC(dual="auto", max_iter=10000)   # as in lib/classifier_search.py


def one_fold(pid):
    t0 = time.time()
    m, y, g, vis, wk, names, info = load_longitudinal(
        apply_per_protein_qc=True, apply_per_sample_qc=True)
    test = g == pid; train = ~test
    mu = m[:, train].mean(axis=1, keepdims=True)
    sd = m[:, train].std(axis=1, keepdims=True)
    ms = (m - mu) / sd
    mtr, ytr, gtr = ms[:, train], y[train], g[train]

    pair_df = scan_all_pairs(mtr, ytr, names, method="svm",
                             cv_for_top=0, groups=gtr)
    seeds = pair_df[pair_df["train_error"] <= SEED_CUTOFF_ERROR]
    if seeds.empty:
        seeds = pair_df[pair_df["train_error"] == pair_df["train_error"].min()]
    name_to_idx = {n: i for i, n in enumerate(names)}
    rows = []
    for _, r in seeds.iterrows():
        pair = (name_to_idx[r["protein_1"]], name_to_idx[r["protein_2"]])
        for b in branching_search(mtr, ytr, names, method="svm",
                                  top_k=TOP_K_FOR_CV,
                                  max_iterations=MAX_ITERATIONS,
                                  seed_pairs=[pair], groups=gtr):
            rows.append({"size": len(b["names"]), "train_error": b["train_error"],
                         "cv_error": b["cv_error"],
                         "proteins": "|".join(sorted(b["names"]))})
    unique = (pd.DataFrame(rows).groupby("proteins", as_index=False)
              .agg(size=("size", "first"), train_error=("train_error", "min"),
                   cv_error=("cv_error", "min"),
                   n_times_found=("proteins", "size"))
              .sort_values(["cv_error", "train_error", "size", "n_times_found"],
                           ascending=[True, True, True, False])
              .reset_index(drop=True))
    unique["rank"] = np.arange(1, len(unique) + 1)
    panels = list(nested28.tied_best(unique)["proteins"])

    S = []
    for p in panels:
        idx = [names.index(q) for q in p.split("|")]
        clf = svm().fit(ms[idx][:, train].T, ytr)
        S.append(clf.decision_function(ms[idx][:, test].T))
    S = np.array(S)
    yte, vte = y[test], vis[test]
    s_pred = np.array([nested28.vote(S[:, j]) for j in range(S.shape[1])])
    p_pred = nested28.vote(S.mean(axis=1))
    base = vte == 1
    v_pred = nested28.vote(S[:, base].mean(axis=1)) if base.any() else None
    print(f"patient {pid}: {len(seeds)} seeds, {len(unique)} panels, "
          f"{time.time() - t0:.0f} s", flush=True)
    return {"excluded_patient": int(pid), "true_class": int(yte[0]),
            "n_samples": int(test.sum()), "has_visit1": bool(base.any()),
            "n_seeds": int(len(seeds)), "n_panels": len(panels),
            "panels": "; ".join(panels),
            "sample_errors": int((s_pred != yte).sum()),
            "patient_pred": p_pred, "visit1_pred": v_pred}


def main():
    m, y, g, *_ = load_longitudinal(apply_per_protein_qc=True,
                                    apply_per_sample_qc=True)
    patients = sorted(np.unique(g).tolist())
    with Pool(len(patients)) as pool:
        rows = pool.map(one_fold, patients)
    df = pd.DataFrame(rows).sort_values("excluded_patient")
    df.to_csv(OUT / "nested_lopo_svm.csv", index=False)

    v1 = df[df["has_visit1"]]
    summary = pd.DataFrame([{
        "samples": nested28.ci(int(df["sample_errors"].sum()), int(df["n_samples"].sum())),
        "patients": nested28.ci(int((df["patient_pred"] != df["true_class"]).sum()), len(df)),
        "visit1": nested28.ci(int((v1["visit1_pred"] != v1["true_class"]).sum()), len(v1)),
    }])
    summary.to_csv(OUT / "nested_lopo_svm_summary.csv", index=False)
    print(df[["excluded_patient", "true_class", "n_seeds", "n_panels",
              "sample_errors", "patient_pred", "visit1_pred", "panels"]].to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
