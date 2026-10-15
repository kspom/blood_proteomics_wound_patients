"""Table S23 -- permutation test of the nested leave-one-patient-out validation.

Asks how often the whole procedure -- panel search without the held-out
patient (as in 25_lopo_sensitivity.py) followed by prediction of that
patient (as in 28_tableS22_nested_lopo.py) -- reaches an error as low as
the observed one when the class labels carry no information.

In each permutation, the healer / non-healer labels are shuffled between
the 16 patients (7 healers and 9 non-healers are kept); all samples of a
patient receive that patient's new label. The full nested procedure is
then repeated on the permuted labels.

Rules fixed before running:
  * seed pairs: training error <= 5, as in the paper. If no pair reaches
    this threshold (expected under random labels), the pairs with the
    minimum training error are used instead; at most MAX_SEEDS pairs
    (the largest number of seeds in the real leave-one-patient-out runs
    was 168).
  * cross-validation of the top pairs, which script 25 computes for
    reporting only, is skipped; it does not affect the search.
  * panel choice and prediction: rule of script 28 (all panels tied at
    rank 1 vote; patient = sign of the mean Protein Healing Score).
  * statistic: nested error over samples (primary) and over patients.
    p = (1 + #permutations with error <= observed) / (1 + N).

Results are appended to output/tableS23_permutation_test.csv as each
permutation finishes, so an interrupted run can be resumed.

Usage:
  python 29_tableS23_permutation_test.py [--n-perm 100] [--workers 22]
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from multiprocessing import Pool
import numpy as np
import pandas as pd

from lib.data_utils import load_longitudinal
from lib.classifier_search import scan_all_pairs
from lib.branching import branching_search

# Re-use the panel choice and prediction rules of 28_*.
import importlib.util
spec = importlib.util.spec_from_file_location(
    "nested28", str(Path(__file__).resolve().parent /
                    "28_tableS22_nested_lopo.py"))
nested28 = importlib.util.module_from_spec(spec); spec.loader.exec_module(nested28)

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)
RUNS = OUT / "tableS23_permutation_test.csv"
SUMMARY = OUT / "permutation_test_summary.csv"

SEED_CUTOFF_ERROR = 5
MAX_SEEDS = 200
TOP_K_FOR_CV = 5
MAX_ITERATIONS = 20
RNG_SEED = 0

DATA = {}


def init_worker():
    m, y, g, vis, wk, names, info = load_longitudinal(
        apply_per_protein_qc=True, apply_per_sample_qc=True)
    DATA.update(m=m, y=y, g=g, vis=vis, names=names)


def permuted_labels(n_perm):
    """Patient-level label permutations, reproducible from RNG_SEED."""
    m, y, g, vis, wk, names, info = load_longitudinal(
        apply_per_protein_qc=True, apply_per_sample_qc=True)
    patients = sorted(np.unique(g).tolist())
    patient_label = np.array([int(y[g == p][0]) for p in patients])
    rng = np.random.default_rng(RNG_SEED)
    perms = []
    for _ in range(n_perm):
        new = dict(zip(patients, rng.permutation(patient_label)))
        perms.append(np.array([new[int(p)] for p in g]))
    return perms, patients


def search_without(excluded, y_perm):
    """Search of script 25 (without excluded patient) on permuted labels."""
    m, g, names = DATA["m"], DATA["g"], DATA["names"]
    keep = g != excluded
    mk, yk, gk = m[:, keep], y_perm[keep], g[keep]
    pair_df = scan_all_pairs(mk, yk, names, method="lda",
                             cv_for_top=0, groups=gk)
    seeds = pair_df[pair_df["train_error"] <= SEED_CUTOFF_ERROR]
    fallback = seeds.empty
    if fallback:
        seeds = pair_df[pair_df["train_error"] == pair_df["train_error"].min()]
    capped = len(seeds) > MAX_SEEDS
    seeds = seeds.head(MAX_SEEDS)

    name_to_idx = {n: i for i, n in enumerate(names)}
    rows = []
    for _, row in seeds.iterrows():
        pair = (name_to_idx[row["protein_1"]], name_to_idx[row["protein_2"]])
        for b in branching_search(mk, yk, names, method="lda",
                                  top_k=TOP_K_FOR_CV,
                                  max_iterations=MAX_ITERATIONS,
                                  seed_pairs=[pair], groups=gk):
            rows.append({"size": len(b["names"]),
                         "train_error": b["train_error"],
                         "cv_error": b["cv_error"],
                         "proteins": "|".join(sorted(b["names"]))})
    unique = (pd.DataFrame(rows)
              .groupby("proteins", as_index=False)
              .agg(size=("size", "first"),
                   train_error=("train_error", "min"),
                   cv_error=("cv_error", "min"),
                   n_times_found=("proteins", "size"))
              .sort_values(["cv_error", "train_error", "size", "n_times_found"],
                           ascending=[True, True, True, False])
              .reset_index(drop=True))
    unique["rank"] = np.arange(1, len(unique) + 1)
    return unique, fallback, capped, len(seeds)


def one_fold(task):
    k, excluded, y_perm = task
    m, g, vis, names = DATA["m"], DATA["g"], DATA["vis"], DATA["names"]
    unique, fallback, capped, n_seeds = search_without(excluded, y_perm)
    panels = list(nested28.tied_best(unique)["proteins"])
    test = g == excluded; train = ~test
    s_pred, p_pred, _ = nested28.predict(panels, m, names, train, test,
                                         y_perm, vis[test])
    return {"perm": k, "excluded_patient": int(excluded),
            "sample_errors": int((s_pred != y_perm[test]).sum()),
            "patient_error": int(p_pred != int(y_perm[test][0])),
            "fallback_seeds": fallback, "capped_seeds": capped,
            "n_seeds": n_seeds}


def summarise(n_perm):
    # rows are appended in order of completion; store them by permutation
    runs = pd.read_csv(RUNS).sort_values("perm").reset_index(drop=True)
    runs.to_csv(RUNS, index=False)
    runs = runs[runs["perm"] < n_perm]
    s22 = pd.read_csv(OUT / "tableS22_nested_lopo.csv")
    obs_s = int(s22["sample_errors"].sum())
    obs_p = int((s22["patient_pred"] != s22["true_class"]).sum())
    N = len(runs)
    summary = pd.DataFrame([
        {"statistic": "nested sample errors (of 68)", "observed": obs_s,
         "null_median": float(runs["sample_errors"].median()),
         "null_min": int(runs["sample_errors"].min()),
         "n_perm": N,
         "p_value": (1 + int((runs["sample_errors"] <= obs_s).sum())) / (1 + N)},
        {"statistic": "nested patient errors (of 16)", "observed": obs_p,
         "null_median": float(runs["patient_errors"].median()),
         "null_min": int(runs["patient_errors"].min()),
         "n_perm": N,
         "p_value": (1 + int((runs["patient_errors"] <= obs_p).sum())) / (1 + N)},
    ])
    summary.to_csv(SUMMARY, index=False)
    print(summary.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--workers", type=int, default=22)
    args = ap.parse_args()

    perms, patients = permuted_labels(args.n_perm)
    done = set(pd.read_csv(RUNS)["perm"]) if RUNS.exists() else set()
    tasks = [(k, p, perms[k]) for k in range(args.n_perm) if k not in done
             for p in patients]
    print(f"{args.n_perm} permutations, {len(done)} already done, "
          f"{len(tasks)} folds to run on {args.workers} workers.", flush=True)

    pending = {}
    with Pool(args.workers, initializer=init_worker) as pool:
        for r in pool.imap_unordered(one_fold, tasks):
            pending.setdefault(r["perm"], []).append(r)
            if len(pending[r["perm"]]) == len(patients):
                folds = pd.DataFrame(pending.pop(r["perm"]))
                row = pd.DataFrame([{
                    "perm": r["perm"],
                    "sample_errors": int(folds["sample_errors"].sum()),
                    "patient_errors": int(folds["patient_error"].sum()),
                    "folds_with_fallback_seeds": int(folds["fallback_seeds"].sum()),
                    "folds_with_capped_seeds": int(folds["capped_seeds"].sum()),
                    "median_n_seeds": float(folds["n_seeds"].median()),
                }])
                row.to_csv(RUNS, mode="a", header=not RUNS.exists(), index=False)
                print(f"perm {r['perm']}: sample errors "
                      f"{int(row['sample_errors'][0])}/68, patient errors "
                      f"{int(row['patient_errors'][0])}/16", flush=True)

    summarise(args.n_perm)


if __name__ == "__main__":
    main()
