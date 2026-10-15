"""Table S22 -- nested leave-one-patient-out validation of the panel search.

Uses the 16 leave-one-patient-out searches of 25_lopo_sensitivity.py, in
which the whole search (pair scan, seed-pair selection, greedy branching
with patient-pair CV) was run without one patient. Here each held-out
patient is predicted by the panel(s) found without them, so the patient
takes no part in selecting proteins or coefficients.

Panel choice in each run (fixed before looking at the predictions): all
panels tied at rank 1 of the search ranking (cv_error, train_error, size,
n_times_found). The method returns a set of equally ranked panels, so all
of them vote.

For each panel, LDA is fitted on all samples of the 15 other patients and
applied to the held-out patient. Predictions at three levels:
  sample   -- every sample of the held-out patient;
  patient  -- sign of the mean Protein Healing Score over the patient's
              samples;
  visit 1  -- the patient's baseline sample only (if it passed QC).
When several panels are tied, they vote at each level; a tied vote is
resolved by the sign of the mean score across panels.

Outputs:
  output/tableS22_nested_lopo.csv       -- one row per held-out patient.
  output/nested_lopo_summary.csv        -- error counts with 95% exact
                                           (Clopper-Pearson) intervals.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from lib.data_utils import load_longitudinal

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)

RANK_KEY = ["cv_error", "train_error", "size", "n_times_found"]


def tied_best(run):
    """Rows sharing the rank-1 value of the search ranking key."""
    best = run.sort_values("rank").iloc[0]
    mask = np.ones(len(run), dtype=bool)
    for k in RANK_KEY:
        mask &= (run[k] == best[k]).to_numpy()
    return run[mask].sort_values("rank")


def panel_scores(panel, m, names, train, test, y):
    idx = [names.index(p) for p in panel.split("|")]
    X = m[idx].T
    clf = LinearDiscriminantAnalysis().fit(X[train], y[train])
    return clf.decision_function(X[test])   # > 0 -> healer (class 1)


def vote(scores):
    """Majority vote over panels; a tie is resolved by the mean score.

    scores: array (n_panels,) of decision values for one prediction.
    """
    votes = (np.asarray(scores) > 0).astype(int)
    k = votes.sum(); n = len(votes)
    if 2 * k != n:
        return int(2 * k > n)
    return int(np.mean(scores) > 0)


def predict(panels, m, names, train, test, y, vis_test):
    """Sample, patient and visit-1 predictions from one or more panels."""
    S = np.array([panel_scores(p, m, names, train, test, y) for p in panels])
    sample_pred = np.array([vote(S[:, j]) for j in range(S.shape[1])])
    patient_pred = vote(S.mean(axis=1))
    base = vis_test == 1
    visit1_pred = vote(S[:, base].mean(axis=1)) if base.any() else None
    return sample_pred, patient_pred, visit1_pred


def ci(err, n):
    lo, hi = binomtest(err, n).proportion_ci(confidence_level=0.95, method="exact")
    return f"{err}/{n} ({100*err/n:.0f}%; 95% CI {100*lo:.0f}-{100*hi:.0f}%)"


def main():
    m, y, g, vis, wk, names, info = load_longitudinal(
        apply_per_protein_qc=True, apply_per_sample_qc=True)
    lopo = pd.read_csv(OUT / "lopo_all_unique_sets.csv")

    rows = []
    for pid in sorted(lopo["excluded_patient"].unique()):
        run = lopo[lopo["excluded_patient"] == pid]
        test = g == pid; train = ~test
        true = int(y[test][0])
        row = {"excluded_patient": int(pid), "true_class": true,
               "n_samples": int(test.sum()),
               "has_visit1": bool((vis[test] == 1).any())}
        panels = list(tied_best(run)["proteins"])
        s_pred, p_pred, v_pred = predict(panels, m, names, train, test,
                                         y, vis[test])
        row["n_panels"] = len(panels)
        row["panels"] = "; ".join(panels)
        row["sample_errors"] = int((s_pred != y[test]).sum())
        row["patient_pred"] = p_pred
        row["visit1_pred"] = v_pred
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tableS22_nested_lopo.csv", index=False)

    n_samp = int(df["n_samples"].sum())
    v1 = df[df["has_visit1"]]
    summary = pd.DataFrame([{
        "samples": ci(int(df["sample_errors"].sum()), n_samp),
        "patients": ci(int((df["patient_pred"] != df["true_class"]).sum()), len(df)),
        "visit1": ci(int((v1["visit1_pred"] != v1["true_class"]).sum()), len(v1)),
    }])
    summary.to_csv(OUT / "nested_lopo_summary.csv", index=False)

    print(df[["excluded_patient", "true_class", "n_samples", "n_panels",
              "sample_errors", "patient_pred", "visit1_pred"]].to_string(index=False))
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
