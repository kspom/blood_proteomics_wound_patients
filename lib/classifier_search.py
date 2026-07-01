"""Greedy forward search with patient-pair cross-validation.

When a `groups` (patient-ID) vector is provided, CV is *pairwise*: every
fold holds out all samples from one healer patient AND all samples from
one non-healer patient simultaneously, trains on the rest, and tests on
the held-out pair. With 7 healers and 9 non-healers there are 7 * 9 =
63 folds and the total number of test predictions equals the number of
samples in held-out patient pairs (528 across the 63 folds in this
cohort).

If groups is None, the function falls back to sample-level LOOCV.

Error metric = total count of misclassified held-out samples summed over
all folds.
"""

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneOut


def _make_classifier(method):
    if method == "lda":
        return LinearDiscriminantAnalysis()
    if method == "svm":
        return LinearSVC(dual="auto", max_iter=10000)
    raise ValueError(f"Unknown classifier method: {method}")


def _train_error(matrix, labels, indices, method):
    X = matrix[list(indices)].T
    clf = _make_classifier(method)
    clf.fit(X, labels)
    pred = clf.predict(X)
    return int(np.sum(pred != labels))


def _pairwise_patient_splits(groups, labels):
    """Yield (train_idx, test_idx) for every (healer, non-healer) patient pair."""
    g = np.asarray(groups)
    y = np.asarray(labels)
    healthy = sorted({int(p) for p in g[y == 0]})
    diseased = sorted({int(p) for p in g[y == 1]})
    all_idx = np.arange(len(g))
    for hp in healthy:
        for dp in diseased:
            test_mask = (g == hp) | (g == dp)
            yield all_idx[~test_mask], all_idx[test_mask]


def _cv_error(matrix, labels, indices, method, groups=None):
    """LOOCV when groups is None; pairwise patient-pair CV otherwise."""
    X = matrix[list(indices)].T
    if groups is None:
        splits = LeaveOneOut().split(X)
    else:
        splits = _pairwise_patient_splits(groups, labels)
    errors = 0
    for train_idx, test_idx in splits:
        clf = _make_classifier(method)
        clf.fit(X[train_idx], labels[train_idx])
        pred = clf.predict(X[test_idx])
        errors += int(np.sum(pred != labels[test_idx]))
    return errors


def scan_all_pairs(matrix, labels, protein_names, method="lda",
                   cv_for_top=None, groups=None):
    """Score every protein pair by training error; CV only the top-k."""
    n = matrix.shape[0]
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            err = _train_error(matrix, labels, (i, j), method)
            rows.append({
                "protein_1": protein_names[i],
                "protein_2": protein_names[j],
                "i": i, "j": j,
                "train_error": err,
                "cv_error": None,
            })
    df = pd.DataFrame(rows).sort_values("train_error").reset_index(drop=True)

    if cv_for_top is None:
        min_err = df["train_error"].min()
        cv_mask = df["train_error"] == min_err
    else:
        cv_mask = df.index < cv_for_top

    for idx in df.index[cv_mask]:
        i, j = int(df.at[idx, "i"]), int(df.at[idx, "j"])
        df.at[idx, "cv_error"] = _cv_error(matrix, labels, (i, j), method,
                                           groups=groups)

    return df.drop(columns=["i", "j"])


def find_best_addition(matrix, labels, protein_names, current_indices,
                       method="lda", top_k=5, log=None, groups=None):
    """Try adding each remaining protein; pick best via CV on the top-k."""
    current = set(current_indices)
    n = matrix.shape[0]
    candidates = [k for k in range(n) if k not in current]

    scored = []
    for cand in candidates:
        idxs = tuple(list(current_indices) + [cand])
        err = _train_error(matrix, labels, idxs, method)
        scored.append((err, cand))
    scored.sort(key=lambda x: x[0])

    top = scored[:top_k]
    top_k_cv = []
    for err, cand in top:
        idxs = tuple(list(current_indices) + [cand])
        cv = _cv_error(matrix, labels, idxs, method, groups=groups)
        top_k_cv.append((cand, err, cv))

    top_k_cv.sort(key=lambda x: (x[2], x[1]))
    winner_cand, winner_train, winner_cv = top_k_cv[0]
    winner_indices = list(current_indices) + [winner_cand]

    if log is not None:
        for cand, err, cv in top_k_cv:
            names = tuple(protein_names[k] for k in list(current_indices) + [cand])
            action = "added" if cand == winner_cand else "top_k_candidate"
            log.add_entry(step="6b", proteins=names, train_error=err,
                          cv_error=cv, action=action)

    return {
        "indices": winner_indices,
        "names": [protein_names[k] for k in winner_indices],
        "train_error": winner_train,
        "cv_error": winner_cv,
        "ranked": scored,
        "top_k_cv": top_k_cv,
    }
