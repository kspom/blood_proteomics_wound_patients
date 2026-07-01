"""Branching greedy forward search.

Implements the procedure described in the paper Methods. Starting from a
seed pair, each iteration:
  1. evaluates every remaining protein as the next addition,
  2. selects the top_k by training error,
  3. cross-validation–reranks those top_k,
  4. *forks* a new branch for every candidate tied at the lowest
     (cv_error, train_error) among the top_k.

Stops when training error reaches zero, when no further improvement is
possible, or when max_iterations is reached. max_branches caps the total
number of simultaneously active branches.
"""

from .classifier_search import find_best_addition, _train_error, _cv_error


def find_tied_best_additions(matrix, labels, protein_names, current_indices,
                             method="lda", top_k=5, groups=None):
    result = find_best_addition(matrix, labels, protein_names, current_indices,
                                method=method, top_k=top_k, log=None,
                                groups=groups)
    top_k_cv = result["top_k_cv"]
    best_train = top_k_cv[0][1]
    best_cv = top_k_cv[0][2]
    tied = [cand for cand, tr, cv in top_k_cv
            if cv == best_cv and tr == best_train]
    return {
        "tied_candidates": tied,
        "train_error": best_train,
        "cv_error": best_cv,
        "top_k_cv": top_k_cv,
    }


def branching_search(matrix, labels, protein_names, method="lda", top_k=5,
                     max_iterations=20, max_branches=20, log=None,
                     seed_pairs=None, groups=None):
    """Multi-branch greedy forward search from given seed pairs."""
    active = []
    assert seed_pairs is not None, "seed_pairs is required"
    for pair in seed_pairs:
        tr = _train_error(matrix, labels, pair, method)
        cv = _cv_error(matrix, labels, pair, method, groups=groups)
        names = [protein_names[k] for k in pair]
        if log is not None:
            log.add_entry(step="seed", proteins=tuple(names),
                          train_error=tr, cv_error=cv,
                          action="user_seed_pair")
        active.append({
            "indices": list(pair),
            "names": names,
            "train_error": tr,
            "cv_error": cv,
            "history": [{"step": "seed", "names": names,
                         "train_error": tr, "cv_error": cv}],
        })

    completed = []
    still_active = []
    for b in active:
        if b["train_error"] == 0:
            b["stop_reason"] = "zero_error"
            completed.append(b)
        else:
            still_active.append(b)
    active = still_active

    for _ in range(max_iterations):
        if not active:
            break

        new_active = []
        for branch in active:
            res = find_tied_best_additions(matrix, labels, protein_names,
                                           branch["indices"],
                                           method=method, top_k=top_k,
                                           groups=groups)
            if res["train_error"] >= branch["train_error"]:
                branch["stop_reason"] = "no_improvement"
                completed.append(branch)
                continue

            for cand in res["tied_candidates"]:
                new_indices = branch["indices"] + [cand]
                new_names = branch["names"] + [protein_names[cand]]
                nb = {
                    "indices": new_indices,
                    "names": new_names,
                    "train_error": res["train_error"],
                    "cv_error": res["cv_error"],
                    "history": branch["history"] + [{
                        "step": "6b",
                        "names": new_names,
                        "train_error": res["train_error"],
                        "cv_error": res["cv_error"],
                    }],
                }
                if log is not None:
                    action = ("branch_split"
                              if len(res["tied_candidates"]) > 1 else "added")
                    log.add_entry(step="6b", proteins=tuple(new_names),
                                  train_error=res["train_error"],
                                  cv_error=res["cv_error"], action=action)

                if res["train_error"] == 0:
                    nb["stop_reason"] = "zero_error"
                    completed.append(nb)
                else:
                    new_active.append(nb)

        seen = set()
        deduped = []
        for b in new_active:
            key = tuple(sorted(b["indices"]))
            if key not in seen:
                seen.add(key)
                deduped.append(b)
        new_active = deduped

        if len(new_active) > max_branches:
            new_active.sort(key=lambda b: (b["cv_error"], b["train_error"]))
            for b in new_active[max_branches:]:
                b["stop_reason"] = "pruned"
                completed.append(b)
            new_active = new_active[:max_branches]

        active = new_active

    for b in active:
        b["stop_reason"] = "max_iterations"
        completed.append(b)

    return completed
