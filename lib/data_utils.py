"""Central data loader for the paper-reproduction pipeline.

Reads the four files in `dataset/`:
  Data.csv               (1326 x 77 matrix of protein intensities)
  ClassLabel.csv         (77 rows, columns: class, patient_id, visit, week,
                          included)
  ProteinNames.txt       (1326 protein names, one per line)
  filtered_proteins.csv  (449 protein names; the per-protein QC pool used
                          in the published paper)

Applies three filters:
  * inclusion:    drop samples whose `included` flag is 0 in
                  ClassLabel.csv (used to remove patients who dropped out
                  due to wound-site serious adverse events unrelated to
                  the natural-history outcome being modelled). Default
                  drops patients #13 and #14.
  * per-protein:  keep only the 449 proteins listed in
                  filtered_proteins.csv (precomputed from the >25%
                  zero-fraction filter).
  * per-sample:   drop samples whose per-sample zero-fraction lies
                  outside [0.10, 0.80].

`apply_per_protein_qc=True` is the default and matches the search/DE
pipeline; `apply_per_protein_qc=False` is used by the longitudinal
Protein-Healing-Score analyses, which only need the sample-level QC
because the panel proteins are specified explicitly.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"

# precomputed per-protein QC: list of 449 proteins kept after the
# >25%-zero-fraction filter (Methods, "Data Filtering").
FILTERED_POOL_FILE = DATA_DIR / "filtered_proteins.csv"

# per-sample zero-fraction band (Methods, "Data Filtering")
PER_SAMPLE_ZERO_LOW = 0.10
PER_SAMPLE_ZERO_HIGH = 0.80


def load_longitudinal(apply_per_protein_qc=True, apply_per_sample_qc=True,
                       apply_inclusion_mask=True):
    """Load the dataset and apply the three filters.

    Returns:
        m       : (n_proteins, n_samples) float intensity matrix
        y       : (n_samples,)  int  class labels (0 = non-healer, 1 = healer)
        groups  : (n_samples,)  int  patient IDs
        visits  : (n_samples,)  int  visit number (1 = baseline)
        weeks   : (n_samples,)  int  week of sample collection (0..32)
        names   : list of n_proteins protein names
        info    : dict of QC summary numbers and dropped-sample details
    """
    matrix_full = pd.read_csv(DATA_DIR / "Data.csv",
                              header=None).to_numpy(dtype=float)
    cl = pd.read_csv(DATA_DIR / "ClassLabel.csv", header=None)
    labels_all = cl.iloc[:, 0].to_numpy(dtype=int)
    groups_all = cl.iloc[:, 1].to_numpy(dtype=int)
    visits_all = cl.iloc[:, 2].to_numpy(dtype=int)
    weeks_all = cl.iloc[:, 3].to_numpy(dtype=int)
    if cl.shape[1] >= 5:
        included_all = cl.iloc[:, 4].to_numpy(dtype=int)
    else:
        included_all = np.ones(matrix_full.shape[1], dtype=int)
    with open(DATA_DIR / "ProteinNames.txt", encoding="utf-8") as f:
        names_full = [ln.strip() for ln in f if ln.strip()]

    # ---- inclusion mask: drop samples flagged as excluded ----
    n_samples_raw = int(matrix_full.shape[1])
    if apply_inclusion_mask:
        inc_keep = included_all.astype(bool)
    else:
        inc_keep = np.ones(matrix_full.shape[1], dtype=bool)
    excluded_patients = sorted({int(p) for p in groups_all[~inc_keep]})
    matrix_full = matrix_full[:, inc_keep]
    labels_all = labels_all[inc_keep]
    groups_all = groups_all[inc_keep]
    visits_all = visits_all[inc_keep]
    weeks_all = weeks_all[inc_keep]
    n_samples_after_inclusion = int(matrix_full.shape[1])

    # ---- per-protein QC: keep the 449 proteins from filtered_proteins.csv ----
    if apply_per_protein_qc:
        with open(FILTERED_POOL_FILE, encoding="utf-8-sig") as f:
            pool = {ln.strip() for ln in f if ln.strip()}
        prot_keep = np.array([n in pool for n in names_full])
    else:
        prot_keep = np.ones(matrix_full.shape[0], dtype=bool)
    per_protein_zf = (matrix_full == 0).mean(axis=1)

    # ---- per-sample QC: keep samples whose zero-fraction is in [0.10, 0.80] ----
    # NOTE: per-sample zero-fraction is computed on the full 1326 proteins,
    # exactly as in the original paper pipeline.
    per_sample_zf = (matrix_full == 0).mean(axis=0)
    samp_keep = ((per_sample_zf >= PER_SAMPLE_ZERO_LOW)
                 & (per_sample_zf <= PER_SAMPLE_ZERO_HIGH)
                 if apply_per_sample_qc else
                 np.ones(matrix_full.shape[1], dtype=bool))

    m = matrix_full[np.ix_(prot_keep, samp_keep)]
    y = labels_all[samp_keep]
    g = groups_all[samp_keep]
    v = visits_all[samp_keep]
    w = weeks_all[samp_keep]
    names = [names_full[i] for i in range(len(names_full)) if prot_keep[i]]

    info = {
        "per_sample_zero_low": PER_SAMPLE_ZERO_LOW,
        "per_sample_zero_high": PER_SAMPLE_ZERO_HIGH,
        "n_proteins_before": int(matrix_full.shape[0]),
        "n_proteins_after": int(prot_keep.sum()),
        "n_samples_raw": n_samples_raw,
        "n_samples_after_inclusion": n_samples_after_inclusion,
        "n_samples_before": n_samples_after_inclusion,
        "n_samples_after": int(samp_keep.sum()),
        "excluded_patients": excluded_patients,
        "n_class0": int((y == 0).sum()),
        "n_class1": int((y == 1).sum()),
        "n_patients": int(len(np.unique(g))),
        "per_sample_zf_all": per_sample_zf,
        "per_protein_zf_all": per_protein_zf,
        "sample_keep_mask": samp_keep,
        "protein_keep_mask": prot_keep,
        "dropped_sample_ids": [int(i) for i in np.where(~samp_keep)[0]],
        "dropped_visits": [int(visits_all[i]) for i in np.where(~samp_keep)[0]],
        "dropped_weeks": [int(weeks_all[i]) for i in np.where(~samp_keep)[0]],
        "dropped_classes": [int(labels_all[i]) for i in np.where(~samp_keep)[0]],
        "dropped_patients": [int(groups_all[i]) for i in np.where(~samp_keep)[0]],
    }
    return m, y, g, v, w, names, info
