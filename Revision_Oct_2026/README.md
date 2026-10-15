# Re-running the code of the previous revision in a different Python environment

## What was done

The code of the previous revision (https://github.com/kspom/blood_proteomics_wound_patients), unchanged and on the same data, was re-run in a new Python environment. The outputs were compared with the main and supplementary tables of the previous revision of the manuscript.

| Package | Previous revision | Re-run |
|---|---|---|
| Python | 3.13.7 | 3.11.17 |
| scikit-learn | 1.8.0 | 1.9.1 |
| numpy | 2.4.4 | 2.4.6 |
| pandas | 3.0.2 | 3.0.6 |
| scipy | 1.17.1 | 1.17.1 |
| statsmodels | 0.14.6 | 0.15.0 |

All scripts ran without errors. The code and the input data were identical; only the library versions differed.

## Why results differ

Between the two scikit-learn versions, every LDA coefficient in Table S7 changes by the same factor, 1.0303 = 68/66 = n/(n − 2). This is consistent with a change in how the within-class covariance is normalised. Rescaling all coefficients alone would not change any prediction. However, the class-prior term in the intercept is not rescaled, so the decision threshold moves slightly. As a result, samples lying almost exactly on the boundary can switch class.

This works as an unplanned perturbation test. A finding that holds up under it does not depend on the numerical details of the classifier. A finding that changes depends on one or two borderline samples.

## What is stable

- Univariate differential abundance (Table 1, Tables S1, S2) and Tables S15, S20: identical.
- The three zero-error panels of the patient-pair (LOPPO) search (Table S5) are identical:
  - CFB + IGHV3-64 + MINPP1 + PI16 + SELL (Complement-5)
  - CFB + IGHV3-64 + IGHV4-4 + MINPP1 + PI16
  - ADIPOQ + FGB + IGHV3-64 + IGLV2-11 (Metabolic-4)
- Top-ranked panels: ranks 1–6 of Table 2 are identical in composition and order.
- Leave-one-patient-out search (Table S19): Core-3 {CFB, IGHV3-64, PI16} recurs in 16/16 runs, the four-protein backbone in 7/16 and Complement-5 in 3/16 in both.
- Panel performance (Table 4): identical for Complement-5, Core-3 and Clinical-3 under all seven schemes.
- Coefficient signs, and therefore the biological direction of every protein, are identical. Magnitudes differ by the constant factor above.

## What is not stable

- Seed pairs at training error ≤ 5 (Table S4): 45 → 42.
- Ranks 7–10 of Table 2: two of four panels are replaced.
- LOOCV search (Table S6): 83 → 93 unique panels, zero-error panels 8 → 6. Metabolic-4 is no longer zero-error under LOOCV.
- Metabolic-4 under LOPO / LOVO (Table 4): 1/68 and 2/68 → 0/68 and 1/68.

## Conclusion

The core result does not depend on numerical details of the classifier implementation: the CFB / IGHV3-64 / PI16 core, the three zero-error panels and the performance of Complement-5. Lower-ranked panels, whose errors differ by one or two samples, are sensitive to them. These lower-ranked panels should be read as examples of alternative configurations, not as specific findings.

## Files

- `previous_revision/`: main (Tables 1-5) and supplementary (Tables S1-S20) tables of the previous revision.
- `rerun/`: the same tables produced by the re-run.
- `compare.py`: script that produces the comparison tables (run from this folder: `python compare.py`).
- `comparison/00_summary.csv`: key counts side by side.
- `comparison/01_table_level.csv`: for each table of the previous revision, whether the re-run reproduces it.
- `comparison/02_top10_panels_LOPPO.csv`: Table 2 side by side.
- `comparison/03_zero_error_panels.csv`: zero-error panels under both search schemes.
- `comparison/04_panel_coefficients.csv`: LDA coefficients of the four named panels.
- `comparison/05_table4_cv_schemes.csv`: Table 4 side by side.
