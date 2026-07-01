"""Tables S8, S9 -- protein occurrence in seed pairs and top-10 panels.

  S8: every protein that appears in any of the 53 seed pairs
      (train_error <= 6 in the LOPPO pair scan), with the count of
      seed pairs in which it participates.
  S9: every protein that appears in any of the 10 top all-68-LOPPO
      panels, with the count of those panels in which it participates.

Reads from output/loppo_pair_scan.csv and
output/table2_top10_panels_LOPPO.csv (both produced by 09_*).
"""

from pathlib import Path
from collections import Counter
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)

SEED_CUTOFF_ERROR = 6


def main():
    pair_scan = pd.read_csv(OUT / "loppo_pair_scan.csv")
    seeds = pair_scan[pair_scan["train_error"] <= SEED_CUTOFF_ERROR]
    n_seeds = len(seeds)

    c1 = Counter()
    for _, r in seeds.iterrows():
        c1[r["protein_1"]] += 1; c1[r["protein_2"]] += 1
    s8 = pd.DataFrame([{"protein": p, "n_seed_pairs": n}
                       for p, n in c1.most_common()])
    s8["fraction_of_seed_pairs"] = (s8["n_seed_pairs"] / n_seeds).round(3)
    s8.to_csv(OUT / "tableS8_protein_freq_seed_pairs.csv", index=False)
    print(f"Saved {OUT/'tableS8_protein_freq_seed_pairs.csv'}  "
          f"({len(s8)} unique proteins across {n_seeds} seed pairs)")

    top10 = pd.read_csv(OUT / "table2_top10_panels_LOPPO.csv")
    n_panels = len(top10)
    c2 = Counter()
    for _, r in top10.iterrows():
        for p in r["proteins"].split("|"):
            c2[p] += 1
    s9 = pd.DataFrame([{"protein": p, "n_panels_top10": n}
                       for p, n in c2.most_common()])
    s9["fraction_of_top10"] = (s9["n_panels_top10"] / n_panels).round(3)
    s9.to_csv(OUT / "tableS9_protein_freq_top10_panels.csv", index=False)
    print(f"Saved {OUT/'tableS9_protein_freq_top10_panels.csv'}  "
          f"({len(s9)} unique proteins across {n_panels} panels)")
    print()
    print(s9.to_string(index=False))


if __name__ == "__main__":
    main()
