"""Table 3 -- panel roster.

Exports the five comparison panels (defined in lib/panels.py) as a CSV.
This is the lookup table the rest of the pipeline uses to cross-
reference panels by name.
"""

from pathlib import Path
import pandas as pd

from lib.panels import PANELS, PANEL_DESCRIPTIONS

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    rows = []
    for name, proteins in PANELS.items():
        rows.append({
            "panel": name,
            "proteins": ", ".join(proteins),
            "size": len(proteins),
            "selection_criterion": PANEL_DESCRIPTIONS[name],
        })
    df = pd.DataFrame(rows)
    out = OUT / "table3_panel_roster.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
