"""Figure S6 -- classification-error trajectories for the top-5 LOPPO panels
during the greedy forward-branching search.

Trajectories are read from output/top5_branch_histories.csv (produced by
12b_top5_branch_histories.py), which records the actual sequence of
protein additions and the train/CV error at each step.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def main():
    hist = pd.read_csv(OUT / "top5_branch_histories.csv")
    panels = list(hist["panel_label"].drop_duplicates())

    fig, axes = plt.subplots(1, 2, figsize=(20, 7.5))
    markers = ["o", "s", "^", "D", "v"]
    linestyles = ["-", "--", "-.", ":", "-"]
    for (label, mk, ls) in zip(panels, markers, linestyles):
        sub = hist[hist["panel_label"] == label].sort_values("step_index")
        order = sub["proteins"].iloc[-1].split("|")
        legend = label + ":  " + " + ".join(order)
        axes[0].plot(sub["size"], sub["train_error"], linestyle=ls, marker=mk,
                     linewidth=2.8, markersize=16, label=legend, alpha=0.9)
        axes[1].plot(sub["size"], sub["cv_error"], linestyle=ls, marker=mk,
                     linewidth=2.8, markersize=16, alpha=0.9)
    for ax in axes:
        ax.set_xticks([2, 3, 4, 5])
        ax.tick_params(axis="both", labelsize=20)
        ax.grid(True, linestyle="--", alpha=0.3)
    axes[0].set_xlabel("Panel size", fontsize=23)
    axes[0].set_ylabel("Training error (/68)", fontsize=23)
    axes[0].set_title("(a) Training error", fontsize=25, loc="left")
    axes[0].legend(fontsize=16, loc="upper right")
    axes[1].set_xlabel("Panel size", fontsize=23)
    axes[1].set_ylabel("CV error  (LOPPO, /528)", fontsize=23)
    axes[1].set_title("(b) Cross-validation error", fontsize=25, loc="left")
    fig.tight_layout()
    out = FIG / "figS6_performance_trajectory.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
