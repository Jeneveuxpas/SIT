"""Plot standalone SiT-XL/2 FID convergence for three training variants."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
})

STEPS = np.array([1.1, 2.0, 3.5])
LABELS = ["100K", "200K", "400K"]
METHODS = ["REPA", "REPA + REPI (w/o consistency)", "REPA + REPI"]
DATA = {
    "REPA": [19.40, 11.10, 7.90],
    "REPA + REPI (w/o consistency)": [12.24, 7.97, 6.75],
    "REPA + REPI": [11.85, 7.47, 6.38],
}
COLORS = {
    "REPA": "#7CC7EA",
    "REPA + REPI (w/o consistency)": "#4A90E2",
    "REPA + REPI": "#FF4040",
}


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    fig, ax = plt.subplots(figsize=(6.2, 4.4), layout="constrained")

    for method in METHODS:
        is_ours = method == "REPI + REPA"
        ax.plot(
            STEPS,
            DATA[method],
            color=COLORS[method],
            marker="o",
            linewidth=4.2 if is_ours else 3.6,
            markersize=9 if is_ours else 8,
            markerfacecolor="white",
            markeredgewidth=2.5 if is_ours else 2.0,
            alpha=0.95 if is_ours else 0.88,
            label=method,
        )

    ax.set_xticks(STEPS)
    ax.set_xticklabels(LABELS)
    ax.set_xlabel("Training Steps", labelpad=8)
    ax.set_ylabel(r"FID Score $\downarrow$", labelpad=8)
    ax.set_xlim(STEPS[0] - 0.18, STEPS[-1] + 0.18)
    ax.set_yscale("log")
    ax.set_ylim(5.5, 21.5)
    ax.set_yticks([6, 8, 10, 15, 20])
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, linestyle="--", linewidth=1.0, color="#c9ccd1", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(1.45)
    ax.spines["bottom"].set_linewidth(1.45)
    ax.tick_params(axis="both", which="major", length=6, width=1.25)

    legend = ax.legend(loc="upper right", frameon=True, labelspacing=0.35, borderpad=0.45)
    legend.get_frame().set_edgecolor("gray")
    legend.get_frame().set_linewidth(1.2)
    legend.get_frame().set_alpha(0.95)
    for text, method in zip(legend.get_texts(), METHODS):
        text.set_color(COLORS[method])
        text.set_fontweight("bold")

    # Vertical distance between each final-value label and its dashed guide.
    label_offsets = {"REPA": 0.35, "REPA + REPI (w/o consistency)": 0.2, "REPA + REPI": -0.3}
    for method in METHODS:
        value = DATA[method][-1]
        ax.hlines(
            value,
            STEPS[0] - 0.18,
            STEPS[-1],
            color=COLORS[method],
            linestyle=(0, (4, 3)),
            linewidth=1.2,
            alpha=0.72,
            zorder=1,
        )
        ax.text(
            STEPS[0] + 0.1,
            value + label_offsets[method],
            f"{value:.2f}",
            color=COLORS[method],
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="center",
        )

    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"fid_chart_xl2_three_methods.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fid_chart_xl2_three_methods.pdf/png")


if __name__ == "__main__":
    main()
