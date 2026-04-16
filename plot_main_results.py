# plot_main_results.py
"""
Publication-quality grouped bar chart.
Methods: LeTra / LeafTrackNet / LeafTrackNet+Fixed / Ours
Metrics: HOTA / AssA / IDF1
Style: CEA / Plant Methods journal standard
"""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams

# ---------------------------------------------------------------
# Global style
# ---------------------------------------------------------------
rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 0,  # no tick marks on x-axis (cleaner)
        "ytick.major.size": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.bottom": True,
        "legend.frameon": True,
        "legend.edgecolor": "#dddddd",
        "legend.fancybox": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# ---------------------------------------------------------------
# Data
# ---------------------------------------------------------------
methods = [
    "LeTra",
    "LeafTrackNet",
    "LeafTrackNet\n+ Fixed constraints",
    "Ours\n(Adaptive weights)",
]

# mean ± std  (std=0 for deterministic methods)
data = {
    "HOTA": {
        "mean": np.array([67.02, 88.03, 89.04, 89.69]),
        "std": np.array([0.04, 0.24, 0.00, 0.10]),
    },
    "AssA": {
        "mean": np.array([54.98, 84.07, 85.98, 87.26]),
        "std": np.array([0.16, 0.49, 0.00, 0.19]),
    },
    "IDF1": {
        "mean": np.array([69.06, 92.90, 93.76, 94.39]),
        "std": np.array([0.10, 0.35, 0.00, 0.09]),
    },
}

metrics = list(data.keys())
n_method = len(methods)
n_metric = len(metrics)

# ---------------------------------------------------------------
# Layout
# ---------------------------------------------------------------
# Academic palette: muted, colour-blind friendly
PALETTE = [
    "#aec7e8",  # light blue  — LeTra
    "#6baed6",  # mid blue    — LeafTrackNet
    "#2171b5",  # dark blue   — + Fixed
    "#084594",  # deep navy   — Ours
]
EDGE_COLOR = "white"

bar_w = 0.18  # single bar width
group_gap = 0.12  # gap between groups (metrics)
x_centers = np.arange(n_metric) * (n_method * bar_w + group_gap)

fig, ax = plt.subplots(figsize=(9.0, 4.0))
fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.22)

# ---------------------------------------------------------------
# Draw bars
# ---------------------------------------------------------------
for mi, method in enumerate(methods):
    offsets = np.arange(n_metric) * 0 + (mi - (n_method - 1) / 2) * bar_w
    xs = x_centers + offsets
    ys = np.array([data[m]["mean"][mi] for m in metrics])
    errs = np.array([data[m]["std"][mi] for m in metrics])

    bars = ax.bar(
        xs,
        ys,
        width=bar_w,
        color=PALETTE[mi],
        edgecolor=EDGE_COLOR,
        linewidth=0.4,
        label=method,
        zorder=3,
    )
    # error bars (only where std > 0)
    for xi, yi, ei in zip(xs, ys, errs):
        if ei > 0:
            ax.errorbar(
                xi,
                yi,
                yerr=ei,
                fmt="none",
                ecolor="#333333",
                elinewidth=0.8,
                capsize=2.5,
                capthick=0.8,
                zorder=4,
            )
    # value labels on top of each bar
    for xi, yi, ei in zip(xs, ys, errs):
        label_y = yi + ei + 0.5
        ax.text(
            xi,
            label_y,
            f"{yi:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color="#333333",
        )

# ---------------------------------------------------------------
# Axes
# ---------------------------------------------------------------
ax.set_xticks(x_centers)
ax.set_xticklabels(metrics, fontsize=10, fontweight="bold")
ax.set_ylabel("Score")
ax.set_ylim(55, 100)
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
ax.grid(axis="y", linestyle=":", linewidth=0.45, color="#d0d0d0", zorder=0)
ax.set_axisbelow(True)

# light vertical separators between metric groups
for xc in x_centers[1:]:
    ax.axvline(
        xc - (bar_w * n_method / 2 + group_gap / 2),
        color="#e0e0e0",
        linewidth=0.5,
        linestyle="-",
        zorder=0,
    )

# ---------------------------------------------------------------
# Legend  (below x-axis, horizontal)
# ---------------------------------------------------------------
handles, labels = ax.get_legend_handles_labels()
leg = ax.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.50, -0.18),
    ncol=4,
    handlelength=1.4,
    handletextpad=0.4,
    columnspacing=1.0,
    borderpad=0.4,
    fontsize=8.5,
)
leg.get_frame().set_linewidth(0.5)

ax.set_title(
    "Tracking performance on CanolaTrack — "
    "progressive improvement with biology-aware constraints",
    loc="left",
    fontweight="bold",
    pad=7,
    fontsize=9.5,
)

# ---------------------------------------------------------------
# Save
# ---------------------------------------------------------------
os.makedirs("figures", exist_ok=True)
out = "figures/main_results_bar.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"Saved → {out}")
