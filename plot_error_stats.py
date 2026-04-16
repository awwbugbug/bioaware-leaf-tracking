# plot_error_stats.py
"""
Per-plant IDSW comparison: LeafTrackNet vs Ours.
Strip plot + box overlay across all 37 test plants.
Output: figures/error_stats.pdf
"""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 0,
        "ytick.major.size": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": True,
        "legend.edgecolor": "#dddddd",
        "legend.fancybox": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

C_BASE = "#6baed6"
C_OURS = "#084594"


def parse_plant_idsw(filepath):
    result = {}
    in_clear = False
    with open(filepath) as f:
        for line in f:
            if "CLEAR:" in line:
                in_clear = True
            if not in_clear:
                continue
            if any(k in line for k in ["HOTA:", "Identity:", "Count:"]):
                in_clear = False
                continue
            stripped = line.strip()
            if not stripped or "COMBINED" in stripped:
                continue
            parts = stripped.split()
            if len(parts) < 14 or not parts[0].startswith("Plant-"):
                continue
            try:
                result[parts[0]] = int(float(parts[13]))
            except:
                continue
    return result


b_idsw = parse_plant_idsw("results/baseline_results.txt")
o_idsw = parse_plant_idsw("results/bio_adaptive_seed42_results.txt")
plants = sorted(set(b_idsw) & set(o_idsw))
b_vals = np.array([b_idsw[p] for p in plants])
o_vals = np.array([o_idsw[p] for p in plants])
print(f"Plants: {len(plants)}, B mean={b_vals.mean():.1f}, O mean={o_vals.mean():.1f}")

np.random.seed(42)
fig, ax = plt.subplots(figsize=(7.0, 4.0))
fig.subplots_adjust(left=0.11, right=0.97, top=0.89, bottom=0.14)

jitter = 0.08
ax.scatter(
    np.random.uniform(1 - jitter, 1 + jitter, len(b_vals)),
    b_vals,
    color=C_BASE,
    alpha=0.55,
    s=18,
    edgecolors="none",
    zorder=3,
)
ax.scatter(
    np.random.uniform(2 - jitter, 2 + jitter, len(o_vals)),
    o_vals,
    color=C_OURS,
    alpha=0.65,
    s=18,
    edgecolors="none",
    zorder=3,
)

for vals, pos, col in [(b_vals, 1, C_BASE), (o_vals, 2, C_OURS)]:
    ax.boxplot(
        vals,
        positions=[pos],
        widths=0.28,
        patch_artist=True,
        zorder=4,
        boxprops=dict(facecolor="none", edgecolor=col, linewidth=1.2),
        medianprops=dict(color=col, linewidth=2.0),
        whiskerprops=dict(color=col, linewidth=1.0),
        capprops=dict(color=col, linewidth=1.0),
        flierprops=dict(marker=""),
    )

ax.scatter([1], [b_vals.mean()], marker="D", s=32, color=C_BASE, zorder=5)
ax.scatter([2], [o_vals.mean()], marker="D", s=32, color=C_OURS, zorder=5)

ax.annotate(
    f"−{b_vals.mean() - o_vals.mean():.0f} IDSW\n(−{(b_vals.mean() - o_vals.mean()) / b_vals.mean() * 100:.0f}%)",
    xy=(1.5, (b_vals.mean() + o_vals.mean()) / 2),
    fontsize=9,
    ha="center",
    va="center",
    color="#c0392b",
    fontweight="bold",
    bbox=dict(
        facecolor="white",
        edgecolor="#c0392b",
        linewidth=0.8,
        pad=3,
        boxstyle="round,pad=0.3",
    ),
)

ax.set_xticks([1, 2])
ax.set_xticklabels(["LeafTrackNet\n(baseline)", "Ours\n(bio-constrained)"], fontsize=10)
ax.set_ylabel("Identity switches per plant (IDSW ↓)")
ax.set_xlim(0.5, 2.7)
ax.set_ylim(-1, max(b_vals.max(), o_vals.max()) + 8)
ax.yaxis.set_major_locator(plt.MultipleLocator(5))
ax.grid(axis="y", linestyle=":", linewidth=0.4, color="#d0d0d0", zorder=0)
ax.set_axisbelow(True)
ax.set_title(
    "Per-plant identity switch distribution across all 37 test plants",
    loc="left",
    fontweight="bold",
    pad=6,
    fontsize=9.5,
)

handles = [
    plt.scatter(
        [], [], color=C_BASE, alpha=0.55, s=18, label="LeafTrackNet (per plant)"
    ),
    plt.scatter([], [], color=C_OURS, alpha=0.65, s=18, label="Ours (per plant)"),
    plt.scatter(
        [], [], marker="D", s=32, color=C_BASE, label=f"Mean: {b_vals.mean():.1f}"
    ),
    plt.scatter(
        [], [], marker="D", s=32, color=C_OURS, label=f"Mean: {o_vals.mean():.1f}"
    ),
]
ax.legend(
    handles=handles,
    loc="upper right",
    fontsize=8.5,
    handletextpad=0.4,
    borderpad=0.5,
    labelspacing=0.35,
)

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/error_stats.pdf", bbox_inches="tight")
print("Saved → figures/error_stats.pdf")
