import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams

rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

labels = ["App.\nonly", "+\u202fPos.", "+\u202fArea", "+\u202fLife"]
x = np.arange(len(labels))
hota = np.array([88.58, 88.85, 89.00, 89.04])
assa = np.array([85.17, 85.64, 85.89, 85.98])
idf1 = np.array([93.37, 93.43, 93.70, 93.76])
idsw = np.array([214, 163, 147, 147])

C_HOTA = "#1f77b4"
C_ASSA = "#2ca02c"
C_IDF1 = "#d62728"
C_IDSW = "#7b2d8b"
MK = dict(
    marker="o",
    markersize=5,
    markeredgewidth=0.5,
    markeredgecolor="white",
    linewidth=1.5,
    clip_on=False,
)

# 更大的图幅，底部留空给 caption-style legend
fig = plt.figure(figsize=(10.0, 3.8))
gs = fig.add_gridspec(
    1, 2, wspace=0.34, left=0.07, right=0.96, top=0.92, bottom=0.30
)  # 底部留大空间
ax_s = fig.add_subplot(gs[0])
ax_i = fig.add_subplot(gs[1])

# ── (a) accuracy ──────────────────────────────────────────────
ax_s.plot(x, hota, color=C_HOTA, **MK)
ax_s.plot(x, assa, color=C_ASSA, **MK)
ax_s.plot(x, idf1, color=C_IDF1, **MK)

# 末端数值标注
for val, col, dy in zip(
    [hota[-1], assa[-1], idf1[-1]], [C_HOTA, C_ASSA, C_IDF1], [0, -4, 2]
):
    ax_s.annotate(
        f"{val:.2f}",
        xy=(x[-1], val),
        xytext=(6, dy),
        textcoords="offset points",
        color=col,
        fontsize=9,
        va="center",
        fontweight="bold",
    )

ax_s.set_xticks(x)
ax_s.set_xticklabels(labels)
ax_s.set_ylabel("Score")
ax_s.set_title("(a) Tracking accuracy metrics", loc="left", fontweight="bold", pad=6)
ax_s.set_ylim(84.0, 95.8)
ax_s.yaxis.set_major_locator(ticker.MultipleLocator(2))
ax_s.grid(axis="y", linestyle=":", linewidth=0.4, color="#d0d0d0")
ax_s.set_xlabel("Cumulative constraints")

# ── (b) IDSW ──────────────────────────────────────────────────
ax_i.bar(x, idsw, width=0.45, color=C_IDSW, alpha=0.12, edgecolor=C_IDSW, linewidth=0.7)
ax_i.plot(x, idsw, color=C_IDSW, **MK)

for xi, val in zip(x, idsw):
    ax_i.annotate(
        str(val),
        xy=(xi, val),
        xytext=(0, 7),
        textcoords="offset points",
        color=C_IDSW,
        fontsize=9.5,
        ha="center",
        fontweight="bold",
    )

# 降幅标注 — 右上角文字，不用箭头压线
ax_i.text(
    3.0,
    228,
    f"\u2212{idsw[0] - idsw[-1]} IDSW",
    ha="right",
    va="top",
    fontsize=9,
    color="#666666",
    style="italic",
    bbox=dict(facecolor="white", edgecolor="none", pad=1),
)

ax_i.set_xticks(x)
ax_i.set_xticklabels(labels)
ax_i.set_ylabel("Identity switches \u2193")
ax_i.set_title("(b) Identity switches", loc="left", fontweight="bold", pad=6)
ax_i.set_ylim(128, 245)
ax_i.yaxis.set_major_locator(ticker.MultipleLocator(20))
ax_i.grid(axis="y", linestyle=":", linewidth=0.4, color="#d0d0d0")
ax_i.set_xlabel("Cumulative constraints")

# ── 底部统一 legend（图下方，三列横排）──────────────────────────
handles = [
    plt.Line2D(
        [0],
        [0],
        color=C_HOTA,
        marker="o",
        markersize=5,
        linewidth=1.5,
        markeredgecolor="white",
        label="HOTA",
    ),
    plt.Line2D(
        [0],
        [0],
        color=C_ASSA,
        marker="o",
        markersize=5,
        linewidth=1.5,
        markeredgecolor="white",
        label="AssA",
    ),
    plt.Line2D(
        [0],
        [0],
        color=C_IDF1,
        marker="o",
        markersize=5,
        linewidth=1.5,
        markeredgecolor="white",
        label="IDF1",
    ),
]
fig.legend(
    handles=handles,
    loc="lower center",
    ncol=3,
    fontsize=10,
    bbox_to_anchor=(0.27, 0.04),  # 左图正下方
    handlelength=1.6,
    handletextpad=0.5,
    columnspacing=1.2,
)

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/ablation_constraints.pdf", bbox_inches="tight")
print("Saved → figures/ablation_constraints.pdf")
