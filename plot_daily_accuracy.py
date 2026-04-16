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
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
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

GT_ROOT = "datasets/CanolaTrack/CanolaTrack/val"
BASELINE = "outputs/baseline/tracks"
OURS = "outputs/bio_adaptive_seed42/tracks"
IOU_THR = 0.5
N_DAYS = 31


def load(path):
    d = {}
    with open(path) as f:
        for line in f:
            p = [x.strip() for x in line.strip().split(",")]
            if len(p) < 6:
                continue
            fr = int(p[0])
            tid = int(p[1])
            x1, y1, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
            d.setdefault(fr, []).append((tid, x1, y1, x1 + w, y1 + h))
    return d


def iou(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ua + ub - inter + 1e-6)


def day_acc(track, gt, frames):
    accs = []
    for fr in frames:
        gd = gt.get(fr, [])
        td = track.get(fr, [])
        if not gd:
            accs.append(np.nan)
            continue
        ok = sum(
            1
            for (gid, *gb) in gd
            if any(iou(gb, tb) >= IOU_THR and tid == gid for (tid, *tb) in td)
        )
        accs.append(ok / len(gd))
    return accs


# collect per-plant per-day accuracy (pad to N_DAYS)
b_mat, o_mat = [], []
for plant in sorted(os.listdir(GT_ROOT)):
    gp = os.path.join(GT_ROOT, plant, "gt", "gt.txt")
    bp = os.path.join(BASELINE, f"{plant}.txt")
    op = os.path.join(OURS, f"{plant}.txt")
    if not all(os.path.exists(p) for p in [gp, bp, op]):
        continue
    gt = load(gp)
    b = load(bp)
    o = load(op)
    frames = sorted(gt.keys())
    ba = day_acc(b, gt, frames)
    oa = day_acc(o, gt, frames)
    # pad
    ba += [np.nan] * (N_DAYS - len(ba))
    oa += [np.nan] * (N_DAYS - len(oa))
    b_mat.append(ba[:N_DAYS])
    o_mat.append(oa[:N_DAYS])

b_arr = np.array(b_mat, dtype=float)
o_arr = np.array(o_mat, dtype=float)
days = np.arange(1, N_DAYS + 1)
b_mean = np.nanmean(b_arr, axis=0)
b_std = np.nanstd(b_arr, axis=0)
o_mean = np.nanmean(o_arr, axis=0)
o_std = np.nanstd(o_arr, axis=0)

print(f"Plants loaded: {len(b_mat)}")
print(f"Baseline mean: {np.nanmean(b_mean):.3f}")
print(f"Ours mean:     {np.nanmean(o_mean):.3f}")

# ── plot ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.0, 3.6))
fig.subplots_adjust(left=0.09, right=0.97, top=0.91, bottom=0.17)

C_B = "#888888"
C_O = "#2166ac"

# std bands
ax.fill_between(
    days,
    np.clip(b_mean - b_std, 0, 1),
    np.clip(b_mean + b_std, 0, 1),
    alpha=0.10,
    color=C_B,
)
ax.fill_between(
    days,
    np.clip(o_mean - o_std, 0, 1),
    np.clip(o_mean + o_std, 0, 1),
    alpha=0.13,
    color=C_O,
)

# mean lines
ax.plot(
    days,
    b_mean,
    color=C_B,
    linewidth=1.4,
    linestyle="--",
    marker="o",
    markersize=3,
    markeredgecolor="white",
    markeredgewidth=0.4,
    label="LeafTrackNet",
    zorder=3,
)
ax.plot(
    days,
    o_mean,
    color=C_O,
    linewidth=1.6,
    marker="o",
    markersize=3,
    markeredgecolor="white",
    markeredgewidth=0.4,
    label="Ours (bio-constrained)",
    zorder=4,
)

ax.set_xlabel("Day")
ax.set_ylabel("Tracking accuracy")
ax.set_title(
    "Average per-day tracking accuracy across all 37 test plants",
    loc="left",
    fontweight="bold",
    pad=6,
)
ax.set_xlim(0.5, N_DAYS + 0.5)
ax.set_ylim(-0.02, 1.12)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
ax.xaxis.set_major_locator(plt.MultipleLocator(5))
ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
ax.grid(axis="y", linestyle=":", linewidth=0.4, color="#d0d0d0")
ax.grid(axis="x", linestyle=":", linewidth=0.3, color="#e8e8e8")

leg = ax.legend(
    loc="lower left",
    fontsize=9.5,
    handlelength=1.8,
    handletextpad=0.5,
    borderpad=0.5,
    labelspacing=0.4,
)
leg.get_frame().set_linewidth(0.5)

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/daily_accuracy.pdf", bbox_inches="tight")
print("Saved → figures/daily_accuracy.pdf")
