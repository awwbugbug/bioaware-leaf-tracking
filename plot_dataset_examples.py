# plot_dataset_examples.py
"""
Dataset overview figure.
Shows 3 plants x 4 time points (Day 1, 10, 20, 31)
with GT bounding boxes and leaf IDs.
Highlights key challenges: occlusion, growth, new leaves.
Output: figures/dataset_examples.pdf
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import rcParams
from PIL import Image

rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
GT_ROOT = "datasets/CanolaTrack/CanolaTrack/val"
DAYS = [1, 10, 20, 31]
PLANTS = ["Plant-153", "Plant-158", "Plant-177"]  # diverse examples
OUT_PATH = "figures/dataset_examples.pdf"

COLORS = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#469990",
    "#9A6324",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#a9a9a9",
    "#dcbeff",
    "#fffac8",
    "#FF6600",
    "#000000",
]


# 专业美观的配色（蓝、绿、紫，带灰度）
PLANT_LABELS = {
    "Plant-153": "Plant A\n(dense canopy)",
    "Plant-158": "Plant B\n(rotation)",
    "Plant-177": "Plant C\n(severe occlusion)",
}
PLANT_LABEL_COLORS = {
    # 加入灰度（RGB混合灰色）
    "Plant-153": (80 / 255, 110 / 255, 150 / 255, 0.72),  # 蓝灰
    "Plant-158": (90 / 255, 120 / 255, 90 / 255, 0.72),  # 绿灰
    "Plant-177": (110 / 255, 90 / 255, 120 / 255, 0.72),  # 紫灰
}


def load_gt(path):
    data = {}
    with open(path) as f:
        for line in f:
            p = [x.strip() for x in line.strip().split(",")]
            if len(p) < 6:
                continue
            fr = int(p[0])
            tid = int(p[1])
            x1, y1, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
            data.setdefault(fr, []).append((tid, x1, y1, x1 + w, y1 + h))
    return data


def color_map(gt):
    all_ids = sorted(set(tid for dets in gt.values() for (tid, *_) in dets))
    return {tid: COLORS[i % len(COLORS)] for i, tid in enumerate(all_ids)}


# ---------------------------------------------------------------
# Figure: 3 rows (plants) x 4 cols (days)
# ---------------------------------------------------------------
n_row, n_col = len(PLANTS), len(DAYS)
fig, axes = plt.subplots(n_row, n_col, figsize=(3.2 * n_col, 3.0 * n_row))
fig.subplots_adjust(
    hspace=0.06, wspace=0.03, left=0.19, right=0.99, top=0.90, bottom=0.06
)

for row, plant in enumerate(PLANTS):
    img_dir = os.path.join(GT_ROOT, plant, "img")
    gt_path = os.path.join(GT_ROOT, plant, "gt", "gt.txt")
    gt = load_gt(gt_path)
    cmap = color_map(gt)
    frames = sorted(gt.keys())

    for col, day in enumerate(DAYS):
        ax = axes[row, col]
        ax.axis("off")

        # find closest available frame to target day
        target_frame = min(frames, key=lambda f: abs(f - day))
        img_path = os.path.join(img_dir, f"{target_frame:08d}.jpg")
        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        # crop centre (remove black/blue border)
        mx, my = int(W * 0.10), int(H * 0.10)
        img_c = img.crop((mx, my, W - mx, H - my))
        cW, cH = img_c.size

        ax.imshow(img_c)

        # draw GT boxes
        dets = gt.get(target_frame, [])
        for tid, x1, y1, x2, y2 in dets:
            cx1 = max(x1 - mx, 0)
            cy1 = max(y1 - my, 0)
            cx2 = min(x2 - mx, cW)
            cy2 = min(y2 - my, cH)
            if cx2 <= 0 or cy2 <= 0 or cx1 >= cW or cy1 >= cH:
                continue
            color = cmap[tid]
            rect = patches.Rectangle(
                (cx1, cy1),
                cx2 - cx1,
                cy2 - cy1,
                linewidth=1.6,
                edgecolor=color,
                facecolor="none",
                zorder=3,
            )
            ax.add_patch(rect)
            ax.text(
                cx1 + 2,
                max(cy1 - 3, 0),
                f"{tid}",
                color="white",
                fontsize=6.5,
                fontweight="bold",
                bbox=dict(facecolor=color, alpha=0.82, pad=0.8, edgecolor="none"),
                zorder=4,
            )

        # column header (day label) — top row only
        if row == 0:
            ax.set_title(f"Day {day}", fontsize=10, fontweight="bold", pad=4)

    # row label on the left，美化字体、加灰、半透明
    axes[row, 0].text(
        -0.22,
        0.5,
        PLANT_LABELS[plant],
        transform=axes[row, 0].transAxes,
        fontsize=11,
        va="center",
        ha="center",
        rotation=90,
        color=PLANT_LABEL_COLORS[plant],
        fontweight="semibold",
        fontname="DejaVu Sans",
        multialignment="center",
        alpha=0.72,
    )


# overall title，减小与图片间距并右移
fig.suptitle(
    "CanolaTrack validation examples — bounding boxes with persistent leaf IDs",
    fontsize=12,
    fontweight="bold",
    y=0.97,
    x=0.58,
    ha="center",
)

os.makedirs("figures", exist_ok=True)
fig.savefig(OUT_PATH, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
