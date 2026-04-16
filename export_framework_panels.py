# export_framework_panels.py
"""
Export clean image panels for framework figure in draw.io:
1. figures/panel_input.png       — raw plant image (no boxes)
2. figures/panel_baseline.png    — baseline with red wrong boxes
3. figures/panel_ours.png        — ours with correct colored boxes

Plant-177, Frame 27 (baseline has 6 wrong IDs).
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image

PLANT = "Plant-177"
FRAME = 27
IMG_DIR = f"datasets/CanolaTrack/CanolaTrack/val/{PLANT}/img"
GT_FILE = f"datasets/CanolaTrack/CanolaTrack/val/{PLANT}/gt/gt.txt"
BASE_FILE = f"outputs/baseline/tracks/{PLANT}.txt"
OUR_FILE = f"outputs/bio_adaptive_seed42/tracks/{PLANT}.txt"

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
]


def load_tracks(path):
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


def correct_ids(track_dets, gt_dets):
    correct = set()
    for gt_id, *gb in gt_dets:
        best, best_tid = 0.3, None
        for tid, *tb in track_dets:
            v = iou(gb, tb)
            if v > best:
                best, best_tid = v, tid
        if best_tid == gt_id:
            correct.add(best_tid)
    return correct


def save_panel(
    img_crop,
    dets,
    cmap,
    wrong_ids,
    title,
    out_path,
    show_title=True,
    lw_correct=2.0,
    lw_wrong=2.8,
):
    """Save a single panel as PNG."""
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.imshow(img_crop)
    ax.axis("off")

    for tid, x1, y1, x2, y2 in dets:
        is_wrong = tid in wrong_ids
        ec = "#FF0000" if is_wrong else cmap.get(tid, "#ffffff")
        lw = lw_wrong if is_wrong else lw_correct
        ax.add_patch(
            mpatches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=lw,
                edgecolor=ec,
                facecolor="none",
                zorder=3,
            )
        )
        ax.text(
            x1 + 2,
            max(y1 - 4, 0),
            f"ID {tid}",
            fontsize=7,
            color="white",
            fontweight="bold",
            bbox=dict(facecolor=ec, pad=1, edgecolor="none", alpha=0.88),
            zorder=4,
        )
        if is_wrong:
            ax.text(
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                "✗",
                color="red",
                fontsize=18,
                fontweight="bold",
                ha="center",
                va="center",
                alpha=0.9,
                zorder=5,
            )

    if show_title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=4, color="#1e293b")

    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor("#FF0000" if wrong_ids else "#059669")
        sp.set_linewidth(2.0)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── Load ──────────────────────────────────────────────────────
os.makedirs("figures", exist_ok=True)

img = Image.open(os.path.join(IMG_DIR, f"{FRAME:08d}.jpg")).convert("RGB")
W, H = img.size
mx, my = int(W * 0.10), int(H * 0.10)
img_c = img.crop((mx, my, W - mx, H - my))
cW, cH = img_c.size

gt = load_tracks(GT_FILE)
b_tr = load_tracks(BASE_FILE)
o_tr = load_tracks(OUR_FILE)

gt_d = [
    (tid, max(x1 - mx, 0), max(y1 - my, 0), min(x2 - mx, cW), min(y2 - my, cH))
    for (tid, x1, y1, x2, y2) in gt.get(FRAME, [])
    if x2 - mx > 0 and y2 - my > 0
]

b_d = [
    (tid, max(x1 - mx, 0), max(y1 - my, 0), min(x2 - mx, cW), min(y2 - my, cH))
    for (tid, x1, y1, x2, y2) in b_tr.get(FRAME, [])
    if x2 - mx > 0 and y2 - my > 0
]

o_d = [
    (tid, max(x1 - mx, 0), max(y1 - my, 0), min(x2 - mx, cW), min(y2 - my, cH))
    for (tid, x1, y1, x2, y2) in o_tr.get(FRAME, [])
    if x2 - mx > 0 and y2 - my > 0
]

# colour maps
all_ids = sorted(set(tid for (tid, *_) in gt_d))
gt_cmap = {tid: COLORS[i % len(COLORS)] for i, tid in enumerate(all_ids)}

b_all = sorted(set(tid for (tid, *_) in b_d))
b_cmap = {tid: COLORS[i % len(COLORS)] for i, tid in enumerate(b_all)}

o_all = sorted(set(tid for (tid, *_) in o_d))
o_cmap = {tid: COLORS[i % len(COLORS)] for i, tid in enumerate(o_all)}

b_correct = correct_ids(b_d, gt_d)
b_wrong = set(tid for (tid, *_) in b_d) - b_correct

o_correct = correct_ids(o_d, gt_d)
o_wrong = set(tid for (tid, *_) in o_d) - o_correct

# ── 1. Raw input (no boxes) ───────────────────────────────────
fig, ax = plt.subplots(figsize=(3.2, 3.2))
fig.subplots_adjust(0, 0, 1, 1)
ax.imshow(img_c)
ax.axis("off")
for sp in ax.spines.values():
    sp.set_visible(True)
    sp.set_edgecolor("#aaaaaa")
    sp.set_linewidth(1.5)
fig.savefig("figures/panel_input.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved → figures/panel_input.png")

# ── 2. Baseline (red wrong boxes) ────────────────────────────
save_panel(
    img_c,
    b_d,
    b_cmap,
    b_wrong,
    f"LeafTrackNet — Day {FRAME}",
    "figures/panel_baseline.png",
)

# ── 3. Ours (all correct) ─────────────────────────────────────
save_panel(img_c, o_d, o_cmap, o_wrong, f"Ours — Day {FRAME}", "figures/panel_ours.png")

print(f"\nBaseline wrong IDs: {b_wrong}")
print(f"Ours    wrong IDs: {o_wrong}")
print("\nDownload all three PNGs from figures/ directory.")
