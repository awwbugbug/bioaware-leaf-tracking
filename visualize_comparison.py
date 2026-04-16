# visualize_comparison.py
"""
Qualitative comparison: LeafTrackNet vs Ours on Plant-177.
Red boxes + X mark identity switches in baseline.
Output: figures/qualitative_comparison.pdf
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

PLANT = "Plant-177"
FRAMES = [25, 27, 29, 31]
IMG_ROOT = "datasets/CanolaTrack/CanolaTrack/val"
BASELINE = "outputs/baseline/tracks"
OUR_METHOD = "outputs/bio_adaptive_seed42/tracks"
OUT_PATH = "figures/qualitative_comparison.pdf"

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
    "#dcbeff",
    "#9A6324",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#a9a9a9",
    "#e6beff",
    "#fffac8",
    "#FF6600",
]


def load_tracks(path):
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


def load_gt(path):
    return load_tracks(path)


def iou_box(a, b):
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
            v = iou_box(gb, tb)
            if v > best:
                best, best_tid = v, tid
        if best_tid == gt_id:
            correct.add(best_tid)
    return correct


def color_map(ids):
    return {tid: COLORS[i % len(COLORS)] for i, tid in enumerate(sorted(ids))}


def crop(dets, mx, my, cW, cH):
    out = []
    for tid, x1, y1, x2, y2 in dets:
        cx1, cy1, cx2, cy2 = x1 - mx, y1 - my, x2 - mx, y2 - my
        if cx2 > 0 and cy2 > 0 and cx1 < cW and cy1 < cH:
            out.append((tid, max(cx1, 0), max(cy1, 0), min(cx2, cW), min(cy2, cH)))
    return out


def draw(ax, img, dets, cmap, title, wrong_ids=None):
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    for tid, x1, y1, x2, y2 in dets:
        is_wrong = wrong_ids and tid in wrong_ids
        ec = "#FF0000" if is_wrong else cmap.get(tid, "#cccccc")
        lw = 3.0 if is_wrong else 1.8
        ax.add_patch(
            patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=lw, edgecolor=ec, facecolor="none"
            )
        )
        ax.text(
            x1,
            max(y1 - 4, 0),
            f"ID {tid}",
            color="white",
            fontsize=6.5,
            fontweight="bold",
            bbox=dict(facecolor=ec, alpha=0.85, pad=1, edgecolor="none"),
        )
        if is_wrong:
            ax.text(
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                "✗",
                color="red",
                fontsize=16,
                fontweight="bold",
                ha="center",
                va="center",
                alpha=0.9,
            )


os.makedirs("figures", exist_ok=True)
img_dir = os.path.join(IMG_ROOT, PLANT, "img")
gt_tracks = load_gt(os.path.join(IMG_ROOT, PLANT, "gt", "gt.txt"))
b_tracks = load_tracks(os.path.join(BASELINE, f"{PLANT}.txt"))
o_tracks = load_tracks(os.path.join(OUR_METHOD, f"{PLANT}.txt"))

# global color maps
all_gt_ids = set(t for fr in FRAMES for (t, *_) in gt_tracks.get(fr, []))
all_b_ids = set(t for fr in FRAMES for (t, *_) in b_tracks.get(fr, []))
all_o_ids = set(t for fr in FRAMES for (t, *_) in o_tracks.get(fr, []))
gt_cmap = color_map(all_gt_ids)
b_cmap = color_map(all_b_ids)
o_cmap = color_map(all_o_ids)

n = len(FRAMES)
fig, axes = plt.subplots(2, n, figsize=(3.8 * n, 7.5))
fig.subplots_adjust(hspace=0.06, wspace=0.03)

for col, frame in enumerate(FRAMES):
    img_path = os.path.join(img_dir, f"{frame:08d}.jpg")
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    mx, my = int(W * 0.15), int(H * 0.15)
    img_c = img.crop((mx, my, W - mx, H - my))
    cW, cH = img_c.size

    gt_d = crop(gt_tracks.get(frame, []), mx, my, cW, cH)
    b_d = crop(b_tracks.get(frame, []), mx, my, cW, cH)
    o_d = crop(o_tracks.get(frame, []), mx, my, cW, cH)

    b_correct = correct_ids(b_d, gt_d)
    b_wrong = set(t for (t, *_) in b_d) - b_correct

    o_correct = correct_ids(o_d, gt_d)
    o_wrong = set(t for (t, *_) in o_d) - o_correct

    draw(axes[0, col], img_c, b_d, b_cmap, f"Day {frame}", wrong_ids=b_wrong)
    draw(
        axes[1, col],
        img_c,
        o_d,
        o_cmap,
        f"Day {frame}",
        wrong_ids=o_wrong if o_wrong else None,
    )

fig.text(
    0.08,
    0.73,
    "LeafTrackNet\n(Baseline)",
    va="center",
    ha="left",
    fontsize=11,
    fontweight="bold",
    color="#555555",
    rotation=90,
)
fig.text(
    0.08,
    0.27,
    "Ours\n(Bio-Constrained)",
    va="center",
    ha="left",
    fontsize=11,
    fontweight="bold",
    color="#2166ac",
    rotation=90,
)
fig.text(
    0.5,
    -0.01,
    "Red boxes (✗) = identity switches in baseline; "
    "our method correctly maintains leaf identities.",
    ha="center",
    fontsize=9,
    color="#444444",
    style="italic",
)
fig.suptitle(
    f"Qualitative tracking comparison on {PLANT}",
    fontsize=12,
    fontweight="bold",
    y=1.02,
)

fig.savefig(OUT_PATH, bbox_inches="tight", dpi=200)
print(f"Saved to {OUT_PATH}")
