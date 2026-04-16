# plot_framework_refined.py  –  improved layout & arrows
import os
from dataclasses import dataclass
from typing import Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# =========================
# Config
# =========================
PLANT = "Plant-177"
FRAME = 25
IMG_DIR = f"datasets/CanolaTrack/CanolaTrack/val/{PLANT}/img"
TRACK_FILE = f"outputs/bio_adaptive_seed42/tracks/{PLANT}.txt"
REAL_COST_MATRIX: Optional[np.ndarray] = None
OUT_DIR = "figures"
OUT_BASENAME = "framework_refined"

TRACK_COLORS = [
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
]


@dataclass(frozen=True)
class Theme:
    gray_fill: str = "#f8fafc"
    gray_edge: str = "#cbd5e1"
    gray_text: str = "#64748b"
    blue_fill: str = "#f0f5ff"
    blue_edge: str = "#2563eb"
    orange_fill: str = "#fff7ed"
    orange_edge: str = "#ea580c"
    text_main: str = "#0f172a"
    text_sub: str = "#6b7280"
    arrow_gray: str = "#94a3b8"
    green_edge: str = "#059669"
    row1_fill: str = "#fef9c3"
    row1_edge: str = "#ca8a04"
    row2_fill: str = "#d1fae5"
    row2_edge: str = "#059669"
    row3_fill: str = "#fee2e2"
    row3_edge: str = "#dc2626"
    row4_fill: str = "#f3e8ff"
    row4_edge: str = "#7c3aed"


TH = Theme()

# =========================
# Layout – all in figure fraction coords
# =========================
FIGSIZE = (15.5, 6.2)

# Image panels (left, bottom, width, height)
PANEL_INPUT = (0.020, 0.25, 0.115, 0.50)
PANEL_DET = (0.155, 0.25, 0.115, 0.50)
PANEL_OUT = (0.870, 0.25, 0.115, 0.50)

# Our Contribution dashed box
CONTRIB = (0.305, 0.12, 0.535, 0.78)

# Inside contribution: cost column (left half), MLP (right half)
# Cost title bar
COST_TITLE = (0.330, 0.72, 0.225, 0.065)
# 4 cost rows
ROW_W = 0.225
ROW_H = 0.062
ROW_X = 0.330
ROW_TOP = 0.635
ROW_GAP = 0.015

# Equation
EQ_Y = 0.265

# Heatmap
HM_RECT = (0.335, 0.30, 0.085, 0.17)

# MLP panel
MLP_RECT = (0.600, 0.17, 0.210, 0.66)

# Memory Bank & Hungarian – bottom row
MEM_BOX = (0.315, 0.035, 0.125, 0.075)
HUNG_BOX = (0.635, 0.035, 0.125, 0.075)


# =========================
# Utilities
# =========================
def load_tracks(path):
    tracks = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [x.strip() for x in line.strip().split(",")]
            if len(parts) < 6:
                continue
            fid = int(parts[0])
            tid = int(parts[1])
            x1, y1, w, h = map(float, parts[2:6])
            tracks.setdefault(fid, []).append((tid, x1, y1, x1 + w, y1 + h))
    return tracks


def crop_center(img, ratio=0.12):
    w, h = img.size
    mx, my = int(w * ratio), int(h * ratio)
    return img.crop((mx, my, w - mx, h - my)), mx, my


def draw_text(ax, x, y, s, fs=8, color=None, bold=False, ha="center", va="center", z=5):
    ax.text(
        x,
        y,
        s,
        fontsize=fs,
        color=color or TH.text_main,
        fontweight="bold" if bold else "normal",
        ha=ha,
        va=va,
        zorder=z,
    )


def draw_round_box(ax, rect, fc, ec, lw=1.0, radius=0.012, ls="-", z=3):
    x, y, w, h = rect
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
        zorder=z,
    )
    ax.add_patch(p)
    return p


def rect_anchor(rect, side="center", t=0.5):
    x, y, w, h = rect
    if side == "left":
        return (x, y + h * t)
    if side == "right":
        return (x + w, y + h * t)
    if side == "top":
        return (x + w * t, y + h)
    if side == "bottom":
        return (x + w * t, y)
    return (x + w * 0.5, y + h * 0.5)


def axes_anchor(ax, side="center", t=0.5):
    bb = ax.get_position()
    return rect_anchor((bb.x0, bb.y0, bb.width, bb.height), side, t)


def draw_arrow(ov, p1, p2, color=None, lw=1.1, ls="-", ms=10, z=2):
    a = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        linestyle=ls,
        color=color or TH.arrow_gray,
        zorder=z,
        connectionstyle="arc3,rad=0.0",
    )
    ov.add_patch(a)


def draw_elbow_arrow(ov, p1, p2, mode="hv", color=None, lw=1.1, ls="-", ms=10, z=2):
    c = color or TH.arrow_gray
    x1, y1 = p1
    x2, y2 = p2
    mid = (x2, y1) if mode == "hv" else (x1, y2)
    ov.plot([x1, mid[0]], [y1, mid[1]], color=c, lw=lw, ls=ls, zorder=z)
    draw_arrow(ov, mid, p2, color=c, lw=lw, ls=ls, ms=ms, z=z)


def draw_polyline_arrow(ov, pts, color=None, lw=1.1, ls="-", ms=10, z=2):
    """Draw a multi-segment polyline ending with an arrowhead."""
    c = color or TH.arrow_gray
    for i in range(len(pts) - 2):
        ov.plot(
            [pts[i][0], pts[i + 1][0]],
            [pts[i][1], pts[i + 1][1]],
            color=c,
            lw=lw,
            ls=ls,
            zorder=z,
        )
    draw_arrow(ov, pts[-2], pts[-1], color=c, lw=lw, ls=ls, ms=ms, z=z)


def make_demo_cost_matrix(n=6, seed=7):
    rng = np.random.default_rng(seed)
    mat = rng.uniform(0.30, 0.90, (n, n))
    np.fill_diagonal(mat, rng.uniform(0.05, 0.18, n))
    return mat


def draw_image_panel(
    fig,
    rect,
    title,
    img_arr,
    dets_draw=None,
    crop_offset=(0, 0),
    crop_size=None,
    color_map=None,
    border="#cbd5e1",
):
    ax = fig.add_axes(rect)
    ax.imshow(img_arr)
    ax.set_xticks([])
    ax.set_yticks([])
    if dets_draw is not None and crop_size is not None:
        mx, my = crop_offset
        cW, cH = crop_size
        for tid, x1, y1, x2, y2 in dets_draw:
            cx1, cy1 = max(x1 - mx, 0), max(y1 - my, 0)
            cx2, cy2 = min(x2 - mx, cW), min(y2 - my, cH)
            if cx2 <= 0 or cy2 <= 0:
                continue
            col = color_map.get(tid, "#fff")
            ax.add_patch(
                mpatches.Rectangle(
                    (cx1, cy1),
                    cx2 - cx1,
                    cy2 - cy1,
                    linewidth=1.45,
                    edgecolor=col,
                    facecolor="none",
                )
            )
            ax.text(
                cx1 + 2,
                max(cy1 - 2, 0),
                str(tid),
                fontsize=5,
                color="white",
                fontweight="bold",
                bbox=dict(facecolor=col, edgecolor="none", pad=0.45, alpha=0.90),
            )
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.9)
        sp.set_edgecolor(border)
    ax.set_title(title, fontsize=8, fontweight="bold", color=TH.text_main, pad=4)
    return ax


def draw_mlp_panel(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="round,pad=0.005,rounding_size=0.03",
            facecolor=TH.orange_fill,
            edgecolor=TH.orange_edge,
            linewidth=1.0,
        )
    )

    ax.text(
        0.50,
        0.94,
        r"Weight Predictor  $\Phi_\theta$",
        ha="center",
        va="center",
        fontsize=8.4,
        fontweight="bold",
        color=TH.text_main,
    )
    ax.text(
        0.50,
        0.85,
        r"input: $\mathbf{f}_{\ell j}\in[0,1]^5$",
        ha="center",
        va="center",
        fontsize=6.8,
        color=TH.text_sub,
    )
    ax.text(
        0.50,
        0.78,
        r"area ratio, $\Delta A/A$, dist., age, $t/T$",
        ha="center",
        va="center",
        fontsize=6.4,
        color=TH.text_sub,
    )

    # MLP nodes
    xs = [0.15, 0.40, 0.65, 0.88]
    shown = [5, 6, 5, 4]
    labels = ["5", "64", "32", "4"]
    node_y = {}
    for i, (x, ns) in enumerate(zip(xs, shown)):
        ys = np.linspace(0.20, 0.64, ns)
        node_y[i] = ys
        for y in ys:
            ax.add_patch(
                plt.Circle(
                    (x, y),
                    0.022,
                    facecolor="#ffedd5",
                    edgecolor=TH.orange_edge,
                    linewidth=0.7,
                    zorder=4,
                )
            )
        if i in (1, 2):
            ax.text(
                x, 0.42, "⋮", fontsize=13, ha="center", va="center", color=TH.text_sub
            )
        ax.text(
            x, 0.11, labels[i], fontsize=7, ha="center", va="center", color=TH.text_sub
        )
    # connections
    for i in range(len(xs) - 1):
        for y1 in node_y[i]:
            chosen = np.linspace(
                0, len(node_y[i + 1]) - 1, min(4, len(node_y[i + 1]))
            ).astype(int)
            for j in chosen:
                ax.plot(
                    [xs[i] + 0.022, xs[i + 1] - 0.022],
                    [y1, node_y[i + 1][j]],
                    color="#fdba74",
                    lw=0.28,
                    alpha=0.45,
                    zorder=2,
                )
    ax.text(0.15, 0.04, "Input", fontsize=6, ha="center", color=TH.text_sub)
    ax.text(
        0.88,
        0.04,
        r"$\boldsymbol{\lambda}_{\ell j}$",
        fontsize=6.8,
        ha="center",
        color=TH.orange_edge,
        fontweight="bold",
    )
    return ax


# =========================
# Main
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    img = Image.open(os.path.join(IMG_DIR, f"{FRAME:08d}.jpg")).convert("RGB")
    img_crop, mx, my = crop_center(img, ratio=0.12)
    cW, cH = img_crop.size

    tracks = load_tracks(TRACK_FILE)
    dets = tracks.get(FRAME, [])
    color_map = {
        tid: TRACK_COLORS[i % len(TRACK_COLORS)]
        for i, (tid, *_) in enumerate(sorted(dets, key=lambda x: x[0]))
    }

    fig = plt.figure(figsize=FIGSIZE)
    ov = fig.add_axes([0, 0, 1, 1])
    ov.set_xlim(0, 1)
    ov.set_ylim(0, 1)
    ov.axis("off")

    # --- Image panels ---
    ax_in = draw_image_panel(
        fig,
        PANEL_INPUT,
        "RGB Image\n$I^t_k$",
        img_crop,
        crop_offset=(mx, my),
        crop_size=(cW, cH),
        color_map=color_map,
        border=TH.gray_edge,
    )
    ax_det = draw_image_panel(
        fig,
        PANEL_DET,
        "Detection +\nEmbedding",
        img_crop,
        dets_draw=dets,
        crop_offset=(mx, my),
        crop_size=(cW, cH),
        color_map=color_map,
        border=TH.gray_edge,
    )
    ax_out = draw_image_panel(
        fig,
        PANEL_OUT,
        "Tracked Leaves\n(Persistent IDs)",
        img_crop,
        dets_draw=dets,
        crop_offset=(mx, my),
        crop_size=(cW, cH),
        color_map=color_map,
        border=TH.green_edge,
    )

    # --- "Inherited" label ---
    draw_text(
        ov,
        0.155,
        0.92,
        "Inherited from LeafTrackNet (unchanged)",
        fs=7.5,
        color=TH.text_sub,
    )

    # --- Our Contribution dashed box ---
    draw_round_box(
        ov,
        CONTRIB,
        fc=TH.blue_fill,
        ec=TH.blue_edge,
        lw=1.3,
        radius=0.010,
        ls="--",
        z=1,
    )
    draw_text(
        ov,
        CONTRIB[0] + CONTRIB[2] / 2,
        CONTRIB[1] + CONTRIB[3] + 0.025,
        "Our Contribution",
        fs=10,
        color=TH.blue_edge,
        bold=True,
    )

    # --- Cost Matrix title ---
    draw_round_box(ov, COST_TITLE, fc="#dbeafe", ec=TH.blue_edge, lw=1.1, radius=0.008)
    draw_text(
        ov,
        COST_TITLE[0] + COST_TITLE[2] / 2,
        COST_TITLE[1] + COST_TITLE[3] / 2,
        r"Biology-Constrained Cost Matrix  $\mathbf{C}$",
        fs=8,
        bold=True,
    )

    # --- 4 cost rows ---
    rows_data = [
        (
            r"$C^{\rm app}_{\ell j}$",
            "Appearance similarity",
            TH.row1_fill,
            TH.row1_edge,
        ),
        (r"$C^{\rm pos}_{\ell j}$", "Spatial continuity", TH.row2_fill, TH.row2_edge),
        (r"$C^{\rm area}_{\ell j}$", "Area growth", TH.row3_fill, TH.row3_edge),
        (
            r"$C^{\rm life}_{\ell j}$",
            "Lifecycle plausibility",
            TH.row4_fill,
            TH.row4_edge,
        ),
    ]
    row_rects = []
    for i, (sym, desc, fc, ec) in enumerate(rows_data):
        ry = ROW_TOP - i * (ROW_H + ROW_GAP)
        r = (ROW_X, ry, ROW_W, ROW_H)
        row_rects.append(r)
        draw_round_box(ov, r, fc=fc, ec=ec, lw=0.9, radius=0.008)
        draw_text(ov, r[0] + 0.058, r[1] + r[3] / 2, sym, fs=8)
        draw_text(ov, r[0] + 0.148, r[1] + r[3] / 2, desc, fs=7.6)

    # --- Equation ---
    last_row = row_rects[-1]
    eq_y = last_row[1] - 0.04
    draw_text(
        ov,
        ROW_X + ROW_W / 2,
        eq_y,
        r"$C_{\ell j}=\boldsymbol{\lambda}_{\ell j}\cdot"
        r"[C^{\rm app},C^{\rm pos},C^{\rm area},C^{\rm life}]^\top$",
        fs=7.7,
        color=TH.blue_edge,
    )

    # --- Heatmap ---
    hm_rect = [HM_RECT[0], HM_RECT[1], HM_RECT[2], HM_RECT[3]]
    ax_hm = fig.add_axes(hm_rect)
    cost_mat = (
        REAL_COST_MATRIX
        if REAL_COST_MATRIX is not None
        else make_demo_cost_matrix(6, 7)
    )
    ax_hm.imshow(cost_mat, cmap="Blues_r", vmin=0, vmax=1, aspect="auto")
    ax_hm.set_xticks([])
    ax_hm.set_yticks([])
    for sp in ax_hm.spines.values():
        sp.set_edgecolor(TH.blue_edge)
        sp.set_linewidth(0.9)
    for i in range(min(cost_mat.shape)):
        ax_hm.add_patch(
            plt.Rectangle(
                (i - 0.5, i - 0.5),
                1,
                1,
                fill=False,
                edgecolor=TH.green_edge,
                linewidth=0.9,
            )
        )
    ax_hm.set_xlabel("Detections", fontsize=5.5, color=TH.text_sub, labelpad=1)
    ax_hm.set_ylabel("Tracks", fontsize=5.5, color=TH.text_sub, labelpad=1)
    ax_hm.set_title(
        "Cost matrix\n(green = low cost)", fontsize=5.5, color=TH.text_sub, pad=2
    )

    # --- MLP panel ---
    ax_mlp = draw_mlp_panel(fig, list(MLP_RECT))

    # --- Memory Bank ---
    draw_round_box(ov, MEM_BOX, fc=TH.gray_fill, ec=TH.gray_edge, lw=1.0, radius=0.008)
    draw_text(
        ov,
        MEM_BOX[0] + MEM_BOX[2] / 2,
        MEM_BOX[1] + MEM_BOX[3] * 0.67,
        "Memory Bank",
        fs=7.5,
        bold=True,
    )
    draw_text(
        ov,
        MEM_BOX[0] + MEM_BOX[2] / 2,
        MEM_BOX[1] + MEM_BOX[3] * 0.28,
        r"$\mathcal{T}^{t-1}$  (track prototypes)",
        fs=6.5,
        color=TH.text_sub,
    )

    # --- Hungarian ---
    draw_round_box(ov, HUNG_BOX, fc=TH.gray_fill, ec=TH.gray_edge, lw=1.0, radius=0.008)
    draw_text(
        ov,
        HUNG_BOX[0] + HUNG_BOX[2] / 2,
        HUNG_BOX[1] + HUNG_BOX[3] * 0.67,
        "Hungarian",
        fs=7.5,
        bold=True,
    )
    draw_text(
        ov,
        HUNG_BOX[0] + HUNG_BOX[2] / 2,
        HUNG_BOX[1] + HUNG_BOX[3] * 0.28,
        "Assignment",
        fs=6.8,
        color=TH.text_sub,
    )

    # ==================================================
    # ARROWS – clean routing
    # ==================================================

    # 1) Input → Detection
    draw_arrow(
        ov,
        axes_anchor(ax_in, "right"),
        axes_anchor(ax_det, "left"),
        color=TH.arrow_gray,
    )

    # 2) Detection → Cost title (elbow: right then down)
    det_exit = axes_anchor(ax_det, "right", 0.55)
    cost_entry = rect_anchor(COST_TITLE, "left", 0.5)
    draw_arrow(ov, det_exit, cost_entry, color=TH.arrow_gray)

    # 3) λ arrow: MLP → Cost title (dashed, from left side of MLP to right side of cost title)
    mlp_left = (MLP_RECT[0], COST_TITLE[1] + COST_TITLE[3] * 0.5)
    cost_right = rect_anchor(COST_TITLE, "right", 0.5)
    draw_arrow(ov, mlp_left, cost_right, color=TH.orange_edge, lw=1.3, ls="--")
    # λ label above the arrow
    mid_x = (mlp_left[0] + cost_right[0]) / 2
    mid_y = (mlp_left[1] + cost_right[1]) / 2 + 0.025
    draw_text(
        ov,
        mid_x,
        mid_y,
        r"$\boldsymbol{\lambda}_{\ell j}$",
        fs=9,
        color=TH.orange_edge,
        bold=True,
    )

    # 4) Cost matrix → Hungarian (vertical down from bottom of heatmap region to Hungarian top)
    cost_bottom = (ROW_X + ROW_W * 0.4, eq_y - 0.025)
    hung_top = rect_anchor(HUNG_BOX, "top", 0.35)
    draw_elbow_arrow(ov, cost_bottom, hung_top, mode="vh", color=TH.arrow_gray)

    # 5) Memory Bank → Hungarian
    draw_arrow(
        ov,
        rect_anchor(MEM_BOX, "right"),
        rect_anchor(HUNG_BOX, "left"),
        color=TH.arrow_gray,
    )
    # label
    draw_text(
        ov,
        (MEM_BOX[0] + MEM_BOX[2] + HUNG_BOX[0]) / 2,
        MEM_BOX[1] + MEM_BOX[3] / 2 + 0.028,
        r"track prototypes $\mathbf{p}^\ell$",
        fs=6.5,
        color=TH.text_sub,
    )

    # 6) Hungarian → Output
    draw_arrow(
        ov,
        rect_anchor(HUNG_BOX, "right"),
        axes_anchor(ax_out, "left", 0.35),
        color=TH.arrow_gray,
    )

    # 7) Update memory: Output bottom → bend left → Memory Bank bottom (dashed feedback)
    out_bot = axes_anchor(ax_out, "bottom", 0.5)
    mem_bot = rect_anchor(MEM_BOX, "bottom", 0.5)
    feedback_y = 0.005
    pts = [out_bot, (out_bot[0], feedback_y), (mem_bot[0], feedback_y), mem_bot]
    draw_polyline_arrow(ov, pts, color=TH.arrow_gray, lw=0.9, ls="--", ms=8)
    draw_text(
        ov,
        (out_bot[0] + mem_bot[0]) / 2,
        feedback_y + 0.018,
        "update memory",
        fs=6.5,
        color=TH.text_sub,
    )

    # --- Save ---
    pdf_path = os.path.join(OUT_DIR, f"{OUT_BASENAME}.pdf")
    png_path = os.path.join(OUT_DIR, f"{OUT_BASENAME}.png")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, bbox_inches="tight", facecolor="white", dpi=600)
    plt.close(fig)
    print(f"Saved -> {pdf_path}")
    print(f"Saved -> {png_path}")


if __name__ == "__main__":
    main()
