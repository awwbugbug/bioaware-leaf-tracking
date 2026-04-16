# analyze_dataset.py
"""
Statistical analysis of CanolaTrack val set.
Computes per-plant and overall statistics to characterise
dataset difficulty and motivate biology-constrained tracking.

Outputs:
  - Per-plant: leaf count per frame, occlusion proxy, ID count
  - Overall: distributions of leaf counts, area change rates,
             spatial displacement between consecutive frames
  - Saves results to results/dataset_analysis.txt
"""

import os
from collections import defaultdict

import numpy as np

VAL_ROOT = "datasets/CanolaTrack/CanolaTrack/val"
OUT_FILE = "results/dataset_analysis.txt"


def load_gt(gt_path):
    """Load gt.txt -> {frame: {leaf_id: [x1,y1,x2,y2]}}"""
    data = defaultdict(dict)
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame = int(parts[0])
            leaf_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            data[frame][leaf_id] = [x, y, x + w, y + h]
    return data


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_centroid(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def iou(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = box_area(a)
    ub = box_area(b)
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def analyze():
    plants = sorted(
        [d for d in os.listdir(VAL_ROOT) if os.path.isdir(os.path.join(VAL_ROOT, d))]
    )

    # global accumulators
    all_leaf_counts = []  # leaves per frame
    all_area_changes = []  # |ΔA/A| between consecutive frames, same leaf
    all_displacements = []  # centroid displacement (px), same leaf
    all_occlusion_rates = []  # per-plant: fraction of frames with IoU>0.1 pairs
    all_id_counts = []  # unique leaf IDs per plant
    all_lifespans = []  # frames each leaf is visible

    per_plant_rows = []

    for plant in plants:
        gt_path = os.path.join(VAL_ROOT, plant, "gt", "gt.txt")
        if not os.path.exists(gt_path):
            continue
        gt = load_gt(gt_path)
        frames = sorted(gt.keys())
        T = len(frames)

        # --- leaf counts per frame ---
        counts = [len(gt[f]) for f in frames]

        # --- unique leaf IDs ---
        all_ids = set()
        for f in frames:
            all_ids.update(gt[f].keys())
        n_ids = len(all_ids)

        # --- leaf lifespans ---
        leaf_frames = defaultdict(list)
        for f in frames:
            for lid in gt[f]:
                leaf_frames[lid].append(f)
        lifespans = [len(v) for v in leaf_frames.values()]

        # --- area change and displacement between consecutive frames ---
        plant_area_changes = []
        plant_displacements = []
        for i in range(T - 1):
            f0, f1 = frames[i], frames[i + 1]
            for lid in gt[f0]:
                if lid in gt[f1]:
                    a0 = box_area(gt[f0][lid])
                    a1 = box_area(gt[f1][lid])
                    if a0 > 0:
                        plant_area_changes.append(abs(a1 - a0) / a0)
                    c0 = box_centroid(gt[f0][lid])
                    c1 = box_centroid(gt[f1][lid])
                    disp = np.sqrt((c0[0] - c1[0]) ** 2 + (c0[1] - c1[1]) ** 2)
                    plant_displacements.append(disp)

        # --- occlusion proxy: fraction of frames with at least one IoU>0.1 pair ---
        occluded_frames = 0
        for f in frames:
            boxes = list(gt[f].values())
            found = False
            for ii in range(len(boxes)):
                for jj in range(ii + 1, len(boxes)):
                    if iou(boxes[ii], boxes[jj]) > 0.1:
                        found = True
                        break
                if found:
                    break
            if found:
                occluded_frames += 1
        occ_rate = occluded_frames / T if T > 0 else 0.0

        # accumulate
        all_leaf_counts.extend(counts)
        all_area_changes.extend(plant_area_changes)
        all_displacements.extend(plant_displacements)
        all_occlusion_rates.append(occ_rate)
        all_id_counts.append(n_ids)
        all_lifespans.extend(lifespans)

        per_plant_rows.append(
            {
                "plant": plant,
                "T": T,
                "n_ids": n_ids,
                "mean_count": np.mean(counts),
                "max_count": max(counts),
                "occ_rate": occ_rate,
                "mean_lifespan": np.mean(lifespans),
            }
        )

    # ----------------------------------------------------------------
    # Print and save results
    # ----------------------------------------------------------------
    lines = []

    lines.append("=" * 60)
    lines.append("CanolaTrack Val Set — Dataset Analysis")
    lines.append("=" * 60)

    lines.append(f"\nNumber of plants:          {len(plants)}")
    lines.append(f"Total annotated frames:    {sum(r['T'] for r in per_plant_rows)}")
    lines.append(f"Total leaf instances:      {sum(all_leaf_counts)}")

    lines.append("\n--- Leaf count per frame ---")
    lines.append(f"  Mean:   {np.mean(all_leaf_counts):.2f}")
    lines.append(f"  Std:    {np.std(all_leaf_counts):.2f}")
    lines.append(f"  Min:    {np.min(all_leaf_counts)}")
    lines.append(f"  Max:    {np.max(all_leaf_counts)}")

    lines.append("\n--- Unique leaf IDs per plant ---")
    lines.append(f"  Mean:   {np.mean(all_id_counts):.2f}")
    lines.append(f"  Std:    {np.std(all_id_counts):.2f}")
    lines.append(f"  Min:    {np.min(all_id_counts)}")
    lines.append(f"  Max:    {np.max(all_id_counts)}")

    lines.append("\n--- Leaf lifespan (frames visible) ---")
    lines.append(f"  Mean:   {np.mean(all_lifespans):.2f}")
    lines.append(f"  Std:    {np.std(all_lifespans):.2f}")
    lines.append(f"  Min:    {np.min(all_lifespans)}")
    lines.append(f"  Max:    {np.max(all_lifespans)}")
    lines.append(
        f"  % leaves visible all 31 days: "
        f"{100 * sum(1 for l in all_lifespans if l == 31) / len(all_lifespans):.1f}%"
    )

    lines.append("\n--- Inter-frame area change rate |ΔA/A| ---")
    lines.append(f"  Mean:   {np.mean(all_area_changes):.4f}")
    lines.append(f"  Std:    {np.std(all_area_changes):.4f}")
    lines.append(f"  Median: {np.median(all_area_changes):.4f}")
    lines.append(
        f"  % pairs with >30% area change: "
        f"{100 * sum(1 for x in all_area_changes if x > 0.3) / len(all_area_changes):.1f}%"
    )

    lines.append("\n--- Inter-frame centroid displacement (px) ---")
    lines.append(f"  Mean:   {np.mean(all_displacements):.2f}")
    lines.append(f"  Std:    {np.std(all_displacements):.2f}")
    lines.append(f"  Median: {np.median(all_displacements):.2f}")
    lines.append(
        f"  % pairs with >100px displacement: "
        f"{100 * sum(1 for x in all_displacements if x > 100) / len(all_displacements):.1f}%"
    )

    lines.append("\n--- Occlusion rate (fraction of frames with IoU>0.1 pair) ---")
    lines.append(f"  Mean:   {np.mean(all_occlusion_rates):.3f}")
    lines.append(f"  Std:    {np.std(all_occlusion_rates):.3f}")
    lines.append(f"  Min:    {np.min(all_occlusion_rates):.3f}")
    lines.append(f"  Max:    {np.max(all_occlusion_rates):.3f}")

    lines.append("\n--- Per-plant summary (first 10) ---")
    lines.append(
        f"{'Plant':<12} {'T':>4} {'IDs':>5} {'MeanLeaves':>11} "
        f"{'OccRate':>9} {'MeanLifespan':>13}"
    )
    lines.append("-" * 60)
    for r in per_plant_rows[:10]:
        lines.append(
            f"{r['plant']:<12} {r['T']:>4} {r['n_ids']:>5} "
            f"{r['mean_count']:>11.2f} {r['occ_rate']:>9.3f} "
            f"{r['mean_lifespan']:>13.2f}"
        )

    output = "\n".join(lines)
    print(output)

    os.makedirs("results", exist_ok=True)
    with open(OUT_FILE, "w") as f:
        f.write(output)
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    analyze()
