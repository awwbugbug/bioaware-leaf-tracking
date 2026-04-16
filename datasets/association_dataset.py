# datasets/association_dataset.py
"""
Association Dataset for Weight Predictor Training.

Constructs pairwise training samples from ground-truth MOT annotations.
Each sample is a triple (f_pos, f_neg, c_pos, c_neg) where:
  - f_pos / f_neg : geometric feature vectors for a correct / incorrect match
  - c_pos / c_neg : cost vectors [c_app, c_pos, c_area, c_life] for each

The weight predictor is trained with a pairwise ranking loss:
  L = max(0, λ·c_pos - λ·c_neg + margin)

so that the weighted cost of a correct match is lower than an incorrect one.

Ground-truth cost construction
-------------------------------
For each consecutive frame pair (t, t+1) in a plant sequence:
  Positive pairs  : same leaf_id across frames
  Negative pairs  : different leaf_id, same frame t+1

Cost terms are precomputed from geometry (no embedding needed at this stage),
since the weight predictor operates purely on geometric features.

Note on appearance cost
-----------------------
We do not load images here. The appearance cost c_app for a pair is
approximated as 0.0 for positive pairs and 1.0 for negative pairs,
consistent with the assumption that a well-trained ReID model already
separates same-leaf / different-leaf embeddings. The predictor learns
to weight the *geometric* terms; appearance always acts as the anchor.
"""

from __future__ import annotations

import os
import random
from typing import List, Tuple

import torch
from models.weight_predictor import build_feature_vector
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Geometry helpers (duplicated locally to avoid cross-module deps)
# ---------------------------------------------------------------------------


def _area(bbox: List[float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _centroid(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _cost_appearance(sim: float) -> float:
    """Approximate appearance cost from cosine similarity."""
    return max(0.0, 1.0 - sim)


def _cost_position(
    track_cx: float,
    track_cy: float,
    det_cx: float,
    det_cy: float,
    image_size: float,
    max_norm_dist: float,
) -> float:
    dist = ((track_cx - det_cx) ** 2 + (track_cy - det_cy) ** 2) ** 0.5
    return min(dist / (max_norm_dist * image_size), 1.0)


def _cost_area(
    track_area: float,
    det_area: float,
    area_shrink_tol: float,
    area_penalty_scale: float,
) -> float:
    if track_area <= 0:
        return 0.0
    shrink_frac = (track_area - det_area) / track_area
    penalty = max(0.0, shrink_frac - area_shrink_tol)
    return min(penalty * area_penalty_scale, 1.0)


def _cost_lifecycle(
    age: int,
    max_age: int,
    track_cx: float,
    track_cy: float,
    det_cx: float,
    det_cy: float,
    image_size: float,
    max_norm_dist: float,
) -> float:
    age_ratio = age / max(max_age, 1)
    if age_ratio == 0:
        return 0.0
    dist = ((track_cx - det_cx) ** 2 + (track_cy - det_cy) ** 2) ** 0.5
    norm_d = min(dist / (max_norm_dist * image_size), 1.0)
    return age_ratio * norm_d


# ---------------------------------------------------------------------------
# GT annotation loader
# ---------------------------------------------------------------------------


def _load_gt(gt_path: str) -> dict:
    """
    Load gt.txt into {frame: {leaf_id: bbox}} structure.
    bbox = [x1, y1, x2, y2]
    """
    data: dict = {}
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame = int(parts[0])
            leaf_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            bbox = [x, y, x + w, y + h]
            data.setdefault(frame, {})[leaf_id] = bbox
    return data


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class AssociationDataset(Dataset):
    """
    Pairwise ranking dataset for the weight predictor.

    For each consecutive frame pair in a plant sequence, we enumerate
    all (track, detection) pairs and label them positive (same ID) or
    negative (different ID). We then form triplets:
      (anchor_track, positive_detection, negative_detection)
    and build feature vectors + cost vectors for each.

    Parameters
    ----------
    train_root       : str   MOT-style root with {plant}/gt/gt.txt
    image_size       : int   Image width = height (default 1200)
    max_age          : int   τ_a for age feature normalisation (default 5)
    max_norm_dist    : float For position / lifecycle cost (default 0.25)
    area_shrink_tol  : float For area cost (default 0.30)
    area_penalty_scale : float (default 1.0)
    length           : int   Virtual epoch size (default 20 000)
    """

    def __init__(
        self,
        train_root: str,
        image_size: int = 1200,
        max_age: int = 5,
        max_norm_dist: float = 0.25,
        area_shrink_tol: float = 0.30,
        area_penalty_scale: float = 1.0,
        length: int = 20_000,
    ):
        self.image_size = float(image_size)
        self.max_age = int(max_age)
        self.max_norm_dist = float(max_norm_dist)
        self.area_shrink_tol = float(area_shrink_tol)
        self.area_penalty_scale = float(area_penalty_scale)
        self._length = int(length)

        # Collect all consecutive-frame triplets
        # Each entry: (track_bbox, pos_bbox, neg_bbox, t, T, track_age)
        self.samples: List[Tuple] = []
        self._build(train_root)

        if not self.samples:
            raise RuntimeError(
                f"No training samples found in {train_root}. "
                "Check that gt/gt.txt files exist and have consecutive frames."
            )

    def _build(self, root: str) -> None:
        plant_dirs = sorted(
            [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        )
        for plant_id in plant_dirs:
            gt_path = os.path.join(root, plant_id, "gt", "gt.txt")
            if not os.path.exists(gt_path):
                continue
            gt = _load_gt(gt_path)
            frames = sorted(gt.keys())
            T = len(frames)
            if T < 2:
                continue

            for idx in range(T - 1):
                t_frame = frames[idx]
                t1_frame = frames[idx + 1]
                tracks = gt[t_frame]  # {leaf_id: bbox} at t
                dets = gt[t1_frame]  # {leaf_id: bbox} at t+1

                # For each track leaf that also appears at t+1 (positive exists)
                for track_id, track_bbox in tracks.items():
                    if track_id not in dets:
                        continue  # no positive match available
                    pos_bbox = dets[track_id]

                    # Negative: any other detection at t+1
                    neg_ids = [lid for lid in dets if lid != track_id]
                    if not neg_ids:
                        continue
                    neg_id = random.choice(neg_ids)
                    neg_bbox = dets[neg_id]

                    # age is 0 for all tracks here (consecutive frames, no gap)
                    self.samples.append((track_bbox, pos_bbox, neg_bbox, idx, T, 0))

    def _build_costs(
        self,
        track_bbox: List[float],
        det_bbox: List[float],
        age: int,
        day_index: int,
        total_days: int,
        is_positive: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (feature_vector [5], cost_vector [4]).
        """
        track_area = _area(track_bbox)
        track_cx, track_cy = _centroid(track_bbox)
        det_area = _area(det_bbox)
        det_cx, det_cy = _centroid(det_bbox)

        feat = build_feature_vector(
            track_area=track_area,
            track_cx=track_cx,
            track_cy=track_cy,
            track_age=age,
            det_area=det_area,
            det_cx=det_cx,
            det_cy=det_cy,
            image_size=self.image_size,
            max_age=self.max_age,
            day_index=day_index,
            total_days=total_days,
        )

        # Appearance cost: fixed at 0.5 for all pairs.
        # We deliberately do NOT use 0/1 proxy here — that would trivially
        # saturate the ranking loss before the MLP can learn geometric weights.
        # The predictor must learn to up-weight geometric terms (position, area)
        # that actually discriminate correct from incorrect matches.
        c_app = 0.5

        c_pos_cost = _cost_position(
            track_cx,
            track_cy,
            det_cx,
            det_cy,
            self.image_size,
            self.max_norm_dist,
        )
        c_area = _cost_area(
            track_area,
            det_area,
            self.area_shrink_tol,
            self.area_penalty_scale,
        )
        c_life = _cost_lifecycle(
            age,
            self.max_age,
            track_cx,
            track_cy,
            det_cx,
            det_cy,
            self.image_size,
            self.max_norm_dist,
        )

        costs = torch.tensor([c_app, c_pos_cost, c_area, c_life], dtype=torch.float32)
        return feat, costs

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int):
        track_bbox, pos_bbox, neg_bbox, day_idx, T, age = random.choice(self.samples)

        feat_pos, cost_pos = self._build_costs(
            track_bbox, pos_bbox, age, day_idx, T, is_positive=True
        )
        feat_neg, cost_neg = self._build_costs(
            track_bbox, neg_bbox, age, day_idx, T, is_positive=False
        )

        return feat_pos, cost_pos, feat_neg, cost_neg
