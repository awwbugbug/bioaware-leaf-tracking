# datasets/growth_reg_triplets.py
"""
LeafTripletDatasetV5 — Growth-Consistency Regularization Training Data.

Compared with the standard triplet datasets (V1-V4), V5 returns two
additional scalar values per sample: the bounding-box area of the anchor
and the positive observation. These are consumed by the training loop to
compute the growth-consistency regularization term:

    L_growth = mean( max(0, g_ap · ‖e_a − e_p‖₂ − ε) )

where g_ap = (A_p − A_a) / (A_a + δ) is the relative area growth rate
of the anchor→positive pair, ‖e_a − e_p‖₂ is the L2 embedding distance,
and ε is a slack margin.

Intuition
---------
When a leaf has grown noticeably (g_ap > 0), the same-identity embeddings
should be close: the network should recognise the leaf despite its growth.
The term penalises cases where g_ap is large but the embedding distance
also remains large, encouraging growth-invariant representations.

When g_ap ≤ 0 (detection noise, slight shrinkage), the clamp makes the
term zero — no spurious gradient.

Sampling strategy
-----------------
Anchor and positive are drawn from the *same* leaf at the *earliest* and
a later time point, so g_ap ≥ 0 on average (growth direction is correct).
Negative sampling is uniform over other leaves on the same plant (same as
V2), keeping the triplet loss component unchanged.
"""

from __future__ import annotations

import random

import torch

from .triplets import _BaseTripletDataset

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _area(bbox) -> float:
    """Bounding-box pixel area from [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class LeafTripletDatasetV5(_BaseTripletDataset):
    """
    Triplet dataset that additionally returns anchor and positive bbox areas
    for growth-consistency loss computation.

    Returns
    -------
    anchor_img  : Tensor [C, H, W]
    pos_img     : Tensor [C, H, W]
    neg_img     : Tensor [C, H, W]
    area_a      : Tensor scalar (float32)  — anchor bbox area in pixels²
    area_p      : Tensor scalar (float32)  — positive bbox area in pixels²

    Parameters
    ----------
    root        : str   MOT-style dataset root (same layout as V1-V4).
    length      : int   Virtual epoch size (default 10 000).
    transform   : callable | None
    """

    def __init__(
        self,
        root: str,
        length: int = 10_000,
        transform=None,
    ):
        super().__init__(root, transform)
        self._length = int(length)

        # Plants that have ≥2 leaves, with at least one leaf having ≥2 observations
        self.valid_plants = [
            plant
            for plant, leaf_keys in self.plant_to_leaf_keys.items()
            if len(leaf_keys) >= 2
            and any(len(self.samples_by_leaf[k]) >= 2 for k in leaf_keys)
        ]

        if not self.valid_plants:
            raise RuntimeError(
                f"No valid plants found in {root}. "
                "Each plant needs ≥2 leaves with ≥2 observations each."
            )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int):
        # --- 1. choose a valid plant ---
        plant_id = random.choice(self.valid_plants)
        leaf_keys = self.plant_to_leaf_keys[plant_id]

        # --- 2. anchor leaf (needs ≥2 observations) ---
        good_keys = [k for k in leaf_keys if len(self.samples_by_leaf[k]) >= 2]
        anchor_key = random.choice(good_keys)
        anchor_samples = self.samples_by_leaf[anchor_key]

        # --- 3. anchor = earlier frame, positive = later frame ---
        # Sort by frame index so anchor is the earlier observation.
        # This guarantees g_ap = (A_p - A_a)/A_a ≥ 0 on average,
        # matching the biological growth direction.
        sorted_samples = sorted(anchor_samples, key=lambda s: s.get("frame", 0))
        # Pick two distinct indices; anchor < positive (time order).
        i, j = sorted(random.sample(range(len(sorted_samples)), 2))
        anchor_sample = sorted_samples[i]
        pos_sample = sorted_samples[j]

        # --- 4. negative: uniform over other leaves on same plant (V2 style) ---
        neg_leaf_keys = [k for k in leaf_keys if k != anchor_key]
        if not neg_leaf_keys:
            other_plants = [p for p in self.valid_plants if p != plant_id]
            if other_plants:
                fallback_plant = random.choice(other_plants)
                neg_leaf_keys = self.plant_to_leaf_keys[fallback_plant]
            else:
                neg_leaf_keys = [anchor_key]
        neg_key = random.choice(neg_leaf_keys)
        neg_sample = random.choice(self.samples_by_leaf[neg_key])

        # --- 5. load image crops ---
        a = self._load_crop(anchor_sample)
        p = self._load_crop(pos_sample)
        n = self._load_crop(neg_sample)

        # --- 6. extract areas ---
        area_a = torch.tensor(_area(anchor_sample["bbox"]), dtype=torch.float32)
        area_p = torch.tensor(_area(pos_sample["bbox"]), dtype=torch.float32)

        return a, p, n, area_a, area_p
