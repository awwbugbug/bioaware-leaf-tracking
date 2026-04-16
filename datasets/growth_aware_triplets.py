# datasets/growth_aware_triplets.py
"""
Growth-Aware Hard Negative Mining for Leaf Re-Identification.

Standard triplet sampling (LeafTrackNet V1/V2) draws negative samples
uniformly from other leaves on the same plant. Many such negatives are
trivially easy (very different area, far away), providing weak gradient
signal and producing embeddings that fail on visually similar leaves.

We instead define a biological similarity score φ between two leaf
observations and use it to *up-weight* hard negatives — those that are
area-similar and spatially close to the anchor, exactly the cases that
cause ID switches in practice.

Biological similarity score
---------------------------
For anchor observation x_a (area A_a, centroid c_a) and negative
candidate observation x_j (area A_j, centroid c_j):

    φ(x_j | x_a) = exp(-γ_A · |A_j − A_a| / max(A_a, A_j))
                 · exp(-γ_s · ‖c_j − c_a‖₂ / r)

where r is the image size (normalisation radius) and γ_A, γ_s are
temperature hyperparameters. Higher φ → biologically more similar →
harder negative.

Sampling procedure
------------------
1. Draw anchor x_a randomly from any valid leaf (≥2 observations).
2. Draw positive x_p from the same leaf at a different time point.
3. For each candidate negative leaf ℓ′ ≠ ℓ on the same plant,
   compute φ for every observation of ℓ′ given x_a.
4. Sample ℓ′ with probability proportional to Σ_j φ(x_j^{ℓ′} | x_a),
   then sample the specific observation proportional to φ.

This concentrates training signal on biologically confusable pairs,
directly addressing the failure modes of long-term leaf tracking.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np

# Re-use the scanner from the official codebase
from .triplets import _BaseTripletDataset

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _area(bbox: List[float]) -> float:
    """Bounding-box area from [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _centroid(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _bio_sim(
    bbox_j: List[float],
    bbox_a: List[float],
    gamma_area: float,
    gamma_spatial: float,
    image_size: float,
) -> float:
    """
    Biological similarity φ(x_j | x_a) ∈ (0, 1].

    Higher value → more area-similar and spatially closer → harder negative.
    """
    A_j = _area(bbox_j)
    A_a = _area(bbox_a)
    denom = max(A_a, A_j, 1e-6)
    area_term = np.exp(-gamma_area * abs(A_j - A_a) / denom)

    cj = _centroid(bbox_j)
    ca = _centroid(bbox_a)
    dist = np.sqrt((cj[0] - ca[0]) ** 2 + (cj[1] - ca[1]) ** 2)
    spatial_term = np.exp(-gamma_spatial * dist / image_size)

    return float(area_term * spatial_term)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class LeafTripletDatasetV4(_BaseTripletDataset):
    """
    Growth-Aware Hard Negative Mining (V4).

    Negative samples are drawn with probability proportional to their
    biological similarity to the anchor, so that the model is trained
    harder on the confusable cases that cause ID switches.

    Parameters
    ----------
    root          : str    MOT-style dataset root (same as V2/V3).
    gamma_area    : float  Temperature for area similarity  (default 2.0).
                           Higher → stronger preference for area-matched negatives.
    gamma_spatial : float  Temperature for spatial similarity (default 2.0).
                           Higher → stronger preference for nearby negatives.
    image_size    : int    Image width/height for spatial normalisation (default 1200).
    length        : int    Virtual dataset length (default 10 000).
    transform     : callable | None
    """

    def __init__(
        self,
        root: str,
        gamma_area: float = 2.0,
        gamma_spatial: float = 2.0,
        image_size: int = 1200,
        length: int = 10_000,
        transform=None,
    ):
        super().__init__(root, transform)

        self.gamma_area = float(gamma_area)
        self.gamma_spatial = float(gamma_spatial)
        self.image_size = float(image_size)
        self._length = int(length)

        # Keep only plants that have ≥2 leaves and at least one leaf with ≥2 samples
        self.valid_plants: List[str] = [
            plant
            for plant, leaf_keys in self.plant_to_leaf_keys.items()
            if len(leaf_keys) >= 2
            and any(len(self.samples_by_leaf[k]) >= 2 for k in leaf_keys)
        ]

        if not self.valid_plants:
            raise RuntimeError(
                f"No valid plants found in {root}. "
                "Each plant needs ≥2 leaves with ≥2 observations."
            )

    # ------------------------------------------------------------------
    # Core sampling
    # ------------------------------------------------------------------

    def _sample_hard_negative(
        self,
        anchor_sample: dict,
        neg_leaf_keys: List[Tuple[str, int]],
    ) -> dict:
        """
        Sample a negative observation using biological similarity weighting.

        Steps
        -----
        1. For every candidate negative leaf ℓ′, compute the max φ over
           all its observations given the anchor. This is the leaf-level
           "hardness score".
        2. Sample ℓ′ with probability proportional to its hardness score
           (softmax-style, via numpy).
        3. Within the chosen ℓ′, sample an observation proportional to
           its per-observation φ score.
        """
        anchor_bbox = anchor_sample["bbox"]

        # --- leaf-level hardness scores ---
        leaf_scores: List[float] = []
        for key in neg_leaf_keys:
            obs_list = self.samples_by_leaf[key]
            max_phi = max(
                _bio_sim(
                    s["bbox"],
                    anchor_bbox,
                    self.gamma_area,
                    self.gamma_spatial,
                    self.image_size,
                )
                for s in obs_list
            )
            leaf_scores.append(max_phi)

        leaf_scores_arr = np.array(leaf_scores, dtype=np.float64)

        # Guard: if all scores are ~0 (degenerate), fall back to uniform
        total = leaf_scores_arr.sum()
        if total < 1e-12:
            leaf_probs = np.ones(len(neg_leaf_keys)) / len(neg_leaf_keys)
        else:
            leaf_probs = leaf_scores_arr / total

        chosen_key = neg_leaf_keys[np.random.choice(len(neg_leaf_keys), p=leaf_probs)]

        # --- observation-level sampling within chosen leaf ---
        obs_list = self.samples_by_leaf[chosen_key]
        obs_scores = np.array(
            [
                _bio_sim(
                    s["bbox"],
                    anchor_bbox,
                    self.gamma_area,
                    self.gamma_spatial,
                    self.image_size,
                )
                for s in obs_list
            ],
            dtype=np.float64,
        )
        obs_total = obs_scores.sum()
        if obs_total < 1e-12:
            obs_probs = np.ones(len(obs_list)) / len(obs_list)
        else:
            obs_probs = obs_scores / obs_total

        return obs_list[np.random.choice(len(obs_list), p=obs_probs)]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int):
        # --- 1. choose a valid plant ---
        plant_id = random.choice(self.valid_plants)
        leaf_keys = self.plant_to_leaf_keys[plant_id]

        # --- 2. choose anchor leaf (must have ≥2 observations) ---
        good_keys = [k for k in leaf_keys if len(self.samples_by_leaf[k]) >= 2]
        anchor_key = random.choice(good_keys)
        anchor_samples = self.samples_by_leaf[anchor_key]

        # --- 3. anchor and positive (same leaf, different frames) ---
        anchor_sample, pos_sample = random.sample(anchor_samples, 2)

        # --- 4. negative via growth-aware hard mining ---
        neg_leaf_keys = [k for k in leaf_keys if k != anchor_key]

        # Edge case: only one leaf on this plant → fall back to uniform
        if not neg_leaf_keys:
            # sample from any other plant
            other_plants = [p for p in self.valid_plants if p != plant_id]
            if other_plants:
                fallback_plant = random.choice(other_plants)
                neg_leaf_keys = self.plant_to_leaf_keys[fallback_plant]
            else:
                neg_leaf_keys = [anchor_key]  # pathological; extremely rare

        neg_sample = self._sample_hard_negative(anchor_sample, neg_leaf_keys)

        # --- 5. load crops ---
        a = self._load_crop(anchor_sample)
        p = self._load_crop(pos_sample)
        n = self._load_crop(neg_sample)
        return a, p, n
