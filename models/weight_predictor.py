# models/weight_predictor.py
"""
Adaptive Constraint Weight Predictor.

Given geometric features of a (track, detection) pair, predicts
per-pair constraint weights λ ∈ R^4 for the four association terms:
  λ_0 : appearance
  λ_1 : position
  λ_2 : area
  λ_3 : lifecycle

Architecture
------------
Input  : f ∈ R^5  — geometric context vector (see below)
Hidden : 5 → 64 → 32 → 4  with ReLU + LayerNorm for stability
Output : sigmoid(·) × λ_max  — element-wise bounded positive weights

Input feature vector f_ij
--------------------------
  f[0] = A_j / A_i                   area ratio          (growth direction)
  f[1] = (A_j - A_i) / (A_i + ε)    relative area change
  f[2] = ‖c_j - c_i‖₂ / r           normalised displacement
  f[3] = age_i / τ_a                 track age ratio      (0 = active)
  f[4] = t / T                       lifecycle position   (0=day1, 1=day31)

All inputs are bounded, making training stable without batch norm.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class WeightPredictor(nn.Module):
    """
    Lightweight MLP that maps geometric context to per-constraint weights.

    Parameters
    ----------
    input_dim  : int    Dimension of the geometric feature vector (default 5).
    hidden_dim : int    Hidden layer width (default 64).
    num_constraints : int  Number of output weights (default 4).
    lambda_max : float  Upper bound for predicted weights (default 2.0).
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        num_constraints: int = 4,
        lambda_max: float = 2.0,
    ):
        super().__init__()
        self.lambda_max = float(lambda_max)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_constraints),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise last layer near zero so weights start close to uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # last layer → small init so sigmoid(0) ≈ 0.5, weights start moderate
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        f : Tensor of shape [..., input_dim]

        Returns
        -------
        λ : Tensor of shape [..., num_constraints], each ∈ (0, lambda_max)
        """
        return torch.sigmoid(self.net(f)) * self.lambda_max


# ---------------------------------------------------------------------------
# Feature construction (stateless utility, used by tracker and dataset)
# ---------------------------------------------------------------------------


def build_feature_vector(
    track_area: float,
    track_cx: float,
    track_cy: float,
    track_age: int,
    det_area: float,
    det_cx: float,
    det_cy: float,
    image_size: float,
    max_age: int,
    day_index: int,
    total_days: int,
) -> torch.Tensor:
    """
    Construct the 5-dimensional geometric feature vector f_ij.

    All components are bounded to [0, 1] or near-bounded, ensuring
    numerical stability without additional normalisation layers.

    Parameters
    ----------
    track_area  : last known area of the track (px²)
    track_cx/cy : last known centroid of the track
    track_age   : consecutive frames without a match (0 = active this frame)
    det_area    : area of the candidate detection
    det_cx/cy   : centroid of the candidate detection
    image_size  : width / height of the image (for normalisation)
    max_age     : τ_a — maximum track age before pruning
    day_index   : current frame index (0-based)
    total_days  : total number of frames in the sequence

    Returns
    -------
    f : Tensor shape [5]
    """
    eps = 1e-6

    # f[0]: area ratio  A_j / A_i  — clipped to [0, 4] then /4 → [0,1]
    area_ratio = det_area / max(track_area, eps)
    f0 = min(area_ratio / 4.0, 1.0)

    # f[1]: relative area change (A_j - A_i) / A_i, clipped to [-1, 1] → [0,1]
    rel_change = (det_area - track_area) / max(track_area, eps)
    f1 = (max(-1.0, min(rel_change, 1.0)) + 1.0) / 2.0

    # f[2]: normalised spatial distance ‖c_j - c_i‖₂ / image_size → [0, 1]
    dist = ((track_cx - det_cx) ** 2 + (track_cy - det_cy) ** 2) ** 0.5
    f2 = min(dist / max(image_size, eps), 1.0)

    # f[3]: track age ratio  age / max_age → [0, 1]
    f3 = min(track_age / max(max_age, 1), 1.0)

    # f[4]: lifecycle position  t / T → [0, 1]
    f4 = day_index / max(total_days - 1, 1)

    return torch.tensor([f0, f1, f2, f3, f4], dtype=torch.float32)
