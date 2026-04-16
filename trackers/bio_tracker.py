# trackers/bio_tracker.py
"""
Biology-Constrained Leaf Tracker with Adaptive Constraint Weighting.

Two operating modes:
  1. Fixed weights  (weight_predictor=None)
     cost = λ_app*(1-sim) + λ_pos*d_pos + λ_area*d_area + λ_life*d_life
     Weights are scalars set at construction time.

  2. Adaptive weights  (weight_predictor=WeightPredictor instance)
     λ_ij = WeightPredictor(f_ij)  — per-pair, context-dependent
     cost_ij = λ_ij · [c_app, c_pos, c_area, c_life]
     The predictor is frozen during inference.

The adaptive mode is the main contribution; fixed mode is used for the
ablation study (A1-A4) and as a fallback.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from models.weight_predictor import build_feature_vector
from scipy.optimize import linear_sum_assignment


def _centroid(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class BioTracker:
    def __init__(
        self,
        reid_model,
        transform,
        device="cuda",
        image_size=1200,
        similarity_threshold=0.4,
        max_age=5,
        update_mode="mean",
        alpha=0.5,
        lambda_app=1.0,
        lambda_pos=0.3,
        lambda_area=0.2,
        lambda_life=0.1,
        weight_predictor=None,
        max_norm_dist=0.25,
        area_shrink_tol=0.30,
        area_penalty_scale=1.0,
        total_days=31,
    ):
        self.reid_model = reid_model.to(device).eval()
        self.transform = transform
        self.device = device
        self.image_size = float(image_size)
        self.similarity_threshold = float(similarity_threshold)
        self.max_age = int(max_age)
        self.update_mode = update_mode.lower()
        assert self.update_mode in {"mean", "ema"}
        self.alpha = float(alpha)
        self.lambda_app = float(lambda_app)
        self.lambda_pos = float(lambda_pos)
        self.lambda_area = float(lambda_area)
        self.lambda_life = float(lambda_life)
        self.weight_predictor = None
        if weight_predictor is not None:
            self.weight_predictor = weight_predictor.to(device).eval()
        self.max_norm_dist = float(max_norm_dist)
        self.area_shrink_tol = float(area_shrink_tol)
        self.area_penalty_scale = float(area_penalty_scale)
        self.total_days = int(total_days)
        self.tracks: List[dict] = []
        self.next_id: int = 1
        self._frame_idx: int = 0

    @torch.no_grad()
    def _embed(self, pil_image, box):
        x1, y1, x2, y2 = box
        patch = pil_image.crop((x1, y1, x2, y2))
        x = self.transform(patch).unsqueeze(0).to(self.device)
        return self.reid_model(x).squeeze(0)

    def _prototype(self, track):
        if self.update_mode == "ema":
            return track["ema_emb"]
        return track["emb_sum"] / track["emb_count"]

    def _update_prototype(self, track, emb):
        if self.update_mode == "ema":
            a = self.alpha
            track["ema_emb"] = (1 - a) * track["ema_emb"] + a * emb
        else:
            track["emb_sum"] += emb
            track["emb_count"] += 1

    def _c_appearance(self, protos, det_embs):
        S = (protos @ det_embs.t()).cpu().numpy()
        return np.clip(1.0 - S, 0.0, 1.0)

    def _c_position(self, tracks, det_boxes):
        T, N = len(tracks), len(det_boxes)
        C = np.zeros((T, N))
        for i, t in enumerate(tracks):
            tcx, tcy = t["centroids"][-1]
            for j, box in enumerate(det_boxes):
                dcx, dcy = _centroid(box)
                d = ((tcx - dcx) ** 2 + (tcy - dcy) ** 2) ** 0.5
                C[i, j] = min(d / (self.max_norm_dist * self.image_size), 1.0)
        return C

    def _c_area(self, tracks, det_boxes):
        T, N = len(tracks), len(det_boxes)
        C = np.zeros((T, N))
        det_areas = [_area(b) for b in det_boxes]
        for i, t in enumerate(tracks):
            last_area = t["areas"][-1]
            if last_area <= 0:
                continue
            for j, da in enumerate(det_areas):
                shrink = (last_area - da) / last_area
                penalty = max(0.0, shrink - self.area_shrink_tol)
                C[i, j] = min(penalty * self.area_penalty_scale, 1.0)
        return C

    def _c_lifecycle(self, tracks, det_boxes):
        T, N = len(tracks), len(det_boxes)
        C = np.zeros((T, N))
        for i, t in enumerate(tracks):
            age_ratio = t["age"] / max(self.max_age, 1)
            if age_ratio == 0:
                continue
            tcx, tcy = t["centroids"][-1]
            for j, box in enumerate(det_boxes):
                dcx, dcy = _centroid(box)
                d = ((tcx - dcx) ** 2 + (tcy - dcy) ** 2) ** 0.5
                norm_d = min(d / (self.max_norm_dist * self.image_size), 1.0)
                C[i, j] = age_ratio * norm_d
        return C

    def _build_cost_matrix(self, protos, det_embs, det_boxes):
        c_app = self._c_appearance(protos, det_embs)
        c_pos = self._c_position(self.tracks, det_boxes)
        c_area = self._c_area(self.tracks, det_boxes)
        c_life = self._c_lifecycle(self.tracks, det_boxes)

        if self.weight_predictor is not None:
            T, N = len(self.tracks), len(det_boxes)
            cost = np.zeros((T, N))
            with torch.no_grad():
                for i, t in enumerate(self.tracks):
                    track_area = t["areas"][-1]
                    track_cx, track_cy = t["centroids"][-1]
                    feats = []
                    for box in det_boxes:
                        f = build_feature_vector(
                            track_area=track_area,
                            track_cx=track_cx,
                            track_cy=track_cy,
                            track_age=t["age"],
                            det_area=_area(box),
                            det_cx=_centroid(box)[0],
                            det_cy=_centroid(box)[1],
                            image_size=self.image_size,
                            max_age=self.max_age,
                            day_index=self._frame_idx,
                            total_days=self.total_days,
                        )
                        feats.append(f)
                    feat_batch = torch.stack(feats, dim=0).to(self.device)
                    lambdas = self.weight_predictor(feat_batch).cpu().numpy()  # [N,4]
                    costs_stack = np.stack(
                        [c_app[i], c_pos[i], c_area[i], c_life[i]], axis=1
                    )  # [N,4]
                    cost[i] = (lambdas * costs_stack).sum(axis=1)
        else:
            cost = (
                self.lambda_app * c_app
                + self.lambda_pos * c_pos
                + self.lambda_area * c_area
                + self.lambda_life * c_life
            )
        return cost

    def _init_track(self, emb, box):
        cx, cy = _centroid(box)
        track = {
            "id": self.next_id,
            "emb_sum": emb.clone(),
            "emb_count": 1,
            "ema_emb": emb.clone(),
            "centroids": [(cx, cy)],
            "areas": [_area(box)],
            "box": box,
            "age": 0,
        }
        self.next_id += 1
        return track

    def _update_track(self, track, emb, box):
        self._update_prototype(track, emb)
        track["centroids"].append(_centroid(box))
        track["areas"].append(_area(box))
        track["box"] = box
        track["age"] = 0

    def update(self, pil_image, detections):
        if len(detections) == 0:
            for t in self.tracks:
                t["age"] += 1
            self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]
            self._frame_idx += 1
            return np.array([], dtype=int), np.array([], dtype=float)

        det_embs = torch.stack(
            [self._embed(pil_image, box) for box in detections], dim=0
        )

        if len(self.tracks) == 0:
            areas = [_area(b) for b in detections]
            order = np.argsort(areas)[::-1]
            ids = np.zeros(len(detections), dtype=int)
            sims = np.ones(len(detections), dtype=float)
            for idx in order:
                track = self._init_track(det_embs[idx], detections[idx])
                self.tracks.append(track)
                ids[idx] = track["id"]
            self._frame_idx += 1
            return ids, sims

        protos = torch.stack([self._prototype(t) for t in self.tracks], dim=0)
        protos = F.normalize(protos, p=2, dim=1)
        cost = self._build_cost_matrix(protos, det_embs, detections)

        t_inds, d_inds = linear_sum_assignment(cost)
        S = (protos @ det_embs.t()).cpu().numpy()

        assigned = np.full(len(detections), -1, dtype=int)
        sims_out = np.zeros(len(detections), dtype=float)
        touched: set = set()

        for ti, di in zip(t_inds, d_inds):
            cos_sim = float(S[ti, di])
            if cos_sim >= self.similarity_threshold:
                assigned[di] = self.tracks[ti]["id"]
                sims_out[di] = cos_sim
                self._update_track(self.tracks[ti], det_embs[di], detections[di])
                touched.add(ti)

        for di in range(len(detections)):
            if assigned[di] == -1:
                track = self._init_track(det_embs[di], detections[di])
                self.tracks.append(track)
                assigned[di] = track["id"]
                sims_out[di] = 1.0

        surviving = []
        for i, t in enumerate(self.tracks):
            if i not in touched:
                t["age"] += 1
            if t["age"] <= self.max_age:
                surviving.append(t)
        self.tracks = surviving

        self._frame_idx += 1
        return assigned, sims_out
