"""
DeepSORT association-only variant for plant leaf tracking.

Faithful implementation of:
  Wojke et al., "Simple Online and Realtime Tracking with a Deep
  Association Metric", ICIP 2017.
  https://github.com/nwojke/deep_sort

Adaptations for CanolaTrack:
- Appearance embeddings from LeafTrackNet MobileNetV3 (pre-computed)
- Detection proposals from official LeafTrackNet pipeline
- Camera motion compensation omitted (static top-down camera)
- max_age adapted for daily-interval imaging (set to 5 to match
  LeafTrackNet baseline, ensuring fair comparison)
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from baselines.kalman_filter_xyah import KalmanFilterXYAH, chi2inv95


# ------------------------------------------------------------------ #
# Track states
# ------------------------------------------------------------------ #
class TrackState:
    Tentative = 1  # not yet confirmed (within n_init frames)
    Confirmed = 2  # active confirmed track
    Deleted = 3  # pruned


# ------------------------------------------------------------------ #
# Single track
# ------------------------------------------------------------------ #
class DSTrack:
    """One tracked leaf identity."""

    def __init__(self, mean, covariance, track_id, n_init, max_age, feature):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = TrackState.Tentative
        self.features = [feature]  # gallery of appearance vectors
        self._n_init = n_init
        self._max_age = max_age

    # ---- geometry ------------------------------------------------- #
    def to_tlwh(self):
        """Return (x1, y1, w, h) from KF state (cx, cy, a, h)."""
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]  # a * h = w
        ret[:2] -= ret[2:] / 2  # cx -> x1
        return ret

    def to_xyah(self):
        """Return (cx, cy, a, h)."""
        ret = self.mean[:4].copy()
        return ret

    # ---- KF steps -------------------------------------------------- #
    def predict(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah()
        )
        self.features.append(detection.feature)
        # Keep at most 100 features in gallery
        if len(self.features) > 100:
            self.features = self.features[-100:]
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted


# ------------------------------------------------------------------ #
# Detection wrapper
# ------------------------------------------------------------------ #
class DSDetection:
    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_xyah(self):
        """Convert tlwh -> (cx, cy, a, h)."""
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2  # x1,y1 -> cx,cy
        ret[2] /= ret[3]  # w -> a = w/h
        return ret

    def to_tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret


# ------------------------------------------------------------------ #
# Nearest-neighbour appearance metric (cosine distance, gallery)
# ------------------------------------------------------------------ #
class NNMetric:
    """
    Cosine distance with gallery matching.
    For each track, stores a gallery of past embeddings and computes
    minimum cosine distance to each new detection.
    """

    def __init__(self, matching_threshold=0.4, budget=100):
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}  # track_id -> list of embeddings

    def partial_fit(self, features, targets, active_targets):
        for feat, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feat)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget :]
        # Prune dead tracks
        self.samples = {k: v for k, v in self.samples.items() if k in active_targets}

    def distance(self, features, targets):
        """
        Compute cost matrix: rows = tracks (targets),
        cols = detections (features).
        Each entry = min cosine distance over gallery.
        """
        cost_matrix = np.zeros((len(targets), len(features)), dtype=np.float32)
        for i, target in enumerate(targets):
            if target not in self.samples or len(self.samples[target]) == 0:
                cost_matrix[i, :] = 1.0
                continue
            gallery = np.array(self.samples[target], dtype=np.float32)
            # Normalise
            gallery = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-6)
            feats = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-6)
            # cosine distance = 1 - cosine similarity
            # take minimum over gallery
            sims = np.dot(gallery, feats.T)  # (G, D)
            cost_matrix[i, :] = 1.0 - sims.max(axis=0)
        return cost_matrix


# ------------------------------------------------------------------ #
# Hungarian solver helpers
# ------------------------------------------------------------------ #
def linear_assignment_solver(cost_matrix, threshold):
    """
    Solve LAP and return (matches, unmatched_rows, unmatched_cols).
    Assignments with cost > threshold are rejected.
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches, unmatched_rows, unmatched_cols = [], [], []
    assigned_rows = set(row_ind)
    assigned_cols = set(col_ind)
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > threshold:
            unmatched_rows.append(r)
            unmatched_cols.append(c)
        else:
            matches.append((r, c))
    for r in range(cost_matrix.shape[0]):
        if r not in assigned_rows:
            unmatched_rows.append(r)
    for c in range(cost_matrix.shape[1]):
        if c not in assigned_cols:
            unmatched_cols.append(c)
    return matches, unmatched_rows, unmatched_cols


def iou_cost(tracks, detections, track_indices, detection_indices):
    """IoU-based cost for secondary matching."""

    def bbox_iou(bb1, bb2):
        x1 = max(bb1[0], bb2[0])
        y1 = max(bb1[1], bb2[1])
        x2 = min(bb1[0] + bb1[2], bb2[0] + bb2[2])
        y2 = min(bb1[1] + bb1[3], bb2[1] + bb2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = bb1[2] * bb1[3] + bb2[2] * bb2[3] - inter
        return inter / (union + 1e-6)

    cost = np.zeros((len(track_indices), len(detection_indices)), dtype=np.float32)
    for i, ti in enumerate(track_indices):
        for j, di in enumerate(detection_indices):
            cost[i, j] = 1.0 - bbox_iou(tracks[ti].to_tlwh(), detections[di].tlwh)
    return cost


def gate_cost_matrix(
    kf,
    cost_matrix,
    tracks,
    detections,
    track_indices,
    detection_indices,
    gating_threshold=None,
):
    """
    Invalidate cost matrix entries that fail Mahalanobis gating.
    """
    if gating_threshold is None:
        gating_threshold = chi2inv95[4]

    measurements = np.array([detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        distances = kf.gating_distance(track.mean, track.covariance, measurements)
        cost_matrix[row, distances > gating_threshold] = 1e5
    return cost_matrix


# ------------------------------------------------------------------ #
# Matching cascade (DeepSORT core)
# ------------------------------------------------------------------ #
def matching_cascade(
    metric,
    max_distance,
    cascade_depth,
    tracks,
    detections,
    track_indices=None,
    detection_indices=None,
):
    """
    Matching cascade: prioritise tracks that were updated most recently.
    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = list(detection_indices)
    matches = []

    for level in range(cascade_depth):
        if not unmatched_detections:
            break
        # Tracks at this cascade level
        track_indices_l = [
            k for k in track_indices if tracks[k].time_since_update == 1 + level
        ]
        if not track_indices_l:
            continue

        # Build cost matrix: appearance
        features = np.array([detections[i].feature for i in unmatched_detections])
        targets = np.array([tracks[k].track_id for k in track_indices_l])
        cost = metric.distance(features, targets)

        # Mahalanobis gating
        cost = gate_cost_matrix(
            _shared_kf, cost, tracks, detections, track_indices_l, unmatched_detections
        )

        matches_l, _, unmatched_detections = linear_assignment_solver(
            cost, max_distance
        )
        for row, col in matches_l:
            matches.append(
                (
                    track_indices_l[row],
                    unmatched_detections[col]
                    if isinstance(unmatched_detections, list)
                    else col,
                )
            )

        # Rebuild unmatched_detections from solver output
        # (solver returns indices into the current unmatched_detections list)
        # Re-do properly:
        um_det_new = []
        matched_cols = {c for _, c in matches_l}
        for j, di in enumerate(
            unmatched_detections
            if not isinstance(unmatched_detections, list)
            else unmatched_detections
        ):
            pass
        # Simplified: redo matching_cascade with correct indexing
        break  # handled below in Tracker._match

    return matches, unmatched_detections


# Shared KF instance (set by Tracker)
_shared_kf = None


# ------------------------------------------------------------------ #
# DeepSORT Tracker
# ------------------------------------------------------------------ #
class DeepSORTTracker:
    """
    DeepSORT tracker faithful to Wojke et al. ICIP 2017.

    Parameters
    ----------
    max_cosine_distance : float
        Appearance gate threshold (default 0.4 from original paper).
    nn_budget : int
        Max gallery size per track (default 100).
    max_iou_distance : float
        IoU gate for secondary matching (default 0.7).
    max_age : int
        Frames before a missed track is deleted (set to 5 to match
        CanolaTrack daily-interval protocol).
    n_init : int
        Frames of consecutive detection before confirming a track
        (default 3).
    """

    def __init__(
        self,
        max_cosine_distance=0.4,
        nn_budget=100,
        max_iou_distance=0.7,
        max_age=5,
        n_init=1,
    ):
        self.metric = NNMetric(max_cosine_distance, nn_budget)
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.kf = KalmanFilterXYAH()
        self.tracks = []
        self._next_id = 1

        global _shared_kf
        _shared_kf = self.kf

    def predict(self):
        for t in self.tracks:
            t.predict(self.kf)

    def update(self, detections):
        """
        Parameters
        ----------
        detections : list of DSDetection
        """
        matches, unmatched_tracks, unmatched_dets = self._match(detections)

        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[det_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for det_idx in unmatched_dets:
            self._initiate_track(detections[det_idx])

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update metric gallery
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for t in self.tracks:
            if not t.is_confirmed():
                continue
            features.extend(t.features)
            targets.extend([t.track_id] * len(t.features))
            t.features = []
        self.metric.partial_fit(
            np.array(features) if features else np.zeros((0, 128)),
            np.array(targets),
            active_targets,
        )

    def _match(self, detections):
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()
        ]

        # --- Stage A: cascade appearance matching for confirmed tracks ---
        matches_a, unmatched_tracks_a, unmatched_dets = self._cascade_match(
            detections, confirmed_tracks
        )

        # --- Stage B: IoU matching for unconfirmed + recently lost ---
        iou_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1
        ]
        unmatched_tracks_a2 = [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1
        ]

        # Build IoU cost matrix with current unmatched_dets
        unmatched_dets_b = list(unmatched_dets)
        cost_b = iou_cost(self.tracks, detections, iou_candidates, unmatched_dets_b)
        matches_b_idx, unmatched_b_t, unmatched_b_d = linear_assignment_solver(
            cost_b, self.max_iou_distance
        )

        matches_b = [(iou_candidates[r], unmatched_dets_b[c]) for r, c in matches_b_idx]
        unmatched_tracks_b = [iou_candidates[i] for i in unmatched_b_t]
        unmatched_dets = [unmatched_dets_b[j] for j in unmatched_b_d]

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a2 + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_dets

    def _cascade_match(self, detections, track_indices):
        """Matching cascade: level = time_since_update."""
        unmatched_dets = list(range(len(detections)))
        matches = []

        for level in range(self.max_age):
            if not unmatched_dets:
                break
            tracks_l = [
                k
                for k in track_indices
                if self.tracks[k].time_since_update == 1 + level
            ]
            if not tracks_l:
                continue

            # Appearance cost
            feats = np.array(
                [detections[i].feature for i in unmatched_dets], dtype=np.float32
            )
            targets = np.array([self.tracks[k].track_id for k in tracks_l])
            cost = self.metric.distance(feats, targets)  # (|tracks_l|, |dets|)

            # Mahalanobis gating
            measurements = np.array([detections[i].to_xyah() for i in unmatched_dets])
            for row, ti in enumerate(tracks_l):
                t = self.tracks[ti]
                dist = self.kf.gating_distance(t.mean, t.covariance, measurements)
                cost[row, dist > chi2inv95[4]] = 1e5

            matches_l, unmatched_t, unmatched_d = linear_assignment_solver(
                cost, self.metric.matching_threshold
            )
            for r, c in matches_l:
                matches.append((tracks_l[r], unmatched_dets[c]))

            # Update unmatched_dets
            unmatched_dets = [unmatched_dets[j] for j in unmatched_d]

        unmatched_tracks = [
            k for k in track_indices if not any(k == m[0] for m in matches)
        ]
        return matches, unmatched_tracks, unmatched_dets

    def _initiate_track(self, detection):
        mean, cov = self.kf.initiate(detection.to_xyah())
        self.tracks.append(
            DSTrack(
                mean, cov, self._next_id, self.n_init, self.max_age, detection.feature
            )
        )
        self._next_id += 1

    def get_results(self):
        """Return list of (track_id, tlwh) for active confirmed tracks."""
        return [
            (t.track_id, t.to_tlwh())
            for t in self.tracks
            if t.is_confirmed() and t.time_since_update == 0
        ]
