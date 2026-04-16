"""
BoT-SORT association-only variant for plant leaf tracking.

Faithful implementation of:
  Aharon et al., "BoT-SORT: Robust Associations Multi-Pedestrian
  Tracking", arXiv:2206.14651, 2022.
  https://github.com/NirAharon/BoT-SORT

Adaptations for CanolaTrack:
- Camera Motion Compensation (CMC/GMC) is omitted: the CanolaTrack
  imaging setup uses a fixed top-down camera; global motion between
  frames is negligible. This is noted explicitly in the paper.
- Appearance embeddings from LeafTrackNet MobileNetV3 (pre-computed).
- track_buffer adapted to max_age=5 for daily-interval protocol.
- with_reid=True (we always use LeafTrackNet embeddings).

Association strategy (BoT-SORT paper, Section 3):
  Stage 1: fuse IoU cost and appearance cost, gate by proximity and
           appearance thresholds, match against high-score detections.
  Stage 2: IoU-only match against low-score detections.
  Stage 3: IoU match for unconfirmed tracks.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from baselines.kalman_filter_xywh import KalmanFilterXYWH


# ------------------------------------------------------------------ #
# Track state
# ------------------------------------------------------------------ #
class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class _IDCounter:
    _count = 0

    @classmethod
    def next_id(cls):
        cls._count += 1
        return cls._count

    @classmethod
    def reset(cls):
        cls._count = 0


# ------------------------------------------------------------------ #
# Single track
# ------------------------------------------------------------------ #
class BoTTrack:
    shared_kalman = KalmanFilterXYWH()

    def __init__(self, tlwh, score, feat=None, feat_history=50):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean = None
        self.covariance = None
        self.is_activated = False
        self.score = float(score)
        self.tracklet_len = 0
        self.state = TrackState.New
        self.track_id = 0
        self.frame_id = 0
        self.start_frame = 0
        self.end_frame = 0

        # Appearance: exponential moving average (alpha=0.9, BoT-SORT paper)
        self.smooth_feat = None
        self.curr_feat = None
        self._feat_alpha = 0.9
        if feat is not None:
            self._update_features(feat)

    def _update_features(self, feat):
        feat = np.asarray(feat, dtype=np.float32)
        feat = feat / (np.linalg.norm(feat) + 1e-6)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = (
                self._feat_alpha * self.smooth_feat + (1 - self._feat_alpha) * feat
            )
        self.smooth_feat = self.smooth_feat / (np.linalg.norm(self.smooth_feat) + 1e-6)

    # ---- geometry ------------------------------------------------- #
    @staticmethod
    def tlwh_to_xywh(tlwh):
        ret = np.asarray(tlwh, dtype=np.float32).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr, dtype=np.float32).copy()
        ret[2:] -= ret[:2]
        return ret

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    # ---- KF ------------------------------------------------------- #
    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(tracks):
        if not tracks:
            return
        multi_mean = np.asarray([t.mean.copy() for t in tracks])
        multi_cov = np.asarray([t.covariance for t in tracks])
        for i, t in enumerate(tracks):
            if t.state != TrackState.Tracked:
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0
        multi_mean, multi_cov = BoTTrack.shared_kalman.multi_predict(
            multi_mean, multi_cov
        )
        for i, (m, c) in enumerate(zip(multi_mean, multi_cov)):
            tracks[i].mean = m
            tracks[i].covariance = c

    def activate(self, kf, frame_id):
        self.kalman_filter = kf
        self.track_id = _IDCounter.next_id()
        self.mean, self.covariance = kf.initiate(self.tlwh_to_xywh(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = frame_id == 1
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh)
        )
        if new_track.curr_feat is not None:
            self._update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.end_frame = frame_id
        if new_id:
            self.track_id = _IDCounter.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.end_frame = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh)
        )
        if new_track.curr_feat is not None:
            self._update_features(new_track.curr_feat)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    def __repr__(self):
        return f"BoTTrack_{self.track_id}_({self.start_frame}-{self.end_frame})"


# ------------------------------------------------------------------ #
# Distance / cost helpers
# ------------------------------------------------------------------ #
def iou_distance(atracks, btracks):
    if not atracks or not btracks:
        return np.zeros((len(atracks), len(btracks)), dtype=np.float32)
    atlbr = np.array([t.tlbr for t in atracks], dtype=np.float32)
    btlbr = np.array([t.tlbr for t in btracks], dtype=np.float32)
    xx1 = np.maximum(atlbr[:, None, 0], btlbr[None, :, 0])
    yy1 = np.maximum(atlbr[:, None, 1], btlbr[None, :, 1])
    xx2 = np.minimum(atlbr[:, None, 2], btlbr[None, :, 2])
    yy2 = np.minimum(atlbr[:, None, 3], btlbr[None, :, 3])
    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    a_area = ((atlbr[:, 2] - atlbr[:, 0]) * (atlbr[:, 3] - atlbr[:, 1]))[:, None]
    b_area = ((btlbr[:, 2] - btlbr[:, 0]) * (btlbr[:, 3] - btlbr[:, 1]))[None, :]
    union = a_area + b_area - inter
    return 1.0 - inter / (union + 1e-6)


def embedding_distance(tracks, detections):
    """
    Cosine distance between track smooth features and detection features.
    Returns shape (len(tracks), len(detections)).
    """
    cost = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for i, t in enumerate(tracks):
        if t.smooth_feat is None:
            cost[i, :] = 1.0
            continue
        for j, d in enumerate(detections):
            if d.curr_feat is None:
                cost[i, j] = 1.0
            else:
                sim = np.dot(t.smooth_feat, d.curr_feat)
                cost[i, j] = max(0.0, 1.0 - sim)
    return cost


def fuse_iou_appearance(
    iou_dists, emb_dists, detections, proximity_thresh, appearance_thresh
):
    """
    BoT-SORT fusion strategy (Section 3.2 of the paper):
      iou_mask  = iou_dist > proximity_thresh (too far → skip appearance)
      emb_mask  = emb_dist > appearance_thresh (too dissimilar)
      fused     = min(iou_dist, emb_dist/2)
      gate out  = 1.0 if iou_mask
    """
    iou_mask = iou_dists > proximity_thresh
    # emb_dists already in [0,1]; paper divides by 2 to normalise
    emb_norm = emb_dists / 2.0
    emb_norm[emb_norm > appearance_thresh] = 1.0
    emb_norm[iou_mask] = 1.0
    fused = np.minimum(iou_dists, emb_norm)
    return fused


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            list(range(cost_matrix.shape[0])),
            list(range(cost_matrix.shape[1])),
        )
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches, u_rows, u_cols = [], [], []
    assigned_r, assigned_c = set(row_ind), set(col_ind)
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches.append([r, c])
        else:
            u_rows.append(r)
            u_cols.append(c)
    for r in range(cost_matrix.shape[0]):
        if r not in assigned_r:
            u_rows.append(r)
    for c in range(cost_matrix.shape[1]):
        if c not in assigned_c:
            u_cols.append(c)
    return np.array(matches, dtype=int), u_rows, u_cols


def joint_stracks(a, b):
    seen = {t.track_id for t in a}
    return list(a) + [t for t in b if t.track_id not in seen]


def sub_stracks(a, b):
    remove = {t.track_id for t in b}
    return [t for t in a if t.track_id not in remove]


def remove_duplicate_stracks(sa, sb, iou_thresh=0.15):
    if not sa or not sb:
        return sa, sb
    pdist = iou_distance(sa, sb)
    pairs = np.where(pdist < iou_thresh)
    da, db = set(), set()
    for p, q in zip(*pairs):
        tp = sa[p].frame_id - sa[p].start_frame
        tq = sb[q].frame_id - sb[q].start_frame
        if tp > tq:
            db.add(q)
        else:
            da.add(p)
    return (
        [t for i, t in enumerate(sa) if i not in da],
        [t for i, t in enumerate(sb) if i not in db],
    )


# ------------------------------------------------------------------ #
# BoT-SORT Tracker
# ------------------------------------------------------------------ #
class BoTSORTTracker:
    """
    BoT-SORT tracker faithful to Aharon et al. arXiv:2206.14651.

    Camera Motion Compensation (CMC) is disabled as CanolaTrack uses
    a fixed static top-down camera (no global frame-to-frame motion).
    This is disclosed in the paper as:
      "We implemented an association-only variant of BoT-SORT with
       CMC omitted, as global motion compensation is inapplicable to
       the static top-down imaging protocol of CanolaTrack."

    Parameters (from original source / paper):
      track_high_thresh  = 0.5
      track_low_thresh   = 0.1
      new_track_thresh   = 0.6
      track_buffer       = 5    (adapted for daily-interval)
      match_thresh       = 0.8
      proximity_thresh   = 0.5
      appearance_thresh  = 0.25
    """

    def __init__(
        self,
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        track_buffer=5,
        match_thresh=0.8,
        proximity_thresh=0.5,
        appearance_thresh=0.25,
    ):
        _IDCounter.reset()
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []

        self.frame_id = 0
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.max_time_lost = track_buffer
        self.match_thresh = match_thresh
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh
        self.kalman_filter = KalmanFilterXYWH()

    def update(self, tlwhs, scores, features):
        """
        Parameters
        ----------
        tlwhs    : array-like (N, 4) – (x1, y1, w, h)
        scores   : array-like (N,)
        features : array-like (N, D) – LeafTrackNet embeddings
        """
        self.frame_id += 1
        activated, refound, lost_new, removed = [], [], [], []

        tlwhs = np.asarray(tlwhs, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        features = np.asarray(features, dtype=np.float32)

        # Split by score
        high_mask = scores > self.track_high_thresh
        low_mask = (scores > self.track_low_thresh) & ~high_mask

        dets_high = [
            BoTTrack(t, s, f)
            for t, s, f in zip(tlwhs[high_mask], scores[high_mask], features[high_mask])
        ]
        dets_low = [BoTTrack(t, s) for t, s in zip(tlwhs[low_mask], scores[low_mask])]

        # Separate confirmed / unconfirmed
        unconfirmed = [t for t in self.tracked_stracks if not t.is_activated]
        tracked = [t for t in self.tracked_stracks if t.is_activated]

        # ---- Stage 1: high-score dets, fused IoU + appearance --------
        pool = joint_stracks(tracked, self.lost_stracks)
        BoTTrack.multi_predict(pool)
        # CMC would go here → omitted (static camera)

        iou_dists = iou_distance(pool, dets_high)
        emb_dists = embedding_distance(pool, dets_high)
        dists1 = fuse_iou_appearance(
            iou_dists,
            emb_dists,
            dets_high,
            self.proximity_thresh,
            self.appearance_thresh,
        )
        m1, u_track, u_det_high = linear_assignment(dists1, self.match_thresh)

        for ti, di in m1:
            t, d = pool[ti], dets_high[di]
            if t.state == TrackState.Tracked:
                t.update(d, self.frame_id)
                activated.append(t)
            else:
                t.re_activate(d, self.frame_id)
                refound.append(t)

        # ---- Stage 2: low-score dets, IoU only -----------------------
        r_tracked = [pool[i] for i in u_track if pool[i].state == TrackState.Tracked]
        dists2 = iou_distance(r_tracked, dets_low)
        m2, u_track2, _ = linear_assignment(dists2, 0.5)

        for ti, di in m2:
            t, d = r_tracked[ti], dets_low[di]
            if t.state == TrackState.Tracked:
                t.update(d, self.frame_id)
                activated.append(t)
            else:
                t.re_activate(d, self.frame_id)
                refound.append(t)

        for i in u_track2:
            t = r_tracked[i]
            if t.state != TrackState.Lost:
                t.mark_lost()
                lost_new.append(t)

        # ---- Stage 3: unconfirmed vs remaining high-score ------------
        remaining = [dets_high[i] for i in u_det_high]
        iou_unc = iou_distance(unconfirmed, remaining)
        emb_unc = embedding_distance(unconfirmed, remaining)
        dists3 = fuse_iou_appearance(
            iou_unc, emb_unc, remaining, self.proximity_thresh, self.appearance_thresh
        )
        m3, u_unc, u_det2 = linear_assignment(dists3, 0.7)

        for ti, di in m3:
            unconfirmed[ti].update(remaining[di], self.frame_id)
            activated.append(unconfirmed[ti])
        for i in u_unc:
            unconfirmed[i].mark_removed()
            removed.append(unconfirmed[i])

        # ---- Init new tracks -----------------------------------------
        for i in u_det2:
            d = remaining[i]
            if d.score >= self.new_track_thresh:
                d.activate(self.kalman_filter, self.frame_id)
                activated.append(d)

        # ---- Prune lost tracks ---------------------------------------
        for t in self.lost_stracks:
            if self.frame_id - t.end_frame > self.max_time_lost:
                t.mark_removed()
                removed.append(t)

        # ---- Merge ---------------------------------------------------
        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refound)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_new)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        return [t for t in self.tracked_stracks if t.is_activated]
