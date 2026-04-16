"""
ByteTrack association-only variant for plant leaf tracking.

Faithful implementation of:
  Zhang et al., "ByteTrack: Multi-Object Tracking by Associating
  Every Detection Box", ECCV 2022.
  https://github.com/ifzhang/ByteTrack

Adaptations for CanolaTrack:
- Detection proposals from official LeafTrackNet pipeline
- track_buffer adapted to max_age=5 (daily-interval protocol)
- No image input required (IoU-only association)
- score thresholds adapted: all proposals have uniform score=1.0,
  so only the high-score stage is active (single-stage IoU matching)
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from baselines.kalman_filter_xywh import KalmanFilterXYWH


# ------------------------------------------------------------------ #
# Track states
# ------------------------------------------------------------------ #
class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


# ------------------------------------------------------------------ #
# ID counter
# ------------------------------------------------------------------ #
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
class BYTrack:
    shared_kalman = KalmanFilterXYWH()

    def __init__(self, tlwh, score):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.score = float(score)
        self.tracklet_len = 0
        self.state = TrackState.New
        self.track_id = 0
        self.frame_id = 0
        self.start_frame = 0
        self.end_frame = 0

    # ---- geometry ------------------------------------------------- #
    @staticmethod
    def tlwh_to_xywh(tlwh):
        ret = np.asarray(tlwh, dtype=np.float32).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @staticmethod
    def xywh_to_tlwh(xywh):
        ret = xywh.copy()
        ret[:2] -= ret[2:] / 2
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
        multi_mean, multi_cov = BYTrack.shared_kalman.multi_predict(
            multi_mean, multi_cov
        )
        for i, (m, c) in enumerate(zip(multi_mean, multi_cov)):
            tracks[i].mean = m
            tracks[i].covariance = c

    def activate(self, kalman_filter, frame_id):
        self.kalman_filter = kalman_filter
        self.track_id = _IDCounter.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xywh(self._tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = frame_id == 1
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh)
        )
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
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    def __repr__(self):
        return f"BYTrack_{self.track_id}_({self.start_frame}-{self.end_frame})"


# ------------------------------------------------------------------ #
# IoU helpers
# ------------------------------------------------------------------ #
def iou_distance(atracks, btracks):
    """Return (1 - IoU) cost matrix."""

    def to_tlbr(tracks):
        return np.array([t.tlbr for t in tracks], dtype=np.float32)

    if not atracks or not btracks:
        return np.zeros((len(atracks), len(btracks)), dtype=np.float32)

    atlbr = to_tlbr(atracks)
    btlbr = to_tlbr(btracks)

    # Vectorised IoU
    xx1 = np.maximum(atlbr[:, None, 0], btlbr[None, :, 0])
    yy1 = np.maximum(atlbr[:, None, 1], btlbr[None, :, 1])
    xx2 = np.minimum(atlbr[:, None, 2], btlbr[None, :, 2])
    yy2 = np.minimum(atlbr[:, None, 3], btlbr[None, :, 3])
    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    a_area = ((atlbr[:, 2] - atlbr[:, 0]) * (atlbr[:, 3] - atlbr[:, 1]))[:, None]
    b_area = ((btlbr[:, 2] - btlbr[:, 0]) * (btlbr[:, 3] - btlbr[:, 1]))[None, :]
    union = a_area + b_area - inter
    iou = inter / (union + 1e-6)
    return 1.0 - iou


def linear_assignment(cost_matrix, thresh):
    """Hungarian solver; reject assignments > thresh."""
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
# ByteTracker
# ------------------------------------------------------------------ #
class ByteTracker:
    """
    ByteTrack tracker faithful to Zhang et al. ECCV 2022.

    Parameters (from original paper / source):
    - track_thresh  = 0.5   (high-score detection threshold)
    - track_buffer  = 30    (adapted to 5 for daily-interval)
    - match_thresh  = 0.8   (IoU threshold for Stage 1)

    Note: because LeafTrackNet proposals use uniform confidence=1.0,
    all detections enter the high-score stage. The low-score stage
    is preserved for completeness but is inactive in this setting.
    """

    def __init__(self, track_thresh=0.5, track_buffer=5, match_thresh=0.8):
        _IDCounter.reset()
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []

        self.frame_id = 0
        self.track_thresh = track_thresh
        self.det_thresh = track_thresh + 0.1
        self.max_time_lost = track_buffer
        self.match_thresh = match_thresh
        self.kalman_filter = KalmanFilterXYWH()

    def update(self, tlwhs, scores):
        """
        Parameters
        ----------
        tlwhs  : list/array of (x1, y1, w, h)
        scores : list/array of detection confidence scores
        """
        self.frame_id += 1
        activated, refound, lost_new, removed = [], [], [], []

        tlwhs = np.asarray(tlwhs, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)

        # Split into high- and low-score detections
        high_mask = scores > self.track_thresh
        low_mask = (scores > 0.1) & ~high_mask
        dets_high = [BYTrack(t, s) for t, s in zip(tlwhs[high_mask], scores[high_mask])]
        dets_low = [BYTrack(t, s) for t, s in zip(tlwhs[low_mask], scores[low_mask])]

        # Separate confirmed vs unconfirmed
        unconfirmed = [t for t in self.tracked_stracks if not t.is_activated]
        tracked = [t for t in self.tracked_stracks if t.is_activated]

        # ---- Stage 1: high-score detections vs all tracked + lost ----
        pool = joint_stracks(tracked, self.lost_stracks)
        BYTrack.multi_predict(pool)

        dists1 = iou_distance(pool, dets_high)
        m1, u_track, u_det_high = linear_assignment(dists1, self.match_thresh)

        for ti, di in m1:
            t, d = pool[ti], dets_high[di]
            if t.state == TrackState.Tracked:
                t.update(d, self.frame_id)
                activated.append(t)
            else:
                t.re_activate(d, self.frame_id)
                refound.append(t)

        # ---- Stage 2: low-score detections vs remaining tracked ------
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

        # ---- Unconfirmed vs remaining high-score detections ----------
        remaining_dets = [dets_high[i] for i in u_det_high]
        dists3 = iou_distance(unconfirmed, remaining_dets)
        m3, u_unc, u_det2 = linear_assignment(dists3, 0.7)

        for ti, di in m3:
            unconfirmed[ti].update(remaining_dets[di], self.frame_id)
            activated.append(unconfirmed[ti])
        for i in u_unc:
            unconfirmed[i].mark_removed()
            removed.append(unconfirmed[i])

        # ---- Init new tracks -----------------------------------------
        for i in u_det2:
            d = remaining_dets[i]
            if d.score >= self.det_thresh:
                d.activate(self.kalman_filter, self.frame_id)
                activated.append(d)

        # ---- Prune lost tracks past max_time_lost --------------------
        for t in self.lost_stracks:
            if self.frame_id - t.end_frame > self.max_time_lost:
                t.mark_removed()
                removed.append(t)

        # ---- Merge state lists ---------------------------------------
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
