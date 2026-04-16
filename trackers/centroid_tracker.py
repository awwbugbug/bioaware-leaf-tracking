# trackers/centroid_tracker.py
"""
Centroid + Hungarian baseline tracker.
输出格式与 LeafTrackNet 完全一致，可直接用 run_eval.py 评估。
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


class CentroidTracker:
    def __init__(self, max_age=5, max_dist=200):
        """
        max_age:  track 连续未匹配多少帧后删除
        max_dist: 质心距离超过此值视为不可匹配
        """
        self.max_age = int(max_age)
        self.max_dist = float(max_dist)

        self.tracks = []  # list of dict: id, cx, cy, age
        self.next_id = 1

    def _get_centroid(self, box):
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def update(self, detections):
        """
        detections: List[[x1, y1, x2, y2]]
        returns: ids (List[int]), sims (List[float])
        """
        if len(detections) == 0:
            for t in self.tracks:
                t["age"] += 1
            self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]
            return [], []

        det_centroids = [self._get_centroid(b) for b in detections]

        if len(self.tracks) == 0:
            ids, sims = [], []
            for cx, cy in det_centroids:
                self.tracks.append({"id": self.next_id, "cx": cx, "cy": cy, "age": 0})
                ids.append(self.next_id)
                sims.append(1.0)
                self.next_id += 1
            return ids, sims

        # 构建距离矩阵 T x N
        T = len(self.tracks)
        N = len(det_centroids)
        dist_matrix = np.zeros((T, N))
        for i, t in enumerate(self.tracks):
            for j, (cx, cy) in enumerate(det_centroids):
                dist_matrix[i, j] = np.sqrt((t["cx"] - cx) ** 2 + (t["cy"] - cy) ** 2)

        t_inds, d_inds = linear_sum_assignment(dist_matrix)

        assigned = [-1] * N
        sims = [0.0] * N
        touched = set()

        for ti, di in zip(t_inds, d_inds):
            if dist_matrix[ti, di] <= self.max_dist:
                assigned[di] = self.tracks[ti]["id"]
                sims[di] = 1.0 - dist_matrix[ti, di] / self.max_dist
                cx, cy = det_centroids[di]
                self.tracks[ti]["cx"] = cx
                self.tracks[ti]["cy"] = cy
                self.tracks[ti]["age"] = 0
                touched.add(ti)

        # 未匹配检测初始化新 track
        for di in range(N):
            if assigned[di] == -1:
                cx, cy = det_centroids[di]
                self.tracks.append({"id": self.next_id, "cx": cx, "cy": cy, "age": 0})
                assigned[di] = self.next_id
                sims[di] = 1.0
                self.next_id += 1

        # 未匹配 track 增加 age
        new_tracks = []
        for i, t in enumerate(self.tracks):
            if i not in touched:
                t["age"] += 1
            if t["age"] <= self.max_age:
                new_tracks.append(t)
        self.tracks = new_tracks

        return assigned, sims
