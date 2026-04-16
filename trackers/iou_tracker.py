# trackers/iou_tracker.py
"""
IoU + Hungarian baseline tracker.
输出格式与 LeafTrackNet 完全一致，可直接用 run_eval.py 评估。
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


class IoUTracker:
    def __init__(self, max_age=5, min_iou=0.1):
        """
        max_age: track 连续未匹配多少帧后删除
        min_iou: IoU 低于此值视为不可匹配
        """
        self.max_age = int(max_age)
        self.min_iou = float(min_iou)

        self.tracks = []  # list of dict: id, box, age
        self.next_id = 1

    def _iou(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

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

        if len(self.tracks) == 0:
            ids, sims = [], []
            for box in detections:
                self.tracks.append({"id": self.next_id, "box": box, "age": 0})
                ids.append(self.next_id)
                sims.append(1.0)
                self.next_id += 1
            return ids, sims

        # 构建 IoU 矩阵 T x N
        T = len(self.tracks)
        N = len(detections)
        iou_matrix = np.zeros((T, N))
        for i, t in enumerate(self.tracks):
            for j, box in enumerate(detections):
                iou_matrix[i, j] = self._iou(t["box"], box)

        # 转为代价矩阵
        cost_matrix = 1.0 - iou_matrix
        t_inds, d_inds = linear_sum_assignment(cost_matrix)

        assigned = [-1] * N
        sims = [0.0] * N
        touched = set()

        for ti, di in zip(t_inds, d_inds):
            if iou_matrix[ti, di] >= self.min_iou:
                assigned[di] = self.tracks[ti]["id"]
                sims[di] = iou_matrix[ti, di]
                self.tracks[ti]["box"] = detections[di]
                self.tracks[ti]["age"] = 0
                touched.add(ti)

        # 未匹配检测初始化新 track
        for di in range(N):
            if assigned[di] == -1:
                self.tracks.append(
                    {"id": self.next_id, "box": detections[di], "age": 0}
                )
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
