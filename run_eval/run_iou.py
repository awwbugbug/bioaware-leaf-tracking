# run_iou.py
"""
用 IoU + Hungarian tracker 在 CanolaTrack val 集上推理。
结果保存到 outputs/iou/tracks/，可直接用 run_eval.py 评估。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tqdm import tqdm
from trackers.iou_tracker import IoUTracker
from utils.io import (
    ensure_dir,
    key_to_plant_frame,
    parse_detection_lines,
    read_detection_json,
    write_mot,
)

PROPOSALS_JSON = "datasets/CanolaTrack/proposals/det_db_val.json"
OUTPUT_DIR = "outputs/iou/tracks"

ensure_dir(OUTPUT_DIR)
det_dict = read_detection_json(PROPOSALS_JSON)

# 按植物分组
plants = {}
for key in det_dict.keys():
    plant, _ = key_to_plant_frame(key)
    plants.setdefault(plant, []).append(key)

for plant, keys in tqdm(plants.items(), desc="Plants"):
    keys = sorted(keys, key=lambda k: key_to_plant_frame(k)[1])
    tracker = IoUTracker(max_age=5, min_iou=0.1)
    lines = []
    for k in keys:
        boxes, _ = parse_detection_lines(det_dict[k])
        plant_id, frame = key_to_plant_frame(k)
        ids, sims = tracker.update(boxes)
        for (x1, y1, x2, y2), tid, s in zip(boxes, ids, sims):
            line = f"{frame}, {int(tid)}, {x1:.2f}, {y1:.2f}, {x2 - x1:.2f}, {y2 - y1:.2f}, {float(s):.4f}, -1, -1, -1"
            lines.append(line)
    write_mot(os.path.join(OUTPUT_DIR, f"{plant}.txt"), lines)

print("Done. Results in", OUTPUT_DIR)
