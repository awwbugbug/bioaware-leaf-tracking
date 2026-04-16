# run_bio_adaptive.py
"""
Inference with Adaptive Biology-Constrained Tracker.

Loads both the ReID model and the trained WeightPredictor,
runs tracking on CanolaTrack val set, outputs to outputs/bio_adaptive/.
"""

import argparse
import os

import torch
import yaml
from models import LeafReIDModel
from models.weight_predictor import WeightPredictor
from PIL import Image
from tqdm import tqdm
from trackers.bio_tracker import BioTracker
from utils.io import (
    ensure_dir,
    key_to_plant_frame,
    parse_detection_lines,
    read_detection_json,
    write_mot,
)
from utils.transforms import get_infer_transform


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument(
        "--checkpoint_path",
        default="datasets/CanolaTrack/weights/LeafTrackNet_CanolaTrack.pth",
    )
    ap.add_argument(
        "--wp_checkpoint",
        default="outputs/weight_predictor/weight_predictor_best.pth",
        help="Path to trained WeightPredictor weights",
    )
    ap.add_argument(
        "--proposals_json", default="datasets/CanolaTrack/proposals/det_db_val.json"
    )
    ap.add_argument("--image_root", default="datasets/CanolaTrack/CanolaTrack/val")
    ap.add_argument("--output_dir", default="outputs/bio_adaptive")
    ap.add_argument("--image_ext", default=".jpg")
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--max_age", type=int, default=5)
    ap.add_argument("--update_mode", default="mean")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--total_days", type=int, default=31)
    ap.add_argument(
        "--lambda_max",
        type=float,
        default=2.0,
        help="Must match the value used during WP training",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ReID model
    reid_model = LeafReIDModel(
        cfg["backbone"], embed_dim=cfg["embed_dim"], pretrained=False
    ).to(device)
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    reid_model.load_state_dict(ckpt)
    reid_model.eval()

    # Weight predictor
    weight_predictor = WeightPredictor(
        input_dim=5,
        hidden_dim=64,
        num_constraints=4,
        lambda_max=args.lambda_max,
    ).to(device)
    wp_ckpt = torch.load(args.wp_checkpoint, map_location=device)
    weight_predictor.load_state_dict(wp_ckpt)
    weight_predictor.eval()
    print(f"Loaded WeightPredictor from {args.wp_checkpoint}")

    tracker_kwargs = dict(
        reid_model=reid_model,
        transform=get_infer_transform(),
        device=str(device),
        similarity_threshold=args.threshold,
        max_age=args.max_age,
        update_mode=args.update_mode,
        alpha=args.alpha,
        weight_predictor=weight_predictor,
        total_days=args.total_days,
    )

    out_tracks = os.path.join(args.output_dir, "tracks")
    ensure_dir(out_tracks)

    det_dict = read_detection_json(args.proposals_json)
    plants: dict = {}
    for key in det_dict:
        plant, _ = key_to_plant_frame(key)
        plants.setdefault(plant, []).append(key)

    for plant, keys in tqdm(sorted(plants.items()), desc="Plants"):
        keys = sorted(keys, key=lambda k: key_to_plant_frame(k)[1])
        tracker = BioTracker(**tracker_kwargs)
        lines = []
        for k in keys:
            boxes, _ = parse_detection_lines(det_dict[k])
            plant_id, frame = key_to_plant_frame(k)
            img_path = os.path.join(
                args.image_root, plant_id, "img", f"{frame:08d}{args.image_ext}"
            )
            if not os.path.exists(img_path):
                continue
            img = Image.open(img_path).convert("RGB")
            ids, sims = tracker.update(img, boxes)
            for (x1, y1, x2, y2), tid, s in zip(boxes, ids, sims):
                line = (
                    f"{frame}, {int(tid)}, "
                    f"{x1:.2f}, {y1:.2f}, {x2 - x1:.2f}, {y2 - y1:.2f}, "
                    f"{float(s):.4f}, -1, -1, -1"
                )
                lines.append(line)
        write_mot(os.path.join(out_tracks, f"{plant}.txt"), lines)

    print(f"Done. Results in {out_tracks}")


if __name__ == "__main__":
    main()
