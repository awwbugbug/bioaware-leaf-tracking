# run_bio.py
"""
Biology-Constrained Leaf Tracker — inference on CanolaTrack val set.

Constraint weights can be set via command-line arguments to support ablation:
  --lambda_app  1.0   appearance only  → set others to 0
  --lambda_pos  0.3   + position
  --lambda_area 0.2   + area
  --lambda_life 0.1   + lifecycle

Output goes to outputs/bio/tracks/, compatible with run_eval.py.
"""

import argparse
import os

import torch
import yaml
from models import LeafReIDModel
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
    ap = argparse.ArgumentParser(
        description="BioTracker inference on CanolaTrack val set."
    )
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument(
        "--checkpoint_path",
        default="datasets/CanolaTrack/weights/LeafTrackNet_CanolaTrack.pth",
    )
    ap.add_argument(
        "--proposals_json", default="datasets/CanolaTrack/proposals/det_db_val.json"
    )
    ap.add_argument("--image_root", default="datasets/CanolaTrack/CanolaTrack/val")
    ap.add_argument("--output_dir", default="outputs/bio")
    ap.add_argument("--image_ext", default=".jpg")

    # tracker hyperparameters
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--max_age", type=int, default=5)
    ap.add_argument("--update_mode", default="mean", choices=["mean", "ema"])
    ap.add_argument("--alpha", type=float, default=0.5)

    # constraint weights  (set to 0.0 to disable for ablation)
    ap.add_argument("--lambda_app", type=float, default=1.0)
    ap.add_argument("--lambda_pos", type=float, default=0.3)
    ap.add_argument("--lambda_area", type=float, default=0.2)
    ap.add_argument("--lambda_life", type=float, default=0.1)

    # biology parameters
    ap.add_argument("--max_norm_dist", type=float, default=0.25)
    ap.add_argument("--area_shrink_tol", type=float, default=0.30)
    ap.add_argument("--area_penalty_scale", type=float, default=1.0)
    return ap.parse_args()


def main():
    args = parse_args()

    # load YAML config for backbone / embed_dim
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load ReID model
    model = LeafReIDModel(
        cfg["backbone"], embed_dim=cfg["embed_dim"], pretrained=False
    ).to(device)
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    tracker_kwargs = dict(
        reid_model=model,
        transform=get_infer_transform(),
        device=str(device),
        similarity_threshold=args.threshold,
        max_age=args.max_age,
        update_mode=args.update_mode,
        alpha=args.alpha,
        lambda_app=args.lambda_app,
        lambda_pos=args.lambda_pos,
        lambda_area=args.lambda_area,
        lambda_life=args.lambda_life,
        max_norm_dist=args.max_norm_dist,
        area_shrink_tol=args.area_shrink_tol,
        area_penalty_scale=args.area_penalty_scale,
    )

    out_tracks = os.path.join(args.output_dir, "tracks")
    ensure_dir(out_tracks)

    det_dict = read_detection_json(args.proposals_json)

    # group keys by plant
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
