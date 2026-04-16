"""
run_baselines.py - see inline comments for details
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

BASE = "/root/autodl-tmp/LeafTrackNet-main"
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "TrackEval"))

import trackeval
from baselines.botsort_tracker import BoTSORTTracker
from baselines.bytetrack_tracker import ByteTracker
from baselines.deepsort_tracker import DeepSORTTracker, DSDetection

VAL_ROOT = os.path.join(BASE, "datasets/CanolaTrack/CanolaTrack/val")
PROPOSALS = os.path.join(BASE, "datasets/CanolaTrack/proposals/det_db_val.json")
SEQMAP = os.path.join(BASE, "seqmap")


def load_all_proposals():
    with open(PROPOSALS) as f:
        raw = json.load(f)
    proposals = {}
    for key, lines in raw.items():
        parts = key.replace("\\", "/").split("/")
        plant = parts[1]
        frame_id = int(os.path.splitext(parts[2])[0])
        dets = []
        for line in lines:
            vals = [float(v) for v in line.strip().split(",")]
            if len(vals) >= 5:
                dets.append(vals[:5])
            elif len(vals) == 4:
                dets.append(vals + [1.0])
        if dets:
            proposals.setdefault(plant, {})[frame_id] = dets
    return proposals


def load_embedding_model(device):
    import torchvision.models as tvm

    class LeafEmbedNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            base = tvm.mobilenet_v3_large(weights=None)
            self.feature_extractor = torch.nn.Sequential(base.features)
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.embedding = torch.nn.Linear(960, 128)
            self.bn = torch.nn.BatchNorm1d(128)
        def forward(self, x):
            f = self.pool(self.feature_extractor(x)).flatten(1)
            return self.bn(self.embedding(f))

    ckpt = os.path.join(BASE,
        "datasets/CanolaTrack/weights/LeafTrackNet_CanolaTrack.pth")
    model = LeafEmbedNet()
    state = torch.load(ckpt, map_location=device)
    filtered = {k:v for k,v in state.items()
                if k.startswith("feature_extractor")
                or k.startswith("embedding")
                or k.startswith("bn")}
    model.load_state_dict(filtered, strict=False)
    print(f"[OK] Loaded LeafTrackNet embedding weights")
    return model.eval().to(device)


TRANSFORM = T.Compose(
    [
        T.Resize((128, 64)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def get_feat(model, img_pil, bbox, device):
    if model is None or img_pil is None:
        return np.zeros(128, dtype=np.float32)
    x1, y1, w, h = bbox
    x2, y2 = min(img_pil.width, int(x1 + w)), min(img_pil.height, int(y1 + h))
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    if x2 <= x1 or y2 <= y1:
        return np.zeros(128, dtype=np.float32)
    t = TRANSFORM(img_pil.crop((x1, y1, x2, y2))).unsqueeze(0).to(device)
    with torch.no_grad():
        f = model(t).squeeze().cpu().numpy()
    return (f / (np.linalg.norm(f) + 1e-6)).astype(np.float32)


def load_img(plant_dir, frame_id):
    for ext in [".png", ".jpg"]:
        p = os.path.join(plant_dir, "img", f"{frame_id:08d}{ext}")
        if os.path.exists(p):
            return Image.open(p).convert("RGB")
    return None


def write_mot(results, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for fr, tid, x1, y1, w, h in sorted(results, key=lambda r: (r[0], r[1])):
            f.write(f"{fr},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")


def run_deepsort(proposals, embed_model, device, output_dir):
    print("\n[DeepSORT]")
    for plant in sorted(proposals):
        plant_dir = os.path.join(VAL_ROOT, plant)
        tracker = DeepSORTTracker(
            max_cosine_distance=0.4,
            nn_budget=100,
            max_iou_distance=0.7,
            max_age=5,
            n_init=1,
        )
        results = []
        for frame_id in sorted(proposals[plant]):
            img = load_img(plant_dir, frame_id)
            dets = [
                DSDetection(
                    [d[0], d[1], d[2], d[3]],
                    d[4],
                    get_feat(embed_model, img, [d[0], d[1], d[2], d[3]], device),
                )
                for d in proposals[plant][frame_id]
            ]
            tracker.predict()
            tracker.update(dets)
            for tid, tlwh in tracker.get_results():
                results.append((frame_id, tid, tlwh[0], tlwh[1], tlwh[2], tlwh[3]))
        write_mot(results, os.path.join(output_dir, "tracks", f"{plant}.txt"))
        print(f"  {plant}: {len(results)} rows")


def run_bytetrack(proposals, embed_model, device, output_dir):
    print("\n[ByteTrack]")
    for plant in sorted(proposals):
        tracker = ByteTracker(track_thresh=0.5, track_buffer=5, match_thresh=0.8)
        results = []
        for frame_id in sorted(proposals[plant]):
            dets = proposals[plant][frame_id]
            active = tracker.update(
                [[d[0], d[1], d[2], d[3]] for d in dets], [d[4] for d in dets]
            )
            for t in active:
                tw = t.tlwh
                results.append((frame_id, t.track_id, tw[0], tw[1], tw[2], tw[3]))
        write_mot(results, os.path.join(output_dir, "tracks", f"{plant}.txt"))
        print(f"  {plant}: {len(results)} rows")


def run_botsort(proposals, embed_model, device, output_dir):
    print("\n[BoT-SORT]")
    for plant in sorted(proposals):
        plant_dir = os.path.join(VAL_ROOT, plant)
        tracker = BoTSORTTracker(
            track_high_thresh=0.5,
            track_low_thresh=0.1,
            new_track_thresh=0.6,
            track_buffer=5,
            match_thresh=0.8,
            proximity_thresh=0.5,
            appearance_thresh=0.25,
        )
        results = []
        for frame_id in sorted(proposals[plant]):
            img = load_img(plant_dir, frame_id)
            dets = proposals[plant][frame_id]
            tlwhs = np.array([[d[0], d[1], d[2], d[3]] for d in dets], dtype=np.float32)
            scores = np.array([d[4] for d in dets], dtype=np.float32)
            feats = np.array(
                [
                    get_feat(embed_model, img, [d[0], d[1], d[2], d[3]], device)
                    for d in dets
                ],
                dtype=np.float32,
            )
            active = tracker.update(tlwhs, scores, feats)
            for t in active:
                tw = t.tlwh
                results.append((frame_id, t.track_id, tw[0], tw[1], tw[2], tw[3]))
        write_mot(results, os.path.join(output_dir, "tracks", f"{plant}.txt"))
        print(f"  {plant}: {len(results)} rows")


def evaluate(tracker_folder, tag):
    print(f"\n[Eval] {tag}")
    eval_cfg = trackeval.Evaluator.get_default_eval_config()
    eval_cfg.update(
        {"USE_PARALLEL": False, "PRINT_CONFIG": False, "PLOT_CURVES": False}
    )
    ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update(
        {
            "GT_FOLDER": VAL_ROOT,
            "TRACKERS_FOLDER": tracker_folder,
            "TRACKERS_TO_EVAL": ["tracks"],
            "TRACKER_SUB_FOLDER": "",
            "SKIP_SPLIT_FOL": True,
            "SPLIT_TO_EVAL": "val",
            "SEQMAP_FILE": SEQMAP,
            "PRINT_CONFIG": False,
        }
    )
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res, _ = trackeval.Evaluator(eval_cfg).evaluate(
            [trackeval.datasets.MotChallenge2DBox(ds_cfg)],
            [
                trackeval.metrics.HOTA(),
                trackeval.metrics.CLEAR(),
                trackeval.metrics.Identity(),
            ],
        )
    try:
        c = res["MotChallenge2DBox"]["tracks"]["COMBINED_SEQ"]["pedestrian"]
        r = dict(
            HOTA=float(c["HOTA"]["HOTA"].mean()),
            DetA=float(c["HOTA"]["DetA"].mean()),
            AssA=float(c["HOTA"]["AssA"].mean()),
            MOTA=float(c["CLEAR"]["MOTA"].mean()),
            IDF1=float(c["Identity"]["IDF1"].mean()),
            IDSW=int(c["CLEAR"]["IDSW"].sum()),
        )
        print(
            f"  HOTA={r['HOTA']:.2f} DetA={r['DetA']:.2f} "
            f"AssA={r['AssA']:.2f} MOTA={r['MOTA']:.2f} "
            f"IDF1={r['IDF1']:.2f} IDSW={r['IDSW']}"
        )
        return r
    except Exception as e:
        print(f"  [ERROR] {e}")
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tracker", default="all", choices=["deepsort", "bytetrack", "botsort", "all"]
    )
    ap.add_argument("--output_root", default="outputs")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    proposals = load_all_proposals()
    print(f"Proposals: {len(proposals)} plants")
    embed_model = load_embedding_model(device)

    to_run = (
        ["deepsort", "bytetrack", "botsort"]
        if args.tracker == "all"
        else [args.tracker]
    )
    all_res = {}
    for name in to_run:
        out_dir = os.path.join(args.output_root, name)
        os.makedirs(os.path.join(out_dir, "tracks"), exist_ok=True)
        {"deepsort": run_deepsort, "bytetrack": run_bytetrack, "botsort": run_botsort}[
            name
        ](proposals, embed_model, device, out_dir)
        all_res[name] = evaluate(out_dir, name.upper())

    print(f"\n{'=' * 65}")
    print(
        f"{'Method':<12}{'HOTA':>8}{'DetA':>8}{'AssA':>8}"
        f"{'MOTA':>8}{'IDF1':>8}{'IDSW':>7}"
    )
    print("-" * 65)
    for name, r in all_res.items():
        if r:
            print(
                f"{name:<12}{r['HOTA']:>8.2f}{r['DetA']:>8.2f}"
                f"{r['AssA']:>8.2f}{r['MOTA']:>8.2f}"
                f"{r['IDF1']:>8.2f}{r['IDSW']:>7}"
            )

    os.makedirs("results", exist_ok=True)
    with open("results/baseline_results.txt", "w") as f:
        f.write("Method,HOTA,DetA,AssA,MOTA,IDF1,IDSW\n")
        for name, r in all_res.items():
            if r:
                f.write(
                    f"{name},{r['HOTA']:.2f},{r['DetA']:.2f},"
                    f"{r['AssA']:.2f},{r['MOTA']:.2f},"
                    f"{r['IDF1']:.2f},{r['IDSW']}\n"
                )
    print("Saved -> results/baseline_results.txt")


if __name__ == "__main__":
    main()
