#!/usr/bin/env python3
"""
eval_v5.py — Isolated evaluation for V5 growth-consistency ablation.

Runs run_bio.py and TrackEval for each alpha variant in its own directory,
prints a summary table at the end.

Usage:
    python eval_v5.py
"""

import os
import subprocess
import sys

sys.path.insert(0, "/root/autodl-tmp/LeafTrackNet-main/TrackEval")
import trackeval

ROOT = "/root/autodl-tmp/LeafTrackNet-main"

VARIANTS = [
    {
        "name": "alpha=0.1",
        "weights": "outputs/v5_alpha0.1_seed42/weights/leaf_reid_e80.pth",
        "out": "outputs/v5_alpha0.1_seed42/tracks",
    },
    {
        "name": "alpha=0.5",
        "weights": "outputs/v5_alpha0.5_seed42/weights/leaf_reid_e80.pth",
        "out": "outputs/v5_alpha0.5_seed42/tracks",
    },
    {
        "name": "alpha=1.0",
        "weights": "outputs/v5_alpha1.0_seed42/weights/leaf_reid_e80.pth",
        "out": "outputs/v5_alpha1.0_seed42/tracks",
    },
]


def run_tracker(weights, output_dir):
    cmd = [
        "python",
        "run_bio.py",
        "--checkpoint_path",
        weights,
        "--output_dir",
        output_dir,
    ]
    print(f"  → Tracking: {weights}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def run_eval(trackers_folder):
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["USE_PARALLEL"] = False
    eval_config["PRINT_CONFIG"] = False
    eval_config["PLOT_CURVES"] = False

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config["GT_FOLDER"] = os.path.join(
        ROOT, "datasets/CanolaTrack/CanolaTrack/val"
    )
    dataset_config["TRACKERS_FOLDER"] = trackers_folder
    dataset_config["TRACKERS_TO_EVAL"] = ["tracks"]
    dataset_config["TRACKER_SUB_FOLDER"] = ""
    dataset_config["SKIP_SPLIT_FOL"] = True
    dataset_config["SPLIT_TO_EVAL"] = "val"
    dataset_config["SEQMAP_FILE"] = os.path.join(ROOT, "seqmap")
    dataset_config["PRINT_CONFIG"] = False

    metrics_list = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR(),
        trackeval.metrics.Identity(),
    ]

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    results, _ = evaluator.evaluate(dataset_list, metrics_list)

    # Extract summary metrics
    res = results["MotChallenge2DBox"]["tracks"]["COMBINED_SEQ"]["pedestrian"]
    hota = res["HOTA"]["HOTA"].mean() * 100
    assa = res["HOTA"]["AssA"].mean() * 100
    idf1 = res["Identity"]["IDF1"] * 100
    idsw = int(res["CLEAR"]["IDSW"])
    deta = res["HOTA"]["DetA"].mean() * 100
    return {"HOTA": hota, "AssA": assa, "IDF1": idf1, "IDSW": idsw, "DetA": deta}


def main():
    results_table = []

    for v in VARIANTS:
        print(f"\n{'=' * 50}")
        print(f"Evaluating {v['name']}")
        print(f"{'=' * 50}")
        os.makedirs(v["out"], exist_ok=True)
        run_tracker(v["weights"], v["out"])
        metrics = run_eval(v["out"])
        results_table.append({"name": v["name"], **metrics})
        print(
            f"  HOTA={metrics['HOTA']:.2f}  AssA={metrics['AssA']:.2f}  "
            f"IDF1={metrics['IDF1']:.2f}  IDSW={metrics['IDSW']}"
        )

    # Summary table
    print(f"\n{'=' * 65}")
    print(f"{'Method':<12} {'HOTA':>7} {'AssA':>7} {'IDF1':>7} {'IDSW':>6}")
    print(f"{'-' * 65}")
    # Baseline reference
    print(f"{'Main(locked)':<12} {'89.69':>7} {'87.26':>7} {'94.39':>7} {'126':>6}")
    for r in results_table:
        print(
            f"{r['name']:<12} {r['HOTA']:>7.2f} {r['AssA']:>7.2f} {r['IDF1']:>7.2f} {r['IDSW']:>6d}"
        )
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
