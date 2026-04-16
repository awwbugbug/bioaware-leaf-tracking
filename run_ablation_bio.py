# run_ablation_bio.py
"""
Systematic ablation over biology constraints.

Runs four configurations in order:
  A1 — appearance only
  A2 — appearance + position
  A3 — appearance + position + area
  A4 — appearance + position + area + lifecycle  (full model)

Results are saved to outputs/ablation/<tag>/tracks/ and evaluated
automatically. A summary table is printed at the end.
"""

import os
import subprocess
import sys

CHECKPOINT = "datasets/CanolaTrack/weights/LeafTrackNet_CanolaTrack.pth"
PROPOSALS = "datasets/CanolaTrack/proposals/det_db_val.json"
IMAGE_ROOT = "datasets/CanolaTrack/CanolaTrack/val"

ABLATIONS = [
    {
        "tag": "A1_app_only",
        "lambda_app": 1.0,
        "lambda_pos": 0.0,
        "lambda_area": 0.0,
        "lambda_life": 0.0,
    },
    {
        "tag": "A2_app_pos",
        "lambda_app": 1.0,
        "lambda_pos": 0.3,
        "lambda_area": 0.0,
        "lambda_life": 0.0,
    },
    {
        "tag": "A3_app_pos_area",
        "lambda_app": 1.0,
        "lambda_pos": 0.3,
        "lambda_area": 0.2,
        "lambda_life": 0.0,
    },
    {
        "tag": "A4_full",
        "lambda_app": 1.0,
        "lambda_pos": 0.3,
        "lambda_area": 0.2,
        "lambda_life": 0.1,
    },
]


def run_inference(cfg: dict) -> str:
    out_dir = os.path.join("outputs", "ablation", cfg["tag"])
    cmd = [
        sys.executable,
        "run_bio.py",
        "--checkpoint_path",
        CHECKPOINT,
        "--proposals_json",
        PROPOSALS,
        "--image_root",
        IMAGE_ROOT,
        "--output_dir",
        out_dir,
        "--lambda_app",
        str(cfg["lambda_app"]),
        "--lambda_pos",
        str(cfg["lambda_pos"]),
        "--lambda_area",
        str(cfg["lambda_area"]),
        "--lambda_life",
        str(cfg["lambda_life"]),
    ]
    print(f"\n{'=' * 60}")
    print(f"Running: {cfg['tag']}")
    print(
        f"  λ_app={cfg['lambda_app']}  λ_pos={cfg['lambda_pos']}"
        f"  λ_area={cfg['lambda_area']}  λ_life={cfg['lambda_life']}"
    )
    print(f"{'=' * 60}")
    subprocess.run(cmd, check=True)
    return out_dir


def run_eval(out_dir: str, tag: str) -> dict:
    """Run TrackEval and parse COMBINED metrics."""
    import sys

    sys.path.insert(0, os.path.abspath("TrackEval"))
    import trackeval  # noqa: E402

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["USE_PARALLEL"] = False
    eval_config["PRINT_CONFIG"] = False
    eval_config["PLOT_CURVES"] = False
    eval_config["PRINT_RESULTS"] = False

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config["GT_FOLDER"] = IMAGE_ROOT
    dataset_config["TRACKERS_FOLDER"] = out_dir
    dataset_config["TRACKERS_TO_EVAL"] = ["tracks"]
    dataset_config["TRACKER_SUB_FOLDER"] = ""
    dataset_config["SKIP_SPLIT_FOL"] = True
    dataset_config["SPLIT_TO_EVAL"] = "val"
    dataset_config["SEQMAP_FILE"] = os.path.abspath("seqmap")
    dataset_config["PRINT_CONFIG"] = False

    metrics_list = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR(),
        trackeval.metrics.Identity(),
    ]

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    results, _ = evaluator.evaluate(dataset_list, metrics_list)

    # extract COMBINED scalars
    combined = results["MotChallenge2DBox"]["tracks"]["COMBINED_SEQ"]["pedestrian"]
    hota = float(np.mean(combined["HOTA"]["HOTA"]))
    deta = float(np.mean(combined["HOTA"]["DetA"]))
    assa = float(np.mean(combined["HOTA"]["AssA"]))
    mota = float(combined["CLEAR"]["MOTA"])
    idf1 = float(combined["Identity"]["IDF1"])
    idsw = int(combined["CLEAR"]["IDSW"])
    return dict(
        tag=tag, HOTA=hota, DetA=deta, AssA=assa, MOTA=mota, IDF1=idf1, IDSW=idsw
    )


def print_table(rows: list) -> None:
    header = f"{'Config':<22} {'HOTA':>7} {'DetA':>7} {'AssA':>7} {'MOTA':>7} {'IDF1':>7} {'IDSW':>6}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['tag']:<22} {r['HOTA']:>7.3f} {r['DetA']:>7.3f} "
            f"{r['AssA']:>7.3f} {r['MOTA']:>7.3f} {r['IDF1']:>7.3f} {r['IDSW']:>6d}"
        )
    print(sep)


if __name__ == "__main__":
    import numpy as np  # needed inside run_eval

    rows = []
    for cfg in ABLATIONS:
        out_dir = run_inference(cfg)
        metrics = run_eval(out_dir, cfg["tag"])
        rows.append(metrics)
        print(
            f"  → HOTA={metrics['HOTA']:.3f}  AssA={metrics['AssA']:.3f}"
            f"  IDF1={metrics['IDF1']:.3f}  IDSW={metrics['IDSW']}"
        )

    print_table(rows)

    # save to file
    os.makedirs("results", exist_ok=True)
    with open("results/ablation_bio.txt", "w") as f:
        header = f"{'Config':<22} {'HOTA':>7} {'DetA':>7} {'AssA':>7} {'MOTA':>7} {'IDF1':>7} {'IDSW':>6}\n"
        f.write(header)
        for r in rows:
            f.write(
                f"{r['tag']:<22} {r['HOTA']:>7.3f} {r['DetA']:>7.3f} "
                f"{r['AssA']:>7.3f} {r['MOTA']:>7.3f} {r['IDF1']:>7.3f} {r['IDSW']:>6d}\n"
            )
    print("\nSaved to results/ablation_bio.txt")
