# collect_results.py
"""
Collect and summarise all tracking experiment results.
Reads from results/*.txt files and prints a clean comparison table.

Usage:
    python3 collect_results.py
"""

import os

import numpy as np

RESULTS_DIR = "results"

# ---------------------------------------------------------------
# Result file registry
# Each entry: (display_name, [list of result .txt files])
# If multiple files → compute mean ± std
# ---------------------------------------------------------------
EXPERIMENTS = [
    ("Centroid + Hungarian", ["centroid_results.txt"]),
    ("IoU + Hungarian", ["iou_results.txt"]),
    ("LeafTrackNet", ["baseline_results.txt"]),
    ("+ Fixed constraints", ["bio_results.txt"]),
    (
        "+ Adaptive WP (Ours)",
        [
            "bio_adaptive_seed42_results.txt",
            "bio_adaptive_seed123_results.txt",
            "bio_adaptive_seed456_results.txt",
        ],
    ),
]


# ---------------------------------------------------------------
# Parser
# ---------------------------------------------------------------


def parse_result_file(filepath):
    """
    Extract HOTA, DetA, AssA, MOTA, IDF1, IDSW from a TrackEval result file.
    Returns dict or None if file not found.
    """
    if not os.path.exists(filepath):
        return None

    metrics = {}
    with open(filepath) as f:
        lines = f.readlines()

    for line in lines:
        if "COMBINED" not in line:
            continue
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        vals = []
        for p in parts[1:]:
            try:
                vals.append(float(p))
            except ValueError:
                break

        # HOTA line: HOTA DetA AssA DetRe DetPr AssRe AssPr LocA ...
        if len(vals) >= 3 and 50 < vals[0] < 100 and 50 < vals[1] < 100:
            if "HOTA" not in metrics:
                metrics["HOTA"] = vals[0]
                metrics["DetA"] = vals[1]
                metrics["AssA"] = vals[2]

        # CLEAR line: MOTA MOTP MODA CLR_Re CLR_Pr ... IDSW
        # MOTA is first value, IDSW is at position 12 (0-indexed from vals)
        if len(vals) >= 13 and "MOTA" not in metrics:
            # MOTA is bounded [-inf, 100], typically 80-100 for good trackers
            if -10 < vals[0] < 100 and vals[0] != metrics.get("HOTA"):
                metrics["MOTA"] = vals[0]
                metrics["IDSW"] = int(vals[12])

        # Identity line: IDF1 IDR IDP IDTP IDFN IDFP
        if len(vals) >= 3 and "IDF1" not in metrics:
            if (
                50 < vals[0] < 100
                and vals[0] != metrics.get("HOTA")
                and vals[0] != metrics.get("MOTA")
            ):
                metrics["IDF1"] = vals[0]

    return metrics if len(metrics) >= 5 else None


def collect(name, files):
    """Collect results from one or more files, return mean ± std."""
    all_metrics = []
    for f in files:
        path = os.path.join(RESULTS_DIR, f)
        m = parse_result_file(path)
        if m:
            all_metrics.append(m)
        else:
            print(f"  [WARNING] Could not parse: {path}")

    if not all_metrics:
        return None

    keys = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW"]
    result = {"name": name, "n": len(all_metrics)}
    for k in keys:
        vals = [m[k] for m in all_metrics if k in m]
        if vals:
            result[f"{k}_mean"] = np.mean(vals)
            result[f"{k}_std"] = np.std(vals)
        else:
            result[f"{k}_mean"] = float("nan")
            result[f"{k}_std"] = 0.0
    return result


def fmt(mean, std, is_int=False):
    """Format mean ± std for display."""
    if np.isnan(mean):
        return "--"
    if std < 0.01:
        if is_int:
            return f"{int(mean)}"
        return f"{mean:.2f}"
    if is_int:
        return f"{int(round(mean))}±{int(round(std))}"
    return f"{mean:.2f}±{std:.2f}"


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------


def main():
    print("\n" + "=" * 90)
    print("Experiment Results Summary")
    print("=" * 90)

    header = f"{'Method':<28} {'HOTA':>12} {'DetA':>10} {'AssA':>10} {'MOTA':>10} {'IDF1':>10} {'IDSW':>8}"
    print(header)
    print("-" * 90)

    rows = []
    for name, files in EXPERIMENTS:
        r = collect(name, files)
        if r is None:
            print(f"  {name:<26}  [no data]")
            continue
        rows.append(r)
        print(
            f"  {r['name']:<26} "
            f"{fmt(r['HOTA_mean'], r['HOTA_std']):>12} "
            f"{fmt(r['DetA_mean'], r['DetA_std']):>10} "
            f"{fmt(r['AssA_mean'], r['AssA_std']):>10} "
            f"{fmt(r['MOTA_mean'], r['MOTA_std']):>10} "
            f"{fmt(r['IDF1_mean'], r['IDF1_std']):>10} "
            f"{fmt(r['IDSW_mean'], r['IDSW_std'], is_int=True):>8}"
        )

    print("=" * 90)

    # Save to file
    out_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(out_path, "w") as f:
        f.write(
            f"{'Method':<28} {'HOTA':>12} {'DetA':>10} {'AssA':>10} {'MOTA':>10} {'IDF1':>10} {'IDSW':>8}\n"
        )
        f.write("-" * 90 + "\n")
        for r in rows:
            f.write(
                f"{r['name']:<28} "
                f"{fmt(r['HOTA_mean'], r['HOTA_std']):>12} "
                f"{fmt(r['DetA_mean'], r['DetA_std']):>10} "
                f"{fmt(r['AssA_mean'], r['AssA_std']):>10} "
                f"{fmt(r['MOTA_mean'], r['MOTA_std']):>10} "
                f"{fmt(r['IDF1_mean'], r['IDF1_std']):>10} "
                f"{fmt(r['IDSW_mean'], r['IDSW_std'], is_int=True):>8}\n"
            )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
