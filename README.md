# Biology-Constrained Leaf Tracking with Adaptive Constraint Weighting

Official implementation of:

> **Biology-Constrained Long-Term Leaf Tracking with Adaptive Constraint Weighting for Top-Down Plant Phenotyping**
> Dongze Zhou, Cong Zhang
> *Plant Methods* (under review)

---

## Overview

We propose a biology-constrained association framework for long-term leaf tracking in top-down plant phenotyping. The method augments appearance-based matching with four explicit growth priors encoded as cost terms in a unified Hungarian assignment matrix:

- **Spatial continuity** — leaf centroids displace slowly between daily observations
- **Area monotonicity** — leaf area is biologically non-decreasing during vegetative growth
- **Lifecycle plausibility** — reappearance of absent tracks is constrained by spatial proximity
- **Appearance similarity** — standard cosine embedding distance

An adaptive **Weight Predictor** (MLP, ~4,000 parameters) is trained with a pairwise ranking loss to predict per-pair constraint weights from geometric features, enabling context-dependent constraint importance.

Evaluated on [CanolaTrack](https://huggingface.co/datasets/shl-shawn/CanolaTrack), our method achieves:

| Method | HOTA | AssA | IDF1 | IDSW |
|--------|-----:|-----:|-----:|-----:|
| LeafTrackNet (baseline) | 88.03±0.24 | 84.07±0.49 | 92.90±0.35 | 228 |
| **Ours** | **89.69±0.10** | **87.26±0.19** | **94.39±0.09** | **126±5** |

Identity switches reduced by **44.7%** over LeafTrackNet, with no retraining of the detection or re-identification backbone.

---

## Requirements

```bash
pip install torch torchvision scipy numpy matplotlib pillow
```

**TrackEval** (for evaluation):
```bash
git clone https://github.com/JonathonLuiten/TrackEval.git
```

---

## Dataset

Download CanolaTrack from HuggingFace:
```bash
python download_canolatrack.py
```

Or manually: https://huggingface.co/datasets/shl-shawn/CanolaTrack

Expected structure:
```
datasets/CanolaTrack/
├── CanolaTrack/
│   ├── train/   (147 plants)
│   └── val/     (37 plants)
├── proposals/
│   └── det_db_val.json
└── weights/
    └── LeafTrackNet_CanolaTrack.pth
```

The official LeafTrackNet weights and detection proposals are available from:
https://github.com/shl-shawn/LeafTrackNet

---

## Usage

### Run biology-constrained tracker (fixed weights)
```bash
python run_bio.py --output_dir outputs/bio
python run_eval.py  # evaluates outputs/bio
```

### Train adaptive Weight Predictor
```bash
python train_weight_predictor.py --seed 42 --output_dir outputs/wp_seed42
```

### Run adaptive tracker (full method)
```bash
python run_bio_adaptive.py \
    --wp_path outputs/wp_seed42/weight_predictor_best.pth \
    --output_dir outputs/bio_adaptive_seed42
python run_eval_adaptive.py
```

### Run general MOT baselines
```bash
python run_baselines.py --tracker bytetrack
python run_baselines.py --tracker deepsort
python run_baselines.py --tracker botsort
python run_baselines.py --tracker all
```

### Reproduce ablation study
```bash
python run_ablation_bio.py
```

### Reproduce sensitivity analysis
```bash
python sensitivity_analysis.py
```

---

## Repository Structure

```
├── trackers/
│   ├── bio_tracker.py          # Fixed-weight biology-constrained tracker
│   └── ...
├── baselines/
│   ├── deepsort_tracker.py     # DeepSORT association-only variant
│   ├── bytetrack_tracker.py    # ByteTrack association-only variant
│   ├── botsort_tracker.py      # BoT-SORT association-only variant
│   ├── kalman_filter_xyah.py   # Kalman filter for DeepSORT
│   └── kalman_filter_xywh.py   # Kalman filter for ByteTrack/BoT-SORT
├── models/
│   ├── weight_predictor.py     # Adaptive Weight Predictor MLP
│   └── leaf_reid.py            # LeafTrackNet ReID model
├── datasets/
│   └── association_dataset.py  # Training data for Weight Predictor
├── run_bio.py                  # Fixed-weight tracker inference
├── run_bio_adaptive.py         # Adaptive tracker inference
├── run_baselines.py            # General MOT baseline runner
├── train_weight_predictor.py   # Weight Predictor training
├── run_eval.py                 # TrackEval evaluation
├── sensitivity_analysis.py     # Hyperparameter sensitivity
└── analyze_dataset.py          # Dataset statistics
```

---

## Key Results

### Comparison with state-of-the-art

| Domain | Method | HOTA | DetA | AssA | MOTA | IDF1 | IDSW |
|--------|--------|-----:|-----:|-----:|-----:|-----:|-----:|
| Naive | Centroid+Hungarian | 67.99 | 91.93 | 50.36 | 90.48 | 67.44 | 438 |
| Naive | IoU+Hungarian | 77.50 | 92.19 | 65.23 | 91.16 | 78.05 | 395 |
| General | DeepSORT | 54.44 | 59.54 | 49.45 | 70.87 | 69.76 | 376 |
| General | ByteTrack | 58.44 | 67.89 | 49.37 | 77.87 | 69.75 | 277 |
| General | BoT-SORT | 58.44 | 67.89 | 49.37 | 77.87 | 69.75 | 278 |
| Plant | LeTra | 67.02±0.04 | 82.03±0.14 | 54.98±0.16 | 82.09±0.19 | 69.06±0.10 | — |
| Plant | LeafTrackNet | 88.03±0.24 | 92.25±0.03 | 84.07±0.49 | 93.64±0.18 | 92.90±0.35 | 228 |
| Plant | **Ours** | **89.69±0.10** | 92.25±0.03 | **87.26±0.19** | **95.39±0.08** | **94.39±0.09** | **126±5** |

Results reported as mean±std over 3 independent training runs (seeds 42, 123, 456).

---

## Citation

```bibtex
@article{zhou2025bioleaftrack,
  title   = {Biology-Constrained Long-Term Leaf Tracking with
             Adaptive Constraint Weighting for Top-Down Plant Phenotyping},
  author  = {Zhou, Dongze and Zhang, Cong},
  journal = {Plant Methods},
  year    = {2025},
  note    = {Under review}
}
```

---

## Acknowledgements

This work builds on [LeafTrackNet](https://github.com/shl-shawn/LeafTrackNet) and
[CanolaTrack](https://huggingface.co/datasets/shl-shawn/CanolaTrack).
Evaluation uses [TrackEval](https://github.com/JonathonLuiten/TrackEval).
