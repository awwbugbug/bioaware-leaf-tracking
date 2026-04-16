---
task_categories:
- image-to-image
- image-feature-extraction
- object-detection
language:
- en
tags:
- plant
- precision agriculture
- plant phenotyping
- tracking
size_categories:
- 10B<n<100B
pretty_name: CanolaTrack
---

# CanolaTrack

**CanolaTrack** is a curated dataset for **leaf-level multi-object tracking (MOT)** and **detection** from top-down RGB imagery of *Brassica napus* (canola) plants. Each sequence records a single plant over time; frames contain annotated **bounding boxes** with **persistent leaf IDs** for tracking. 

- For baseline methods and a reference pipeline built on CanolaTrack, see **LeafTrackNet** (training, inference, and TrackEval integration) in our [Github repo](https://github.com/shl-shawn/LeafTrackNet).

---

## Dataset Summary

- **Domain:** Plant phenotyping (leaf-level analysis, time series)
- **Modalities:** RGB images (top-down)
- **Use cases:** Multi-object tracking (leaf IDs), detection, re-identification
- **Content:** Sequences of a single plant over days; each frame has MOT-style annotations
- **Annotations:** `gt/gt.txt` per sequence with **frame**, **leaf_id**, **x**, **y**, **w**, **h** (pixels)
- **Extras:** YOLOv10 **proposals JSONs** and **LeafTrackNet model weights**for reproducible tracking baselines

---

## Repository Structure
```
CanolaTrack/ 
│  ├── train/
│  │   └── <plant_id>/
│  │         ├── gt/gt.txt # CSV: frame,id,x,y,w,h,,,*
│  │         └── img/{frame:08d}.jpg
│  └──val/
│     └── <plant_id>/
│            ├── gt/gt.txt
│            └── img/{frame:08d}.jpg
proposals/ # detection proposals for standardized benchmarking
│     ├── det_db_train.json
│     └── det_db_val.json
weights/ # detctors and tracker weights
      └── <files>
```

## Supported Tasks and Benchmarks

- **Multi-Object Tracking (MOT)** at the **leaf** level  
- **Object Detection** (per-frame leaf boxes)
- **Leaf Segmentation** (per-frame leaf masks)

---

## How to Cite
Please cite the dataset and the accompanying papers:

```bib
@article{leaftracknet2025,
  title={LeafTrackNet: A Deep Learning Framework for Robust Leaf Tracking in Top-Down Plant Phenotyping},
  year={2025},
  author = {},
  url    = {}
}
```

> CanolaTrack dataset© BASF SE 2025. This dataset may be freely used for non-commercial research and educational purposes.

