# train_weight_predictor.py
"""
Train the Adaptive Constraint Weight Predictor.

Loss: Pairwise ranking loss (hinge)
  L = mean( max(0, λ·c_pos - λ·c_neg + margin) )

where λ = WeightPredictor(f) and c is the 4-dim cost vector.

This enforces: weighted cost of a correct match < weighted cost of an
incorrect match, by at least `margin`. The predictor learns to assign
higher weights to the constraints that are most discriminative given
the geometric context of each pair.

Usage
-----
python3 train_weight_predictor.py \
    --train_root datasets/CanolaTrack/CanolaTrack/train \
    --output_dir outputs/weight_predictor \
    --epochs 100 \
    --lr 1e-3
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.optim as optim
from datasets.association_dataset import AssociationDataset
from models.weight_predictor import WeightPredictor
from torch.utils.data import DataLoader
from tqdm import tqdm


def ranking_loss(
    predictor: WeightPredictor,
    feat_pos: torch.Tensor,  # [B, 5]
    cost_pos: torch.Tensor,  # [B, 4]
    feat_neg: torch.Tensor,  # [B, 5]
    cost_neg: torch.Tensor,  # [B, 4]
    margin: float,
) -> torch.Tensor:
    """
    Pairwise ranking loss.

    λ_pos = predictor(f_pos),  λ_neg = predictor(f_neg)
    score_pos = (λ_pos * c_pos).sum(dim=1)
    score_neg = (λ_neg * c_neg).sum(dim=1)
    L = mean( max(0, score_pos - score_neg + margin) )

    Note: we use the pair-specific weights (λ depends on f, not just c),
    reflecting that the optimal constraint weighting is context-dependent.
    """
    lambda_pos = predictor(feat_pos)  # [B, 4]
    lambda_neg = predictor(feat_neg)  # [B, 4]

    score_pos = (lambda_pos * cost_pos).sum(dim=1)  # [B]
    score_neg = (lambda_neg * cost_neg).sum(dim=1)  # [B]

    loss = torch.clamp(score_pos - score_neg + margin, min=0.0)
    return loss.mean()


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    dataset = AssociationDataset(
        train_root=args.train_root,
        image_size=args.image_size,
        max_age=args.max_age,
        max_norm_dist=args.max_norm_dist,
        area_shrink_tol=args.area_shrink_tol,
        length=args.dataset_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Dataset: {len(dataset.samples)} raw pairs → virtual length {len(dataset)}")

    # model
    predictor = WeightPredictor(
        input_dim=5,
        hidden_dim=args.hidden_dim,
        num_constraints=4,
        lambda_max=args.lambda_max,
    ).to(device)
    print(f"WeightPredictor params: {sum(p.numel() for p in predictor.parameters())}")

    optimizer = optim.Adam(predictor.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        predictor.train()
        running, seen = 0.0, 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for feat_pos, cost_pos, feat_neg, cost_neg in pbar:
            feat_pos = feat_pos.to(device)
            cost_pos = cost_pos.to(device)
            feat_neg = feat_neg.to(device)
            cost_neg = cost_neg.to(device)

            optimizer.zero_grad()
            loss = ranking_loss(
                predictor,
                feat_pos,
                cost_pos,
                feat_neg,
                cost_neg,
                margin=args.margin,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
            optimizer.step()

            bs = feat_pos.size(0)
            running += loss.item() * bs
            seen += bs
            pbar.set_postfix({"loss": f"{running / seen:.6f}"})

        scheduler.step()
        epoch_loss = running / seen

        # save best
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                predictor.state_dict(),
                os.path.join(args.output_dir, "weight_predictor_best.pth"),
            )

        # periodic save
        if epoch % args.save_period == 0 or epoch == args.epochs:
            torch.save(
                predictor.state_dict(),
                os.path.join(args.output_dir, f"weight_predictor_e{epoch}.pth"),
            )

    print(f"\nTraining complete. Best loss: {best_loss:.6f}")
    print(f"Weights saved to {args.output_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="datasets/CanolaTrack/CanolaTrack/train")
    ap.add_argument("--output_dir", default="outputs/weight_predictor")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--lambda_max", type=float, default=2.0)
    ap.add_argument("--image_size", type=int, default=1200)
    ap.add_argument("--max_age", type=int, default=5)
    ap.add_argument("--max_norm_dist", type=float, default=0.25)
    ap.add_argument("--area_shrink_tol", type=float, default=0.30)
    ap.add_argument("--dataset_length", type=int, default=20000)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--save_period", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import random

    import numpy as np

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train(args)
