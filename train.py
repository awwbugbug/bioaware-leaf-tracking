import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import yaml
from datasets import (
    LeafTripletDataset,
    LeafTripletDatasetV2,
    LeafTripletDatasetV3,
    LeafTripletDatasetV4,
)
from datasets.growth_reg_triplets import LeafTripletDatasetV5
from models import LeafReIDModel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from utils.dist import get_local_rank, is_ddp, is_main_process, set_seed
from utils.transforms import get_train_transform


def build_dataset(cfg):
    tfm = get_train_transform()
    strategy = cfg["triplet_strategy"].lower()

    if strategy == "v1":
        return LeafTripletDataset(cfg["train_root"], transform=tfm)
    if strategy == "v2":
        return LeafTripletDatasetV2(cfg["train_root"], transform=tfm)
    if strategy == "v3":
        return LeafTripletDatasetV3(
            cfg["train_root"],
            window_size=cfg["window_size"],
            transform=tfm,
        )
    if strategy == "v4":
        return LeafTripletDatasetV4(
            cfg["train_root"],
            gamma_area=cfg.get("gamma_area", 2.0),
            gamma_spatial=cfg.get("gamma_spatial", 2.0),
            image_size=cfg.get("image_size", 1200),
            length=cfg.get("dataset_length", 10000),
            transform=tfm,
        )
    if strategy == "v5":
        return LeafTripletDatasetV5(
            cfg["train_root"],
            length=cfg.get("dataset_length", 10000),
            transform=tfm,
        )
    raise ValueError(
        f"triplet_strategy must be one of [v1, v2, v3, v4, v5], got '{strategy}'"
    )


def growth_consistency_loss(
    ea: torch.Tensor,
    ep: torch.Tensor,
    area_a: torch.Tensor,
    area_p: torch.Tensor,
    epsilon: float = 0.3,
) -> torch.Tensor:
    """
    Growth-Consistency Regularization Loss.

    For each anchor-positive pair where the leaf has grown (A_p > A_a),
    penalise large embedding distance:

        L_growth = mean( max(0,  g_ap · ‖e_a − e_p‖₂  −  ε) )

    where g_ap = (A_p − A_a) / (A_a + δ) is the relative area growth rate.

    Parameters
    ----------
    ea, ep   : Tensor [B, D]  — anchor and positive embeddings
    area_a   : Tensor [B]     — anchor bbox areas (pixels²)
    area_p   : Tensor [B]     — positive bbox areas (pixels²)
    epsilon  : float          — slack margin (default 0.3)

    Returns
    -------
    Scalar loss tensor.
    """
    # Relative area growth rate: positive when leaf grew, near-zero otherwise
    g_ap = (area_p - area_a) / (area_a + 1e-6)  # [B]
    g_ap = torch.clamp(g_ap, min=0.0, max=1.0)

    # L2 distance between anchor and positive embeddings
    dist_ap = torch.norm(ea - ep, p=2, dim=1)  # [B]

    # Penalise: growing leaf pairs should have small embedding distance
    loss = torch.clamp(g_ap * dist_ap - epsilon, min=0.0)
    return loss.mean()


def train(cfg):
    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # growth-reg hyperparams (only used when strategy == v5)
    alpha_growth = float(cfg.get("alpha_growth", 0.5))
    epsilon_growth = float(cfg.get("epsilon_growth", 0.3))
    use_growth_reg = cfg["triplet_strategy"].lower() == "v5"

    # model
    model = LeafReIDModel(cfg["backbone"], embed_dim=cfg["embed_dim"], pretrained=True)

    if is_ddp():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(local_rank)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    # data
    dataset = build_dataset(cfg)
    if is_ddp():
        sampler = DistributedSampler(dataset, shuffle=True)
        loader = DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            sampler=sampler,
            num_workers=cfg["num_workers"],
            pin_memory=True,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
            pin_memory=True,
        )

    # loss / optimiser
    criterion = nn.TripletMarginLoss(margin=0.3)
    optimizer = optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    # training loop
    for epoch in range(1, cfg["epochs"] + 1):
        if is_ddp():
            sampler.set_epoch(epoch)
        running, seen = 0.0, 0
        running_triplet, running_growth = 0.0, 0.0
        iterator = loader
        if get_local_rank() == 0:
            iterator = tqdm(loader, desc=f"Epoch {epoch}/{cfg['epochs']}")

        for batch in iterator:
            if use_growth_reg:
                a, p, n, area_a, area_p = batch
                a, p, n = a.to(device), p.to(device), n.to(device)
                area_a = area_a.to(device)
                area_p = area_p.to(device)
            else:
                a, p, n = batch
                a, p, n = a.to(device), p.to(device), n.to(device)

            optimizer.zero_grad()
            ea, ep, en = model(a), model(p), model(n)

            l_triplet = criterion(ea, ep, en)

            if use_growth_reg:
                l_growth = growth_consistency_loss(
                    ea, ep, area_a, area_p, epsilon=epsilon_growth
                )
                loss = l_triplet + alpha_growth * l_growth
            else:
                l_growth = torch.tensor(0.0)
                loss = l_triplet

            loss.backward()
            optimizer.step()

            bs = a.size(0)
            running += loss.item() * bs
            running_triplet += l_triplet.item() * bs
            running_growth += l_growth.item() * bs
            seen += bs

            if get_local_rank() == 0:
                postfix = {"loss": f"{running / seen:.4f}"}
                if use_growth_reg:
                    postfix["tri"] = f"{running_triplet / seen:.4f}"
                    postfix["growth"] = f"{running_growth / seen:.4f}"
                iterator.set_postfix(postfix)

        if is_main_process() and (
            epoch % cfg["save_period"] == 0 or epoch == cfg["epochs"]
        ):
            ckpt_dir = os.path.join(cfg["output_dir"], "weights")
            os.makedirs(ckpt_dir, exist_ok=True)
            state = model.module.state_dict() if is_ddp() else model.state_dict()
            torch.save(state, os.path.join(ckpt_dir, f"leaf_reid_e{epoch}.pth"))

    if is_ddp():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train_root")
    parser.add_argument("--output_dir")
    parser.add_argument("--backbone")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--triplet_strategy")
    parser.add_argument("--window_size", type=int)
    parser.add_argument("--gamma_area", type=float)
    parser.add_argument("--gamma_spatial", type=float)
    parser.add_argument(
        "--alpha_growth", type=float, help="Weight of growth-consistency loss (v5 only)"
    )
    parser.add_argument(
        "--epsilon_growth",
        type=float,
        help="Slack margin for growth-consistency loss (v5 only)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    for k, v in vars(args).items():
        if k != "config" and v is not None:
            cfg[k] = v

    train(cfg)
