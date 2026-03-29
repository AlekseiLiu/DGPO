"""Phase 1: Pretrain diffusion model on NAS benchmark graphs.

Trains a GraphTransformer to denoise corrupted NAS benchmark graphs via
cross-entropy loss on node and edge types.

Usage:
    python src/train_pretrain.py --config configs/pretrain_nb201.yaml [--seed 42] [--epochs 100] [--limit 1000]
    python src/train_pretrain.py --config configs/pretrain_nb201.yaml --resume outputs/.../checkpoints/last.pt
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# Ensure companion/src resolves correctly
_companion_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _companion_root not in sys.path:
    sys.path.insert(0, _companion_root)

from src.dataset import load_nb101, load_nb201, DatasetInfo
from src.diffusion import (
    PredefinedNoiseScheduleDiscrete,
    MarginalUniformTransition,
    apply_noise,
    forward_pass,
)
from src.model import GraphTransformer
from src.utils import PlaceHolder, load_config, namespace_to_dict


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Marginals computation
# ---------------------------------------------------------------------------

def compute_marginals(train_loader, dataset_info: DatasetInfo) -> dict:
    """Compute empirical marginal distributions from training data."""
    node_counts = torch.zeros(dataset_info.node_types)
    edge_counts = torch.zeros(dataset_info.edge_types)
    graph_size_counts = torch.zeros(dataset_info.max_nodes + 1)
    total_nodes = 0
    total_edges = 0

    for batch in train_loader:
        X, E, node_mask = batch["X"], batch["E"], batch["node_mask"]
        bs = X.shape[0]
        for i in range(bs):
            n_nodes = node_mask[i].sum().item()
            graph_size_counts[int(n_nodes)] += 1
            # Node type counts from one-hot X
            node_counts += X[i, :int(n_nodes)].sum(dim=0)
            total_nodes += n_nodes
            # Edge type counts from one-hot E (only real nodes)
            n = int(n_nodes)
            edge_counts += E[i, :n, :n].sum(dim=(0, 1))
            total_edges += n * n

    node_marginals = node_counts / total_nodes if total_nodes > 0 else node_counts
    edge_marginals = edge_counts / total_edges if total_edges > 0 else edge_counts
    size_marginals = graph_size_counts / graph_size_counts.sum() if graph_size_counts.sum() > 0 else graph_size_counts

    return {
        "node_types": node_marginals,
        "edge_types": edge_marginals,
        "node_counts": size_marginals,
    }


# ---------------------------------------------------------------------------
# Loss computation (faithful to TrainLossDiscrete)
# ---------------------------------------------------------------------------

def compute_train_loss(pred_X, pred_E, true_X, true_E, lambda_train):
    """Weighted CE loss on node and edge types.

    Args:
        pred_X: (bs, n, dx) logits
        pred_E: (bs, n, n, de) logits
        true_X: (bs, n, dx) one-hot targets
        true_E: (bs, n, n, de) one-hot targets
        lambda_train: [edge_weight, y_weight]
    """
    # Flatten and filter masked positions
    flat_true_X = true_X.reshape(-1, true_X.size(-1))
    flat_pred_X = pred_X.reshape(-1, pred_X.size(-1))
    mask_X = (flat_true_X != 0.0).any(dim=-1)
    flat_true_X = flat_true_X[mask_X]
    flat_pred_X = flat_pred_X[mask_X]

    flat_true_E = true_E.reshape(-1, true_E.size(-1))
    flat_pred_E = pred_E.reshape(-1, pred_E.size(-1))
    mask_E = (flat_true_E != 0.0).any(dim=-1)
    flat_true_E = flat_true_E[mask_E]
    flat_pred_E = flat_pred_E[mask_E]

    # CE: target is argmax of one-hot
    loss_X = F.cross_entropy(flat_pred_X, flat_true_X.argmax(dim=-1)) if flat_true_X.numel() > 0 else torch.tensor(0.0)
    loss_E = F.cross_entropy(flat_pred_E, flat_true_E.argmax(dim=-1)) if flat_true_E.numel() > 0 else torch.tensor(0.0)

    return loss_X + lambda_train[0] * loss_E


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, val_loader, noise_schedule, transition_model, T, lambda_train, device):
    """Compute average validation NLL."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for batch in val_loader:
        X = batch["X"].to(device)
        E = batch["E"].to(device)
        y = batch["y"].to(device)
        node_mask = batch["node_mask"].to(device)

        # Mask clean data so padding positions are zeroed (matching original validation_step)
        clean = PlaceHolder(X=X, E=E, y=y).mask(node_mask)
        X, E = clean.X, clean.E

        noisy_data = apply_noise(X, E, y, node_mask, noise_schedule, transition_model, T, training=False)
        pred = forward_pass(model, noisy_data, node_mask)
        loss = compute_train_loss(pred.X, pred.E, X, E, lambda_train)
        total_loss += loss.item()
        total_batches += 1

    model.train()
    return total_loss / max(total_batches, 1)


# ---------------------------------------------------------------------------
# Build model from config
# ---------------------------------------------------------------------------

def build_model(cfg, dataset_info: DatasetInfo):
    """Construct GraphTransformer from config and dataset info."""
    num_node_types = dataset_info.node_types
    num_edge_types = dataset_info.edge_types

    # Input dims: noisy features + extra features (timestep=1 for y)
    input_dims = {"X": num_node_types, "E": num_edge_types, "y": 1}
    output_dims = {"X": num_node_types, "E": num_edge_types, "y": 0}

    hidden_dims = {
        "dx": cfg.model.hidden_dims.dx,
        "de": cfg.model.hidden_dims.de,
        "dy": cfg.model.hidden_dims.dy,
        "n_head": cfg.model.hidden_dims.n_head,
        "dim_ffX": cfg.model.hidden_dims.dim_ffX,
        "dim_ffE": cfg.model.hidden_dims.dim_ffE,
        "dim_ffy": cfg.model.hidden_dims.dim_ffy,
        "max_nodes": dataset_info.max_nodes,
    }
    hidden_mlp_dims = {
        "X": cfg.model.hidden_mlp_dims.X,
        "E": cfg.model.hidden_mlp_dims.E,
        "y": cfg.model.hidden_mlp_dims.y,
    }

    return GraphTransformer(
        n_layers=cfg.model.n_layers,
        input_dims=input_dims,
        hidden_mlp_dims=hidden_mlp_dims,
        hidden_dims=hidden_dims,
        output_dims=output_dims,
    )


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, optimizer, epoch, val_nll, config_dict, marginals, dataset_info_dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_nll": val_nll,
        "config": config_dict,
        "marginals": marginals,
        "dataset_info": dataset_info_dict,
    }, path)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pretrain diffusion model on NAS benchmark")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--limit", type=int, default=None, help="Limit dataset size (for debugging)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    config_dict = namespace_to_dict(cfg)

    # Apply CLI overrides
    if args.seed is not None:
        cfg.general.seed = args.seed
        config_dict["general"]["seed"] = args.seed
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
        config_dict["train"]["epochs"] = args.epochs

    seed = cfg.general.seed
    set_seed(seed)

    # Determine limit
    limit = args.limit
    if limit is None:
        raw_limit = getattr(cfg.dataset, "limit", "all")
        if raw_limit != "all" and raw_limit is not None:
            limit = int(raw_limit)

    # Record effective limit in config for provenance
    if limit is not None:
        config_dict["dataset"]["limit"] = limit

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Output directory
    run_name = f"{cfg.general.name}_seed{seed}"
    output_dir = os.path.join("outputs", run_name)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    print(f"Config saved to {config_path}")
    print(f"Config: {json.dumps(config_dict, indent=2, default=str)}")

    # Load dataset
    data_dir = cfg.dataset.data_dir
    split_seed = getattr(cfg.dataset, "split_seed", 0)
    filter_threshold = getattr(cfg.dataset, "filter_threshold", None)
    batch_size = cfg.train.batch_size

    if cfg.dataset.name == "nasbench101":
        train_loader, val_loader, _, dataset_info = load_nb101(
            data_dir, split_seed=split_seed, filter_threshold=filter_threshold,
            limit=limit, batch_size=batch_size,
        )
    elif cfg.dataset.name == "nasbench201":
        reward_dataset = getattr(cfg.dataset, "reward_dataset", "cifar10-valid")
        train_loader, val_loader, _, dataset_info = load_nb201(
            data_dir, split_seed=split_seed, filter_threshold=filter_threshold,
            limit=limit, batch_size=batch_size, reward_dataset=reward_dataset,
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset.name}")

    print(f"Dataset: {cfg.dataset.name} | train batches: {len(train_loader)} | val batches: {len(val_loader)}")
    print(f"  node_types={dataset_info.node_types}, edge_types={dataset_info.edge_types}, max_nodes={dataset_info.max_nodes}")

    # Compute marginals from training data
    marginals = compute_marginals(train_loader, dataset_info)
    print(f"  node marginals: {marginals['node_types']}")
    print(f"  edge marginals: {marginals['edge_types']}")
    print(f"  node_counts dist: {marginals['node_counts']}")

    dataset_info_dict = {
        "num_node_types": dataset_info.node_types,
        "num_edge_types": dataset_info.edge_types,
        "max_nodes": dataset_info.max_nodes,
    }

    # Build model
    model = build_model(cfg, dataset_info).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")

    # Noise schedule and transition model
    T = cfg.model.diffusion_steps
    noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.noise_schedule, T)
    noise_schedule = noise_schedule.to(device)
    transition_model = MarginalUniformTransition(
        x_marginals=marginals["node_types"],
        e_marginals=marginals["edge_types"],
        y_classes=0,
    )

    # Optimizer
    lambda_train = cfg.model.lambda_train
    lr = cfg.train.lr
    weight_decay = getattr(cfg.train, "weight_decay", 1e-12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, amsgrad=True, weight_decay=weight_decay)

    # Training params
    epochs = cfg.train.epochs
    check_val_every = getattr(cfg.train, "check_val_every_n_epochs", 50)

    # Resume
    start_epoch = 0
    best_val_nll = float("inf")
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_nll = ckpt.get("val_nll", float("inf"))
        print(f"  Resumed at epoch {start_epoch}, best_val_nll={best_val_nll:.4f}")

    # JSONL log: append on resume, truncate on fresh run
    log_path = os.path.join(output_dir, "train_log.jsonl")
    log_mode = "a" if args.resume else "w"
    log_file = open(log_path, log_mode)

    # Training loop
    t_start = time.time()
    print(f"\nStarting training: {epochs} epochs, lr={lr}, T={T}, lambda_train={lambda_train}")

    model.train()
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            X = batch["X"].to(device)
            E = batch["E"].to(device)
            y = batch["y"].to(device)
            node_mask = batch["node_mask"].to(device)

            # Mask clean data so padding positions are zeroed (matching original training_step)
            clean = PlaceHolder(X=X, E=E, y=y).mask(node_mask)
            X, E = clean.X, clean.E

            noisy_data = apply_noise(X, E, y, node_mask, noise_schedule, transition_model, T)
            pred = forward_pass(model, noisy_data, node_mask)
            loss = compute_train_loss(pred.X, pred.E, X, E, lambda_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / max(num_batches, 1)

        # Validation
        val_nll = None
        if (epoch + 1) % check_val_every == 0 or epoch == epochs - 1:
            val_nll = validate(model, val_loader, noise_schedule, transition_model, T, lambda_train, device)
            if val_nll < best_val_nll:
                best_val_nll = val_nll
                save_checkpoint(
                    os.path.join(ckpt_dir, "best.pt"),
                    model, optimizer, epoch, best_val_nll,
                    config_dict, marginals, dataset_info_dict,
                )
            print(f"  Epoch {epoch}: val_nll={val_nll:.4f} (best={best_val_nll:.4f})")

        elapsed = time.time() - t_start
        log_entry = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "val_nll": round(val_nll, 6) if val_nll is not None else None,
            "lr": lr,
            "elapsed_sec": round(elapsed, 1),
        }
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f} ({epoch_time:.1f}s)")

    # Save last checkpoint
    save_checkpoint(
        os.path.join(ckpt_dir, "last.pt"),
        model, optimizer, epochs - 1, best_val_nll,
        config_dict, marginals, dataset_info_dict,
    )

    log_file.close()
    print(f"\nTraining complete. Checkpoints in {ckpt_dir}")
    print(f"  best_val_nll={best_val_nll:.4f}")


if __name__ == "__main__":
    main()
