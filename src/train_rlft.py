"""Phase 2: DGPO RL fine-tuning of pretrained diffusion model.

Implements Directed Graph Policy Optimization (DGPO) — the paper's core
contribution.  Fine-tunes a pretrained graph diffusion model using
REINFORCE-style policy gradients where the reward is the generated
architecture's validation accuracy on a NAS benchmark.

Usage:
    python src/train_rlft.py --config configs/rlft_nb201_cifar10.yaml \
        --resume path/to/pretrain.pt [--seed 42] [--epochs 60]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# Ensure companion/src resolves correctly
_companion_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _companion_root not in sys.path:
    sys.path.insert(0, _companion_root)

from src.diffusion import (
    MarginalUniformTransition,
    PredefinedNoiseScheduleDiscrete,
    forward_pass,
    sample_p_zs_given_zt,
)
from src.model import GraphTransformer
from src.rewards import RewardComputer
from src.utils import PlaceHolder, load_config, namespace_to_dict, sample_discrete_feature_noise


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
# Model construction from checkpoint
# ---------------------------------------------------------------------------

def build_model_from_checkpoint(ckpt: dict, rlft_cfg) -> GraphTransformer:
    """Reconstruct model using dataset_info from checkpoint, then load weights.

    Validates architecture match between pretrain checkpoint and RL-FT config.
    """
    ckpt_config = ckpt["config"]
    ckpt_model = ckpt_config["model"]

    # Step 1: Validate architecture match
    for key in ["n_layers", "diffusion_steps"]:
        ckpt_val = ckpt_model[key]
        rlft_val = getattr(rlft_cfg.model, key)
        if ckpt_val != rlft_val:
            raise ValueError(
                f"Architecture mismatch: pretrain {key}={ckpt_val}, rlft {key}={rlft_val}"
            )
    # Validate hidden_dims (all keys that affect model architecture)
    ckpt_hd = ckpt_model["hidden_dims"]
    rlft_hd = namespace_to_dict(rlft_cfg.model.hidden_dims)
    for k in ["dx", "de", "dy", "n_head", "dim_ffX", "dim_ffE", "dim_ffy"]:
        if ckpt_hd.get(k) != rlft_hd.get(k):
            raise ValueError(
                f"Architecture mismatch: pretrain hidden_dims.{k}={ckpt_hd.get(k)}, "
                f"rlft hidden_dims.{k}={rlft_hd.get(k)}"
            )

    # Step 2: Reconstruct model from checkpoint's dataset_info
    ds_info = ckpt["dataset_info"]
    num_node_types = ds_info["num_node_types"]
    num_edge_types = ds_info["num_edge_types"]
    max_nodes = ds_info["max_nodes"]

    input_dims = {"X": num_node_types, "E": num_edge_types, "y": 1}
    output_dims = {"X": num_node_types, "E": num_edge_types, "y": 0}

    hidden_dims = {
        "dx": ckpt_hd["dx"],
        "de": ckpt_hd["de"],
        "dy": ckpt_hd["dy"],
        "n_head": ckpt_hd["n_head"],
        "dim_ffX": ckpt_hd["dim_ffX"],
        "dim_ffE": ckpt_hd["dim_ffE"],
        "dim_ffy": ckpt_hd["dim_ffy"],
        "max_nodes": max_nodes,
    }
    hidden_mlp_dims = ckpt_model["hidden_mlp_dims"]

    model = GraphTransformer(
        n_layers=ckpt_model["n_layers"],
        input_dims=input_dims,
        hidden_mlp_dims=hidden_mlp_dims,
        hidden_dims=hidden_dims,
        output_dims=output_dims,
    )

    # Load pretrained weights
    model.load_state_dict(ckpt["model_state_dict"])
    return model


# ---------------------------------------------------------------------------
# Layer freezing
# ---------------------------------------------------------------------------

def freeze_transformer_layers(model: nn.Module, frac: float) -> Dict[str, Any]:
    """Freeze the bottom fraction of transformer layers in-place.

    Returns metadata about frozen/trainable parameter counts.
    """
    layers = getattr(model, "tf_layers", None)
    if layers is None:
        raise RuntimeError("Model has no 'tf_layers' attribute — cannot freeze layers")

    total_layers = len(layers)
    layers_to_freeze = min(total_layers, int(math.ceil(frac * total_layers)))

    frozen_params = 0
    frozen_layer_names = []
    for idx in range(layers_to_freeze):
        for param in layers[idx].parameters():
            if param.requires_grad:
                frozen_params += param.numel()
                param.requires_grad_(False)
        frozen_layer_names.append(f"tf_layers.{idx}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if trainable_params <= 0:
        raise RuntimeError("All parameters frozen — at least one must remain trainable")

    return {
        "frozen_layers": layers_to_freeze,
        "total_layers": total_layers,
        "frozen_params": frozen_params,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_layer_names": frozen_layer_names,
    }


# ---------------------------------------------------------------------------
# Trajectory sampling (full reverse diffusion with trajectory storage)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_trajectories(
    model: nn.Module,
    batch_size: int,
    T: int,
    limit_dist: PlaceHolder,
    node_dist: torch.Tensor,
    noise_schedule: PredefinedNoiseScheduleDiscrete,
    transition_model: MarginalUniformTransition,
    Xdim_output: int,
    Edim_output: int,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Sample trajectories via full reverse diffusion, storing all intermediate states.

    Returns:
        X_traj: list of T+1 tensors, each (bs, n, dx) one-hot — from t=T down to t=0.
        E_traj: list of T+1 tensors, each (bs, n, n, de) one-hot — from t=T down to t=0.
        node_mask: (bs, n) boolean mask.
        graphs: list of (node_types, edge_types) tuples — final graphs with DAG extraction.
    """
    # Sample number of nodes per graph
    n_nodes = torch.multinomial(node_dist.float(), batch_size, replacement=True).to(device)
    n_max = torch.max(n_nodes).item()

    # Build masks
    arange = torch.arange(n_max, device=device).unsqueeze(0).expand(batch_size, -1)
    node_mask = arange < n_nodes.unsqueeze(1)

    # Sample from limit distribution (pure noise)
    z_T = sample_discrete_feature_noise(limit_dist=limit_dist, node_mask=node_mask)
    X_t, E_t, y_t = z_T.X, z_T.E, z_T.y

    X_traj: List[torch.Tensor] = [X_t.detach().cpu()]
    E_traj: List[torch.Tensor] = [E_t.detach().cpu()]

    # Reverse diffusion: t = T, T-1, ..., 1
    for s_int in reversed(range(0, T)):
        s_array = s_int * torch.ones((batch_size, 1), device=device).type_as(y_t)
        t_array = s_array + 1
        s_norm = s_array / T
        t_norm = t_array / T

        sampled_s, _ = sample_p_zs_given_zt(
            s_norm, t_norm, X_t, E_t, y_t, node_mask,
            model, noise_schedule, transition_model,
            Xdim_output, Edim_output,
        )
        X_t, E_t, y_t = sampled_s.X, sampled_s.E, sampled_s.y

        X_traj.append(X_t.detach().cpu())
        E_traj.append(E_t.detach().cpu())

    # Extract final graphs with DAG recovery (upper-triangular edges)
    final = sampled_s.mask(node_mask, collapse=True)
    graphs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(batch_size):
        n_i = int(n_nodes[i].item())
        node_types = final.X[i, :n_i].to(torch.int64).cpu()
        # Upper-triangular extraction for DAG recovery
        edge_mat = final.E[i, :n_i, :n_i].to(torch.int64).cpu()
        edge_mat = torch.triu(edge_mat, diagonal=1)
        graphs.append((node_types, edge_mat))

    return X_traj, E_traj, node_mask, graphs


# ---------------------------------------------------------------------------
# DGPO core: advantage computation
# ---------------------------------------------------------------------------

def compute_advantages(
    rewards: torch.Tensor,
    running_mean: float,
    running_std: float,
) -> torch.Tensor:
    """Compute clipped advantages from rewards using running statistics.

    All rewards passed here are already filtered to valid-only (no -1 sentinels).
    Returns zero advantages if all rewards are identical (degenerate batch).
    """
    advantages = torch.zeros_like(rewards)

    # Degenerate batch: all rewards identical → no gradient signal
    if rewards.numel() < 2 or (rewards.max() - rewards.min()).item() < 1e-8:
        return advantages

    mean = torch.tensor(running_mean, device=rewards.device, dtype=rewards.dtype)
    std = torch.tensor(running_std, device=rewards.device, dtype=rewards.dtype).clamp_min(1e-8)
    advantages = ((rewards - mean) / std).clamp_(-5.0, 5.0)

    return advantages


# ---------------------------------------------------------------------------
# DGPO core: policy gradient loss for a single timestep slice
# ---------------------------------------------------------------------------

def ppo_step_slice(
    model: nn.Module,
    X_t: torch.Tensor,
    E_t: torch.Tensor,
    X_0: torch.Tensor,
    E_0: torch.Tensor,
    node_mask: torch.Tensor,
    t_int: torch.Tensor,
    advantages: torch.Tensor,
    T: int,
    lambda_e: float,
) -> torch.Tensor:
    """Compute advantage-weighted CE loss for one noisy timestep slice (REINFORCE-style).

    Note: named ``ppo_step_slice`` to match the original codebase; the actual
    algorithm is a REINFORCE-style policy gradient (no ratio clipping).

    Args:
        model: GraphTransformer.
        X_t, E_t: noisy state at timestep t (one-hot).
        X_0, E_0: clean target state (one-hot).
        node_mask: (bs, n) boolean mask.
        t_int: (bs, 1) integer timestep.
        advantages: (bs,) advantage weights.
        T: total diffusion steps.
        lambda_e: edge loss weight.

    Returns:
        Scalar loss tensor.
    """
    device = X_t.device
    batch_size = X_t.shape[0]
    t_float = (t_int / float(T)).to(device)

    # Build noisy data dict for forward pass
    y_zero = torch.zeros(batch_size, 0, device=device, dtype=X_t.dtype)
    z_t = PlaceHolder(X=X_t, E=E_t, y=y_zero).type_as(X_t).mask(node_mask)
    noisy_data = {
        "t": t_float,
        "X_t": z_t.X,
        "E_t": z_t.E,
        "y_t": z_t.y,
        "node_mask": node_mask,
    }
    pred = forward_pass(model, noisy_data, node_mask)

    # Advantage-weighted CE loss
    reweight = advantages.reshape(batch_size, 1)
    ce = nn.CrossEntropyLoss(reduction="none")

    # Node loss
    flat_true_X = X_0.reshape(-1, X_0.size(-1))
    flat_pred_X = pred.X.reshape(-1, pred.X.size(-1))
    mask_X = (flat_true_X != 0.0).any(dim=-1)
    target_X = torch.argmax(flat_true_X, dim=-1)
    loss_X = ce(flat_pred_X, target_X)
    loss_X = loss_X.reshape(batch_size, -1) * reweight
    loss_X = loss_X.view(-1)[mask_X]

    # Edge loss
    flat_true_E = E_0.reshape(-1, E_0.size(-1))
    flat_pred_E = pred.E.reshape(-1, pred.E.size(-1))
    mask_E = (flat_true_E != 0.0).any(dim=-1)
    target_E = torch.argmax(flat_true_E, dim=-1)
    loss_E = ce(flat_pred_E, target_E)
    loss_E = loss_E.reshape(batch_size, -1) * reweight
    loss_E = loss_E.view(-1)[mask_E]

    loss_X_mean = loss_X.mean() if loss_X.numel() else torch.zeros((), device=device, dtype=X_t.dtype)
    loss_E_mean = loss_E.mean() if loss_E.numel() else torch.zeros((), device=device, dtype=X_t.dtype)

    return loss_X_mean + lambda_e * loss_E_mean


# ---------------------------------------------------------------------------
# Distribution sampling (for Fig 2 reproduction)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_distribution_snapshot(
    model: nn.Module,
    n_target: int,
    reward_computer: RewardComputer,
    T: int,
    limit_dist: PlaceHolder,
    node_dist: torch.Tensor,
    noise_schedule: PredefinedNoiseScheduleDiscrete,
    transition_model: MarginalUniformTransition,
    Xdim_output: int,
    Edim_output: int,
    device: torch.device,
    max_attempts: int = 1500,
) -> List[Dict[str, Any]]:
    """Sample architectures and return their accuracies for distribution logging.

    Returns list of dicts with keys: sample_idx, val_acc, is_valid.
    """
    model.eval()
    results: List[Dict[str, Any]] = []
    total_sampled = 0
    batch_size = min(100, n_target)

    while len(results) < n_target and total_sampled < max_attempts:
        n_sample = min(batch_size, n_target - len(results) + 50)
        _, _, _, graphs = sample_trajectories(
            model, n_sample, T, limit_dist, node_dist,
            noise_schedule, transition_model, Xdim_output, Edim_output, device,
        )
        total_sampled += n_sample

        rewards, metas = reward_computer.compute_rewards(graphs)
        for r, meta in zip(rewards, metas):
            is_valid = r > -0.5
            results.append({
                "sample_idx": len(results),
                "val_acc": meta.get("validation_accuracy") if is_valid else None,
                "is_valid": is_valid,
            })
            if len(results) >= n_target:
                break

    model.train()
    return results


def write_dist_samples_csv(
    csv_path: str, epoch: int, samples: List[Dict[str, Any]], write_header: bool = False,
):
    """Append distribution sample rows to CSV."""
    mode = "w" if write_header else "a"
    with open(csv_path, mode, newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "sample_idx", "val_acc", "is_valid"])
        for s in samples:
            writer.writerow([
                epoch,
                s["sample_idx"],
                f"{s['val_acc']:.6f}" if s["val_acc"] is not None else "",
                "true" if s["is_valid"] else "false",
            ])


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------

def save_rlft_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_acc_mean: float,
    config_dict: dict,
    marginals: dict,
    dataset_info_dict: dict,
    pretrain_checkpoint: str,
    frozen_layers: List[str],
    reward_stats: Dict[str, float],
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_acc_mean": best_val_acc_mean,
        "config": config_dict,
        "marginals": marginals,
        "dataset_info": dataset_info_dict,
        "pretrain_checkpoint": pretrain_checkpoint,
        "frozen_layers": frozen_layers,
        "reward_stats": reward_stats,
    }, path)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DGPO RL fine-tuning")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--resume", required=True, help="Path to pretrained checkpoint")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Config
    # ---------------------------------------------------------------
    cfg = load_config(args.config)
    config_dict = namespace_to_dict(cfg)

    if args.seed is not None:
        cfg.general.seed = args.seed
        config_dict["general"]["seed"] = args.seed
    if args.epochs is not None:
        cfg.rlft.epochs = args.epochs
        config_dict["rlft"]["epochs"] = args.epochs

    seed = cfg.general.seed
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---------------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------------
    run_name = f"{cfg.general.name}_seed{seed}"
    output_dir = os.path.join("outputs", run_name)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save resolved config
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    print(f"Config saved to {config_path}")

    # ---------------------------------------------------------------
    # Phase Handoff: Load pretrain checkpoint
    # ---------------------------------------------------------------
    print(f"\nLoading pretrain checkpoint: {args.resume}")
    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)

    # Step 1-2: Validate architecture and reconstruct model
    model = build_model_from_checkpoint(ckpt, cfg)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")

    # Step 3: Load marginals from checkpoint (NOT recomputed from data)
    marginals = ckpt["marginals"]
    dataset_info_dict = ckpt["dataset_info"]
    print(f"Marginals loaded from checkpoint:")
    print(f"  node_types: {marginals['node_types']}")
    print(f"  edge_types: {marginals['edge_types']}")

    num_node_types = dataset_info_dict["num_node_types"]
    num_edge_types = dataset_info_dict["num_edge_types"]
    max_nodes = dataset_info_dict["max_nodes"]

    # Noise schedule and transition model (from marginals)
    T = cfg.model.diffusion_steps
    noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.noise_schedule, T).to(device)
    transition_model = MarginalUniformTransition(
        x_marginals=marginals["node_types"],
        e_marginals=marginals["edge_types"],
        y_classes=0,
    )

    # Limit distribution for sampling
    limit_dist = PlaceHolder(
        X=marginals["node_types"],
        E=marginals["edge_types"],
        y=torch.ones(0),
    )
    node_dist = marginals["node_counts"]

    # Step 4: Freeze layers, create fresh optimizer
    fix_frac = cfg.rlft.fix_frac
    freeze_info = freeze_transformer_layers(model, fix_frac)
    print(f"Layer freezing (frac={fix_frac}):")
    print(f"  frozen: {freeze_info['frozen_layers']}/{freeze_info['total_layers']} layers")
    print(f"  trainable: {freeze_info['trainable_params']:,} / {freeze_info['total_params']:,} params")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.rlft.lr,
        weight_decay=cfg.rlft.weight_decay,
    )
    print(f"Fresh optimizer created (lr={cfg.rlft.lr}, NOT loaded from pretrain)")

    # Step 5: Initialize reward API
    data_dir = cfg.dataset.data_dir
    reward_dataset = getattr(cfg.dataset, "reward_dataset", "cifar10-valid")
    reward_computer = RewardComputer(
        benchmark=cfg.dataset.name,
        data_dir=data_dir,
        reward_dataset=reward_dataset,
    )

    # ---------------------------------------------------------------
    # Training hyperparameters
    # ---------------------------------------------------------------
    epochs = cfg.rlft.epochs
    batch_size = cfg.rlft.batch_size
    sampleloop = getattr(cfg.rlft, "sampleloop", 1)
    innerloop = getattr(cfg.rlft, "innerloop", 4)
    ppo_sr = getattr(cfg.rlft, "ppo_sr", 0.3)
    step_freq = max(1, getattr(cfg.rlft, "step_freq", 2))
    grad_clip = getattr(cfg.rlft, "grad_clip", 1.0)
    lambda_e = float(cfg.model.lambda_train[0]) if cfg.model.lambda_train else 1.0
    inverse = getattr(cfg.general, "inverse", False)

    # Distribution sampling config
    dist_enabled = getattr(cfg.rlft, "dist_sample_enabled", False)
    dist_every_k = getattr(cfg.rlft, "dist_sample_every_k", 10)
    dist_n_target = getattr(cfg.rlft, "dist_sample_n_target", 300)

    # Running reward statistics (EMA)
    running_mean = 0.0
    running_std = 1.0
    ema_alpha = 0.1  # EMA smoothing factor

    # Timestep subsampling count
    time_budget = max(1, min(T, int(math.ceil(T * ppo_sr))))
    scale = float(time_budget * step_freq)

    # ---------------------------------------------------------------
    # Logging setup
    # ---------------------------------------------------------------
    log_path = os.path.join(output_dir, "train_log.jsonl")
    dist_csv_path = os.path.join(output_dir, "dist_samples.csv")
    best_val_acc_mean = 0.0

    # Truncate logs from any prior run in the same output directory
    open(log_path, "w").close()

    print(f"\nStarting DGPO training: {epochs} epochs, K={batch_size}, "
          f"innerloop={innerloop}, ppo_sr={ppo_sr}, inverse={inverse}")

    # ---------------------------------------------------------------
    # Epoch loop
    # ---------------------------------------------------------------
    t_start = time.time()
    model.train()

    for epoch in range(epochs):
        epoch_start = time.time()

        # =============================================================
        # STEP 1: Sample trajectories (no_grad)
        # =============================================================
        all_samples: List[Dict[str, Any]] = []
        epoch_rewards: List[float] = []
        epoch_val_accs: List[float] = []
        n_total = 0
        n_invalid = 0
        n_dedup = 0
        seen_hashes: set = set()  # epoch-scoped dedup

        model.eval()
        for _ in range(sampleloop):
            X_traj, E_traj, node_mask, graphs = sample_trajectories(
                model, batch_size, T, limit_dist, node_dist,
                noise_schedule, transition_model,
                num_node_types, num_edge_types, device,
            )

            # Compute rewards (always positive val_acc or -1 for invalid)
            raw_rewards, metas = reward_computer.compute_rewards(graphs)
            n_total += len(raw_rewards)

            # Filter on raw (pre-negation) rewards, dedup, then apply inverse
            valid_indices = []
            rewards_list: List[float] = list(raw_rewards)

            for i, (r, meta) in enumerate(zip(raw_rewards, metas)):
                if r <= -0.5:
                    n_invalid += 1
                    continue

                # Deduplicate by architecture identity (epoch-scoped)
                arch_id = meta.get("spec_hash") or meta.get("arch_str") or ""
                if arch_id:
                    if arch_id in seen_hashes:
                        n_dedup += 1
                        continue
                    seen_hashes.add(arch_id)

                # Track val_acc stats from the original (un-negated) reward
                val_acc = meta.get("validation_accuracy")
                if val_acc is not None:
                    epoch_val_accs.append(val_acc)

                # Apply inverse *after* validity check
                effective_reward = -r if inverse else r
                rewards_list[i] = effective_reward
                epoch_rewards.append(effective_reward)
                valid_indices.append(i)

            if not valid_indices:
                print(f"  [WARN] Epoch {epoch}: no valid samples in this sampleloop iteration")
                continue

            # Build trajectory tensors for valid samples
            T_len = len(X_traj)
            rewards_t = torch.tensor([rewards_list[i] for i in valid_indices], dtype=torch.float32)

            # Stack and index only valid samples
            X_noise = torch.stack([X_traj[t] for t in range(T_len - 1)])[:, valid_indices]
            E_noise = torch.stack([E_traj[t] for t in range(T_len - 1)])[:, valid_indices]
            X_0 = X_traj[-1][valid_indices]
            E_0 = E_traj[-1][valid_indices]
            valid_mask = node_mask[valid_indices].detach().cpu()

            # Build time matrix and shuffle per-sample
            batch_eff = len(valid_indices)
            base_time = torch.arange(T, 0, -1, dtype=torch.float32)
            times = base_time.unsqueeze(1).repeat(1, batch_eff)
            for gi in range(batch_eff):
                perm = torch.randperm(T)
                X_noise[:, gi] = X_noise[perm, gi]
                E_noise[:, gi] = E_noise[perm, gi]
                times[:, gi] = times[perm, gi]

            all_samples.append({
                "X_noise": X_noise,
                "E_noise": E_noise,
                "X_0": X_0,
                "E_0": E_0,
                "node_mask": valid_mask,
                "rewards": rewards_t,
                "time": times,
            })
        model.train()

        # Update running reward statistics
        if epoch_rewards:
            batch_mean = float(np.mean(epoch_rewards))
            batch_std = float(np.std(epoch_rewards))
            if epoch == 0:
                running_mean = batch_mean
                running_std = max(batch_std, 1e-8)
            else:
                running_mean = (1 - ema_alpha) * running_mean + ema_alpha * batch_mean
                running_std = max((1 - ema_alpha) * running_std + ema_alpha * batch_std, 1e-8)

        # =============================================================
        # STEP 2: Policy update (GDPO)
        # =============================================================
        loss_entries: List[float] = []
        grad_norms: List[float] = []

        if not all_samples:
            print(f"  [WARN] Epoch {epoch}: no valid samples, skipping policy update")
        else:
            for inner_idx in range(innerloop):
                optimizer.zero_grad()
                accum_steps = 0

                for sample in all_samples:
                    rewards_s = sample["rewards"].to(device)
                    advantages = compute_advantages(rewards_s, running_mean, running_std)

                    # Skip update if all advantages are zero (all rewards identical)
                    if not torch.any(advantages != 0):
                        continue

                    X_noise_s = sample["X_noise"].to(device)
                    E_noise_s = sample["E_noise"].to(device)
                    X_0_s = sample["X_0"].to(device)
                    E_0_s = sample["E_0"].to(device)
                    mask_s = sample["node_mask"].to(device)
                    time_s = sample["time"].to(device)

                    # Subsample timesteps
                    timestep_indices = torch.randperm(T)[:time_budget]

                    for idx in timestep_indices.tolist():
                        loss = ppo_step_slice(
                            model,
                            X_t=X_noise_s[idx],
                            E_t=E_noise_s[idx],
                            X_0=X_0_s,
                            E_0=E_0_s,
                            node_mask=mask_s,
                            t_int=time_s[idx].unsqueeze(1),
                            advantages=advantages,
                            T=T,
                            lambda_e=lambda_e,
                        )

                        loss = loss / scale
                        loss_entries.append(loss.detach().item())
                        loss.backward()

                        accum_steps += 1
                        if accum_steps == step_freq:
                            if grad_clip > 0:
                                gn = torch.nn.utils.clip_grad_norm_(
                                    trainable_params, max_norm=grad_clip
                                ).item()
                                grad_norms.append(gn)
                            optimizer.step()
                            optimizer.zero_grad()
                            accum_steps = 0

                # Flush remaining accumulated gradients
                if accum_steps > 0:
                    if grad_clip > 0:
                        gn = torch.nn.utils.clip_grad_norm_(
                            trainable_params, max_norm=grad_clip
                        ).item()
                        grad_norms.append(gn)
                    optimizer.step()
                    optimizer.zero_grad()

        # =============================================================
        # STEP 3: Distribution sampling (periodic)
        # =============================================================
        if dist_enabled and (epoch % dist_every_k == 0):
            print(f"  Distribution sampling at epoch {epoch}...")
            dist_samples = sample_distribution_snapshot(
                model, dist_n_target, reward_computer,
                T, limit_dist, node_dist, noise_schedule, transition_model,
                num_node_types, num_edge_types, device,
            )
            write_header = (epoch == 0)
            write_dist_samples_csv(dist_csv_path, epoch, dist_samples, write_header=write_header)
            n_valid_dist = sum(1 for s in dist_samples if s["is_valid"])
            print(f"  Wrote {n_valid_dist}/{len(dist_samples)} valid samples to {dist_csv_path}")

        # =============================================================
        # STEP 4: Logging
        # =============================================================
        elapsed = time.time() - t_start
        epoch_time = time.time() - epoch_start
        mean_loss = float(np.mean(loss_entries)) if loss_entries else 0.0
        mean_grad = float(np.mean(grad_norms)) if grad_norms else 0.0
        n_kept = len(epoch_rewards)  # after invalid + dedup filtering
        invalid_rate = n_invalid / n_total if n_total > 0 else 0.0

        reward_mean = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
        reward_std = float(np.std(epoch_rewards)) if epoch_rewards else 0.0
        reward_max = float(np.max(epoch_rewards)) if epoch_rewards else 0.0
        val_acc_mean = float(np.mean(epoch_val_accs)) if epoch_val_accs else 0.0
        val_acc_max = float(np.max(epoch_val_accs)) if epoch_val_accs else 0.0

        log_entry = {
            "epoch": epoch,
            "reward_mean": round(reward_mean, 6),
            "reward_std": round(reward_std, 6),
            "reward_max": round(reward_max, 6),
            "val_acc_mean": round(val_acc_mean, 6),
            "val_acc_max": round(val_acc_max, 6),
            "loss": round(mean_loss, 6),
            "invalid_rate": round(invalid_rate, 4),
            "grad_norm": round(mean_grad, 6),
            "n_valid": n_kept,
            "n_total": n_total,
            "elapsed_sec": round(elapsed, 1),
        }

        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        print(f"Epoch {epoch}: reward={reward_mean:.4f}±{reward_std:.4f} "
              f"max={reward_max:.4f} val_acc={val_acc_mean:.4f} "
              f"loss={mean_loss:.4f} invalid={invalid_rate:.1%} ({epoch_time:.1f}s)")

        # =============================================================
        # STEP 5: Checkpointing
        # =============================================================
        reward_stats = {"running_mean": running_mean, "running_std": running_std}

        # Save every epoch (RL-FT is short)
        save_rlft_checkpoint(
            os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"),
            model, optimizer, epoch, best_val_acc_mean,
            config_dict, marginals, dataset_info_dict,
            pretrain_checkpoint=os.path.abspath(args.resume),
            frozen_layers=freeze_info["frozen_layer_names"],
            reward_stats=reward_stats,
        )

        # Save best
        if val_acc_mean > best_val_acc_mean:
            best_val_acc_mean = val_acc_mean
            save_rlft_checkpoint(
                os.path.join(ckpt_dir, "best.pt"),
                model, optimizer, epoch, best_val_acc_mean,
                config_dict, marginals, dataset_info_dict,
                pretrain_checkpoint=os.path.abspath(args.resume),
                frozen_layers=freeze_info["frozen_layer_names"],
                reward_stats=reward_stats,
            )

    print(f"\nDGPO training complete. best_val_acc_mean={best_val_acc_mean:.4f}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
