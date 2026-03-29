"""Evaluation pipeline: sample architectures from a trained model, compute metrics.

Two modes:
  1. Single-run: sample N architectures, compute stats, output JSON.
     python src/evaluate.py --checkpoint <path> --config <path> --n-samples 300 --seed 42

  2. Aggregate: collect per-seed JSONs, compute mean+-std, compare against expected.
     python src/evaluate.py --aggregate --results-dir outputs/ \
         --pattern "eval_rlft_*_seed*/results.json" \
         --expected expected_outputs/experiment_1_nb201.json \
         --output outputs/experiment_1_nb201_summary.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure companion/src resolves correctly
_companion_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _companion_root not in sys.path:
    sys.path.insert(0, _companion_root)

from src.diffusion import (
    MarginalUniformTransition,
    PredefinedNoiseScheduleDiscrete,
    sample_p_zs_given_zt,
)
from src.model import GraphTransformer
from src.rewards import RewardComputer
from src.utils import PlaceHolder, load_config, sample_discrete_feature_noise


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
# Model loading (reuses train_rlft pattern)
# ---------------------------------------------------------------------------

def build_model_from_checkpoint(ckpt: dict) -> GraphTransformer:
    """Reconstruct model from checkpoint metadata and load weights."""
    ckpt_config = ckpt["config"]
    ckpt_model = ckpt_config["model"]
    ds_info = ckpt["dataset_info"]

    num_node_types = ds_info["num_node_types"]
    num_edge_types = ds_info["num_edge_types"]
    max_nodes = ds_info["max_nodes"]

    input_dims = {"X": num_node_types, "E": num_edge_types, "y": 1}
    output_dims = {"X": num_node_types, "E": num_edge_types, "y": 0}

    ckpt_hd = ckpt_model["hidden_dims"]
    hidden_dims = {
        "dx": ckpt_hd["dx"], "de": ckpt_hd["de"], "dy": ckpt_hd["dy"],
        "n_head": ckpt_hd["n_head"], "dim_ffX": ckpt_hd["dim_ffX"],
        "dim_ffE": ckpt_hd["dim_ffE"], "dim_ffy": ckpt_hd["dim_ffy"],
        "max_nodes": max_nodes,
    }

    model = GraphTransformer(
        n_layers=ckpt_model["n_layers"],
        input_dims=input_dims,
        hidden_mlp_dims=ckpt_model["hidden_mlp_dims"],
        hidden_dims=hidden_dims,
        output_dims=output_dims,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model


# ---------------------------------------------------------------------------
# Sampling (simplified from train_rlft — no trajectory storage needed)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_architectures(
    model: torch.nn.Module,
    n_samples: int,
    T: int,
    limit_dist: PlaceHolder,
    node_dist: torch.Tensor,
    noise_schedule: PredefinedNoiseScheduleDiscrete,
    transition_model: MarginalUniformTransition,
    Xdim_output: int,
    Edim_output: int,
    device: torch.device,
    batch_size: int = 64,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Sample architectures via reverse diffusion.

    Returns list of (node_types, edge_types) tuples.
    """
    model.eval()
    all_graphs: List[Tuple[torch.Tensor, torch.Tensor]] = []

    while len(all_graphs) < n_samples:
        bs = min(batch_size, n_samples - len(all_graphs))

        # Sample number of nodes
        n_nodes = torch.multinomial(node_dist.float(), bs, replacement=True).to(device)
        n_max = torch.max(n_nodes).item()
        arange = torch.arange(n_max, device=device).unsqueeze(0).expand(bs, -1)
        node_mask = arange < n_nodes.unsqueeze(1)

        # Start from noise
        z_T = sample_discrete_feature_noise(limit_dist=limit_dist, node_mask=node_mask)
        X_t, E_t, y_t = z_T.X, z_T.E, z_T.y

        # Reverse diffusion
        for s_int in reversed(range(0, T)):
            s_array = s_int * torch.ones((bs, 1), device=device).type_as(y_t)
            t_array = s_array + 1
            s_norm = s_array / T
            t_norm = t_array / T

            sampled_s, _ = sample_p_zs_given_zt(
                s_norm, t_norm, X_t, E_t, y_t, node_mask,
                model, noise_schedule, transition_model,
                Xdim_output, Edim_output,
            )
            X_t, E_t, y_t = sampled_s.X, sampled_s.E, sampled_s.y

        # Extract final graphs with DAG recovery
        final = sampled_s.mask(node_mask, collapse=True)
        for i in range(bs):
            n_i = int(n_nodes[i].item())
            node_types = final.X[i, :n_i].to(torch.int64).cpu()
            edge_mat = final.E[i, :n_i, :n_i].to(torch.int64).cpu()
            edge_mat = torch.triu(edge_mat, diagonal=1)
            all_graphs.append((node_types, edge_mat))

    return all_graphs[:n_samples]


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(
    rewards: List[float],
    metas: List[Dict[str, Any]],
    n_samples: int,
    filter_threshold: Optional[float] = None,
    pretrain_samples: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 50,
) -> Dict[str, Any]:
    """Compute evaluation metrics from reward results.

    Returns a dict with all metrics. OOD metrics are included only when
    filter_threshold is provided.
    """
    valid_accs = []
    valid_ids = set()
    for r, m in zip(rewards, metas):
        if r > -0.5:  # valid
            acc = m.get("validation_accuracy")
            if acc is not None:
                valid_accs.append(acc)
            aid = m.get("spec_hash") or m.get("arch_str") or m.get("index")
            if aid is not None:
                valid_ids.add(aid)

    n_valid = sum(1 for r in rewards if r > -0.5)
    n_unique = len(valid_ids)

    result: Dict[str, Any] = {
        "n_samples": n_samples,
        "n_valid": n_valid,
        "n_unique": n_unique,
        "valid_rate": n_valid / n_samples if n_samples > 0 else 0.0,
    }

    if valid_accs:
        result["max_val_acc"] = float(np.max(valid_accs))
        result["mean_val_acc"] = float(np.mean(valid_accs))
        result["std_val_acc"] = float(np.std(valid_accs))
    else:
        result["max_val_acc"] = 0.0
        result["mean_val_acc"] = 0.0
        result["std_val_acc"] = 0.0

    # OOD metrics (only when filter_threshold is provided)
    if filter_threshold is not None:
        # Strict > to match reference ood_verification.py:180
        above = [a for a in valid_accs if a > filter_threshold]
        result["threshold_crossing_rate"] = len(above) / len(valid_accs) if valid_accs else 0.0
        result["best_generated_acc"] = result["max_val_acc"]
        result["extrapolation_pp"] = result["max_val_acc"] - filter_threshold

        # Top-K OOD discovery (requires pretrain samples)
        if pretrain_samples is not None:
            pretrain_ids = set()
            for s in pretrain_samples:
                aid = s.get("arch_id") or s.get("spec_hash") or s.get("arch_str")
                if aid is not None:
                    pretrain_ids.add(aid)

            # Sort valid samples by accuracy, take top-K
            valid_with_ids = []
            for r, m in zip(rewards, metas):
                if r > -0.5:
                    acc = m.get("validation_accuracy")
                    aid = m.get("spec_hash") or m.get("arch_str") or m.get("index")
                    if acc is not None and aid is not None:
                        valid_with_ids.append({"acc": acc, "id": aid})
            valid_with_ids.sort(key=lambda x: x["acc"], reverse=True)
            top_k_items = valid_with_ids[:top_k]

            if top_k_items:
                novel_count = sum(1 for item in top_k_items if item["id"] not in pretrain_ids)
                result["ood_discovery_rate"] = novel_count / len(top_k_items)

                # OOD lift: P(above threshold | top-K) / P(above threshold | all valid) - 1
                p_top_k = sum(1 for item in top_k_items if item["acc"] > filter_threshold) / len(top_k_items)
                p_all = result["threshold_crossing_rate"]
                result["ood_lift"] = (p_top_k / p_all - 1.0) if p_all > 0 else None
            else:
                result["ood_discovery_rate"] = None
                result["ood_lift"] = None
        else:
            result["ood_discovery_rate"] = None
            result["ood_lift"] = None

    return result


# ---------------------------------------------------------------------------
# Sample loading (JSON and CSV, matching ood_verification.py conventions)
# ---------------------------------------------------------------------------

def _load_samples(path: str) -> List[Dict[str, Any]]:
    """Load samples from JSON or CSV file.

    JSON: expects {"samples": [...]} (BaselineOutput) or raw list.
    CSV: expects header row with at least arch_id/arch_str and val_acc columns.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        samples = []
        with open(path, newline="") as f:
            lines = [line for line in f if not line.startswith("#")]
        if not lines:
            return []
        reader = csv.DictReader(lines)
        for row in reader:
            # Normalize arch_id: prefer arch_index, fall back to arch_str, arch_id
            aid = None
            for key in ("arch_index", "arch_id"):
                if key in row and row[key]:
                    try:
                        candidate = int(row[key])
                        if candidate >= 0:
                            aid = candidate
                            break
                    except (ValueError, TypeError):
                        pass
            if aid is None:
                aid = row.get("arch_str") or row.get("arch_id")
            row["arch_id"] = aid
            # Normalize val_acc
            val_acc = row.get("val_acc")
            if val_acc is not None and val_acc != "":
                try:
                    row["val_acc"] = float(val_acc)
                except (ValueError, TypeError):
                    row["val_acc"] = None
            # Normalize is_valid
            is_valid = row.get("is_valid", "true")
            if isinstance(is_valid, str):
                row["is_valid"] = is_valid.lower() == "true"
            samples.append(row)
        return samples

    # Default: JSON
    with open(path) as f:
        pdata = json.load(f)
    if isinstance(pdata, dict):
        return pdata.get("samples", [])
    if isinstance(pdata, list):
        return pdata
    return []


# ---------------------------------------------------------------------------
# Single-run evaluation
# ---------------------------------------------------------------------------

def run_single_evaluation(args) -> Dict[str, Any]:
    """Run evaluation on a single checkpoint."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Build model
    model = build_model_from_checkpoint(ckpt)
    model = model.to(device)
    model.eval()

    # Setup diffusion components from checkpoint
    marginals = ckpt["marginals"]
    ds_info = ckpt["dataset_info"]
    num_node_types = ds_info["num_node_types"]
    num_edge_types = ds_info["num_edge_types"]

    cfg = load_config(args.config)
    T = cfg.model.diffusion_steps

    # Validate diffusion_steps matches checkpoint to avoid silent inconsistency
    ckpt_T = ckpt["config"]["model"].get("diffusion_steps")
    if ckpt_T is not None and ckpt_T != T:
        raise ValueError(
            f"Diffusion steps mismatch: config has {T}, checkpoint was trained with {ckpt_T}. "
            f"Use a config with matching diffusion_steps."
        )

    noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.noise_schedule, T).to(device)
    transition_model = MarginalUniformTransition(
        x_marginals=marginals["node_types"],
        e_marginals=marginals["edge_types"],
        y_classes=0,
    )

    limit_dist = PlaceHolder(
        X=marginals["node_types"],
        E=marginals["edge_types"],
        y=torch.ones(0),
    )
    node_dist = marginals["node_counts"]

    # Sample architectures
    print(f"Sampling {args.n_samples} architectures...")
    graphs = sample_architectures(
        model, args.n_samples, T, limit_dist, node_dist,
        noise_schedule, transition_model,
        num_node_types, num_edge_types, device,
    )
    print(f"Generated {len(graphs)} graphs")

    # Compute rewards
    data_dir = cfg.dataset.data_dir
    reward_dataset = getattr(cfg.dataset, "reward_dataset", "cifar10-valid")
    reward_computer = RewardComputer(
        benchmark=cfg.dataset.name,
        data_dir=data_dir,
        reward_dataset=reward_dataset,
    )

    print("Computing rewards...")
    rewards, metas = reward_computer.compute_rewards(graphs)

    # Load pretrain samples if provided (supports JSON and CSV)
    pretrain_samples = None
    if args.pretrain_samples:
        pretrain_samples = _load_samples(args.pretrain_samples)

    # Compute metrics
    result = compute_metrics(
        rewards, metas, args.n_samples,
        filter_threshold=args.filter_threshold,
        pretrain_samples=pretrain_samples,
    )

    result["seed"] = args.seed
    result["checkpoint"] = os.path.abspath(args.checkpoint)
    result["config"] = os.path.abspath(args.config)

    return result


# ---------------------------------------------------------------------------
# Aggregate mode
# ---------------------------------------------------------------------------

def run_aggregate(args) -> Dict[str, Any]:
    """Collect per-seed results, compute mean+-std, compare against expected."""
    # Find result files
    pattern = os.path.join(args.results_dir, args.pattern)
    result_files = sorted(glob.glob(pattern, recursive=True))

    if not result_files:
        print(f"ERROR: No files matching pattern: {pattern}")
        sys.exit(1)

    print(f"Found {len(result_files)} result files:")
    for f in result_files:
        print(f"  {f}")

    # Group by task (extract from path: eval_rlft_nb201_cifar10_seed42 -> cifar10)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for fpath in result_files:
        with open(fpath) as f:
            data = json.load(f)

        # Try to extract task name from the directory name
        dirname = os.path.basename(os.path.dirname(fpath))
        # Pattern: eval_rlft_{benchmark}_{task}_seed{N}
        parts = dirname.split("_")
        task = "default"
        for i, p in enumerate(parts):
            if p.startswith("seed"):
                # Task is everything between benchmark and seed
                # e.g. eval_rlft_nb201_cifar10_seed42 -> cifar10
                # e.g. eval_rlft_nb101_seed42 -> default
                task_parts = parts[3:i]  # skip eval, rlft, benchmark
                if task_parts:
                    task = "_".join(task_parts)
                break

        groups.setdefault(task, []).append(data)

    # Compute per-task aggregates
    summary: Dict[str, Any] = {"tasks": {}}

    for task, results in sorted(groups.items()):
        task_summary: Dict[str, Any] = {"n_seeds": len(results), "seeds": []}
        metrics_to_agg = ["max_val_acc", "mean_val_acc", "valid_rate",
                          "threshold_crossing_rate", "ood_discovery_rate",
                          "extrapolation_pp", "ood_lift"]

        for metric in metrics_to_agg:
            values = [r[metric] for r in results if metric in r and r[metric] is not None]
            if values:
                task_summary[f"{metric}_mean"] = float(np.mean(values))
                task_summary[f"{metric}_std"] = float(np.std(values))

        for r in results:
            task_summary["seeds"].append(r.get("seed"))

        summary["tasks"][task] = task_summary

    # Compare against expected if provided
    all_pass = True
    if args.expected:
        with open(args.expected) as f:
            expected = json.load(f)

        expected_metrics = expected.get("metrics", {})
        comparisons: List[Dict[str, Any]] = []

        for metric_key, spec in expected_metrics.items():
            exp_val = spec["expected"]
            tolerance = spec["tolerance"]
            unit = spec.get("unit", "")

            # Find matching aggregate value
            # metric_key format: cifar10_max_val_acc -> task=cifar10, metric=max_val_acc
            parts = metric_key.rsplit("_", 2)
            if len(parts) >= 3:
                # Try task_metric split
                for task in summary["tasks"]:
                    prefix = task + "_"
                    if metric_key.startswith(prefix):
                        metric_name = metric_key[len(prefix):]
                        agg_key = f"{metric_name}_mean"
                        if agg_key in summary["tasks"][task]:
                            actual = summary["tasks"][task][agg_key]
                            # Convert to percent if needed
                            if unit == "percent":
                                actual_cmp = actual * 100.0
                            else:
                                actual_cmp = actual
                            passed = abs(actual_cmp - exp_val) <= tolerance
                            if not passed:
                                all_pass = False
                            comp = {
                                "metric": metric_key,
                                "expected": exp_val,
                                "tolerance": tolerance,
                                "actual": round(actual_cmp, 4),
                                "pass": passed,
                            }
                            comparisons.append(comp)
                            status = "PASS" if passed else "FAIL"
                            print(f"  {metric_key}: {actual_cmp:.4f} vs {exp_val}+-{tolerance} [{status}]")
                            break

            # Also check non-task-prefixed metrics (e.g. valid_rate)
            if not any(c["metric"] == metric_key for c in comparisons):
                for task in summary["tasks"]:
                    agg_key = f"{metric_key}_mean"
                    if agg_key in summary["tasks"][task]:
                        actual = summary["tasks"][task][agg_key]
                        if unit == "percent":
                            actual_cmp = actual * 100.0
                        else:
                            actual_cmp = actual
                        passed = abs(actual_cmp - exp_val) <= tolerance
                        if not passed:
                            all_pass = False
                        comp = {
                            "metric": metric_key,
                            "expected": exp_val,
                            "tolerance": tolerance,
                            "actual": round(actual_cmp, 4),
                            "pass": passed,
                        }
                        comparisons.append(comp)
                        status = "PASS" if passed else "FAIL"
                        print(f"  {metric_key}: {actual_cmp:.4f} vs {exp_val}+-{tolerance} [{status}]")

        # Fail if any expected metric was not matched
        matched_keys = {c["metric"] for c in comparisons}
        unmatched = set(expected_metrics.keys()) - matched_keys
        if unmatched:
            all_pass = False
            for key in sorted(unmatched):
                print(f"  {key}: NOT FOUND in aggregated results [FAIL]")
                comparisons.append({
                    "metric": key,
                    "expected": expected_metrics[key]["expected"],
                    "tolerance": expected_metrics[key]["tolerance"],
                    "actual": None,
                    "pass": False,
                })

        summary["comparisons"] = comparisons
        summary["all_pass"] = all_pass

    # Write output
    output_path = args.output
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary written to {output_path}")

    if args.expected:
        if all_pass:
            print("\nAll metrics PASS.")
        else:
            print("\nSome metrics FAILED.")
            sys.exit(1)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DGPO model")

    # Mode selection
    parser.add_argument("--aggregate", action="store_true",
                        help="Aggregate mode: collect per-seed results")

    # Single-run arguments
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, help="Path to YAML config")
    parser.add_argument("--n-samples", type=int, default=300, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--filter-threshold", type=float, default=None,
                        help="If set, compute OOD metrics")
    parser.add_argument("--pretrain-samples", type=str, default=None,
                        help="Path to pretrain baseline JSON for OOD discovery rate")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")

    # Aggregate arguments
    parser.add_argument("--results-dir", type=str, default="outputs/",
                        help="Root directory containing result files")
    parser.add_argument("--pattern", type=str, default="eval_*/results.json",
                        help="Glob pattern for result files")
    parser.add_argument("--expected", type=str, default=None,
                        help="Path to expected output JSON for pass/fail comparison")

    args = parser.parse_args()

    if args.aggregate:
        run_aggregate(args)
    else:
        if not args.checkpoint or not args.config:
            parser.error("--checkpoint and --config are required for single-run mode")

        result = run_single_evaluation(args)

        # Print result
        print("\n" + json.dumps(result, indent=2))

        # Save to file
        output_path = args.output
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
