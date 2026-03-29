"""Baseline computation: random search and pretrained-only sampling.

Provides reference lines for Figures 1, 3, and 4:
- Random search: sample N architectures uniformly from the benchmark.
- Pretrained-only: sample N architectures from a pretrained model (before RL-FT).

Both modes run over multiple seeds and produce per-seed + aggregate output files.

Usage:
    python src/baselines.py --mode random --benchmark nb201 --n-samples 300 --seeds 42 123 456
    python src/baselines.py --mode pretrained --benchmark nb201 --checkpoint <path> \
        --n-samples 300 --seeds 42 123 456
    python src/baselines.py --mode pretrained --benchmark nb201 --checkpoint <path> \
        --variant filtered --n-samples 300 --seeds 42 123 456
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# Ensure companion/src resolves correctly
_companion_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _companion_root not in sys.path:
    sys.path.insert(0, _companion_root)

from src.rewards import RewardComputer


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
# Random search baseline
# ---------------------------------------------------------------------------

def random_search_baseline(
    benchmark: str,
    reward_computer: RewardComputer,
    n_samples: int,
    seeds: List[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Sample N architectures uniformly from the benchmark API per seed.

    Samples from architectures that actually exist in the benchmark:
    - NB101: uniform sample from ~423k hashes in the tfrecord.
    - NB201: uniform sample from 15,625 architecture indices.

    All sampled architectures are valid by construction (they exist in the
    benchmark). This matches the paper's random search baseline definition.

    Returns (per_seed_results, aggregate).
    """
    per_seed: List[Dict[str, Any]] = []

    for seed in seeds:
        set_seed(seed)
        samples = _query_random_from_benchmark(
            benchmark, reward_computer, n_samples, seed,
        )
        valid_accs = [s["val_acc"] for s in samples if s["val_acc"] is not None]
        n_valid = len(valid_accs)

        result: Dict[str, Any] = {
            "seed": seed,
            "n_samples": n_samples,
            "n_valid": n_valid,
            "valid_rate": n_valid / n_samples if n_samples > 0 else 0.0,
            "max_val_acc": float(np.max(valid_accs)) if valid_accs else 0.0,
            "mean_val_acc": float(np.mean(valid_accs)) if valid_accs else 0.0,
            "std_val_acc": float(np.std(valid_accs)) if valid_accs else 0.0,
            "samples": samples,
        }
        per_seed.append(result)
        print(f"  seed={seed}: max_val_acc={result['max_val_acc']:.4f} "
              f"mean_val_acc={result['mean_val_acc']:.4f} "
              f"valid={n_valid}/{n_samples}")

    aggregate = _aggregate_results(per_seed, seeds, n_samples)
    return per_seed, aggregate


def _query_random_from_benchmark(
    benchmark: str,
    reward_computer: RewardComputer,
    n_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Sample uniformly from benchmark architectures and query accuracies.

    NB101: sample hashes from the loaded API, query metrics directly.
    NB201: sample indices from {0..15624}, query via unified cache or API.
    """
    rng = random.Random(seed)

    if benchmark == "nasbench101":
        nb101_api = reward_computer._nb101_api
        all_hashes = list(nb101_api.hash_iterator())
        sampled = rng.sample(all_hashes, min(n_samples, len(all_hashes)))

        samples = []
        for h in sampled:
            try:
                fixed, computed = nb101_api.get_metrics_from_hash(h)
                val_acc = RewardComputer._extract_nb101_val_acc(computed)
                test_acc = RewardComputer._extract_nb101_test_acc(computed)
                samples.append({
                    "arch_id": h,
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                    "is_valid": val_acc is not None,
                })
            except Exception:
                samples.append({
                    "arch_id": h, "val_acc": None, "test_acc": None,
                    "is_valid": False,
                })
        return samples

    elif benchmark == "nasbench201":
        total_archs = 15625  # 5^6 op choices
        sampled_indices = rng.sample(range(total_archs), min(n_samples, total_archs))

        if reward_computer._nb201_lookup is not None:
            entries = reward_computer._nb201_lookup
            arch_strs = list(entries.keys())
            ds = reward_computer.reward_dataset
            samples = []
            for idx in sampled_indices:
                if idx < len(arch_strs):
                    arch_str = arch_strs[idx]
                    entry = entries[arch_str]
                    ds_metrics = entry.get(ds, {})
                    samples.append({
                        "arch_id": entry.get("index", idx),
                        "arch_str": arch_str,
                        "val_acc": ds_metrics.get("val_acc"),
                        "test_acc": ds_metrics.get("test_acc"),
                        "is_valid": ds_metrics.get("val_acc") is not None,
                    })
                else:
                    samples.append({
                        "arch_id": idx, "val_acc": None, "is_valid": False,
                    })
            return samples
        else:
            nb201_api = reward_computer._nb201_api
            ds = reward_computer.reward_dataset
            samples = []
            for idx in sampled_indices:
                try:
                    arch_str = nb201_api.arch(idx)
                    info = nb201_api.get_more_info(
                        idx, ds, iepoch=None,
                        use_12epochs_result=False, is_random=False,
                    )
                    val_acc_raw = info.get("valid-accuracy") or info.get("est-valid-accuracy")
                    if val_acc_raw is not None:
                        val_acc = float(val_acc_raw)
                        if val_acc > 1.0:
                            val_acc /= 100.0
                    else:
                        val_acc = None
                    samples.append({
                        "arch_id": idx,
                        "arch_str": arch_str,
                        "val_acc": val_acc,
                        "is_valid": val_acc is not None,
                    })
                except Exception:
                    samples.append({
                        "arch_id": idx, "val_acc": None, "is_valid": False,
                    })
            return samples

    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


# ---------------------------------------------------------------------------
# Pretrained-only baseline
# ---------------------------------------------------------------------------

def pretrained_only_baseline(
    benchmark: str,
    checkpoint_path: str,
    reward_computer: RewardComputer,
    n_samples: int,
    seeds: List[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Sample N architectures from pretrained model per seed.

    Uses the same sampling code as evaluate.py.

    Returns (per_seed_results, aggregate).
    """
    from src.evaluate import build_model_from_checkpoint, sample_architectures
    from src.diffusion import (
        MarginalUniformTransition,
        PredefinedNoiseScheduleDiscrete,
    )
    from src.utils import PlaceHolder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint once
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Validate benchmark matches checkpoint
    ckpt_dataset = ckpt["config"].get("dataset", {}).get("name", "")
    expected_dataset = benchmark  # "nasbench101" or "nasbench201"
    if ckpt_dataset and ckpt_dataset != expected_dataset:
        raise ValueError(
            f"Benchmark mismatch: --benchmark implies {expected_dataset}, "
            f"but checkpoint was trained on {ckpt_dataset}"
        )

    marginals = ckpt["marginals"]
    ds_info = ckpt["dataset_info"]
    T = ckpt["config"]["model"]["diffusion_steps"]

    noise_schedule = PredefinedNoiseScheduleDiscrete(
        ckpt["config"]["model"]["noise_schedule"], T
    ).to(device)
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

    per_seed: List[Dict[str, Any]] = []

    for seed in seeds:
        set_seed(seed)

        # Rebuild model each seed for clean state
        model = build_model_from_checkpoint(ckpt)
        model = model.to(device)
        model.eval()

        graphs = sample_architectures(
            model, n_samples, T, limit_dist, node_dist,
            noise_schedule, transition_model,
            ds_info["num_node_types"], ds_info["num_edge_types"],
            device,
        )

        rewards, metas = reward_computer.compute_rewards(graphs)
        result = _summarize_rewards(rewards, metas, n_samples, seed)

        # Include per-sample data for OOD analysis downstream
        samples = []
        for r, m in zip(rewards, metas):
            samples.append({
                "arch_id": m.get("spec_hash") or m.get("index"),
                "arch_str": m.get("arch_str"),
                "val_acc": m.get("validation_accuracy"),
                "is_valid": r > -0.5,
            })
        result["samples"] = samples

        per_seed.append(result)
        print(f"  seed={seed}: max_val_acc={result['max_val_acc']:.4f} "
              f"mean_val_acc={result['mean_val_acc']:.4f} "
              f"valid={result['n_valid']}/{n_samples}")

    aggregate = _aggregate_results(per_seed, seeds, n_samples)
    return per_seed, aggregate


# ---------------------------------------------------------------------------
# Bootstrap E[max(K)] / E[min(K)] for Fig 4 reference lines
# ---------------------------------------------------------------------------

def bootstrap_expected_extremes(
    val_accs: List[float],
    K: int = 15,
    n_resamples: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute bootstrap E[max(K)] and E[min(K)] from a pool of accuracies.

    Args:
        val_accs: pool of valid accuracies (from all seeds combined).
        K: sample size per resample (default 15, matching RL-FT batch_size).
        n_resamples: number of bootstrap resamples.
        seed: RNG seed for reproducibility.

    Returns:
        {"e_max_k": float, "e_min_k": float, "K": int, "n_resamples": int}
    """
    if len(val_accs) < K:
        return {"e_max_k": None, "e_min_k": None, "K": K, "n_resamples": n_resamples}

    rng = np.random.RandomState(seed)
    pool = np.array(val_accs)
    maxes = []
    mins = []
    for _ in range(n_resamples):
        subset = rng.choice(pool, size=K, replace=True)
        maxes.append(subset.max())
        mins.append(subset.min())

    return {
        "e_max_k": float(np.mean(maxes)),
        "e_min_k": float(np.mean(mins)),
        "K": K,
        "n_resamples": n_resamples,
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _summarize_rewards(
    rewards: List[float],
    metas: List[Dict[str, Any]],
    n_samples: int,
    seed: int,
) -> Dict[str, Any]:
    """Compute summary statistics from rewards."""
    valid_accs = []
    for r, m in zip(rewards, metas):
        if r > -0.5:
            acc = m.get("validation_accuracy")
            if acc is not None:
                valid_accs.append(acc)

    n_valid = sum(1 for r in rewards if r > -0.5)
    return {
        "seed": seed,
        "n_samples": n_samples,
        "n_valid": n_valid,
        "valid_rate": n_valid / n_samples if n_samples > 0 else 0.0,
        "max_val_acc": float(np.max(valid_accs)) if valid_accs else 0.0,
        "mean_val_acc": float(np.mean(valid_accs)) if valid_accs else 0.0,
        "std_val_acc": float(np.std(valid_accs)) if valid_accs else 0.0,
    }


def _aggregate_results(
    per_seed: List[Dict[str, Any]],
    seeds: List[int],
    n_samples: int,
) -> Dict[str, Any]:
    """Compute mean+-std over seeds for aggregate output."""
    max_accs = [r["max_val_acc"] for r in per_seed]
    mean_accs = [r["mean_val_acc"] for r in per_seed]

    # Collect all valid accuracies for bootstrap
    all_valid_accs = []
    for r in per_seed:
        if "samples" in r:
            for s in r["samples"]:
                if s.get("is_valid") and s.get("val_acc") is not None:
                    all_valid_accs.append(s["val_acc"])

    result = {
        "seeds": seeds,
        "n_samples_per_seed": n_samples,
        "mean_max_val_acc": float(np.mean(max_accs)),
        "std_max_val_acc": float(np.std(max_accs)),
        "mean_mean_val_acc": float(np.mean(mean_accs)),
        "std_mean_val_acc": float(np.std(mean_accs)),
    }

    # Bootstrap E[max(K)] / E[min(K)]
    if all_valid_accs:
        bootstrap = bootstrap_expected_extremes(all_valid_accs)
        result["bootstrap"] = bootstrap

    return result


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_result(path: str, data: Dict[str, Any]):
    """Write JSON result file, creating directories as needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Written: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Baseline computation for DGPO experiments")
    parser.add_argument("--mode", choices=["random", "pretrained"], required=True,
                        help="Baseline mode")
    parser.add_argument("--benchmark", choices=["nb101", "nb201"], required=True,
                        help="NAS benchmark")
    parser.add_argument("--n-samples", type=int, default=300, help="Samples per seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                        help="Random seeds (default: 42 123 456)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Pretrained checkpoint (required for --mode pretrained)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data directory (overrides config default)")
    parser.add_argument("--reward-dataset", type=str, default="cifar10-valid",
                        help="Reward dataset for NB201 (default: cifar10-valid)")
    parser.add_argument("--variant", type=str, default=None,
                        help="Variant suffix for output files (e.g. 'filtered')")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: outputs/baselines/)")

    args = parser.parse_args()

    if args.mode == "pretrained" and not args.checkpoint:
        parser.error("--checkpoint is required for --mode pretrained")

    # Determine benchmark name for RewardComputer
    benchmark_name = "nasbench101" if args.benchmark == "nb101" else "nasbench201"

    # Determine data_dir
    data_dir = args.data_dir
    if data_dir is None:
        if args.benchmark == "nb101":
            data_dir = "data/NASBench101/raw"
        else:
            data_dir = "data/NASBench201/raw"

    # Initialize reward computer
    reward_computer = RewardComputer(
        benchmark=benchmark_name,
        data_dir=data_dir,
        reward_dataset=args.reward_dataset,
    )

    # Build output path components
    variant_suffix = f"_{args.variant}" if args.variant else ""
    base_name = f"{args.mode}_{args.benchmark}{variant_suffix}"
    output_dir = args.output_dir or "outputs/baselines"

    # Run baseline
    print(f"\n=== {args.mode.title()} baseline: {args.benchmark}{variant_suffix} ===")
    print(f"  n_samples={args.n_samples}, seeds={args.seeds}")

    if args.mode == "random":
        per_seed, aggregate = random_search_baseline(
            benchmark_name, reward_computer, args.n_samples, args.seeds,
        )
    else:
        per_seed, aggregate = pretrained_only_baseline(
            benchmark_name, args.checkpoint, reward_computer,
            args.n_samples, args.seeds,
        )

    # Write per-seed files
    for result in per_seed:
        seed = result["seed"]
        path = os.path.join(output_dir, f"{base_name}_seed{seed}.json")
        _write_result(path, result)

    # Write aggregate file
    agg_path = os.path.join(output_dir, f"{base_name}_aggregate.json")
    _write_result(agg_path, aggregate)

    print(f"\n=== Aggregate: max_val_acc={aggregate['mean_max_val_acc']:.4f}"
          f"+-{aggregate['std_max_val_acc']:.4f} ===")


if __name__ == "__main__":
    main()
