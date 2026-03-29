#!/usr/bin/env python
"""Build unified NB201 cache from the full NASBench-201 API.

Extracts per-architecture metrics for all datasets into a single lightweight
file (~3-5 MB) that replaces the need to load the full API (~12.7 GB RSS).

Run once on a high-RAM machine:
    python scripts/build_nb201_cache.py \
        --api-path ../data/NASBench201/raw/NAS-Bench-201-v1_1-096897.pth \
        --output ../data/NASBench201/raw/nb201_unified.pt

The output file is used by both dataset.py (pretraining) and rewards.py (RL-FT).
"""
from __future__ import annotations

import argparse
import datetime
import os
import random
import sys
import time
from typing import List, Optional

import numpy as np
import torch

# Ensure companion/src resolves
_companion_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _companion_root not in sys.path:
    sys.path.insert(0, _companion_root)

from src.ops import NB201_OP_PRIMITIVES, NB201_NAME_TO_IDX


# Datasets to query.  For cifar10-valid we also query "cifar10" for test_acc.
_REWARD_DATASETS = ["cifar10-valid", "cifar100", "ImageNet16-120"]


def _parse_arch_str(arch_str: str) -> List[int]:
    """Parse NB201 arch_str into node type indices [8].

    Format: |op~0|+|op~0|op~1|+|op~0|op~1|op~2|
    Returns: [input, op1, op2, op3, op4, op5, op6, output] as int indices.
    """
    nodes = ["input"]
    for step in arch_str.split("+"):
        for node in step.strip("|").split("|"):
            op_name, _ = node.split("~")
            nodes.append(op_name)
    nodes.append("output")
    return [NB201_NAME_TO_IDX[op] for op in nodes]


def _normalize_acc(value) -> Optional[float]:
    """Normalize NB201 accuracy to [0, 1]. Returns None if missing.

    NB201 API always returns accuracy as percentages (0-100 range),
    so we unconditionally divide by 100.
    """
    if value is None:
        return None
    return float(value) / 100.0


def build_cache(api_path: str, output_path: str):
    from vendored.nasbench201.api import NASBench201API

    print(f"Loading NB201 API from {api_path}...")
    t0 = time.time()
    nb201 = NASBench201API(api_path, verbose=False)
    print(f"  Loaded in {time.time() - t0:.1f}s, {len(nb201)} architectures")

    entries = {}
    t0 = time.time()

    for index in range(len(nb201)):
        info = nb201.query_meta_info_by_index(index)
        arch_str = info.arch_str
        x_indices = _parse_arch_str(arch_str)

        entry = {"index": index, "x_indices": x_indices}

        for ds in _REWARD_DATASETS:
            try:
                result = nb201.get_more_info(
                    index, ds, iepoch=None,
                    use_12epochs_result=False, is_random=False,
                )
                val_acc = _normalize_acc(result.get("valid-accuracy")
                                         or result.get("est-valid-accuracy"))
                test_acc = _normalize_acc(result.get("test-accuracy"))
            except Exception as e:
                print(f"  [WARN] index={index} ds={ds}: {e}")
                val_acc, test_acc = None, None

            # For cifar10-valid, get test_acc from "cifar10" (full-train evaluation)
            if ds == "cifar10-valid":
                try:
                    cifar10_result = nb201.get_more_info(
                        index, "cifar10", iepoch=None,
                        use_12epochs_result=False, is_random=False,
                    )
                    test_acc = _normalize_acc(cifar10_result.get("test-accuracy"))
                except Exception:
                    pass  # keep test_acc from cifar10-valid query

            entry[ds] = {"val_acc": val_acc, "test_acc": test_acc}

        entries[arch_str] = entry

        if (index + 1) % 5000 == 0:
            print(f"  Processed {index + 1}/{len(nb201)} ({time.time() - t0:.1f}s)")

    elapsed = time.time() - t0
    print(f"\nBuilt {len(entries)} entries in {elapsed:.1f}s")

    # Assemble output
    cache = {
        "version": 1,
        "benchmark": "nasbench201",
        "build_info": {
            "source": os.path.basename(api_path),
            "date": datetime.datetime.now().isoformat(),
            "is_random": False,
            "num_entries": len(entries),
        },
        "entries": entries,
    }

    # Summary statistics
    print("\nPer-dataset statistics:")
    for ds in _REWARD_DATASETS:
        accs = [e[ds]["val_acc"] for e in entries.values() if e[ds]["val_acc"] is not None]
        taccs = [e[ds]["test_acc"] for e in entries.values() if e[ds]["test_acc"] is not None]
        print(f"  {ds}:")
        print(f"    val_acc:  n={len(accs)}, min={min(accs):.4f}, max={max(accs):.4f}, mean={np.mean(accs):.4f}")
        if taccs:
            print(f"    test_acc: n={len(taccs)}, min={min(taccs):.4f}, max={max(taccs):.4f}, mean={np.mean(taccs):.4f}")

    # Spot-check: verify 5 random entries against direct API queries
    print("\nSpot-check (5 random entries):")
    sample_keys = random.sample(list(entries.keys()), min(5, len(entries)))
    for arch_str in sample_keys:
        entry = entries[arch_str]
        idx = nb201.query_index_by_arch(arch_str)
        direct = nb201.get_more_info(int(idx), "cifar10-valid", iepoch=None,
                                     use_12epochs_result=False, is_random=False)
        direct_val = _normalize_acc(direct.get("valid-accuracy"))
        cached_val = entry["cifar10-valid"]["val_acc"]
        match = "OK" if abs((direct_val or 0) - (cached_val or 0)) < 1e-6 else "MISMATCH"
        print(f"  [{match}] {arch_str[:40]}... cached={cached_val:.4f} direct={direct_val:.4f}")

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(cache, output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nSaved to {output_path} ({file_size_mb:.1f} MB)")

    assert len(entries) == 15625, f"Expected 15625 entries, got {len(entries)}"
    print("All checks passed.")


def main():
    parser = argparse.ArgumentParser(description="Build unified NB201 cache")
    parser.add_argument("--api-path", required=True,
                        help="Path to NAS-Bench-201-v1_1-096897.pth (or v1_0)")
    parser.add_argument("--output", default="data/NASBench201/raw/nb201_unified.pt",
                        help="Output path for nb201_unified.pt")
    args = parser.parse_args()

    if not os.path.isfile(args.api_path):
        print(f"ERROR: API file not found: {args.api_path}")
        sys.exit(1)

    build_cache(args.api_path, args.output)


if __name__ == "__main__":
    main()
