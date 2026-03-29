"""Reward computation via vendored NB101/NB201 APIs.

Wraps the vendored NASBench APIs to compute validation-accuracy rewards
for generated architectures. Each graph is converted from DiGress format
(node_types, adjacency) to the API-specific format, queried, and returned
as (reward_float, metadata_dict).

Invalid graphs receive reward = -1.0.  For inverse experiments, reward
negation happens in the training script, not here.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Ensure companion root is on sys.path
_companion_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _companion_root not in sys.path:
    sys.path.insert(0, _companion_root)

from src.ops import (
    NB101_OP_PRIMITIVES,
    NB201_OP_PRIMITIVES,
    NB201_NUM_VERTICES,
)


# ---------------------------------------------------------------------------
# Graph → API format conversion (inlined from legacy_adapter)
# ---------------------------------------------------------------------------

def _nb101_to_adj_ops(
    node_types: torch.Tensor, adjacency: torch.Tensor
) -> Tuple[np.ndarray, List[str]]:
    """Convert DiGress graph to NB101 (matrix, ops) for API query.

    Args:
        node_types: (n,) integer node type indices.
        adjacency: (n, n) integer adjacency matrix (possibly symmetric).

    Returns:
        (upper_triangular_matrix, ops_list) ready for api.ModelSpec.
    """
    if isinstance(adjacency, torch.Tensor):
        adj = adjacency.cpu().numpy()
    else:
        adj = np.asarray(adjacency)
    adj_upper = np.triu(adj.astype(int), k=1)

    if isinstance(node_types, torch.Tensor):
        indices = node_types.cpu().tolist()
    else:
        indices = list(node_types)

    ops = []
    for idx in indices:
        if 0 <= idx < len(NB101_OP_PRIMITIVES):
            ops.append(NB101_OP_PRIMITIVES[idx])
        else:
            raise ValueError(f"NB101 node type index {idx} out of range [0, {len(NB101_OP_PRIMITIVES)})")

    # Enforce input/output placement
    if len(ops) >= 2:
        ops[0] = "input"
        ops[-1] = "output"

    return adj_upper, ops


def _nb201_to_arch_str(node_types: torch.Tensor) -> str:
    """Convert DiGress NB201 node types to architecture string.

    NB201 arch string format: |op~0|+|op~0|op~1|+|op~0|op~1|op~2|
    DiGress NB201 has 8 nodes: 0(input), 1..6(intermediate), 7(output).
    We use nodes 1-6 for the arch string.
    """
    if isinstance(node_types, torch.Tensor):
        indices = node_types.cpu().numpy()
    else:
        indices = np.array(node_types)

    ops = []
    for i in indices:
        if 0 <= i < len(NB201_OP_PRIMITIVES):
            ops.append(NB201_OP_PRIMITIVES[i])
        else:
            ops.append("none")

    if len(ops) != NB201_NUM_VERTICES:
        return ""

    steps = ['0', '0', '1', '0', '1', '2']
    try:
        node_1 = '|' + ops[1] + '~' + steps[0] + '|'
        node_2 = '|' + ops[2] + '~' + steps[1] + '|' + ops[3] + '~' + steps[2] + '|'
        node_3 = '|' + ops[4] + '~' + steps[3] + '|' + ops[5] + '~' + steps[4] + '|' + ops[6] + '~' + steps[5] + '|'
        return node_1 + '+' + node_2 + '+' + node_3
    except IndexError:
        return ""


# ---------------------------------------------------------------------------
# RewardComputer
# ---------------------------------------------------------------------------

def _make_meta(
    valid: bool,
    error: Optional[str] = None,
    spec_hash: Optional[str] = None,
    arch_str: Optional[str] = None,
    index: Optional[int] = None,
    validation_accuracy: Optional[float] = None,
    test_accuracy: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a metadata dict with a stable schema for all outcomes."""
    return {
        "valid": valid,
        "error": error,
        "spec_hash": spec_hash,
        "arch_str": arch_str,
        "index": index,
        "validation_accuracy": validation_accuracy,
        "test_accuracy": test_accuracy,
    }


class RewardComputer:
    """Unified reward computation for NB101 and NB201.

    Wraps the vendored APIs to provide a consistent interface.
    Caches API instances (loaded once, queried many times).
    """

    def __init__(self, benchmark: str, data_dir: str, reward_dataset: str = "cifar10-valid"):
        """
        Args:
            benchmark: "nasbench101" or "nasbench201".
            data_dir: path to data directory (parent of benchmark-specific files).
            reward_dataset: for NB201, one of "cifar10-valid", "cifar100", "ImageNet16-120".
        """
        self.benchmark = benchmark
        self.data_dir = data_dir
        self.reward_dataset = reward_dataset
        self._nb101_api = None
        self._nb201_api = None
        self._nb201_lookup: Optional[Dict[str, Any]] = None  # unified cache
        self._cache: Dict[Any, Tuple[float, Dict[str, Any]]] = {}

        self._init_api()

    def _init_api(self):
        if self.benchmark == "nasbench101":
            self._init_nb101()
        elif self.benchmark == "nasbench201":
            self._init_nb201()
        else:
            raise ValueError(f"Unknown benchmark: {self.benchmark}")

    def _init_nb101(self):
        from vendored.nasbench import api as nb101_api
        self._nb101_module = nb101_api

        # Find the tfrecord file
        tfrecord_path = os.path.join(self.data_dir, "nasbench_only108.tfrecord")
        if not os.path.exists(tfrecord_path):
            # Try parent data dir patterns
            alt = os.path.join(self.data_dir, "NASBench101", "raw", "nasbench_only108.tfrecord")
            if os.path.exists(alt):
                tfrecord_path = alt
            else:
                raise FileNotFoundError(
                    f"NB101 tfrecord not found at {tfrecord_path} or {alt}. "
                    f"Set data_dir to the directory containing nasbench_only108.tfrecord."
                )

        print(f"[RewardComputer] Loading NB101 API from {tfrecord_path}...")
        self._nb101_api = nb101_api.NASBench(tfrecord_path)
        print("[RewardComputer] NB101 API loaded.")

    def _init_nb201(self):
        # Strategy 1: Try unified cache (lightweight, ~3-5 MB)
        _UNIFIED_NAME = "nb201_unified.pt"
        _SEARCH_DIRS = [
            self.data_dir,
            os.path.join(self.data_dir, ".."),
            os.path.join(self.data_dir, "NASBench201", "raw"),
            os.path.join(self.data_dir, "NASBench201"),
        ]
        for d in _SEARCH_DIRS:
            candidate = os.path.join(d, _UNIFIED_NAME)
            if os.path.isfile(candidate):
                print(f"[RewardComputer] Loading unified NB201 cache from {candidate}...")
                data = torch.load(candidate, map_location="cpu", weights_only=False)
                entries = data["entries"]
                if not entries:
                    raise ValueError(
                        "[RewardComputer] Unified cache has no entries (empty or corrupted). "
                        "Rebuild with: python scripts/build_nb201_cache.py"
                    )

                # Validate a bounded sample of entries (up to 50) for the requested dataset
                _entries_list = list(entries.values())
                _check = _entries_list[:min(50, len(_entries_list))]
                _n_missing_key = sum(1 for e in _check if self.reward_dataset not in e)
                _n_missing_val = sum(
                    1 for e in _check
                    if self.reward_dataset in e and e[self.reward_dataset].get("val_acc") is None
                )
                if _n_missing_key == len(_check):
                    available = [k for k in _check[0] if isinstance(_check[0][k], dict)]
                    raise ValueError(
                        f"[RewardComputer] Unified cache missing '{self.reward_dataset}'. "
                        f"Available: {available}. Rebuild with: "
                        f"python scripts/build_nb201_cache.py"
                    )
                if _n_missing_val > 0:
                    raise ValueError(
                        f"[RewardComputer] Unified cache has '{self.reward_dataset}' "
                        f"but {_n_missing_val}/{len(_check)} sampled entries have "
                        f"val_acc=None (incomplete cache). Rebuild with full API."
                    )

                self._nb201_lookup = entries
                print(f"[RewardComputer] Unified cache loaded: {len(self._nb201_lookup)} entries "
                      f"(reward_dataset={self.reward_dataset})")
                return

        # Strategy 2: Fall back to full API (requires ~12.7 GB RSS)
        from vendored.nasbench201 import api as nb201_api
        self._nb201_module = nb201_api

        _NB201_FILENAMES = [
            "NAS-Bench-201-v1_1-096897.pth",
            "NAS-Bench-201-v1_0-e61699.pth",
        ]
        api_path = None
        for d in _SEARCH_DIRS:
            for fname in _NB201_FILENAMES:
                candidate = os.path.join(d, fname)
                if os.path.exists(candidate):
                    api_path = candidate
                    break
            if api_path:
                break
        if api_path is None:
            raise FileNotFoundError(
                f"NB201 unified cache or API .pth not found. "
                f"Searched dirs: {_SEARCH_DIRS}. "
                f"Build the cache with: python scripts/build_nb201_cache.py"
            )

        print(f"[RewardComputer] Loading NB201 API from {api_path} (WARNING: ~12.7 GB RSS)...")
        self._nb201_api = nb201_api.NASBench201API(api_path, verbose=False)
        print("[RewardComputer] NB201 API loaded.")

    def compute_rewards(
        self, graphs: Sequence[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        """Compute rewards for a batch of (node_types, adjacency) pairs.

        Returns:
            (rewards, metadata_list) where each reward is val_acc in [0,1]
            or -1.0 for invalid graphs.
        """
        rewards = []
        metas = []
        for node_types, adjacency in graphs:
            r, m = self._compute_single(node_types, adjacency)
            rewards.append(r)
            metas.append(m)
        return rewards, metas

    def _compute_single(
        self, node_types: torch.Tensor, adjacency: torch.Tensor
    ) -> Tuple[float, Dict[str, Any]]:
        if self.benchmark == "nasbench101":
            return self._compute_nb101(node_types, adjacency)
        else:
            return self._compute_nb201(node_types, adjacency)

    def _compute_nb101(
        self, node_types: torch.Tensor, adjacency: torch.Tensor
    ) -> Tuple[float, Dict[str, Any]]:
        api_mod = self._nb101_module

        try:
            matrix, ops = _nb101_to_adj_ops(node_types, adjacency)
        except (ValueError, IndexError):
            return -1.0, _make_meta(False, error="conversion_failed")

        try:
            spec = api_mod.ModelSpec(matrix=matrix, ops=ops)
            spec_hash = self._nb101_api._hash_spec(spec)
        except (ValueError, TypeError):
            return -1.0, _make_meta(False, error="invalid_spec")

        # Check cache
        cached = self._cache.get(spec_hash)
        if cached is not None:
            return cached

        try:
            fixed_metrics, computed_metrics = self._nb101_api.get_metrics_from_spec(spec)
        except api_mod.OutOfDomainError:
            result = (-1.0, _make_meta(False, error="out_of_domain", spec_hash=spec_hash))
            self._cache[spec_hash] = result
            return result
        except (KeyError, ValueError):
            result = (-1.0, _make_meta(False, error="spec_not_found", spec_hash=spec_hash))
            self._cache[spec_hash] = result
            return result

        # Extract final validation accuracy (average across trials at epoch 108)
        val_acc = self._extract_nb101_val_acc(computed_metrics)
        if val_acc is None:
            result = (-1.0, _make_meta(False, error="missing_metrics", spec_hash=spec_hash))
            self._cache[spec_hash] = result
            return result

        result = (val_acc, _make_meta(
            True, spec_hash=spec_hash,
            validation_accuracy=val_acc,
            test_accuracy=self._extract_nb101_test_acc(computed_metrics),
        ))
        self._cache[spec_hash] = result
        return result

    @staticmethod
    def _extract_nb101_val_acc(computed_metrics: dict) -> Optional[float]:
        """Extract final validation accuracy from NB101 computed_metrics."""
        # computed_metrics is {epoch: [trial_dicts]}
        # We want epoch 108 (the final epoch in NB101)
        for epoch in [108, 36, 12]:
            trials = computed_metrics.get(epoch)
            if trials:
                accs = [t.get("final_validation_accuracy") for t in trials
                        if t.get("final_validation_accuracy") is not None]
                if accs:
                    return float(np.mean(accs))
        return None

    @staticmethod
    def _extract_nb101_test_acc(computed_metrics: dict) -> Optional[float]:
        for epoch in [108, 36, 12]:
            trials = computed_metrics.get(epoch)
            if trials:
                accs = [t.get("final_test_accuracy") for t in trials
                        if t.get("final_test_accuracy") is not None]
                if accs:
                    return float(np.mean(accs))
        return None

    def _compute_nb201(
        self, node_types: torch.Tensor, adjacency: torch.Tensor
    ) -> Tuple[float, Dict[str, Any]]:
        try:
            arch_str = _nb201_to_arch_str(node_types)
        except (ValueError, IndexError):
            return -1.0, _make_meta(False, error="conversion_failed")

        if not arch_str:
            return -1.0, _make_meta(False, error="invalid_arch_str")

        # Check cache (tuple key avoids string-collision risk)
        cache_key = (arch_str, self.reward_dataset)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        # --- Unified cache path (lightweight, no API needed) ---
        if self._nb201_lookup is not None:
            entry = self._nb201_lookup.get(arch_str)
            if entry is None:
                result = (-1.0, _make_meta(False, error="arch_not_found", arch_str=arch_str))
                self._cache[cache_key] = result
                return result

            ds_metrics = entry.get(self.reward_dataset, {})
            val_acc = ds_metrics.get("val_acc")
            if val_acc is None:
                result = (-1.0, _make_meta(False, error="missing_valid_accuracy", arch_str=arch_str))
                self._cache[cache_key] = result
                return result

            result = (val_acc, _make_meta(
                True, arch_str=arch_str, index=entry.get("index"),
                validation_accuracy=val_acc,
                test_accuracy=ds_metrics.get("test_acc"),
            ))
            self._cache[cache_key] = result
            return result

        # --- Full API path (fallback, requires ~12.7 GB RSS) ---
        idx = self._nb201_api.query_index_by_arch(arch_str)
        if idx in (-1, None):
            result = (-1.0, _make_meta(False, error="arch_not_found", arch_str=arch_str))
            self._cache[cache_key] = result
            return result

        try:
            info = self._nb201_api.get_more_info(
                int(idx), self.reward_dataset, iepoch=None,
                use_12epochs_result=False, is_random=False,
            )
        except Exception:
            result = (-1.0, _make_meta(False, error="query_failed", arch_str=arch_str))
            self._cache[cache_key] = result
            return result

        val_acc_raw = info.get("valid-accuracy", None)
        if val_acc_raw is None:
            val_acc_raw = info.get("est-valid-accuracy", None)
        if val_acc_raw is None:
            result = (-1.0, _make_meta(False, error="missing_valid_accuracy", arch_str=arch_str))
            self._cache[cache_key] = result
            return result

        # NB201 API always returns percentages (0-100)
        val_acc = float(val_acc_raw) / 100.0

        result = (val_acc, _make_meta(
            True, arch_str=arch_str, index=int(idx),
            validation_accuracy=val_acc,
            test_accuracy=_normalize_acc(info.get("test-accuracy")),
        ))
        self._cache[cache_key] = result
        return result


def _normalize_acc(val: Any) -> Optional[float]:
    """Normalize NB201 accuracy to [0, 1].

    NB201 API always returns percentages (0-100), so unconditionally divide.
    """
    if val is None:
        return None
    return float(val) / 100.0
