"""Data loading for NB101 and NB201 benchmarks.

Replaces the 5-layer legacy data pipeline with direct loading functions.
Uses a two-tier strategy:
  1. Try to load from a pre-processed cache (torch.save'd list of PyG Data objects,
     produced by the original codebase or by this module on first run).
  2. If no cache exists, load from the vendored API, process, and save cache.

Produces dense tensors: X [n_max, num_node_types] one-hot,
E [n_max, n_max, num_edge_types] one-hot (symmetrized / undirected).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import sys as _sys
# Ensure companion root is on sys.path so `src.ops` resolves regardless of
# how this file is invoked (python src/dataset.py, python -m src.dataset,
# or import from companion root).
_companion_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _companion_root not in _sys.path:
    _sys.path.insert(0, _companion_root)

from src.ops import (  # noqa: E402
    ADJACENCY_NB201,
    NB101_NAME_TO_IDX,
    NB101_NUM_VERTICES,
    NB101_OP_PRIMITIVES,
    NB201_NAME_TO_IDX,
    NB201_NUM_VERTICES,
    NB201_OP_PRIMITIVES,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GraphData:
    """Single graph as dense tensors."""
    X: torch.Tensor       # [n, num_node_types] one-hot float32
    E: torch.Tensor       # [n, n, num_edge_types] one-hot float32
    val_acc: float         # validation accuracy


@dataclass
class DatasetInfo:
    """Metadata about a loaded dataset."""
    node_types: int
    edge_types: int
    max_nodes: int
    node_distribution: torch.Tensor   # [num_node_types] marginal
    edge_marginals: torch.Tensor      # [num_edge_types] marginal


class GraphDataset(Dataset):
    """Simple dataset wrapping a list of GraphData."""

    def __init__(self, graphs: List[GraphData]):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def _collate_fn(batch: List[GraphData]) -> Dict[str, torch.Tensor]:
    """Pad graphs to max_nodes and stack into dense batches."""
    B = len(batch)
    max_n = max(g.X.shape[0] for g in batch)
    num_node_types = batch[0].X.shape[1]
    num_edge_types = batch[0].E.shape[2]

    X = torch.zeros(B, max_n, num_node_types)
    E = torch.zeros(B, max_n, max_n, num_edge_types)
    E[:, :, :, 0] = 1.0  # no-edge channel for padding positions
    y = torch.zeros(B, 0)
    node_mask = torch.zeros(B, max_n, dtype=torch.bool)

    for i, g in enumerate(batch):
        n = g.X.shape[0]
        X[i, :n] = g.X
        E[i, :n, :n] = g.E
        node_mask[i, :n] = True

    return {"X": X, "E": E, "y": y, "node_mask": node_mask}


def _symmetrize_adjacency(adj: np.ndarray) -> np.ndarray:
    """Make adjacency undirected: adj[i,j] = adj[j,i] = max(adj[i,j], adj[j,i])."""
    return np.maximum(adj, adj.T)


def _adj_to_edge_onehot(adj: np.ndarray, num_edge_types: int = 2) -> torch.Tensor:
    """Convert binary adjacency to [n, n, num_edge_types] one-hot.
    Index 0 = no edge, index 1 = edge. Symmetrized (undirected)."""
    sym = _symmetrize_adjacency(adj)
    n = sym.shape[0]
    E = torch.zeros(n, n, num_edge_types)
    edge_mask = torch.from_numpy(sym.astype(np.int64))
    E[:, :, 0] = 1.0 - edge_mask.float()
    E[:, :, 1] = edge_mask.float()
    return E


def _edge_index_to_dense_adj(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    """Convert PyG edge_index [2, E] to dense adjacency [n, n]."""
    adj = np.zeros((num_nodes, num_nodes), dtype=np.int32)
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    adj[src, dst] = 1
    return adj


# ---------------------------------------------------------------------------
# Statistics & splitting
# ---------------------------------------------------------------------------

def _compute_dataset_info(
    graphs: List[GraphData], num_node_types: int, num_edge_types: int, max_nodes: int
) -> DatasetInfo:
    """Compute marginal distributions from a list of graphs."""
    node_counts = torch.zeros(num_node_types)
    edge_counts = torch.zeros(num_edge_types)
    total_nodes = 0
    total_edges = 0

    for g in graphs:
        node_counts += g.X.sum(dim=0)
        total_nodes += g.X.shape[0]
        edge_counts += g.E.sum(dim=(0, 1))
        total_edges += g.E.shape[0] * g.E.shape[1]

    node_distribution = node_counts / total_nodes if total_nodes > 0 else node_counts
    edge_marginals = edge_counts / total_edges if total_edges > 0 else edge_counts

    return DatasetInfo(
        node_types=num_node_types,
        edge_types=num_edge_types,
        max_nodes=max_nodes,
        node_distribution=node_distribution,
        edge_marginals=edge_marginals,
    )


def _split_data(
    graphs: List[GraphData], ratios: Tuple[float, ...], seed: int
) -> List[List[GraphData]]:
    """Split graphs into subsets according to ratios, with deterministic shuffling."""
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(len(graphs), generator=g).tolist()
    shuffled = [graphs[i] for i in perm]

    n = len(graphs)
    counts = [int(ratio * n) for ratio in ratios]
    # Assign remainder to the first non-zero split to avoid silent data loss
    remainder = n - sum(counts)
    if remainder > 0:
        for i, ratio in enumerate(ratios):
            if ratio > 0:
                counts[i] += remainder
                break
    splits = []
    start = 0
    for count in counts:
        splits.append(shuffled[start:start + count])
        start += count
    return splits


def _make_loaders(
    graphs: List[GraphData],
    split_ratios: Tuple[float, ...],
    split_seed: int,
    num_node_types: int,
    num_edge_types: int,
    max_nodes: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, DatasetInfo]:
    """Split graphs and create DataLoaders + DatasetInfo."""
    splits = _split_data(graphs, split_ratios, split_seed)
    train_data, val_data = splits[0], splits[1]
    test_data = splits[2] if len(splits) > 2 else []

    dataset_info = _compute_dataset_info(graphs, num_node_types, num_edge_types, max_nodes)

    train_loader = DataLoader(GraphDataset(train_data), batch_size=batch_size, shuffle=True, collate_fn=_collate_fn)
    val_loader = DataLoader(GraphDataset(val_data), batch_size=batch_size, shuffle=False, collate_fn=_collate_fn)
    test_loader = DataLoader(GraphDataset(test_data), batch_size=batch_size, shuffle=False, collate_fn=_collate_fn)

    return train_loader, val_loader, test_loader, dataset_info


# ---------------------------------------------------------------------------
# Cache conversion: PyG Data (legacy cache) -> GraphData (companion format)
# ---------------------------------------------------------------------------

def _pyg_to_graph_data_nb101(pyg_data, num_node_types: int, num_edge_types: int) -> GraphData:
    """Convert a legacy-cached PyG Data object (NB101) to GraphData."""
    # x: [n] int64 op indices; x_binary: [n, 5] one-hot (may exist)
    if hasattr(pyg_data, 'x_binary') and pyg_data.x_binary is not None:
        X = pyg_data.x_binary.float()
    else:
        x_indices = pyg_data.x.long()
        X = F.one_hot(x_indices, num_classes=num_node_types).float()

    # Pad to NB101_NUM_VERTICES if needed
    n = X.shape[0]
    if n < NB101_NUM_VERTICES:
        pad = NB101_NUM_VERTICES - n
        X = F.pad(X, (0, 0, 0, pad), value=0)
        # Set padding nodes to "output" one-hot
        X[n:, NB101_NAME_TO_IDX["output"]] = 1.0

    # Edge features from edge_index
    adj = _edge_index_to_dense_adj(pyg_data.edge_index, NB101_NUM_VERTICES)
    E = _adj_to_edge_onehot(adj, num_edge_types)

    val_acc = float(pyg_data.val_acc.item()) if hasattr(pyg_data, 'val_acc') else 0.0
    return GraphData(X=X, E=E, val_acc=val_acc)


def _pyg_to_graph_data_nb201(pyg_data, num_node_types: int, num_edge_types: int,
                              E_template: torch.Tensor) -> GraphData:
    """Convert a legacy-cached PyG Data object (NB201) to GraphData."""
    if hasattr(pyg_data, 'x_binary') and pyg_data.x_binary is not None:
        X = pyg_data.x_binary.float()
    else:
        x_indices = pyg_data.x.long()
        X = F.one_hot(x_indices, num_classes=num_node_types).float()

    E = E_template.clone()
    val_acc = float(pyg_data.val_acc.item()) if hasattr(pyg_data, 'val_acc') else 0.0
    return GraphData(X=X, E=E, val_acc=val_acc)


# ---------------------------------------------------------------------------
# Cache search: find existing cache files (legacy or companion)
# ---------------------------------------------------------------------------

_NB101_CACHE_SEARCH_PATHS = [
    # companion local cache
    ("companion_processed", "processed", "nb101_cache.pt"),
    # original codebase legacy cache (relative to data_dir)
    ("legacy_raw", ".", "cache"),
    ("legacy_raw_parent", "..", "cache"),
    # original codebase processed cache
    ("legacy_processed_sibling", "../processed", "cache"),
]

_NB201_CACHE_SEARCH_PATHS = [
    # unified cache (preferred — lightweight, all datasets)
    ("unified", ".", "nb201_unified.pt"),
    ("unified_parent", "..", "nb201_unified.pt"),
    # companion local cache
    ("companion_processed", "processed", "nb201_cache.pt"),
    # original codebase legacy cache (relative to data_dir)
    ("legacy_processed", ".", "cache_cifar10_valid_converged"),
    ("legacy_processed_parent", "..", "cache_cifar10_valid_converged"),
    ("legacy_processed_sibling", "../processed", "cache_cifar10_valid_converged"),
]


def _find_cache(data_dir: str, search_paths: list, extra_dirs: Optional[List[str]] = None) -> Optional[str]:
    """Search for existing cache files in data_dir and optional extra directories."""
    dirs_to_search = [data_dir] + (extra_dirs or [])
    for base_dir in dirs_to_search:
        if not os.path.isdir(base_dir):
            continue
        for name, subdir, filename in search_paths:
            path = os.path.join(base_dir, subdir, filename)
            if os.path.isfile(path):
                return os.path.abspath(path)
            real = os.path.realpath(path)
            if os.path.isfile(real):
                return real
    return None


def _extra_cache_dirs(data_dir: str, benchmark: str) -> List[str]:
    """Compute extra directories to search for caches.

    When companion/data/<benchmark>/raw is the data_dir, the legacy cache
    may live under <repo_root>/data/<benchmark>/raw instead.
    """
    _MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    companion_root = os.path.dirname(_MODULE_DIR)
    repo_root = os.path.dirname(companion_root)
    extras = []
    for sub in [f"data/{benchmark}/raw", f"data/{benchmark}"]:
        d = os.path.join(repo_root, sub)
        if os.path.isdir(d) and os.path.abspath(d) != os.path.abspath(data_dir):
            extras.append(d)
    return extras


# ---------------------------------------------------------------------------
# NB101 loading
# ---------------------------------------------------------------------------

def _load_nb101_from_api(data_dir: str, limit: Optional[int] = None) -> list:
    """Load raw PyG-like data from vendored NB101 API. Returns list of dicts."""
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    from vendored.nasbench.api import NASBench

    tfrecord_path = os.path.join(data_dir, "nasbench_only108.tfrecord")
    if not os.path.isfile(tfrecord_path):
        raise FileNotFoundError(f"NB101 tfrecord not found at {tfrecord_path}")

    print(f"Loading NB101 API from {tfrecord_path} (this takes ~60s)...")
    nb = NASBench(tfrecord_path)

    epoch = 108
    raw_data = []
    count = 0
    for unique_hash in nb.hash_iterator():
        fixed_metrics, computed_metrics = nb.get_metrics_from_hash(unique_hash)
        matrix = fixed_metrics["module_adjacency"]
        ops = fixed_metrics["module_operations"]
        num_nodes = len(ops)

        val_acc = np.mean([
            computed_metrics[epoch][i]["final_validation_accuracy"]
            for i in range(len(computed_metrics[epoch]))
        ])
        test_acc = np.mean([
            computed_metrics[epoch][i]["final_test_accuracy"]
            for i in range(len(computed_metrics[epoch]))
        ])

        node_indices = torch.tensor([NB101_NAME_TO_IDX[op] for op in ops], dtype=torch.long)
        if num_nodes < NB101_NUM_VERTICES:
            node_indices = F.pad(node_indices, (0, NB101_NUM_VERTICES - num_nodes),
                                 value=NB101_NAME_TO_IDX["output"])
            pad_matrix = np.zeros((NB101_NUM_VERTICES, NB101_NUM_VERTICES), dtype=int)
            pad_matrix[:num_nodes, :num_nodes] = matrix
            matrix = pad_matrix

        edge_index = torch.tensor(np.array(np.nonzero(matrix)), dtype=torch.long)
        x_binary = F.one_hot(node_indices, num_classes=len(NB101_OP_PRIMITIVES))

        raw_data.append({
            "x": node_indices,
            "x_binary": x_binary,
            "edge_index": edge_index,
            "val_acc": torch.tensor([val_acc]),
            "acc": torch.tensor([test_acc]),
            "num_nodes": num_nodes,
        })

        count += 1
        if limit is not None and count >= limit:
            break

    return raw_data


def load_nb101(
    data_dir: str,
    split_seed: int = 0,
    filter_threshold: Optional[float] = None,
    limit: Optional[int] = None,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, DataLoader, DatasetInfo]:
    """Load NB101 dataset.

    Searches for a pre-processed cache first (from original codebase or previous run).
    Falls back to loading from vendored API + generating cache.

    Args:
        data_dir: Directory containing nasbench_only108.tfrecord (or a cache)
        split_seed: Random seed for train/val/test split
        filter_threshold: If set, keep only architectures with val_acc < threshold
        limit: If set, load only this many graphs (for debugging)
        batch_size: Batch size for DataLoaders

    Returns:
        (train_loader, val_loader, test_loader, dataset_info)
    """
    num_node_types = len(NB101_OP_PRIMITIVES)  # 5
    num_edge_types = 2

    # Strategy 1: Try to find an existing cache
    cache_path = _find_cache(data_dir, _NB101_CACHE_SEARCH_PATHS,
                             extra_dirs=_extra_cache_dirs(data_dir, "NASBench101"))

    if cache_path is not None:
        print(f"Loading NB101 from cache: {cache_path}")
        raw_data = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(f"  Loaded {len(raw_data)} cached graphs")
    else:
        # Strategy 2: Load from vendored API (slow, requires TF)
        raw_data = _load_nb101_from_api(data_dir, limit=limit)

        # Save companion cache for future runs (only if full dataset)
        if limit is None:
            cache_dir = os.path.join(data_dir, "processed")
            os.makedirs(cache_dir, exist_ok=True)
            cache_out = os.path.join(cache_dir, "nb101_cache.pt")
            print(f"Saving NB101 companion cache to {cache_out}")
            torch.save(raw_data, cache_out)

    # Convert to GraphData (handles both legacy PyG Data and new dict format)
    graphs: List[GraphData] = []
    for item in raw_data:
        # Legacy cache: PyG Data objects; new cache: dicts
        if hasattr(item, 'x'):
            # PyG Data or namespace-like object
            g = _pyg_to_graph_data_nb101(item, num_node_types, num_edge_types)
        else:
            # Dict format (our cache)
            x_binary = item["x_binary"]
            edge_index = item["edge_index"]
            adj = _edge_index_to_dense_adj(edge_index, NB101_NUM_VERTICES)
            g = GraphData(
                X=x_binary.float(),
                E=_adj_to_edge_onehot(adj, num_edge_types),
                val_acc=float(item["val_acc"].item()),
            )

        if filter_threshold is not None and g.val_acc >= filter_threshold:
            continue
        graphs.append(g)

        if limit is not None and len(graphs) >= limit:
            break

    print(f"Prepared {len(graphs)} NB101 graphs" +
          (f" (filtered < {filter_threshold})" if filter_threshold else ""))

    return _make_loaders(graphs, (0.9, 0.05, 0.05), split_seed,
                         num_node_types, num_edge_types, NB101_NUM_VERTICES, batch_size)


# ---------------------------------------------------------------------------
# NB201 loading
# ---------------------------------------------------------------------------

def _load_nb201_from_api(data_dir: str, limit: Optional[int] = None) -> list:
    """Load raw data from vendored NB201 API. Returns list of dicts."""
    from vendored.nasbench201.api import NASBench201API

    pth_candidates = [
        os.path.join(data_dir, "NAS-Bench-201-v1_1-096897.pth"),
        os.path.join(data_dir, "NAS-Bench-201-v1_0-e61699.pth"),
    ]
    pth_path = None
    for p in pth_candidates:
        if os.path.isfile(p):
            pth_path = p
            break
    if pth_path is None:
        raise FileNotFoundError(f"NB201 .pth file not found. Searched: {pth_candidates}")

    print(f"Loading NB201 API from {pth_path} (requires ~5GB RAM)...")
    nb201 = NASBench201API(pth_path)

    raw_data = []
    count = 0
    for index in range(len(nb201)):
        info = nb201.query_meta_info_by_index(index)
        arch_str = info.arch_str

        # 200-epoch converged average (use_12epochs_result=False, is_random=False)
        # Matches original NASBench201.train_and_eval() which stores valid_acc_avg
        more_info = nb201.get_more_info(index, "cifar10-valid", None, False, False)
        val_acc = more_info["valid-accuracy"] / 100.0

        # Parse arch_str: |op~0|+|op~0|op~1|+|op~0|op~1|op~2|
        nodes = ["input"]
        for step in arch_str.split("+"):
            for node in step.strip("|").split("|"):
                op_name, _ = node.split("~")
                nodes.append(op_name)
        nodes.append("output")

        node_indices = torch.tensor([NB201_NAME_TO_IDX[op] for op in nodes], dtype=torch.long)
        x_binary = F.one_hot(node_indices, num_classes=len(NB201_OP_PRIMITIVES))
        edge_index = torch.tensor(np.array(np.nonzero(ADJACENCY_NB201)), dtype=torch.long)

        raw_data.append({
            "x": node_indices,
            "x_binary": x_binary,
            "edge_index": edge_index,
            "val_acc": torch.tensor([val_acc]),
            "num_nodes": NB201_NUM_VERTICES,
        })

        count += 1
        if limit is not None and count >= limit:
            break

    return raw_data


def load_nb201(
    data_dir: str,
    split_seed: int = 0,
    filter_threshold: Optional[float] = None,
    limit: Optional[int] = None,
    batch_size: int = 64,
    reward_dataset: str = "cifar10-valid",
) -> Tuple[DataLoader, DataLoader, DataLoader, DatasetInfo]:
    """Load NB201 dataset.

    Searches for caches in priority order: unified cache (nb201_unified.pt),
    companion cache, legacy cache. Falls back to vendored API if none found.

    Args:
        data_dir: Directory containing NB201 data files (cache or .pth)
        split_seed: Random seed for train/val split
        filter_threshold: If set, keep only architectures with val_acc < threshold
        limit: If set, load only this many graphs (for debugging)
        batch_size: Batch size for DataLoaders
        reward_dataset: Which dataset's val_acc to use for GraphData.val_acc
            (affects filtering). One of "cifar10-valid", "cifar100", "ImageNet16-120".

    Returns:
        (train_loader, val_loader, test_loader, dataset_info)
        Note: test_loader is empty for NB201 (small dataset, 90/10 split)
    """
    num_node_types = len(NB201_OP_PRIMITIVES)  # 7
    num_edge_types = 2
    E_template = _adj_to_edge_onehot(ADJACENCY_NB201, num_edge_types)

    # Strategy 1: Try to find an existing cache
    cache_path = _find_cache(data_dir, _NB201_CACHE_SEARCH_PATHS,
                             extra_dirs=_extra_cache_dirs(data_dir, "NASBench201"))

    if cache_path is not None:
        print(f"Loading NB201 from cache: {cache_path}")
        raw_data = torch.load(cache_path, map_location="cpu", weights_only=False)
        _is_unified = isinstance(raw_data, dict) and raw_data.get("version")
        _n = len(raw_data.get("entries", {})) if _is_unified else len(raw_data)
        print(f"  Loaded {_n} cached graphs" + (" (unified)" if _is_unified else ""))
    else:
        # Strategy 2: Load from vendored API (slow, ~5GB RAM)
        raw_data = _load_nb201_from_api(data_dir, limit=limit)

        # Save companion cache for future runs (only if full dataset)
        if limit is None:
            cache_dir = os.path.join(data_dir, "processed")
            os.makedirs(cache_dir, exist_ok=True)
            cache_out = os.path.join(cache_dir, "nb201_cache.pt")
            print(f"Saving NB201 companion cache to {cache_out}")
            torch.save(raw_data, cache_out)

    # Convert to GraphData — detect format and branch
    graphs: List[GraphData] = []
    is_unified = isinstance(raw_data, dict) and raw_data.get("version")

    if is_unified:
        # Validate that the requested dataset exists in the cache
        if not raw_data["entries"]:
            raise ValueError(
                "Unified cache has no entries (empty or corrupted). "
                "Rebuild with: python scripts/build_nb201_cache.py"
            )
        # Validate a bounded sample of entries (up to 50) for the requested dataset
        _entries_list = list(raw_data["entries"].values())
        _check_entries = _entries_list[:min(50, len(_entries_list))]
        _n_missing_key = 0
        _n_missing_val = 0
        for _e in _check_entries:
            if reward_dataset not in _e:
                _n_missing_key += 1
            elif _e[reward_dataset].get("val_acc") is None:
                _n_missing_val += 1
        if _n_missing_key == len(_check_entries):
            available = [k for k in _check_entries[0] if isinstance(_check_entries[0][k], dict)]
            raise ValueError(
                f"Unified cache does not contain metrics for '{reward_dataset}'. "
                f"Available datasets: {available}. "
                f"Rebuild cache with: python scripts/build_nb201_cache.py"
            )
        if _n_missing_val > 0:
            raise ValueError(
                f"Unified cache has '{reward_dataset}' but {_n_missing_val}/{len(_check_entries)} "
                f"sampled entries have val_acc=None. "
                f"Cache was likely built from legacy data (cifar10-valid only). "
                f"Rebuild with full API: python scripts/build_nb201_cache.py"
            )

        print(f"  Using unified cache (reward_dataset={reward_dataset})")
        for entry in raw_data["entries"].values():
            x_indices = torch.tensor(entry["x_indices"], dtype=torch.long)
            x_binary = F.one_hot(x_indices, num_classes=num_node_types).float()
            val_acc = entry[reward_dataset]["val_acc"]

            if filter_threshold is not None and val_acc >= filter_threshold:
                continue
            graphs.append(GraphData(X=x_binary, E=E_template.clone(), val_acc=val_acc))

            if limit is not None and len(graphs) >= limit:
                break
    else:
        for item in raw_data:
            if hasattr(item, 'x'):
                # Legacy PyG Data object
                g = _pyg_to_graph_data_nb201(item, num_node_types, num_edge_types, E_template)
            else:
                # Dict format (companion cache)
                g = GraphData(
                    X=item["x_binary"].float(),
                    E=E_template.clone(),
                    val_acc=float(item["val_acc"].item()),
                )

            if filter_threshold is not None and g.val_acc >= filter_threshold:
                continue
            graphs.append(g)

            if limit is not None and len(graphs) >= limit:
                break

    print(f"Prepared {len(graphs)} NB201 graphs" +
          (f" (filtered < {filter_threshold})" if filter_threshold else ""))

    return _make_loaders(graphs, (0.9, 0.1, 0.0), split_seed,
                         num_node_types, num_edge_types, NB201_NUM_VERTICES, batch_size)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    companion_root = os.path.dirname(script_dir)

    print("=" * 60)
    print("Dataset Smoke Tests")
    print("=" * 60)

    # Repo root (parent of companion/)
    repo_root = os.path.dirname(companion_root)

    def _find_data_dir(subdirs: list) -> str:
        """Search common locations for benchmark data."""
        candidates = []
        for sub in subdirs:
            candidates.append(os.path.join(companion_root, "data", sub))
            candidates.append(os.path.join(repo_root, "data", sub))
        for c in candidates:
            if os.path.isdir(c):
                return c
        return candidates[0]  # fallback to first candidate

    # --- NB201 test ---
    nb201_data_dir = _find_data_dir(["NASBench201/raw", "NASBench201"])
    print(f"\n--- NB201 (data_dir={nb201_data_dir}) ---")
    train, val, test, info = load_nb201(nb201_data_dir, limit=10, batch_size=5)
    batch = next(iter(train))
    X, E = batch["X"], batch["E"]
    print(f"X: {X.shape} (expect [B, 8, 7]), E: {E.shape} (expect [B, 8, 8, 2])")
    assert X.shape[1:] == (8, 7), f"X shape mismatch: {X.shape}"
    assert E.shape[1:] == (8, 8, 2), f"E shape mismatch: {E.shape}"
    assert len(test) == 0, f"Expected empty test split, got {len(test)}"
    print("NB201 smoke test PASSED!")

    # --- NB101 test (only if tfrecord available or cache exists) ---
    nb101_data_dir = _find_data_dir(["NASBench101/raw", "NASBench101"])
    print(f"\n--- NB101 (data_dir={nb101_data_dir}) ---")
    train, val, test, info = load_nb101(nb101_data_dir, limit=10, batch_size=5)
    batch = next(iter(train))
    X, E = batch["X"], batch["E"]
    print(f"X: {X.shape} (expect [B, 7, 5]), E: {E.shape} (expect [B, 7, 7, 2])")
    assert X.shape[1:] == (7, 5), f"X shape mismatch: {X.shape}"
    assert E.shape[1:] == (7, 7, 2), f"E shape mismatch: {E.shape}"
    print("NB101 smoke test PASSED!")
