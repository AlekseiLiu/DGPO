#!/usr/bin/env python3
"""Pre-flight check: verify data files, imports, and cache integrity before running experiments."""

import os
import sys

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"

errors = 0
warnings = 0


def check(label, ok, msg_fail, warn_only=False):
    global errors, warnings
    if ok:
        print(f"  [{PASS}] {label}")
    elif warn_only:
        warnings += 1
        print(f"  [{WARN}] {label}: {msg_fail}")
    else:
        errors += 1
        print(f"  [{FAIL}] {label}: {msg_fail}")


def main():
    global errors, warnings

    # Determine repo root (script lives in scripts/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    # Default data dirs (relative to repo_root, matching config defaults)
    nb201_dir = os.path.join(repo_root, "data", "NASBench201", "raw")
    nb101_dir = os.path.join(repo_root, "data", "NASBench101", "raw")

    # Allow override via env vars
    nb201_dir = os.environ.get("NB201_DATA_DIR", nb201_dir)
    nb101_dir = os.environ.get("NB101_DATA_DIR", nb101_dir)

    # --- 1. Import checks ---
    print("\n1. Import checks")

    try:
        import torch
        check("torch", True, "")
        torch_ver = torch.__version__
        check(f"torch version ({torch_ver})", True, "")
    except ImportError:
        check("torch", False, "torch not installed")

    try:
        import yaml
        check("pyyaml", True, "")
    except ImportError:
        check("pyyaml", False, "pyyaml not installed")

    # Source imports (must run from repo root or have it on path)
    sys.path.insert(0, repo_root)

    src_modules = [
        ("src.model", "GraphTransformer"),
        ("src.diffusion", "PredefinedNoiseScheduleDiscrete"),
        ("src.dataset", "load_nb201"),
        ("src.rewards", "RewardComputer"),
        ("src.evaluate", "sample_architectures"),
        ("src.baselines", "random_search_baseline"),
        ("src.ops", "NB101_OP_PRIMITIVES"),
        ("src.utils", "PlaceHolder"),
        ("src.train_pretrain", None),
        ("src.train_rlft", None),
    ]

    for mod_name, attr in src_modules:
        try:
            mod = __import__(mod_name, fromlist=[attr] if attr else [""])
            if attr:
                assert hasattr(mod, attr), f"{attr} not found in {mod_name}"
            check(f"{mod_name}" + (f".{attr}" if attr else ""), True, "")
        except Exception as e:
            check(f"{mod_name}" + (f".{attr}" if attr else ""), False, str(e))

    # --- 2. Data file checks ---
    print("\n2. Data file checks")

    # NB201
    nb201_api_path = os.path.join(nb201_dir, "NAS-Bench-201-v1_1-096897.pth")
    nb201_cache_path = os.path.join(nb201_dir, "nb201_unified.pt")

    check("NB201 API file", os.path.isfile(nb201_api_path),
          f"not found at {nb201_api_path}", warn_only=True)
    check("NB201 unified cache", os.path.isfile(nb201_cache_path),
          f"not found at {nb201_cache_path}")

    # NB101
    nb101_tfrecord = os.path.join(nb101_dir, "nasbench_only108.tfrecord")
    check("NB101 tfrecord", os.path.isfile(nb101_tfrecord),
          f"not found at {nb101_tfrecord}", warn_only=True)

    # --- 3. NB201 cache integrity ---
    print("\n3. NB201 cache integrity")

    if os.path.isfile(nb201_cache_path):
        try:
            import torch
            cache = torch.load(nb201_cache_path, weights_only=False)
            entries = cache.get("entries", cache)  # v1 format has 'entries' key
            n_entries = len(entries)
            check(f"Cache entries: {n_entries}", n_entries > 100, f"only {n_entries} entries")

            # Check a sample entry for all 3 datasets
            sample_key = next(iter(entries))
            entry = entries[sample_key]
            datasets_to_check = ["cifar10-valid", "cifar100", "ImageNet16-120"]
            for ds in datasets_to_check:
                if ds in entry:
                    val = entry[ds].get("val_acc") if isinstance(entry[ds], dict) else None
                    check(f"  {ds} val_acc present", val is not None,
                          f"val_acc is None — cache incomplete for {ds}")
                else:
                    check(f"  {ds} key present", False, f"key '{ds}' missing from cache entry")
        except Exception as e:
            check("Cache loadable", False, str(e))
    else:
        check("Cache integrity (skipped)", False, "cache file not found", warn_only=True)

    # --- 4. Config file checks ---
    print("\n4. Config files")

    config_dir = os.path.join(repo_root, "configs")
    if os.path.isdir(config_dir):
        import yaml
        yaml_files = [f for f in os.listdir(config_dir) if f.endswith(".yaml")]
        check(f"Found {len(yaml_files)} YAML configs", len(yaml_files) > 0, "no configs found")
        for yf in sorted(yaml_files):
            path = os.path.join(config_dir, yf)
            try:
                with open(path) as f:
                    yaml.safe_load(f)
                check(f"  {yf} parses", True, "")
            except Exception as e:
                check(f"  {yf} parses", False, str(e))
    else:
        check("configs/ directory", False, f"not found at {config_dir}")

    # --- 5. Experiment scripts ---
    print("\n5. Experiment scripts")

    scripts = [
        "run_experiment_1_nb201.sh",
        "run_experiment_1_nb101.sh",
        "run_experiment_2_nb201.sh",
        "run_experiment_2_nb101.sh",
        "run_inverse_control.sh",
    ]
    for s in scripts:
        path = os.path.join(script_dir, s)
        exists = os.path.isfile(path)
        executable = os.access(path, os.X_OK) if exists else False
        check(f"{s} exists", exists, "missing")
        if exists:
            check(f"  executable", executable, "not executable (chmod +x)")

    # --- Summary ---
    print(f"\n{'='*50}")
    print(f"Results: {errors} errors, {warnings} warnings")
    if errors == 0:
        print(f"[{PASS}] Preflight passed — ready to run experiments.")
    else:
        print(f"[{FAIL}] Preflight failed — fix errors above before proceeding.")
    return 1 if errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
