#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$REPO_ROOT/data"
PYTHON_BIN="${PYTHON:-python}"

cleanup() {
    if [ -n "${NB101_TMP_FILE:-}" ] && [ -f "$NB101_TMP_FILE" ]; then
        rm -f "$NB101_TMP_FILE"
    fi
}

trap cleanup EXIT

echo "==> DGPO Companion Setup"
echo "    Repo root: $REPO_ROOT"

# --- NAS-Bench-201 data ---
NB201_DIR="$DATA_DIR/NASBench201/raw"
NB201_FILE="$NB201_DIR/NAS-Bench-201-v1_1-096897.pth"
NB201_RELATIVE_PATH="data/NASBench201/raw/NAS-Bench-201-v1_1-096897.pth"

echo ""
echo "==> Checking NAS-Bench-201 data..."
mkdir -p "$NB201_DIR"
if [ -L "$NB201_FILE" ] && [ ! -e "$NB201_FILE" ]; then
    echo "    WARNING: $NB201_RELATIVE_PATH is a broken symlink. Removing."
    rm "$NB201_FILE"
fi
if [ -f "$NB201_FILE" ]; then
    echo "    Found: $NB201_RELATIVE_PATH"
else
    echo ""
    echo "    WARNING: NAS-Bench-201 data file not found."
    echo "    NB201 experiments will not work until you download it."
    echo ""
    echo "    Download NAS-Bench-201-v1_1-096897.pth from the NAS-Bench-201 GitHub repository"
    echo "    (https://github.com/D-X-Y/NAS-Bench-201) and place it at $NB201_RELATIVE_PATH"
    echo "    Then run: make build-nb201-cache"
    echo ""
fi

# --- Install Python package in editable mode ---
echo ""
echo "==> Installing Python package (editable)..."
"$PYTHON_BIN" -m pip install -e "$REPO_ROOT"

# --- NAS-Bench-101 data ---
NB101_DIR="$DATA_DIR/NASBench101/raw"
NB101_FILE="$NB101_DIR/nasbench_only108.tfrecord"
NB101_URL="https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord"

echo ""
echo "==> Checking NAS-Bench-101 data..."
if [ -s "$NB101_FILE" ]; then
    echo "    Found: $NB101_FILE (skipping download)"
else
    echo "    Downloading NAS-Bench-101 tfrecord (~500 MB)..."
    mkdir -p "$NB101_DIR"
    NB101_TMP_FILE="$(mktemp "$NB101_DIR/nasbench_only108.tfrecord.tmp.XXXXXX")"
    curl -fL --retry 3 --retry-delay 2 -o "$NB101_TMP_FILE" "$NB101_URL"
    mv "$NB101_TMP_FILE" "$NB101_FILE"
    NB101_TMP_FILE=""
    echo "    Downloaded: $NB101_FILE"
fi

echo ""
echo "==> Setup complete."
echo "    NB101 data: $NB101_FILE"
echo "    NB201 data: $NB201_FILE"
