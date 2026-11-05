"""Pytest setup: make the repo under test importable and register markers."""
import os
import sys
from pathlib import Path

# Must be set before protobuf is first imported (vendored NB101 API); see src/dataset.py.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

REPO_ROOT = Path(__file__).resolve().parents[1]

# Always import `src` / `vendored` from this repo, never from an editable install.
sys.path.insert(0, str(REPO_ROOT))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: needs the NB101 tfrecord (set NB101_TFRECORD)"
    )
