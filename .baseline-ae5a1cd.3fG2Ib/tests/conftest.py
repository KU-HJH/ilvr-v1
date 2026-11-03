"""Repository-wide pytest compatibility for the local Python build marker."""

from __future__ import annotations

import os
import platform


# Deterministic CUDA tests must configure cuBLAS before any test initializes CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


_ORIGINAL_PYTHON_VERSION = platform.python_version
if _ORIGINAL_PYTHON_VERSION().endswith("+"):
    platform.python_version = lambda: _ORIGINAL_PYTHON_VERSION().rstrip("+")


def pytest_unconfigure(config) -> None:
    del config
    platform.python_version = _ORIGINAL_PYTHON_VERSION
