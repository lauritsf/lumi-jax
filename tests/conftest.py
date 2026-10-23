"""Shared pytest config: repo-root import path and auto-skip markers.

Markers (declared in pyproject):
  gpu       -- requires a ROCm GPU backend
  multigpu  -- requires all 8 GCDs (LUMI-G node)
  so(name)  -- requires <name>, resolved the same way the kernels themselves resolve it

Tests must never fail at COLLECTION off-GPU/off-container -- import GPU
deps lazily inside fixtures or test functions.
"""
import os
import sys

import pytest

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Resolve a kernel the way the kernels do, via _lib.so_path(), NOT by checking the checkout:
# `ROOT/lumi_kernels/lib` is wrong in both directions once the images ship kernels, skipping
# tests whose kernel the image has and admitting tests whose kernel the checkout does not.
def _so_available(name):
    try:
        from lumi_kernels import _lib
    except Exception:
        return False
    try:
        _lib.so_path(name)
        return True
    except FileNotFoundError:
        return False


def _search_dirs_str():
    """For the skip message: say WHERE we looked, not just that it is missing."""
    try:
        from lumi_kernels import _lib
        return ", ".join(_lib.search_dirs())
    except Exception:
        return "<lumi_kernels not importable>"


def _gpu_count():
    try:
        import jax

        return len(jax.devices()) if jax.default_backend() == "gpu" else 0
    except Exception:
        return 0


def pytest_collection_modifyitems(config, items):
    ngpu = _gpu_count()
    for item in items:
        if item.get_closest_marker("gpu") and ngpu == 0:
            item.add_marker(pytest.mark.skip(reason="no ROCm GPU backend"))
        if item.get_closest_marker("multigpu") and ngpu < 8:
            item.add_marker(pytest.mark.skip(reason=f"needs 8 GCDs, have {ngpu}"))
        m = item.get_closest_marker("so")
        if m and not _so_available(m.args[0]):
            item.add_marker(pytest.mark.skip(
                reason=f"{m.args[0]} not found in any of {_search_dirs_str()} "
                       "(build via csrc/*/build_*.sh, or use an image that ships it)"))
