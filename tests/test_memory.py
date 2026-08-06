"""Allocate a large buffer to confirm HBM is properly accessible."""
import pytest

import jax.numpy as jnp

pytestmark = pytest.mark.gpu


def test_4gb_matmul():
    """Allocate 4 GB and multiply it, to prove HBM is reachable.

    BUDGET ~1400 s ON THE 0.11.0 LINE, against ~90 s on 0.9.1. The difference is
    XLA's GEMM autotune sweep, not the multiply: 32768**2 sits at the extreme end of
    a cost curve that is under 10 s at 8192**2 and ~100 s at 16384**2, and steady
    execution is identical on both lines.

    So size the job's wall clock against ~1400 s, or hand the process a persisted
    autotune cache, which takes it to ~3 s with throughput fully intact:

        jax.config.update("jax_compilation_cache_dir", "<dir on scratch>")

    Deliberately not done here: a test must not write a cache into whichever
    filesystem is running it, and entries are keyed per fusion, so it would help
    this one shape and nothing else. The repo sets no pytest timeout, so this does
    not fail on the clock -- it just takes the time.
    """
    size_gb = 4
    elems = size_gb * (1024 ** 3) // 4
    side = int(elems ** 0.5)
    x = jnp.ones((side, side), dtype=jnp.float32)
    jnp.dot(x, x.T).block_until_ready()
