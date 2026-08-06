"""Multi-GPU pmap + all-reduce. Exercises RCCL."""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

pytestmark = [pytest.mark.gpu, pytest.mark.multigpu]


def test_psum_allgather():
    n = jax.device_count()
    x = jnp.arange(n, dtype=jnp.float32)

    got = jax.pmap(lambda v: jax.lax.psum(v, "i"), axis_name="i")(x)
    np.testing.assert_allclose(np.asarray(got), np.full((n,), float(x.sum())))

    got = jax.pmap(lambda v: jax.lax.all_gather(v, "i"), axis_name="i")(x)
    np.testing.assert_allclose(np.asarray(got)[0], np.asarray(x))
