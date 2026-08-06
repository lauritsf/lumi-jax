"""Same JIT-compiled fn run twice should give identical results."""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

pytestmark = pytest.mark.gpu


def test_jit_repeat_identical():
    @jax.jit
    def f(x):
        for _ in range(10):
            x = jnp.tanh(x @ x.T) + x
        return x

    x = jax.random.normal(jax.random.key(42), (256, 256))
    np.testing.assert_array_equal(np.asarray(f(x)), np.asarray(f(x)))
