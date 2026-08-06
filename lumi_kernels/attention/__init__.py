"""Memory-efficient / fused attention for JAX on LUMI (MI250X, gfx90a). Layout (B,S,H,D).

  reference.naive_reference  -- explicit O(S^2) einsum, correctness reference
  reference.xla_dpa          -- baseline: jax.nn.dot_product_attention (XLA math)
  chunked.chunked_attention  -- pure-JAX online-softmax, O(S) memory
"""
from . import reference  # noqa: F401

__all__ = ["reference"]

try:  # chunked has no extra deps, but import defensively so reference always loads
    from . import chunked  # noqa: F401

    __all__.append("chunked")
except Exception:  # pragma: no cover
    pass
