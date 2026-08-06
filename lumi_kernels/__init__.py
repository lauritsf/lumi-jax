"""Hand-built XLA-FFI kernels for JAX on LUMI MI250X (gfx90a, ROCm).

Modules import lazily: nothing here touches the GPU or dlopens a .so.
"""

__version__ = "0.1.0"
