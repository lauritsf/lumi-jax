"""Locate built kernel shared objects.

Reads take the first hit PER LIBRARY of $LUMI_KERNELS_LIB (override), <this package>/lib
(a local build), then /opt/lumi-jax/lumi_kernels/lib (the image's prebuilt set). Per
library, not per directory, so building one kernel locally gives you your build plus the
image's others rather than hiding them. Writes always go to the checkout's lib/.
"""

import os

LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")

# Where both image recipes install. Absent on a bare host, hence the skip-missing search.
IMAGE_LIB_DIR = "/opt/lumi-jax/lumi_kernels/lib"


def search_dirs() -> list:
    """Directories consulted for a prebuilt library, in order."""
    dirs = []
    override = os.environ.get("LUMI_KERNELS_LIB")
    if override:
        dirs.append(override)
    dirs.append(LIB_DIR)
    if IMAGE_LIB_DIR not in dirs:
        dirs.append(IMAGE_LIB_DIR)
    return dirs


def lib_path(name: str) -> str:
    """Where `name` lands after a build (may not exist yet). Always the checkout's lib/,
    never the image's: this names a build OUTPUT.

    NOT for loading. Loading through it ignores $LUMI_KERNELS_LIB and the image, so a
    checkout with stale .so files silently wins and one without them fails even though the
    image ships the kernel. Use so_path()/have() to load.
    """
    return os.path.join(LIB_DIR, name)


def have(name: str) -> bool:
    """True if `name` is resolvable anywhere on the search path."""
    try:
        so_path(name)
        return True
    except FileNotFoundError:
        return False


def so_path(name: str) -> str:
    tried = []
    for d in search_dirs():
        path = os.path.join(d, name)
        if os.path.exists(path):
            return path
        tried.append(path)
    raise FileNotFoundError(
        f"{name} not found -- run the matching csrc/*/build_*.sh, "
        f"or use an image that ships it (lumi-jax-lean-j*-latest.sif has four of the "
        f"five, lumi-jax-torch-j091-latest.sif has all five). Looked in: "
        + ", ".join(tried)
    )
