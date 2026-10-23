"""Require a JAX trace with GPU execution events and the candidate ROCr mapping."""
import gzip
import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

out = pathlib.Path(sys.argv[1])
assert jax.devices()[0].platform == "gpu"
x = jnp.ones((512, 512), dtype=jnp.float32)
f = jax.jit(lambda a: a @ a)
f(x).block_until_ready()
with jax.profiler.trace(str(out)):
    for _ in range(20):
        result = f(x)
    result.block_until_ready()
np.testing.assert_array_equal(np.asarray(result), np.full((512, 512), 512, np.float32))
paths = list(out.rglob("*.trace.json.gz"))
assert paths, "No trace produced"
gpu_events = []
for path in paths:
    with gzip.open(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    gpu_pids = {e["pid"] for e in events if e.get("name") == "process_name"
                and "GPU" in e.get("args", {}).get("name", "").upper()}
    gpu_events.extend(e for e in events if e.get("pid") in gpu_pids and e.get("ph") == "X")
assert gpu_events, "Trace has no GPU execution events"
mappings = {line.split()[-1] for line in pathlib.Path("/proc/self/maps").read_text().splitlines()
            if "libhsa-runtime" in line}
assert mappings and all(p.startswith("/opt/rocm") for p in mappings), mappings
print("JAX PROFILER CAPTURE OK", len(gpu_events), sorted(mappings))
