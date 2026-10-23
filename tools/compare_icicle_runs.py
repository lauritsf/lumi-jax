#!/usr/bin/env python3
"""Compare complete saved scan trajectories with a qualified reference."""
import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("runs", type=Path, nargs="+")
args = parser.parse_args()
records = []
with np.load(args.reference, allow_pickle=False) as reference:
    for path in args.runs:
        record = {"file": str(path), "reference": str(args.reference), "arrays": {}}
        with np.load(path, allow_pickle=False) as actual:
            assert set(actual.files) == set(reference.files), path
            for key in reference.files:
                expected, value = reference[key], actual[key]
                assert expected.shape == value.shape and expected.dtype == value.dtype, (path, key)
                exact = np.array_equal(expected, value, equal_nan=True)
                finite = np.isfinite(expected) & np.isfinite(value)
                delta = np.abs(expected[finite].astype(np.float64) - value[finite].astype(np.float64))
                record["arrays"][key] = {
                    "exact": bool(exact), "bitwise_equal": expected.tobytes() == value.tobytes(),
                    "elements": int(value.size),
                    "max_abs_diff": float(delta.max()) if delta.size else 0.0,
                }
        record["exact"] = all(a["exact"] for a in record["arrays"].values())
        record["bitwise_equal"] = all(a["bitwise_equal"] for a in record["arrays"].values())
        records.append(record)
args.output.write_text(json.dumps(records, indent=2) + "\n")
for record in records:
    print(record["file"], "EXACT" if record["exact"] else "DIFFERENT")
assert all(record["exact"] for record in records), "Trajectory mismatch; inspect comparison output"
