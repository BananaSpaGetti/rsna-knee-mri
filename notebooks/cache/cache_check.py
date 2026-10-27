# RSNA Knee: tiny inspector for a built pixel cache.
#
# Loads cache_config.json + cache_index.csv from CACHE_DIR (build_cache.py's
# OUT, or a mounted cache dataset) plus one shard's .npy, and prints the
# recipe knobs (version, n_slots, slot_layout, slices_per_slot, size), the
# shard/row counts, that one shard's shape/dtype, and the filled_slots
# histogram - a quick sanity check before burning a train kernel on a freshly
# built cache. No dependency on src/preprocess.py: only reads what
# build_cache.py already wrote.
#
# Attach: the cache build's own kernel output (this script is meant to run
# right after build_cache.py, against the same /kaggle/working, or against a
# mounted cache dataset by pointing CACHE_DIR at it).

import csv
import json
import os
from collections import Counter

import numpy as np

CACHE_DIR = "/kaggle/working"


def main():
    with open(os.path.join(CACHE_DIR, "cache_config.json")) as fh:
        cfg = json.load(fh)
    with open(os.path.join(CACHE_DIR, "cache_index.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))

    print(f"version={cfg.get('version')} n_slots={cfg.get('n_slots')} "
          f"slot_layout={cfg.get('slot_layout')} "
          f"slices_per_slot={cfg.get('slices_per_slot')} size={cfg.get('size')}")

    shards = sorted({int(r["shard"]) for r in rows})
    print(f"shards={len(shards)} rows={len(rows)}")

    if shards:
        shard0 = shards[0]
        arr = np.load(os.path.join(CACHE_DIR, f"cache_{shard0:03d}.npy"))
        print(f"shard {shard0}: shape={arr.shape} dtype={arr.dtype}")
    else:
        print("no shards found")

    hist = Counter(
        int(float(r["filled_slots"])) for r in rows
        if r.get("filled_slots") not in (None, ""))
    print(f"filled_slots histogram: {dict(sorted(hist.items()))}")


if __name__ == "__main__":
    main()
