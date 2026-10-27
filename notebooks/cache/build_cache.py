# RSNA Knee: build the uint8 pixel cache on Kaggle CPU (default recipe: v4g -- cardinal slice order, laterality canonicalisation, LUT + crop window).
# Attach: competition data + rsna-knee-src -- see docs/code-notes.md "cache build overview" for shard layout.

import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

def find_input(*candidates):
    """Return the first existing path among old/new Kaggle mount layouts."""
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(candidates)


SRC = find_input(
    "/kaggle/input/rsna-67knee-src",
    "/kaggle/input/datasets/patimakornsrinamen/rsna-67knee-src",
    # pre-rename slug, kept so the kernel runs against either dataset name
    "/kaggle/input/rsna-knee-src",
    "/kaggle/input/datasets/patimakornsrinamen/rsna-knee-src",
)
sys.path.insert(0, SRC)
from preprocess import (build_study_tensor, read_series_table, recipe_kwargs,
                        n_slots_for, RECIPE_V4, RECIPE_V5, RECIPE_V4G)

COMP = find_input(
    "/kaggle/input/rsna-knee-abnormality-detection",
    "/kaggle/input/competitions/rsna-knee-abnormality-detection",
)
OUT = "/kaggle/working"
SHARD_SIZE = 250
WORKERS = 4
TINY = False  # True -> only 3 studies, for pipeline validation
# SHARD_RANGE selects this kernel's shard slice from the global numbering -- see docs/code-notes.md "SHARD_RANGE shard slice"
SHARD_RANGE = (0, 9)

# RECIPE is dumped to cache_config.json and read via recipe_kwargs()/n_slots_for() -- see docs/code-notes.md "RECIPE dict contract"
RECIPE = RECIPE_V4G
# RECIPE = RECIPE_V5
# RECIPE = RECIPE_V4  # v4 build switch: reproduces the pre-v5 6-slot cache
SIZE = RECIPE["size"]
N_SLOTS_VAL = n_slots_for(RECIPE)
# max slot depth: v5's slices_per_slot is per-slot, v2-v4 use one n_slices
N_SL = (RECIPE["n_slices"] if RECIPE.get("slices_per_slot") is None
        else max(RECIPE["slices_per_slot"]))
# N_SL must match n_slices or a recipe would build slices no window covers -- see docs/code-notes.md "N_SL matches n_slices"
assert N_SL == RECIPE["n_slices"], (
    f"recipe n_slices={RECIPE['n_slices']} != deepest slot {N_SL}")


def process_study(args):
    study_uid, series_rows = args
    try:
        t, filled = build_study_tensor(
            os.path.join(COMP, "train_series", study_uid), series_rows,
            **recipe_kwargs(RECIPE))
    except Exception as e:
        print(f"WARN study {study_uid} failed ({type(e).__name__}); zeros",
              flush=True)
        t, filled = np.zeros((N_SLOTS_VAL, N_SL, SIZE, SIZE), np.uint8), 0
    return study_uid, t, filled


def main():
    by_study = read_series_table(os.path.join(COMP, "train_series.csv"))
    studies_all = sorted(by_study)
    lo, hi = SHARD_RANGE
    if TINY:
        # TINY keeps the global-shard prefix so shard_no still lands on the real shard -- see docs/code-notes.md "TINY smoke prefix handling"
        studies = studies_all[:lo * SHARD_SIZE] + studies_all[
            lo * SHARD_SIZE:lo * SHARD_SIZE + 3]
    else:
        studies = studies_all
    print(f"{len(studies)} studies, shard={SHARD_SIZE}, shards [{lo}, {hi}), "
          f"workers={WORKERS}, recipe={RECIPE}", flush=True)

    import json
    # cache_config.json is written first so surviving shards stay interpretable -- see docs/code-notes.md "cache_config write order"
    with open(os.path.join(OUT, "cache_config.json"), "w") as fh:
        json.dump({**RECIPE, "n_slots": N_SLOTS_VAL}, fh)

    n_empty = 0
    t0 = time.time()
    n_part = len(studies[lo * SHARD_SIZE:hi * SHARD_SIZE])  # computed once
    # index is flushed after every shard so a mid-build crash leaves the
    # completed shards usable; filled_slots lets training drop empty studies
    with open(os.path.join(OUT, "cache_index.csv"), "w", newline="") as idx:
        w = csv.writer(idx)
        w.writerow(["StudyInstanceUID", "shard", "row", "filled_slots"])
        n_done = 0
        for shard_no, start in enumerate(range(0, len(studies), SHARD_SIZE)):
            if not (lo <= shard_no < hi):
                continue
            chunk = studies[start:start + SHARD_SIZE]
            buf = np.zeros((len(chunk), N_SLOTS_VAL, N_SL, SIZE, SIZE),
                           np.uint8)
            order = {uid: i for i, uid in enumerate(chunk)}
            filled_of = {}
            with ProcessPoolExecutor(max_workers=WORKERS) as ex:
                for uid, tensor, filled in ex.map(
                        process_study, [(u, by_study[u]) for u in chunk]):
                    buf[order[uid]] = tensor
                    filled_of[uid] = filled
            np.save(os.path.join(OUT, f"cache_{shard_no:03d}.npy"), buf)
            for uid in chunk:
                w.writerow([uid, shard_no, order[uid], filled_of[uid]])
            idx.flush()
            n_empty += sum(1 for u in chunk if filled_of[u] == 0)
            n_done += len(chunk)
            rate = n_done / (time.time() - t0)
            eta = (n_part - n_done) / rate / 60 if rate else 0
            print(f"shard {shard_no}: {n_done}/{n_part} "
                  f"({rate:.1f} studies/s, eta {eta:.0f} min)", flush=True)

    print(f"DONE {n_done} studies in shards [{lo}, {hi}) ({n_empty} all-zero) "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
