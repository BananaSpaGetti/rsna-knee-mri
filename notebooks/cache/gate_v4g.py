# RSNA Knee: v4g gate kernel (Kaggle CPU script). Part 1: RECIPE_V4 byte-identity on the first 3 studies
# (tiny_v4.npy + md5). Part 2: 20-study RECIPE_V4G smoke with per-study / per-series receipts and v4 vs v4g `filled`.
# Attach: competition data + rsna-knee-src; see docs/specs/geometry-v4g.md Task 10c.

import hashlib
import os
import sys
from collections import Counter

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
import preprocess
from preprocess import (build_study_tensor, read_series_table, recipe_kwargs,
                        RECIPE_V4, RECIPE_V4G)

COMP = os.environ.get("GATE_COMP") or find_input(
    "/kaggle/input/rsna-knee-abnormality-detection",
    "/kaggle/input/competitions/rsna-knee-abnormality-detection",
)
OUT = os.environ.get("GATE_OUT", "/kaggle/working")
N_PART1 = 3
N_PART2 = 20


def series_header_stats(series_dir, suid):
    """(n_files, slope_range, n_nonidentity) from one header-only read per file."""
    d = os.path.join(series_dir, suid)
    files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".dcm")]
    recs = preprocess._read_slice_headers(files)
    slopes = [r["slope"] for r in recs]
    inters = [r["intercept"] for r in recs]
    non_id = any(s != 1.0 or i != 0.0 for s, i in zip(slopes, inters))
    rng = (f"{min(slopes):g}..{max(slopes):g}/i{min(inters):g}..{max(inters):g}"
           if recs else "none")
    return len(files), rng, non_id


def main():
    by_study = read_series_table(os.path.join(COMP, "train_series.csv"))
    studies_all = sorted(by_study)

    # Part 1: byte identity of the switched-off module on the first 3 studies
    n1 = min(N_PART1, len(studies_all))
    kw4 = recipe_kwargs(RECIPE_V4)
    n_slots = 6
    tiny = np.zeros((n1, n_slots, RECIPE_V4["n_slices"], RECIPE_V4["size"],
                     RECIPE_V4["size"]), np.uint8)
    for i, uid in enumerate(studies_all[:n1]):
        t, filled = build_study_tensor(os.path.join(COMP, "train_series", uid),
                                       by_study[uid], **kw4)
        tiny[i] = t
        print(f"PART1 v4 {uid[-8:]} filled={filled}", flush=True)
    np.save(os.path.join(OUT, "tiny_v4.npy"), tiny)
    print(f"PART1 tiny_v4.npy shape={tiny.shape} md5="
          f"{hashlib.md5(np.ascontiguousarray(tiny).tobytes()).hexdigest()}",
          flush=True)

    # Part 2: v4g smoke
    n2 = min(N_PART2, len(studies_all))
    kw4g = recipe_kwargs(RECIPE_V4G)
    sides = Counter()
    flipped = Counter()
    methods = Counter()
    n_mono1 = n_nonid = n_multi = n_rejected = n_series = 0
    cmp_rows = []
    for uid in studies_all[:n2]:
        sdir = os.path.join(COMP, "train_series", uid)
        rc = {}
        t, filled = build_study_tensor(sdir, by_study[uid], **kw4g, receipts=rc)
        _, filled_v4 = build_study_tensor(sdir, by_study[uid], **kw4)
        cmp_rows.append((uid, filled_v4, filled))
        sag = rc.get("sag_x_median")
        sag_s = "None" if sag is None else f"{sag:.2f}"
        print(f"{uid[-8:]} side={rc.get('side')} source={rc.get('side_source')} "
              f"tags={rc.get('tags')} sag_x_med={sag_s} filled={filled}",
              flush=True)
        sides[rc.get("side_source")] += 1
        fluid_of = {r["SeriesInstanceUID"]: r["Fluid_Sensitive"]
                    for r in by_study[uid]}
        for suid, s in rc.get("series", {}).items():
            n_files, slope_rng, non_id = series_header_stats(sdir, suid)
            phot = ",".join(s["photometric_values"]) or "None"
            md = s["min_delta"]
            md_s = "None" if md is None else f"{md:.3f}"
            print(f"  {s['plane']} fluid={fluid_of.get(suid)} files={n_files} "
                  f"method={s['method']} flipped_normal={s['flipped_normal']} "
                  f"flip={s['flip']} reversed={s['reversed']} photometric={phot} "
                  f"slope_range={slope_rng} n_multiframe={s['n_multiframe_files']} "
                  f"n_lat_conflict={s['n_lat_conflict']} min_delta={md_s}",
                  flush=True)
            n_series += 1
            methods[s["method"]] += 1
            if s["flipped_normal"]:
                flipped[s["plane"]] += 1
            if "MONOCHROME1" in s["photometric_values"]:
                n_mono1 += 1
            if non_id:
                n_nonid += 1
            n_multi += s["n_multiframe_files"]
            if s["method"].startswith("rejected"):
                n_rejected += 1

    print("SUMMARY", flush=True)
    print(f"  studies={n2} series={n_series}")
    print(f"  side_sources={dict(sides)}")
    print(f"  flipped_normal_by_plane={dict(flipped)}")
    print(f"  methods={dict(methods)}")
    print(f"  monochrome1_series={n_mono1}")
    print(f"  series_slope_ne_1_or_intercept_ne_0={n_nonid}")
    print(f"  multiframe_files={n_multi}")
    print(f"  rejected_series={n_rejected}")
    print("FILLED v4 vs v4g (a drop = rejection)", flush=True)
    for uid, a, b in cmp_rows:
        print(f"  {uid[-8:]} v4={a} v4g={b}{'  DROP' if b < a else ''}")
    print(f"  studies_with_drop={sum(1 for _, a, b in cmp_rows if b < a)}",
          flush=True)


if __name__ == "__main__":
    main()
