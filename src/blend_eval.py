"""Rank-blend gold-58 predictions from two or more model arms (families) and
report per-label / macro AUC for each arm and for the equal-weight blend.

This is information only - nothing is selected here. Round 4's lesson was
that a post-processing table tuned on the 58 gold studies can lose on the
LB, so the CLI below prints a comparison table and writes nothing.

`pct_rank` and `rank_blend` are pure numpy so the inference kernel can import
them without pulling in torch. `auc_per_label` (which reaches sklearn via
tune_pooling -> model -> torch) is imported lazily inside main() so that
`from blend_eval import rank_blend` stays torch-free.

Usage: py src/blend_eval.py <dir_arm1> <dir_arm2> [more dirs...]

Each dir holds one or more gold_windows*.npz files (study_probs
(n_gold, n_folds, 12), y (n_gold, 12), uids (n_gold,)) written by the
training kernel - a merged dir (several kernels folded together) may hold
gold_windows_a.npz, gold_windows_b.npz, ... instead of a single
gold_windows.npz; all matching files in a dir are concatenated on the fold
axis - plus an optional run_config.json. Dirs are grouped by family =
run_config.json["family"] if present, else "dinov2s" (the default family for
runs that predate the "family" key) if run_config.json exists at all, else
the dir's basename; several dirs of the same family (e.g. a two-kernel arm)
have their study_probs concatenated on the fold axis.

Round 6 adds an optional `--weights <blend_weights.json>` flag (as written by
blend_weights.py) that prints the AUC of that JSON's per-label weighted
blend alongside the equal-weight blend - information only, nothing is
selected here either.
"""

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import LABELS


def pct_rank(col: np.ndarray) -> np.ndarray:
    """Percentile rank of each value in col, in (0, 1], average ranks for ties."""
    col = np.asarray(col, dtype=float)
    if not np.all(np.isfinite(col)):
        raise ValueError("pct_rank: input contains non-finite values")
    u, inv, cnt = np.unique(col, return_inverse=True, return_counts=True)
    hi = np.cumsum(cnt)
    avg = (hi - cnt + 1 + hi) / 2
    return avg[inv] / len(col)


def rank_blend(mats: list, weights=None) -> np.ndarray:
    """Blend over families of the per-column pct_rank of each mat.
    Each mat in mats is (n, k); all mats must share the same shape.

    weights is None (default): equal-weight mean over families - the
    original round-4/5 behaviour, unchanged.

    weights given: array-like of shape (len(mats), k), one row per family
    and one column per label. Must be non-negative and each column must sum
    to > 0; it is normalised per column here, so callers may pass
    unnormalised non-negative weights (e.g. raw AUC-derived scores)."""
    if not mats:
        raise ValueError("rank_blend: mats is empty")
    shape = mats[0].shape
    for m in mats:
        if m.shape != shape:
            raise ValueError(f"rank_blend: shape mismatch {m.shape} vs {shape}")
    n, k = shape
    ranked = np.empty((len(mats), n, k))
    for i, m in enumerate(mats):
        for c in range(k):
            ranked[i, :, c] = pct_rank(m[:, c])
    if weights is None:
        return ranked.mean(axis=0)
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(mats), k):
        raise ValueError(f"rank_blend: weights shape {w.shape} != {(len(mats), k)}")
    if np.any(w < 0):
        raise ValueError("rank_blend: weights must be non-negative")
    col_sums = w.sum(axis=0)
    if np.any(col_sums <= 0):
        raise ValueError("rank_blend: each weights column must sum to > 0")
    w = w / col_sums
    return (ranked * w[:, None, :]).sum(axis=0)


def _family_of(d):
    """run_config.json["family"] if present, else "dinov2s" - the default
    family for runs recorded before the "family" key existed - as long as
    run_config.json exists at all. The dir's basename is the fallback only
    when there is NO run_config.json in the dir."""
    cfg_path = os.path.join(d, "run_config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        fam = cfg.get("family")
        return fam if fam else "dinov2s"
    return os.path.basename(os.path.normpath(d))


def _load_fold_of(d):
    """fold_of.json contents (uid -> fold mapping or array), or None if
    absent or unparseable (a note is printed for the unparseable case)."""
    path = os.path.join(d, "fold_of.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        print(f"note: could not parse fold_of.json in {d}: {e}")
        return None


def main():
    from tune_pooling import auc_per_label

    argv = sys.argv[1:]
    dirs, weights_path = [], None
    i = 0
    while i < len(argv):
        if argv[i] == "--weights":
            i += 1
            if i >= len(argv):
                sys.exit("--weights requires a path argument")
            weights_path = argv[i]
        else:
            dirs.append(argv[i])
        i += 1
    if len(dirs) < 2:
        sys.exit("usage: py src/blend_eval.py <dir_arm1> <dir_arm2> [more dirs...] "
                  "[--weights <blend_weights.json>]")

    families = {}  # family -> [dirs]
    for d in dirs:
        families.setdefault(_family_of(d), []).append(d)
    fam_names = list(families.keys())

    fam_probs, fam_y, fam_uids, fam_fold_of = {}, {}, {}, {}
    for fam, fam_dirs in families.items():
        study_probs_list, y, uids = [], None, None
        for d in fam_dirs:
            npz_paths = sorted(glob.glob(os.path.join(d, "gold_windows*.npz")))
            if not npz_paths:
                raise FileNotFoundError(f"no gold_windows*.npz found in {d}")
            for p in npz_paths:
                data = np.load(p)
                study_probs_list.append(data["study_probs"])
                if y is None:
                    y, uids = data["y"], data["uids"]
                else:
                    assert np.array_equal(uids, data["uids"]), (
                        f"uids mismatch within family {fam} across dirs")
                    assert np.array_equal(y, data["y"]), (
                        f"y mismatch within family {fam} across dirs")
        sp = np.concatenate(study_probs_list, axis=1)  # (n_gold, n_folds_total, 12)
        fam_probs[fam] = sp.mean(axis=1)  # (n_gold, 12)
        fam_y[fam], fam_uids[fam] = y, uids
        fold_of = None
        for d in fam_dirs:
            fo = _load_fold_of(d)
            if fo is not None:
                fold_of = fo
                break
        fam_fold_of[fam] = fold_of

    ref_fam = fam_names[0]
    ref_uids, ref_y = fam_uids[ref_fam], fam_y[ref_fam]
    for fam in fam_names[1:]:
        assert np.array_equal(fam_uids[fam], ref_uids), (
            f"uids mismatch: {fam} vs {ref_fam}")
        assert np.array_equal(fam_y[fam], ref_y), f"y mismatch: {fam} vs {ref_fam}"

    for i, fam_a in enumerate(fam_names):
        for fam_b in fam_names[i + 1:]:
            fo_a, fo_b = fam_fold_of[fam_a], fam_fold_of[fam_b]
            if fo_a is not None and fo_b is not None and fo_a != fo_b:
                print(f"note: fold_of.json differs between {fam_a} and {fam_b}")

    y = ref_y
    mats = [fam_probs[fam] for fam in fam_names]
    aucs = {fam: auc_per_label(y, fam_probs[fam]) for fam in fam_names}
    aucs["blend"] = auc_per_label(y, rank_blend(mats))
    cols = fam_names + ["blend"]

    if weights_path:
        with open(weights_path) as fh:
            wcfg = json.load(fh)
        w = np.empty((len(fam_names), len(LABELS)))
        for c, name in enumerate(LABELS):
            fam_w = wcfg["weights"].get(name, {})
            for r, fam in enumerate(fam_names):
                if fam not in fam_w:
                    sys.exit(f"--weights: family {fam!r} missing for label "
                              f"{name!r} in {weights_path}")
                w[r, c] = fam_w[fam]
        aucs["w-blend"] = auc_per_label(y, rank_blend(mats, weights=w))
        cols.append("w-blend")

    print(f"{'label':<18} " + "  ".join(f"{c:>8}" for c in cols))
    for name in LABELS:
        cells = [f"{aucs[c][name]:.4f}" if name in aucs[c] else "    skip"
                 for c in cols]
        print(f"{name:<18} " + "  ".join(f"{cell:>8}" for cell in cells))

    print()
    for c in cols:
        scored = list(aucs[c].keys())
        macro = float(np.mean([aucs[c][n] for n in scored])) if scored else float("nan")
        print(f"macro AUC {c:<8}: {macro:.4f}")


if __name__ == "__main__":
    main()
