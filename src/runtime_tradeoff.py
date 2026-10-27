"""Offline fold-count trade-off curve from the stored gold-58 per-window
predictions (gold_windows*.npz).

Only the fold axis is computable offline: study_probs (n_gold, n_models, 12)
is the model's own attention pooling over windows, already collapsed per
fold/model, so averaging it over a subset of folds reproduces exactly what
running only that subset at inference would score. The window axis itself
(pooling over the 30 stored windows inside probs) is NOT reproduced by a
plain mean - see the round-8 plan - so this module does not touch it.

Usage: py src/runtime_tradeoff.py <weights_dir>... [--out <csv>]
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tune_pooling import auc_per_label

# Recorded gold-58 macro AUC for each arm, from merging all folds' study_probs
# (round-6/round-8 plans). run_config.json does not carry this for every arm
# (v6c's is null there), so the known values are kept here and matched by
# dataset-metadata.json's title (or the directory name as a fallback).
RECORDED_GOLD58 = {"v6c": 0.8933, "v4b": 0.8758}

# AUC weight in the efficiency score `-2.19 * AUC + hours / 9` (round-8 plan):
# one hour of runtime costs 0.051 AUC.
EFFICIENCY_COEF = -2.19


def _recorded_key(weights_dir):
    """Which arm a weights dir is, from dataset-metadata.json's title
    (falling back to the directory name), so the reproduction check below
    is matched against the right recorded number."""
    meta_path = os.path.join(weights_dir, "dataset-metadata.json")
    title = ""
    if os.path.isfile(meta_path):
        with open(meta_path) as fh:
            title = json.load(fh).get("title", "")
    haystack = title or os.path.basename(os.path.normpath(weights_dir))
    for key in RECORDED_GOLD58:
        if key in haystack:
            return key
    return None


def load_gold(weights_dir):
    """Load every gold_windows*.npz in sorted-filename order and concatenate
    study_probs along the model/fold axis. Returns (study_probs, y) where
    study_probs is (n_gold, n_folds, 12) and y is (n_gold, 12)."""
    paths = sorted(glob.glob(os.path.join(weights_dir, "gold_windows*.npz")))
    if not paths:
        raise SystemExit(f"no gold_windows*.npz found in {weights_dir}")
    parts = []
    y = None
    for p in paths:
        d = np.load(p)
        parts.append(d["study_probs"])
        if y is None:
            y = d["y"]
        elif not np.array_equal(y, d["y"]):
            raise SystemExit(f"gold-58 y mismatch between {paths[0]} and {p}")
    return np.concatenate(parts, axis=1), y


def macro_auc(y, scores):
    aucs = auc_per_label(y, scores)
    return float(np.mean(list(aucs.values())))


def fold_table(weights_dir):
    """Returns (rows, all_folds_auc, recorded, key). Asserts the all-folds
    cell reproduces the recorded gold-58 AUC before returning, when a
    recorded value is known for this arm."""
    study_probs, y = load_gold(weights_dir)
    n_gold, n_folds, n_labels = study_probs.shape

    key = _recorded_key(weights_dir)
    recorded = RECORDED_GOLD58.get(key)

    all_folds_auc = macro_auc(y, study_probs.mean(axis=1))
    if recorded is not None:
        assert abs(all_folds_auc - recorded) < 0.001, (
            f"{weights_dir}: all-folds gold-58 macro AUC {all_folds_auc:.4f} "
            f"does not reproduce the recorded value {recorded:.4f} for arm "
            f"'{key}' to within 0.001")

    rows = []
    for i in range(n_folds):
        auc_i = macro_auc(y, study_probs[:, i, :])
        rows.append({
            "weights_dir": weights_dir, "kind": "single_fold", "k": i + 1,
            "n_folds": n_folds, "auc": auc_i, "rel_cost": 1.0 / n_folds,
            "eff_delta": EFFICIENCY_COEF * (auc_i - all_folds_auc),
        })
    for k in range(1, n_folds + 1):
        auc_k = macro_auc(y, study_probs[:, :k, :].mean(axis=1))
        rows.append({
            "weights_dir": weights_dir, "kind": "first_k", "k": k,
            "n_folds": n_folds, "auc": auc_k, "rel_cost": k / n_folds,
            "eff_delta": EFFICIENCY_COEF * (auc_k - all_folds_auc),
        })
    return rows, all_folds_auc, recorded, key


def print_table(weights_dir, rows, all_folds_auc, recorded, key):
    print(f"\n=== {weights_dir} ===")
    if recorded is not None:
        print(f"all-folds gold-58 macro AUC {all_folds_auc:.4f} checked "
              f"against recorded value {recorded:.4f} for arm '{key}'")
    else:
        print(f"all-folds gold-58 macro AUC {all_folds_auc:.4f} "
              f"(no recorded value known for this arm - not checked)")
    print(f"{'kind':<12} {'k':>2} {'n_folds':>7}  {'auc':>6}  "
          f"{'rel_cost':>8}  {'eff_delta':>9}")
    for r in rows:
        print(f"{r['kind']:<12} {r['k']:>2} {r['n_folds']:>7}  "
              f"{r['auc']:.4f}  {r['rel_cost']:>8.3f}  {r['eff_delta']:>+9.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Offline fold-count trade-off curve from stored "
                     "gold-58 study_probs (fold axis only).")
    parser.add_argument("weights_dirs", nargs="+",
                         help="one or more weights dirs, each holding "
                              "gold_windows*.npz")
    parser.add_argument("--out", help="write all rows to this CSV path")
    args = parser.parse_args()

    tables = [fold_table(d) for d in args.weights_dirs]

    for weights_dir, (rows, all_folds_auc, recorded, key) in zip(
            args.weights_dirs, tables):
        print_table(weights_dir, rows, all_folds_auc, recorded, key)

    if args.out:
        fieldnames = ["weights_dir", "kind", "k", "n_folds", "auc",
                      "rel_cost", "eff_delta"]
        with open(args.out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for _, (rows, _, _, _) in zip(args.weights_dirs, tables):
                for r in rows:
                    writer.writerow(r)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
