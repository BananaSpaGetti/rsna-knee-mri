"""Merge several Kaggle kernel run dirs - each having trained a subset of the
5-way fold split - into one weights directory: checkpoints copied in,
oof.npy stitched together fold by fold, run_config.json spliced. Round 5 did
this by hand (checkpoints copied, run_config spliced, oof.npy nan-fill-unioned
by eye); this script replaces that.

Usage: py src/merge_fold_runs.py --labels <consensus_v1.csv> <out_dir> <run_dir>...

Each run_dir holds `{family}_fold{k}.pt` per k in that run's `folds_trained`,
`run_config.json`, `oof.npy` (n_train, 12) with untrained rows zero-init,
`fold_of.json` (uid -> fold; insertion order == oof row order), and
`gold_windows.npz`. The split is seeded so every run's fold_of.json must be
identical.
"""

import argparse
import csv
import glob
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import LABELS
from tune_pooling import auc_per_label

# Keys that must agree across every run being merged (they describe the
# recipe/cache/arch, not which folds a run happened to cover). A key absent
# from EVERY run's config is skipped (older run, field not written yet); a
# key present in some but not all is an error, not a silent skip.
EQUAL_KEYS = (
    "arch", "family", "label_file", "version", "size", "crop_mm",
    "unique_slices", "n_slices", "window", "windows_per_slot",
    "lr_backbone", "unfreeze", "warmup_epochs", "epochs",
)

# Per-fold dicts (fold -> value) that a run's config may carry; merged by
# taking, for each fold in the union, the entry from whichever run trained it.
PER_FOLD_KEYS = ("epochs_done", "sec_per_epoch", "stopped_early")


def _run_name(run_dir):
    return os.path.basename(os.path.normpath(run_dir))


def _load_json(path):
    with open(path) as fh:
        return json.load(fh)


def _load_labels(path):
    """{uid: np.ndarray(12,)} from a consensus-style labels csv (LABELS order)."""
    labels = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            labels[row["StudyInstanceUID"]] = np.array(
                [float(row[name]) for name in LABELS], dtype=np.float32)
    return labels


def _check_equal_keys(configs, names):
    """Assert EQUAL_KEYS agree across all run configs. A key absent from
    every config is skipped; present in some but not all, or disagreeing
    where present, is a SystemExit."""
    for key in EQUAL_KEYS:
        present = [(n, c[key]) for n, c in zip(names, configs) if key in c]
        if not present:
            continue
        if len(present) < len(configs):
            missing = [n for n in names if n not in [p[0] for p in present]]
            sys.exit(f"run_config key {key!r} missing from run(s): {missing}")
        ref_val = present[0][1]
        odd = [(n, v) for n, v in present if v != ref_val]
        if odd:
            sys.exit(f"run_config key {key!r} disagrees across runs: {present}")


def _merged_per_fold(configs, run_folds, key):
    """{fold: value} for a PER_FOLD_KEYS entry, built from whichever run
    trained that fold; empty if no run's config set the key."""
    merged = {}
    for cfg, folds in zip(configs, run_folds):
        d = cfg.get(key)
        if not d:
            continue
        for k in folds:
            if str(k) in d:
                merged[k] = d[str(k)]
            elif k in d:
                merged[k] = d[k]
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Merge split-fold Kaggle training run dirs into one "
                     "weights directory (checkpoints, oof.npy, run_config.json).")
    parser.add_argument("--labels", required=True,
                         help="consensus_v1.csv path, for the recomputed oof_llm_auc")
    parser.add_argument("--force", action="store_true",
                         help="overwrite an out_dir that already has checkpoints")
    parser.add_argument("out_dir")
    parser.add_argument("run_dirs", nargs="+", metavar="run_dir")
    args = parser.parse_args()

    if os.path.isdir(args.out_dir):
        existing = glob.glob(os.path.join(args.out_dir, "*_fold*.pt"))
        if existing and not args.force:
            sys.exit(f"{args.out_dir} already has checkpoints "
                      f"(e.g. {os.path.basename(existing[0])}); use --force to overwrite")
    os.makedirs(args.out_dir, exist_ok=True)

    names = [_run_name(d) for d in args.run_dirs]
    fold_ofs = [_load_json(os.path.join(d, "fold_of.json")) for d in args.run_dirs]
    ref_fold_of = fold_ofs[0]
    for name, fo in zip(names[1:], fold_ofs[1:]):
        if fo != ref_fold_of:
            sys.exit(f"fold_of.json mismatch: run {name!r} differs from "
                      f"reference {names[0]!r}")

    uids = list(ref_fold_of)
    fold = np.array([ref_fold_of[u] for u in uids])

    configs = [_load_json(os.path.join(d, "run_config.json")) for d in args.run_dirs]
    oofs = []
    for name, d in zip(names, args.run_dirs):
        oof = np.load(os.path.join(d, "oof.npy"))
        if oof.shape != (len(uids), 12):
            sys.exit(f"run {name!r}: oof.npy shape {oof.shape} != "
                      f"{(len(uids), 12)}")
        oofs.append(oof)

    _check_equal_keys(configs, names)

    epochs_done_values = set()
    for cfg in configs:
        ed = cfg.get("epochs_done")
        if ed:
            epochs_done_values.update(ed.values())
    if len(epochs_done_values) > 1:
        print(f"WARN: epochs_done differs across runs: {sorted(epochs_done_values)}")

    split_folds = configs[0]["split_folds"]

    merged_oof = np.full((len(uids), 12), np.nan, dtype=np.float32)
    claimed_by = {}
    run_folds = []  # per run, folds it actually claimed here (== folds_trained)
    for name, d, cfg, run_oof in zip(names, args.run_dirs, configs, oofs):
        family = cfg["family"]
        folds = []
        for k in cfg["folds_trained"]:
            if k in claimed_by:
                sys.exit(f"fold {k} claimed by both {claimed_by[k]!r} and "
                          f"{name!r}")
            mask = fold == k
            if not np.any(run_oof[mask]):
                sys.exit(f"run {name!r}: fold {k} is in folds_trained but its "
                          f"oof rows are all zero (claimed but never written)")
            merged_oof[mask] = run_oof[mask]
            src_pt = os.path.join(d, f"{family}_fold{k}.pt")
            if not os.path.isfile(src_pt):
                sys.exit(f"run {name!r}: missing checkpoint {src_pt}")
            shutil.copy(src_pt, os.path.join(args.out_dir, f"{family}_fold{k}.pt"))
            claimed_by[k] = name
            folds.append(k)
        run_folds.append(folds)

    union_folds = sorted(claimed_by)
    if len(union_folds) < split_folds:
        missing = sorted(set(range(split_folds)) - set(union_folds))
        print(f"WARN: fold coverage {len(union_folds)}/{split_folds}; "
              f"folds {missing} untrained (oof rows stay NaN)")

    for name, d in zip(names, args.run_dirs):
        shutil.copy(os.path.join(d, "gold_windows.npz"),
                    os.path.join(args.out_dir, f"gold_windows_{name}.npz"))

    shutil.copy(os.path.join(args.run_dirs[0], "fold_of.json"),
                os.path.join(args.out_dir, "fold_of.json"))

    labels_by_uid = _load_labels(args.labels)
    trained_mask = ~np.isnan(merged_oof).any(axis=1)
    trained_uids = [u for u, keep in zip(uids, trained_mask) if keep]
    for u in trained_uids:
        if u not in labels_by_uid:
            sys.exit(f"uid {u!r} (a trained row) missing from labels csv "
                      f"{args.labels}")
    y = np.stack([labels_by_uid[u] for u in trained_uids])
    aucs = auc_per_label(y, merged_oof[trained_mask])
    macro = float(np.mean(list(aucs.values()))) if aucs else float("nan")

    merged_cfg = {
        **configs[0],
        "folds_trained": union_folds,
        "n_folds": len(union_folds),
        "merged_from": dict(zip(names, run_folds)),
        "use_tta": all(cfg["use_tta"] for cfg in configs),
        "use_tta_per_run": {n: c["use_tta"] for n, c in zip(names, configs)},
        "gold58_auc": None,
        "tta_gold58_auc": None,
        "oof_llm_auc": round(macro, 4),
    }
    for key in PER_FOLD_KEYS:
        merged = _merged_per_fold(configs, run_folds, key)
        if merged:
            merged_cfg[key] = merged

    with open(os.path.join(args.out_dir, "run_config.json"), "w") as fh:
        json.dump(merged_cfg, fh)
    np.save(os.path.join(args.out_dir, "oof.npy"), merged_oof)

    print(f"{'label':<18} {'auc':>8}")
    for name in LABELS:
        cell = f"{aucs[name]:.4f}" if name in aucs else "skip"
        print(f"{name:<18} {cell:>8}")
    print()
    print(f"macro AUC: {macro:.4f}")
    print(f"fold coverage {len(union_folds)}/{split_folds}")


if __name__ == "__main__":
    main()
