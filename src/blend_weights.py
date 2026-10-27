"""Compute per-label rank-blend weights across model families from their
out-of-fold (OOF) predictions on the training studies, and write them to a
JSON the inference kernel reads via blend_eval.rank_blend(weights=...).

Weights are derived ONLY from OOF AUC against the consensus labels on the
training studies (fold_of.json / oof.npy in each run dir) - NEVER from the
58 gold studies, which is the mistake round 4/5 made. The weighted-blend
OOF macro AUC printed at the end is information only: it is a sanity check
computed on the same rows the weights were fit on, not a selection signal
and not comparable to a gold-58 number.

Usage: py src/blend_weights.py --labels <consensus_v1.csv> --out
       <blend_weights.json> [--allow-partial] <run_dir> [<run_dir> ...]

Each run dir holds fold_of.json (uid -> fold, dict order matches oof.npy's
rows), oof.npy ((n_train, 12) float32; a row for a fold this run did not
train is NaN in a merged dir or all-zero in a single-kernel dir), and
run_config.json (family, folds_trained, split_folds - see blend_eval.py's
_family_of for the family default). Family is taken via blend_eval._family_of
so both scripts agree on it; two run dirs mapping to the same family is an
error - pass a merged dir instead of the per-kernel split dirs.
"""

import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blend_eval import _family_of, rank_blend
from preprocess import LABELS
from tune_pooling import auc_per_label


def _load_run(d):
    """(uids, fold_of, oof, folds_trained, split_folds) for one run dir.
    uids is fold_of.json's key order, matching oof.npy's row order."""
    with open(os.path.join(d, "fold_of.json")) as fh:
        fold_of = json.load(fh)
    uids = list(fold_of.keys())
    oof = np.load(os.path.join(d, "oof.npy"))
    with open(os.path.join(d, "run_config.json")) as fh:
        cfg = json.load(fh)
    folds_trained, split_folds = cfg.get("folds_trained"), cfg.get("split_folds")
    if folds_trained is None or split_folds is None:
        # pre-round-5 run_configs (e.g. weights-v4b) only carry n_folds; those
        # runs trained every fold of their own split, so derive both from it
        n_folds = cfg.get("n_folds")
        if n_folds is None:
            sys.exit(f"{d}: run_config.json missing folds_trained/split_folds/n_folds")
        folds_trained = list(range(int(n_folds)))
        split_folds = int(n_folds)
        print(f"note: {d}: no folds_trained/split_folds in run_config.json, "
              f"assuming all {n_folds} folds trained", flush=True)
    return uids, fold_of, oof, folds_trained, split_folds


def _labels_from_csv(path, uids):
    """(len(uids), 12) binary array (>= 0.5) from consensus_v1.csv, in uids
    order; a hard error if a uid is missing from the CSV."""
    rows = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows[row["StudyInstanceUID"]] = row
    y = np.empty((len(uids), len(LABELS)), dtype=float)
    for i, uid in enumerate(uids):
        if uid not in rows:
            sys.exit(f"uid {uid} not found in {path}")
        for c, name in enumerate(LABELS):
            y[i, c] = float(rows[uid][name])
    return (y >= 0.5).astype(float)


def main():
    argv = sys.argv[1:]
    labels_path, out_path, allow_partial, dirs = None, None, False, []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--labels":
            i += 1
            labels_path = argv[i]
        elif a == "--out":
            i += 1
            out_path = argv[i]
        elif a == "--allow-partial":
            allow_partial = True
        else:
            dirs.append(a)
        i += 1
    if not labels_path or not out_path or not dirs:
        sys.exit("usage: py src/blend_weights.py --labels <consensus_v1.csv> "
                  "--out <blend_weights.json> [--allow-partial] <run_dir> ...")

    runs = {}  # family -> {dir, uids, fold_of, oof, folds_trained, split_folds}
    for d in dirs:
        fam = _family_of(d)
        uids, fold_of, oof, folds_trained, split_folds = _load_run(d)
        if fam in runs:
            sys.exit(f"duplicate family {fam!r}: {runs[fam]['dir']} and {d} - "
                      f"pass a merged dir, not the per-kernel split dirs")
        runs[fam] = dict(dir=d, uids=uids, fold_of=fold_of, oof=oof,
                          folds_trained=set(folds_trained), split_folds=split_folds)
    fam_names = list(runs.keys())

    ref_fam = fam_names[0]
    ref_uids = runs[ref_fam]["uids"]
    for fam in fam_names[1:]:
        if runs[fam]["uids"] != ref_uids:
            sys.exit(f"uid list mismatch between {ref_fam} and {fam}")
    uids = ref_uids

    split_folds_vals = {runs[f]["split_folds"] for f in fam_names}
    if len(split_folds_vals) > 1:
        sys.exit(f"split_folds mismatch across runs: {split_folds_vals}")
    split_folds = split_folds_vals.pop()

    union_folds = set()
    for f in fam_names:
        union_folds |= runs[f]["folds_trained"]
    missing = sorted(set(range(split_folds)) - union_folds)
    if missing:
        msg = f"folds {missing} of {split_folds} not covered by any run"
        if not allow_partial:
            sys.exit(msg)
        print(f"WARN: {msg} (--allow-partial)")

    y = _labels_from_csv(labels_path, uids)

    used_mask = {}
    for f in fam_names:
        r = runs[f]
        fold_arr = np.array([r["fold_of"][u] for u in uids])
        trained = np.isin(fold_arr, list(r["folds_trained"]))
        finite = np.all(np.isfinite(r["oof"]), axis=1)
        used_mask[f] = trained & finite

    aucs = {f: auc_per_label(y[used_mask[f]], runs[f]["oof"][used_mask[f]])
            for f in fam_names}

    weights = {name: {} for name in LABELS}
    for name in LABELS:
        raw = {f: max(aucs[f].get(name, 0.5) - 0.5, 0.0) for f in fam_names}
        total = sum(raw.values())
        if total <= 0:
            print(f"WARN: all families score <= 0.5 AUC for {name!r}; "
                  f"using equal weights")
            for f in fam_names:
                weights[name][f] = 1.0 / len(fam_names)
        else:
            for f in fam_names:
                weights[name][f] = raw[f] / total

    shared_mask = np.all([used_mask[f] for f in fam_names], axis=0)
    w_matrix = np.array([[weights[name][f] for name in LABELS] for f in fam_names])
    mats = [runs[f]["oof"][shared_mask] for f in fam_names]
    blend = rank_blend(mats, weights=w_matrix)
    blend_aucs = auc_per_label(y[shared_mask], blend)

    print(f"{'label':<18} " + "  ".join(f"{f + ' auc':>12} {f + ' w':>7}"
                                          for f in fam_names))
    for name in LABELS:
        cells = []
        for f in fam_names:
            a = f"{aucs[f][name]:.4f}" if name in aucs[f] else "      skip"
            w = f"{weights[name][f]:.4f}"
            cells.append(f"{a:>12} {w:>7}")
        print(f"{name:<18} " + "  ".join(cells))

    print()
    for f in fam_names:
        scored = list(aucs[f].keys())
        macro = float(np.mean([aucs[f][n] for n in scored])) if scored else float("nan")
        print(f"macro AUC {f:<10} (own OOF rows, n={int(used_mask[f].sum())}): "
              f"{macro:.4f}")

    scored = list(blend_aucs.keys())
    macro_blend = float(np.mean([blend_aucs[n] for n in scored])) if scored else float("nan")
    print(f"macro AUC weighted blend (shared OOF rows, n={int(shared_mask.sum())}): "
          f"{macro_blend:.4f}  [information only - not a gold-58 selection]")

    out = {
        "version": 1,
        "families": fam_names,
        "n_studies": len(uids),
        "weights": weights,
        "auc": aucs,
    }
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"weights written to {out_path}")


if __name__ == "__main__":
    main()
