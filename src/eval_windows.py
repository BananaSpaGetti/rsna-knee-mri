"""Paired OOF comparison: does dense (overlapping, per-slice) 3-slice window
scoring beat the shipped non-overlapping 3-slice grouping (model.study_windows),
using the already-trained CoAtNet-2 fold checkpoints unchanged?

For each fold checkpoint, every study assigned to that fold (fold_of.json) is
scored twice with the SAME model: once with model.study_windows(vol, 3) (the
"stride 3" scheme - reproduces the training run's oof.npy) and once with
model.study_windows(vol, 3, stride=1) (the "stride 1" scheme - a window
centred on every slice, Raptor's K_EVAL-style dense scoring; the stride
option grew out of this script's original local copy of that scheme). Both schemes report p_attn (the
attention head's sigmoid over all of a study's windows, owner=zeros); no TTA
(run_config.json's use_tta is false for this arm).

Usage:
    py src/eval_windows.py --weights C:/kaggle/staging/weights-v6c \\
        --labels data/labels-consensus/consensus_v1.csv --strides 3,1 \\
        [--limit N] [--out docs/eval-windows.md]

The train script (notebooks/train/train_dinov2.py) is not imported: its
module body runs find_cache_parts()/load_index() etc. against hardcoded
/kaggle/input paths as soon as it is imported (no `if __name__ ==
"__main__"` guard around that scan), which is a Kaggle-only layout. The
pieces needed here (cache index loading, macro AUC) are re-derived instead,
reusing src/model.py and src/tune_pooling.py where those modules make it a
plain import (tune_pooling.auc_per_label has no import-time side effects).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Model, study_windows  # noqa: E402
from preprocess import LABELS  # noqa: E402
from tune_pooling import auc_per_label  # noqa: E402

DEFAULT_CACHE_ROOT = "C:/kaggle/input"


def find_cache_parts(root: str, marker: str = "cache_index.csv",
                     skip=("competitions", "rsna-knee-abnormality-detection")):
    """Every directory under `root` holding `marker` (deterministic, sorted
    walk, same pruning as train_dinov2.find_cache_parts) - a v4 cache ships
    as two directories (disjoint global shard ranges)."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath == root:
            dirnames[:] = sorted(d for d in dirnames if d not in skip)
        else:
            dirnames.sort()
        if marker in filenames:
            found.append(dirpath)
    if not found:
        raise FileNotFoundError(f"{marker} not under {root} (skipped {skip})")
    return sorted({os.path.realpath(d) for d in found})


def load_cache_index(parts):
    """({uid: (shard, row)}, {shard: part dir}, cache_config dict) merged
    across cache parts. Mirrors train_dinov2.load_index's shard/uid dedupe
    checks; rows with filled_slots == 0 (all-zero study) are dropped."""
    cache_cfg = None
    for p in parts:
        with open(os.path.join(p, "cache_config.json")) as fh:
            cfg = json.load(fh)
        if cache_cfg is None:
            cache_cfg = cfg
        else:
            assert cfg == cache_cfg, (
                f"cache_config.json differs between {parts[0]} and {p}")

    index, shard_seen_in, seen_uids = {}, {}, set()
    for p in parts:
        with open(os.path.join(p, "cache_index.csv"), newline="") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            uid = r.get("StudyInstanceUID")
            if not uid or r.get("shard") in (None, "") or r.get("row") in (None, ""):
                continue
            shard = int(r["shard"])
            prev = shard_seen_in.setdefault(shard, p)
            assert prev == p, f"shard {shard} appears in both {prev} and {p}"
            assert uid not in seen_uids, f"study {uid} appears in more than one cache part"
            seen_uids.add(uid)
            fs = r.get("filled_slots")
            if fs not in (None, "") and int(float(fs)) == 0:
                continue
            index[uid] = (shard, int(r["row"]))

    shard_dir = {}
    for shard, p in shard_seen_in.items():
        path = os.path.join(p, f"cache_{shard:03d}.npy")
        assert os.path.exists(path), f"{path} missing (listed in {p}'s cache_index.csv)"
        shard_dir[shard] = p
    return index, shard_dir, cache_cfg


class Shards:
    """Memory-maps cache_XXX.npy on demand (same as train_dinov2.Shards)."""

    def __init__(self, shard_dir):
        self.shard_dir = shard_dir
        self._maps = {}

    def get(self, shard, row):
        if shard not in self._maps:
            self._maps[shard] = np.load(
                os.path.join(self.shard_dir[shard], f"cache_{shard:03d}.npy"),
                mmap_mode="r")
        return self._maps[shard][row]


def score_study(model, vol, device, use_amp, stride):
    """p_attn (12,) for one study/scheme: same math as model.predict_study
    with window=3 (stride 3) or its dense stride=k scheme (any other
    stride), no TTA (this arm's run_config.json has use_tta=false)."""
    # stride 3 = the training-time non-overlapping grouping (model.py's
    # stride=None); any other value = model.study_windows' dense scheme.
    x, slots, owner = study_windows(vol, group=3,
                                    stride=None if stride == 3 else stride)
    x, slots, owner = x.to(device), slots.to(device), owner.to(device)
    with torch.no_grad(), torch.amp.autocast(device, dtype=torch.float16, enabled=use_amp):
        h = model.embed(x, slots)
        logits = model.head(h, owner, 1)
    return torch.sigmoid(logits).float().cpu().numpy()[0]


def macro(aucs: dict) -> float:
    return float(np.mean(list(aucs.values()))) if aucs else float("nan")


def bootstrap_macro_diff(y, probs_base, probs_challenger, n_boot=1000, seed=0):
    """Paired bootstrap (resample studies with replacement, same resample
    for both schemes) of the macro-AUC difference (challenger - base).
    Returns (mean, std) over n_boot resamples."""
    rng = np.random.default_rng(seed)
    n = y.shape[0]
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        auc_base = auc_per_label(yb, probs_base[idx])
        auc_chal = auc_per_label(yb, probs_challenger[idx])
        common = sorted(set(auc_base) & set(auc_chal))
        diffs[b] = (np.mean([auc_chal[k] for k in common])
                    - np.mean([auc_base[k] for k in common]))
    return float(diffs.mean()), float(diffs.std())


def load_labels_csv(path):
    """{uid: (12,) float32} in preprocess.LABELS order, binarised later."""
    labels = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            labels[r["StudyInstanceUID"]] = np.array(
                [float(r[c]) for c in LABELS], dtype=np.float32)
    return labels


def build_report(strides, per_label_aucs, macro_aucs, boot_mean, boot_std,
                 repro_mean, repro_max, n_studies, wall_s, weights_name):
    lines = [f"# Window-scheme OOF comparison ({weights_name})", ""]
    lines.append(f"Studies scored: {n_studies}; wall time: {wall_s/60:.1f} min")
    lines.append("")
    lines.append(f"stride-3 reproduction of saved oof.npy: mean abs diff "
                f"{repro_mean:.5f}, max abs diff {repro_max:.5f}")
    lines.append("")
    header = "| label | " + " | ".join(f"stride{s}" for s in strides) + " |"
    lines.append(header)
    lines.append("|---" * (len(strides) + 1) + "|")
    all_labels = sorted(set().union(*[set(d) for d in per_label_aucs.values()]))
    for name in all_labels:
        row = [name] + [f"{per_label_aucs[s].get(name, float('nan')):.4f}" for s in strides]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("| **MACRO** | " + " | ".join(f"{macro_aucs[s]:.4f}" for s in strides) + " |")
    lines.append("")
    lines.append(f"paired bootstrap macro diff (stride{strides[-1]} - stride{strides[0]}, "
                f"1000 resamples): mean {boot_mean:+.4f}, std {boot_std:.4f}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--strides", default="3,1")
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N studies per fold")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    ap.add_argument("--cache-version", default=None,
                    help="keep only cache parts whose cache_config.json version matches "
                         "(v4 and v4g parts can share one root)")
    ap.add_argument("--folds", default=None,
                    help="comma-separated folds to score (default: all in fold_of)")
    ap.add_argument("--device", default=None,
                    help="cuda | cpu (default: cuda when available)")
    args = ap.parse_args()

    strides = [int(s) for s in args.strides.split(",")]

    run_cfg = json.load(open(os.path.join(args.weights, "run_config.json")))
    fold_of = json.load(open(os.path.join(args.weights, "fold_of.json")))
    uids_order = list(fold_of.keys())  # == oof.npy row order (insertion order)
    n_studies = len(uids_order)
    row_of_uid = {u: i for i, u in enumerate(uids_order)}

    labels_map = load_labels_csv(args.labels)
    missing = [u for u in uids_order if u not in labels_map]
    assert not missing, f"{len(missing)} fold_of uids missing from {args.labels}"
    y_all = np.stack([labels_map[u] for u in uids_order])

    oof_saved = np.load(os.path.join(args.weights, "oof.npy"))
    assert oof_saved.shape == (n_studies, len(LABELS)), (
        f"oof.npy shape {oof_saved.shape} != ({n_studies}, {len(LABELS)})")

    cache_parts = find_cache_parts(args.cache_root)
    if args.cache_version is not None:
        cache_parts = [d for d in cache_parts
                       if json.load(open(os.path.join(d, "cache_config.json"))).get("version")
                       == args.cache_version]
        assert cache_parts, f"no cache part under {args.cache_root} with version {args.cache_version!r}"
    print(f"cache parts ({len(cache_parts)}): {cache_parts}", flush=True)
    index, shard_dir, cache_cfg = load_cache_index(cache_parts)
    shards = Shards(shard_dir)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device == "cuda"
    img_size = run_cfg["size"]
    arch = run_cfg["arch"]
    family = run_cfg["family"]
    drop_path = run_cfg.get("drop_path")
    unfreeze = run_cfg.get("unfreeze")
    print(f"device={device} arch={arch!r} img_size={img_size} family={family!r}",
          flush=True)

    results = {s: np.full((n_studies, len(LABELS)), np.nan, dtype=np.float32)
              for s in strides}
    processed = np.zeros(n_studies, dtype=bool)

    folds = sorted(set(fold_of.values()))
    if args.folds is not None:
        folds = [int(f) for f in args.folds.split(",")]
    t0 = time.time()
    for fold in folds:
        ckpt_path = os.path.join(args.weights, f"{family}_fold{fold}.pt")
        model = Model(pretrained=False, img_size=img_size, arch=arch,
                     grad_ckpt=False, unfreeze=unfreeze, drop_path=drop_path,
                     n_slots=None, spatial_pool=run_cfg.get("spatial_pool")).to(device)
        sd = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(sd)
        model.eval()

        fold_uids = [u for u in uids_order if fold_of[u] == fold]
        if args.limit is not None:
            fold_uids = fold_uids[:args.limit]
        print(f"fold {fold}: {len(fold_uids)} studies", flush=True)

        for n_done, u in enumerate(fold_uids, 1):
            i = row_of_uid[u]
            shard, row = index[u]
            vol = np.asarray(shards.get(shard, row))
            for s in strides:
                results[s][i] = score_study(model, vol, device, use_amp, s)
            processed[i] = True
            if n_done % 200 == 0:
                print(f"  fold {fold}: {n_done}/{len(fold_uids)} "
                      f"({time.time()-t0:.0f}s elapsed)", flush=True)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    wall_s = time.time() - t0
    n_processed = int(processed.sum())
    print(f"scored {n_processed}/{n_studies} studies in {wall_s:.0f}s", flush=True)

    # reproduction check: stride-3 scheme vs the training run's saved oof.npy
    repro_mean = repro_max = float("nan")
    if 3 in results:
        diff = np.abs(results[3][processed] - oof_saved[processed])
        repro_mean, repro_max = float(diff.mean()), float(diff.max())
        print(f"stride-3 vs saved oof.npy: mean abs diff {repro_mean:.5f}, "
              f"max abs diff {repro_max:.5f} (over {n_processed} studies)",
              flush=True)

    y_proc = y_all[processed]
    per_label_aucs, macro_aucs = {}, {}
    for s in strides:
        aucs = auc_per_label(y_proc, results[s][processed])
        per_label_aucs[s] = aucs
        macro_aucs[s] = macro(aucs)
        print(f"stride {s} macro AUC: {macro_aucs[s]:.4f} "
              f"({len(aucs)}/{len(LABELS)} labels scored)", flush=True)
        for name, a in aucs.items():
            print(f"  {name}: {a:.4f}", flush=True)

    boot_mean = boot_std = float("nan")
    if len(strides) >= 2:
        boot_mean, boot_std = bootstrap_macro_diff(
            y_proc, results[strides[0]][processed], results[strides[-1]][processed])
        print(f"paired bootstrap macro diff (stride{strides[-1]} - "
              f"stride{strides[0]}, 1000 resamples): mean {boot_mean:+.4f}, "
              f"std {boot_std:.4f}", flush=True)

    for s in strides:
        out_path = os.path.join(args.weights, f"oof_stride{s}_eval.npy")
        np.save(out_path, results[s])
        print(f"wrote {out_path}", flush=True)

    if args.out:
        report = build_report(strides, per_label_aucs, macro_aucs, boot_mean,
                              boot_std, repro_mean, repro_max, n_processed, wall_s,
                              os.path.basename(os.path.normpath(args.weights)))
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(report)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
