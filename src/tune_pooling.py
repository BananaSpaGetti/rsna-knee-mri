"""Tune per-label window pooling (attn/mean/max/top2) on the gold-58 saved
per-window probs (gold_windows.npz); writes the chosen table into
run_config.json in the same directory.

Usage: py src/tune_pooling.py <weights_dir>
"""

import json
import os
import sys

import numpy as np
try:
    from sklearn.metrics import roc_auc_score
except ImportError:  # sklearn's compiled extension blocked (Smart App Control) or not installed
    from auc import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import POOLERS
from preprocess import LABELS

# Prior table from the survey of public 0.93+ notebooks; labels absent here
# default to attn (the model's own attention pooling over all windows).
PRIOR_TABLE = {
    "Fracture": "max", "Contusion": "max", "Medial Meniscus": "max",
    "Lateral Meniscus": "max", "Baker's": "max",
    "ACL": "top2", "MCL": "top2",
}
assert set(PRIOR_TABLE) <= set(LABELS), "PRIOR_TABLE has a label not in LABELS"
MARGIN = 0.002


def auc_per_label(y, scores):
    """{label: auc}; a label whose gold column is single-class is omitted."""
    aucs = {}
    for c, name in enumerate(LABELS):
        t = (y[:, c] >= 0.5).astype(int)
        if t.min() == t.max():
            continue
        aucs[name] = roc_auc_score(t, scores[:, c])
    return aucs


def choose_table(mode_aucs):
    """mode_aucs: {mode: {label: auc}}. Per label: start from attn; switch to
    the prior's mode if it beats attn by more than MARGIN; switch to any
    other mode only if it beats BOTH attn and the prior's mode by more than
    MARGIN. Returns {label: mode} for every label (skipped labels -> attn).

    A label is skipped (kept at attn) if ANY mode's AUC for it is missing
    (single-class gold column) or NaN - the same guard applied uniformly to
    every mode, not just attn, so a NaN slipping into mean/max/top2 cannot
    silently win a comparison."""
    modes = ("attn", "mean", "max", "top2")
    chosen = {}
    for name in LABELS:
        aucs = [mode_aucs.get(m, {}).get(name) for m in modes]
        if any(a is None or np.isnan(a) for a in aucs):
            chosen[name] = "attn"
            continue
        attn_a = mode_aucs["attn"][name]
        prior_m = PRIOR_TABLE.get(name, "attn")
        prior_a = mode_aucs[prior_m][name]
        pick = prior_m if prior_a - attn_a > MARGIN else "attn"
        others = [m for m in ("mean", "max", "top2") if m != prior_m]
        qualifying = [m for m in others
                     if mode_aucs[m][name] - attn_a > MARGIN
                     and mode_aucs[m][name] - prior_a > MARGIN]
        if qualifying:
            pick = max(qualifying, key=lambda m: mode_aucs[m][name])
        chosen[name] = pick
    return chosen


def main():
    weights_dir = sys.argv[1]
    d = np.load(os.path.join(weights_dir, "gold_windows.npz"))
    probs, study_probs, y = d["probs"], d["study_probs"], d["y"]
    cfg_path = os.path.join(weights_dir, "run_config.json")
    with open(cfg_path) as fh:
        cfg = json.load(fh)

    mode_scores = {"attn": study_probs.mean(axis=1)}  # (n_gold, n_folds, 12) -> (n_gold, 12)
    for m, fn in POOLERS.items():
        mode_scores[m] = fn(probs, axis=2).mean(axis=1)
    mode_aucs = {m: auc_per_label(y, s) for m, s in mode_scores.items()}

    chosen = choose_table(mode_aucs)

    modes = ("attn", "mean", "max", "top2")
    print(f"{'label':<18} {'pos':>3}  " + "  ".join(f"{m:>6}" for m in modes)
          + "  chosen")
    for c, name in enumerate(LABELS):
        pos = int((y[:, c] >= 0.5).sum())
        cells = [f"{mode_aucs[m][name]:.4f}" if name in mode_aucs[m] else "  skip"
                for m in modes]
        print(f"{name:<18} {pos:>3}  " + "  ".join(f"{cell:>6}" for cell in cells)
              + f"  {chosen[name]}")

    scored = list(mode_aucs["attn"].keys())
    macro_attn = float(np.mean([mode_aucs["attn"][n] for n in scored]))
    macro_chosen = float(np.mean([mode_aucs[chosen[n]][n] for n in scored]))
    print(f"macro AUC all-attn: {macro_attn:.4f}")
    print(f"macro AUC chosen:   {macro_chosen:.4f}")
    assert macro_chosen >= macro_attn - 1e-9, (
        f"chosen table ({macro_chosen:.4f}) scored worse than all-attn "
        f"({macro_attn:.4f}) - selection logic is broken")

    cfg["window_pool"] = chosen
    with open(cfg_path, "w") as fh:
        json.dump(cfg, fh)
    print("window_pool written to", cfg_path)
    print(json.dumps({"window_pool": chosen}, indent=2))


if __name__ == "__main__":
    main()
