"""Multi-teacher pseudo labels: blend the LLM report consensus with our own
image models' out-of-fold predictions, and fill report silence with the
image teacher where silence carries no information.

Why (docs/label-audit.md, docs/competitors.md 2026-09-25): every account above
0.945 on the public LB credits label work, not backbones; stevenleehans showed
that imputing "not addressed" cells lifts the label set's gold-58 AUC for
Synovitis (+0.056) but hurts Baker's / Contusion, because report silence is
informative for some findings (unreported Baker's is almost always absent)
and uninformative for others (a third of unreported Synovitis is present).

Inputs
  data/labels-consensus/consensus_v1.csv     y and w_ columns, 4407 studies
  data/labels-pilkwang/report_labels_v2.csv  <label>__verdict == "UNK" = silent
  data/labels-stevenleehans/llm_labels_full.csv  value == 0.5 = silent (the raw
                    reader; v4_blend has no 0.5 cells left)
  --run <dir> ...   merged run dirs (oof.npy + fold_of.json + run_config.json +
                    gold_windows*.npz); each is one image teacher, all five
                    folds trained, split identical across runs
  data/gold58.csv   radiologist labels, evaluation only (never a target)

Rule (pseudo_v1)
  silent(cell)   = pilkwang UNK and stevenleehans 0.5 (a missing reader row
                   counts as silent for that reader)
  teacher(cell)  = mean over runs of the OOF probability (train rows only)
  FILL labels    = labels whose gold-58 positive rate when silent is at least
                   SILENT_RATIO_MIN times the rate when addressed - silence
                   is not evidence of absence there
  addressed cell = MIX_W * consensus + (1 - MIX_W) * teacher, weight unchanged
  silent cell    = teacher if the label is in FILL, else consensus unchanged;
                   filled cells get w = 0.15 + 0.85 * (0.65 * agreement +
                   0.35 * certainty) with agreement = 1 - |teacher_a - teacher_b|
                   (the make_labels.py weight formula, readers -> teachers)
  gold-58 rows   = consensus unchanged (no OOF exists for holdout studies;
                   they are never trained on anyway)

Every variant is scored on gold-58 as a LABEL SET (its values on the 58 gold
studies vs the radiologists), using gold_windows predictions as the teacher
component there. That table justifies the rule; the LB decides whether a
model trained on the file is better (nothing selected on gold-58 ships
without an LB check).

Usage:
  py src/pseudo_labels.py --run C:/kaggle/staging/weights-v6c \\
      --run C:/kaggle/staging/weights-v4b \\
      --out data/labels-consensus/pseudo_v1.csv --report docs/pseudo-labels.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
try:
    from sklearn.metrics import roc_auc_score
except ImportError:  # sklearn's compiled extension blocked (Smart App Control) or not installed
    from auc import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, HERE)
from preprocess import LABELS  # noqa: E402

MIX_W = 0.5              # consensus share in an addressed cell
SILENT_RATIO_MIN = 0.3   # silent / addressed gold-positive rate -> fill
UID = "StudyInstanceUID"


def auc_or_nan(t, s):
    t = (np.asarray(t, dtype=float) >= 0.5).astype(int)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(s)
    if ok.sum() == 0 or t[ok].min() == t[ok].max():
        return float("nan")
    return roc_auc_score(t[ok], s[ok])


def load_silence(uids: pd.Index) -> np.ndarray:
    pk = pd.read_csv(os.path.join(DATA, "labels-pilkwang", "report_labels_v2.csv")).set_index(UID)
    st = pd.read_csv(os.path.join(DATA, "labels-stevenleehans", "llm_labels_full.csv")).set_index(UID)
    pk_unk = np.stack([pk.reindex(uids)[f"{l}__verdict"].fillna("UNK").eq("UNK").to_numpy()
                       for l in LABELS], axis=1)
    st_v = st.reindex(uids)[LABELS].to_numpy(dtype=float)
    st_unk = ~np.isfinite(st_v) | (st_v == 0.5)
    return pk_unk & st_unk


def load_teacher_oof(run_dir: str, uids: pd.Index) -> np.ndarray:
    """(n_uids, 12) OOF probabilities aligned on uids; NaN where the run has no
    row (gold-58 holdout) or the fold was not trained."""
    cfg = json.load(open(os.path.join(run_dir, "run_config.json")))
    fold_of = json.load(open(os.path.join(run_dir, "fold_of.json")))
    run_uids = list(fold_of)
    oof = np.load(os.path.join(run_dir, "oof.npy")).astype(float)
    trained = set(cfg.get("folds_trained") or range(int(cfg.get("split_folds", cfg.get("n_folds", 5)))))
    if len(trained) < int(cfg.get("split_folds", cfg.get("n_folds", 5))):
        raise SystemExit(f"{run_dir}: only folds {sorted(trained)} trained - need all of them")
    ok = np.array([fold_of[u] in trained for u in run_uids])
    oof[~ok] = np.nan
    frame = pd.DataFrame(oof, index=run_uids, columns=LABELS)
    return frame.reindex(uids).to_numpy(dtype=float), fold_of


def load_teacher_gold(run_dir: str, gold_uids: list[str]) -> np.ndarray:
    mats = []
    for p in sorted(glob.glob(os.path.join(run_dir, "gold_windows*.npz"))):
        z = np.load(p, allow_pickle=True)
        zu = list(z["uids"])
        order = [zu.index(u) for u in gold_uids]
        sp = z["study_probs"][order]
        for k in range(sp.shape[1]):
            mats.append(sp[:, k, :])
    if not mats:
        raise SystemExit(f"no gold_windows*.npz in {run_dir}")
    return np.stack(mats).mean(axis=0)


def per_label_table(rows: dict[str, np.ndarray], gold: np.ndarray) -> pd.DataFrame:
    tab = pd.DataFrame({name: [auc_or_nan(gold[:, j], s[:, j]) for j in range(len(LABELS))]
                        for name, s in rows.items()}, index=LABELS)
    tab.loc["macro"] = tab.mean(axis=0)
    return tab


def md_table(df: pd.DataFrame, nd: int = 3) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(["label"] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append("—" if (isinstance(v, float) and np.isnan(v)) else
                         (f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v)))
        out.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="append", required=True, help="merged run dir (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    if len(args.run) < 2:
        raise SystemExit("need at least two --run teachers (agreement needs a pair)")

    cons = pd.read_csv(os.path.join(DATA, "labels-consensus", "consensus_v1.csv")).set_index(UID)
    uids = cons.index
    y = cons[LABELS].to_numpy(dtype=float)
    w = cons[[f"w_{l}" for l in LABELS]].to_numpy(dtype=float)
    gold_df = pd.read_csv(os.path.join(DATA, "gold58.csv")).set_index(UID)[LABELS]
    gold_uids = [u for u in gold_df.index if u in cons.index]
    gold = gold_df.loc[gold_uids].to_numpy(dtype=float)
    gold_pos = np.array([u in set(gold_uids) for u in uids])

    silent = load_silence(uids)

    teachers, ref_fold_of = [], None
    for r in args.run:
        t, fo = load_teacher_oof(r, uids)
        if ref_fold_of is not None and fo != ref_fold_of:
            raise SystemExit(f"{r}: fold_of.json differs from {args.run[0]}")
        ref_fold_of = fo
        teachers.append(t)
    teach = np.stack(teachers)                       # (n_runs, n, 12)
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        teacher = np.nanmean(teach, axis=0)          # NaN on gold rows
        agreement = np.clip(1 - np.nanmax(teach, axis=0) + np.nanmin(teach, axis=0), 0, 1)
    has_teacher = np.isfinite(teacher).all(axis=1)
    n_train = int(has_teacher.sum())
    assert not (has_teacher & gold_pos).any(), "a gold-58 study has OOF rows - holdout broken"

    teacher_gold = np.mean([load_teacher_gold(r, gold_uids) for r in args.run], axis=0)
    cons_gold = cons.loc[gold_uids, LABELS].to_numpy(dtype=float)
    silent_gold = silent[[list(uids).index(u) for u in gold_uids]]

    # --- silence informativeness ---------------------------------------
    info = []
    for j, l in enumerate(LABELS):
        sg, tr = silent_gold[:, j], silent[:, j] & has_teacher
        g_sil = gold[sg, j].mean() if sg.any() else np.nan
        g_add = gold[~sg, j].mean() if (~sg).any() else np.nan
        t_sil = (teacher[tr, j] >= 0.5).mean() if tr.any() else np.nan
        t_add = (teacher[~silent[:, j] & has_teacher, j] >= 0.5).mean()
        ratio = g_sil / g_add if (g_add and np.isfinite(g_sil)) else np.nan
        info.append({"label": l, "silent_frac_train": float(silent[has_teacher, j].mean()),
                     "n_silent_gold": int(sg.sum()), "gold_pos_silent": g_sil,
                     "gold_pos_addressed": g_add, "ratio": ratio,
                     "teacher_pos_silent": t_sil, "teacher_pos_addressed": t_add})
    info = pd.DataFrame(info).set_index("label")
    fill = [l for l in LABELS if np.isfinite(info.loc[l, "ratio"]) and info.loc[l, "ratio"] >= SILENT_RATIO_MIN]
    fill_mask = np.array([l in fill for l in LABELS])

    # --- label-set variants on gold-58 -------------------------------------
    def variant(c, t, s, mix, fill_silent):
        out = mix * c + (1 - mix) * t if mix is not None else c.copy()
        if fill_silent is not None:
            m = s & fill_silent[None, :]
            out[m] = t[m]
        return out

    all_fill = np.ones(len(LABELS), dtype=bool)
    variants_gold = {
        "consensus": cons_gold,
        "teacher": teacher_gold,
        "mix50": variant(cons_gold, teacher_gold, silent_gold, MIX_W, None),
        "fill_all": variant(cons_gold, teacher_gold, silent_gold, None, all_fill),
        "fill_FILL": variant(cons_gold, teacher_gold, silent_gold, None, fill_mask),
        "pseudo_v1": variant(cons_gold, teacher_gold, silent_gold, MIX_W, fill_mask),
    }
    tab = per_label_table(variants_gold, gold)

    # --- build the file ------------------------------------------------------
    out_y, out_w = y.copy(), w.copy()
    tr = has_teacher
    out_y[tr] = MIX_W * y[tr] + (1 - MIX_W) * teacher[tr]
    filled = silent & fill_mask[None, :] & tr[:, None]
    out_y[filled] = teacher[filled]
    certainty = np.clip(2 * np.abs(teacher - 0.5), 0, 1)
    w_teacher = 0.15 + 0.85 * (0.65 * agreement + 0.35 * certainty)
    out_w[filled] = w_teacher[filled]
    out = cons.copy()
    out[LABELS] = out_y
    out[[f"w_{l}" for l in LABELS]] = out_w
    out.reset_index().to_csv(args.out, index=False, float_format="%.6f", lineterminator="\n")

    pos_before = (y[tr] >= 0.5).mean(axis=0)
    pos_after = (out_y[tr] >= 0.5).mean(axis=0)
    moved = np.abs(out_y[tr] - y[tr])
    summary = pd.DataFrame({"fill": ["yes" if f else "" for f in fill_mask],
                            "silent_frac": info["silent_frac_train"].to_numpy(),
                            "cells_filled": filled.sum(axis=0),
                            "pos_rate_before": pos_before, "pos_rate_after": pos_after,
                            "mean_abs_change": moved.mean(axis=0)}, index=LABELS)

    lines = [
        "# Pseudo labels v1 — multi-teacher blend of the report consensus and our image models",
        "",
        f"Generated by `src/pseudo_labels.py` from `consensus_v1.csv` and {len(args.run)} image teachers "
        f"({', '.join(os.path.basename(r.rstrip('/\\')) for r in args.run)}), MIX_W = {MIX_W}, "
        f"SILENT_RATIO_MIN = {SILENT_RATIO_MIN}. Train rows with a teacher: {n_train}; gold-58 rows kept as consensus.",
        "",
        "## Is report silence informative? (gold-58 positive rate when the readers did not address the finding)",
        "",
        md_table(info.rename(columns={"silent_frac_train": "silent frac (train)", "n_silent_gold": "n silent (gold)",
                                      "gold_pos_silent": "gold+ given silent", "gold_pos_addressed": "gold+ given addressed",
                                      "ratio": "ratio", "teacher_pos_silent": "teacher+ given silent (train)",
                                      "teacher_pos_addressed": "teacher+ given addressed (train)"})),
        "",
        f"FILL (ratio >= {SILENT_RATIO_MIN}): {', '.join(fill) if fill else 'none'}. "
        "For every other label an unreported finding stays at the consensus value (low), because silence there means absent.",
        "",
        "## Label-set quality on gold-58 (AUC of the label values vs the radiologists; teacher = gold_windows ensemble)",
        "",
        md_table(tab),
        "",
        "`fill_all` fills silence for every label, `fill_FILL` only for the FILL labels, `pseudo_v1` = mix50 on addressed cells + fill on FILL labels (the shipped rule).",
        "",
        "## What changed in the training file (train rows)",
        "",
        md_table(summary, nd=3),
        "",
        "Weights: addressed cells keep the consensus weight; filled cells get 0.15 + 0.85 * (0.65 * teacher agreement + 0.35 * teacher certainty).",
        "",
        "Decision rule for shipping: a model trained on this file must beat the same arm trained on consensus_v1 on the public LB (fold 0, fast preset: 0.898); gold-58 only justifies the rule, it does not select it.",
        "",
    ]
    with open(args.report, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {args.out} and {args.report}")
    print(tab.round(4).to_string())
    print("FILL:", fill)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
