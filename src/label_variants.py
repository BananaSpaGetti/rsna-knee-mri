"""Label-file variants for controlled relabelling experiments.

Each variant reads data/labels-consensus/consensus_v1.csv and writes a sibling
file that differs in exactly one documented way, so a training run with
LABEL_FILE flipped to the variant is a one-factor experiment against the
same arm, cache, fold split and seed.

Variants:
  frac   Fracture = max(Fracture, FRAC_FACTOR * Contusion), FRAC_FACTOR = 0.7. The label audit
         (docs/label-audit.md, 2026-09-15) found the readers call Fracture on
         7 % of studies while the radiologist holdout has 31 %, and that the
         missed cases are reported as bone-marrow oedema / contusion; the
         teacher's Contusion column predicts radiologist Fracture better
         (0.86) than the Fracture column itself (0.82). Where the max comes
         from Contusion the weight cell becomes max(w_Fracture, w_Contusion).
         0.7 moves the Fracture positive rate (>= 0.5) from 7.1 % to 18.8 %; 0.5
         would leave it at 7.1 % (only soft targets change), 1.0 gives 20.6 %.

  uteacher  On "unaddressed" cells (pilkwang verdict UNK AND stevenleehans
         llm_labels_full == 0.5, 23 % of all cells; consensus there is y ~ 0.18,
         w ~ 0.76 - report silence read as a soft negative) the target becomes
         the image teacher's out-of-fold probability (--teacher <oof .npy> in
         the row order of --fold-of <fold_of.json>); weights unchanged; cells
         of studies absent from fold_of (gold-58) untouched. Tests whether the
         radiologist test labels are better approximated by the images than by
         report silence on the findings reports skip (Synovitis 84 %, Fracture
         56 %, Baker's 46 % of studies).
  teach4   Effusion, Synovitis, Contusion, Medial OA: every cell <- image-teacher
         OOF with w = 1 (image models beat report labels on those four on gold-58,
         TranBaDat2607 ledger); other findings unchanged.
  teach    Generalised teach4: --labels "A,B,C" picks the findings, --name the
         output suffix (consensus_v1_<name>.csv). Used for t6 (teach4 minus
         Contusion plus Lateral OA / Baker's / Fracture - the findings where the
         rs+rs2+t4 ensemble beats consensus_v1 on gold-58 by > 0.02, 2026-10-01)
         and t4b (teach4 columns, teacher = mean of rs / rs2 / t4 dense OOF).
  uignore  Same cells, w = 0 (CheXpert "U-Ignore"): the model learns those
         findings from addressed reports only. Brackets uteacher from the
         other side; the current consensus is close to "U-Zeros".

Usage: py src/label_variants.py frac  -> data/labels-consensus/consensus_v1_frac.csv
       py src/label_variants.py uteacher --teacher C:/kaggle/staging/weights-rs/oof_stride1_eval.npy            --fold-of C:/kaggle/staging/weights-rs/fold_of.json
       py src/label_variants.py uignore
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import LABELS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "labels-consensus", "consensus_v1.csv")
FRAC_FACTOR = 0.7


def frac(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cand = FRAC_FACTOR * df["Contusion"].to_numpy(float)
    f = df["Fracture"].to_numpy(float)
    take = cand > f
    out["Fracture"] = np.where(take, cand, f)
    out["w_Fracture"] = np.where(take, np.maximum(df["w_Fracture"], df["w_Contusion"]),
                                 df["w_Fracture"])
    n = int(take.sum())
    print(f"frac: Fracture raised on {n} of {len(df)} studies "
          f"({n / len(df):.1%}); positive rate {np.mean(f >= 0.5):.3f} -> "
          f"{np.mean(out['Fracture'].to_numpy(float) >= 0.5):.3f}")
    return out


def unaddressed_mask(df: pd.DataFrame) -> np.ndarray:
    """(n, 12) bool: pilkwang verdict UNK and stevenleehans llm_labels_full
    == 0.5 for that study/finding (both readers say the report is silent)."""
    ids = pd.Index(df["StudyInstanceUID"])
    pk = pd.read_csv(os.path.join(ROOT, "data", "labels-pilkwang", "report_labels_v2.csv")
                     ).set_index("StudyInstanceUID").reindex(ids)
    sf = pd.read_csv(os.path.join(ROOT, "data", "labels-stevenleehans", "llm_labels_full.csv")
                     ).set_index("StudyInstanceUID").reindex(ids)
    unk = np.stack([(pk[f"{l}__verdict"] == "UNK").to_numpy() for l in LABELS], axis=1)
    s5 = np.stack([(sf[l] == 0.5).to_numpy() for l in LABELS], axis=1)
    return unk & s5


def uteacher(df: pd.DataFrame, teacher: str, fold_of: str) -> pd.DataFrame:
    oof = np.load(teacher)
    order = list(pd.read_json(fold_of, typ="series").index)
    assert oof.shape == (len(order), len(LABELS)), (oof.shape, len(order))
    row_of = {u: i for i, u in enumerate(order)}
    out = df.copy()
    mask = unaddressed_mask(df)
    idx = np.array([row_of.get(u, -1) for u in df["StudyInstanceUID"]])
    have = idx >= 0
    mask &= have[:, None]
    n_cells = 0
    for c, l in enumerate(LABELS):
        y = out[l].to_numpy(float)
        t = np.full(len(df), np.nan)
        t[have] = oof[idx[have], c]
        y = np.where(mask[:, c], t, y)
        out[l] = y
        n_cells += int(mask[:, c].sum())
        print(f"uteacher {l:17s}: {mask[:, c].sum():5d} cells ({mask[:, c].mean():.0%}) "
              f"consensus mean {df[l].to_numpy(float)[mask[:, c]].mean():.3f} -> "
              f"teacher mean {t[mask[:, c]].mean():.3f}")
    print(f"uteacher: {n_cells} of {mask.size} cells replaced ({n_cells / mask.size:.1%}); "
          f"{int((~have).sum())} studies without teacher rows left untouched")
    return out


def uignore(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mask = unaddressed_mask(df)
    for c, l in enumerate(LABELS):
        w = out[f"w_{l}"].to_numpy(float)
        out[f"w_{l}"] = np.where(mask[:, c], 0.0, w)
    print(f"uignore: {int(mask.sum())} of {mask.size} cells ({mask.mean():.1%}) get weight 0; "
          f"per label {dict(zip(LABELS, np.round(mask.mean(0), 2)))}")
    return out


TEACH4 = ("Effusion", "Synovitis", "Contusion", "Medial OA")


def teach(df: pd.DataFrame, teacher: str, fold_of: str, labels=TEACH4,
          tag: str = "teach4") -> pd.DataFrame:
    """Full teacher replacement (y = image-teacher OOF, w = 1) on EVERY cell of
    the four findings where image models beat every report-label table on the
    radiologist-labelled gold-58 (TranBaDat2607 ledger A02: Effusion 0.95 vs
    0.83, Synovitis 0.80 vs 0.69, Contusion 0.92 vs 0.82, Medial OA 0.98 vs
    0.94 - "gold labels are image-based; report labels carry systematic
    noise on those findings"). The other eight findings keep consensus_v1.
    Differs from pseudo_v1 (0.5/0.5 mix on all twelve) in both scope and
    strength. Studies absent from fold_of (gold-58) untouched. `labels` /
    `tag` generalise it to any finding subset (variant "teach")."""
    oof = np.load(teacher)
    order = list(pd.read_json(fold_of, typ="series").index)
    assert oof.shape == (len(order), len(LABELS)), (oof.shape, len(order))
    row_of = {u: i for i, u in enumerate(order)}
    idx = np.array([row_of.get(u, -1) for u in df["StudyInstanceUID"]])
    have = idx >= 0
    out = df.copy()
    for l in labels:
        c = LABELS.index(l)
        y = out[l].to_numpy(float).copy()
        y[have] = oof[idx[have], c]
        out[l] = y
        w = out[f"w_{l}"].to_numpy(float).copy()
        w[have] = 1.0
        out[f"w_{l}"] = w
        print(f"{tag} {l:12s}: consensus mean {df[l].to_numpy(float)[have].mean():.3f} -> "
              f"teacher mean {y[have].mean():.3f}; pos-rate (>=0.5) "
              f"{np.mean(df[l].to_numpy(float)[have] >= 0.5):.3f} -> {np.mean(y[have] >= 0.5):.3f}")
    return out


def teach4(df: pd.DataFrame, teacher: str, fold_of: str) -> pd.DataFrame:
    return teach(df, teacher, fold_of, TEACH4, "teach4")


VARIANTS = {"frac": frac, "uteacher": uteacher, "uignore": uignore, "teach4": teach4,
            "teach": teach}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=sorted(VARIANTS))
    ap.add_argument("--teacher", help="uteacher: OOF probabilities .npy (n_studies, 12)")
    ap.add_argument("--fold-of", help="uteacher: fold_of.json whose key order is the .npy row order")
    ap.add_argument("--labels", help="teach: comma-separated findings to replace")
    ap.add_argument("--name", help="teach: output suffix -> consensus_v1_<name>.csv")
    args = ap.parse_args()
    name = args.variant
    df = pd.read_csv(SRC)
    if name == "teach":
        if not (args.teacher and args.fold_of and args.labels and args.name):
            raise SystemExit("teach needs --teacher, --fold-of, --labels and --name")
        labels = tuple(s.strip() for s in args.labels.split(","))
        bad = [l for l in labels if l not in LABELS]
        assert not bad, f"unknown findings {bad}"
        out = teach(df, args.teacher, args.fold_of, labels, args.name)
        name = args.name
    elif name in ("uteacher", "teach4"):
        if not (args.teacher and args.fold_of):
            raise SystemExit(f"{name} needs --teacher and --fold-of")
        out = VARIANTS[name](df, args.teacher, args.fold_of)
    else:
        out = VARIANTS[name](df)
    dst = SRC.replace("consensus_v1.csv", f"consensus_v1_{name}.csv")
    out.to_csv(dst, index=False, float_format="%.6f", lineterminator="\n")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
