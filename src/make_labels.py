"""Consensus label builder: blend three independent LLM readers into one
pseudo-label set with per-cell weights.

Readers (all bare `<Finding>` columns, probabilities in [0, 1]):
    labels-pilkwang/report_labels_v2.csv
    labels-stevenleehans/llm_labels_v4_blend.csv
    labels-lixin73/labels_llm_gpt56sol.csv

Aligned on the 4407 study ids of train.csv (left join -> NaN where a reader
is missing a study). Per study/finding, over the stack of the three reader
values (cube, shape (3, n_studies, 12)):

    y            = nanmean(cube, axis=0)
    disagreement = nanmean(|cube - y|, axis=0)
    agreement    = clip(1 - 2*disagreement, 0, 1)
    certainty    = clip(2*|y - 0.5|, 0, 1)
    w            = 0.15 + 0.85 * (0.65*agreement + 0.35*certainty)

Writes data/labels-consensus/consensus_v1.csv: StudyInstanceUID, the 12
LABELS columns (y), and 12 w_<label> weight columns. The 58 gold studies are
not treated specially -- they get the same consensus formula as everyone
else, gold-58 is a pure holdout elsewhere in this project.

Run: py src/make_labels.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from preprocess import LABELS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

# stevenleehans contributes v4_blend, not v2: measured against gold-58, the
# three-reader blend scores 0.8948 with v4_blend and 0.8919 with v2, against
# 0.8927 for v4_blend alone. Every 2-, 3- and 4-reader combination was ranked
# before this choice; see the round-4 plan Log.
READERS = [
    ("pilkwang", os.path.join(DATA, "labels-pilkwang", "report_labels_v2.csv")),
    ("stevenleehans", os.path.join(DATA, "labels-stevenleehans", "llm_labels_v4_blend.csv")),
    ("lixin73", os.path.join(DATA, "labels-lixin73", "labels_llm_gpt56sol.csv")),
]


def load_reader(name: str, path: str, study_ids: pd.Index) -> pd.DataFrame:
    df = pd.read_csv(path).set_index("StudyInstanceUID")
    if df.index.duplicated().any():
        raise ValueError(f"{name}: duplicate study ids in {path}")
    return df.reindex(study_ids)[LABELS]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", action="append", default=[], metavar="NAME=CSV",
                    help="override the reader set (repeatable); default = the three public readers")
    ap.add_argument("--out", default="consensus_v1.csv",
                    help="file name under data/labels-consensus/ (default consensus_v1.csv)")
    args = ap.parse_args()
    readers = [tuple(r.split("=", 1)) for r in args.reader] or READERS

    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    study_ids = pd.Index(train["StudyInstanceUID"], name="StudyInstanceUID")

    cube = np.stack(
        [load_reader(name, path, study_ids).to_numpy(dtype=float) for name, path in readers],
        axis=0,
    )  # (n_readers, n_studies, 12)
    print("readers:", ", ".join(n for n, _ in readers))

    n_finite = np.isfinite(cube).sum(axis=0)
    if (n_finite < 2).any():
        n_bad = int((n_finite < 2).sum())
        raise ValueError(
            f"{n_bad} study/finding cells have fewer than two finite reader values"
        )

    y = np.nanmean(cube, axis=0)
    disagreement = np.nanmean(np.abs(cube - y), axis=0)
    agreement = np.clip(1 - 2 * disagreement, 0, 1)
    certainty = np.clip(2 * np.abs(y - 0.5), 0, 1)
    w = 0.15 + 0.85 * (0.65 * agreement + 0.35 * certainty)

    out = pd.DataFrame(y, index=study_ids, columns=LABELS)
    for i, l in enumerate(LABELS):
        out[f"w_{l}"] = w[:, i]
    out = out.reset_index()

    out_dir = os.path.join(DATA, "labels-consensus")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.out)
    out.to_csv(out_path, index=False, float_format="%.6f")

    print(f"Wrote {len(out)} rows to {out_path}")
    print("Mean weight per finding:")
    for l in LABELS:
        print(f"  {l}: {out[f'w_{l}'].mean():.4f}")


if __name__ == "__main__":
    main()
