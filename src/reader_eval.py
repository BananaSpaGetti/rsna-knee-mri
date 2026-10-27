"""Score every report reader (public and ours) against the 58 radiologist-labelled gold
studies: per-finding AUC of the reader's score (YES 1 / NO 0 / UNK 0.5, or the reader's own
probability), UNK rate, and coverage of the gold rows. Information only - gold-58 is a
holdout; nothing is selected here.

Usage: py src/reader_eval.py [--reader NAME=path.csv ...] [--out docs/reader-eval.md]
The public readers and consensus_v1 are always included; `--reader` adds ours
(data/labels-ours/<name>.csv from llm_reader.py). A reader missing some gold rows is
scored on the rows it has (n column).
"""

import argparse
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

PUBLIC = {
    "pilkwang_v2": os.path.join(DATA, "labels-pilkwang", "report_labels_v2.csv"),
    "steven_full": os.path.join(DATA, "labels-stevenleehans", "llm_labels_full.csv"),
    "steven_v4blend": os.path.join(DATA, "labels-stevenleehans", "llm_labels_v4_blend.csv"),
    "lixin_gpt56": os.path.join(DATA, "labels-lixin73", "labels_llm_gpt56sol.csv"),
    "consensus_v1": os.path.join(DATA, "labels-consensus", "consensus_v1.csv"),
}


def load_scores(path):
    df = pd.read_csv(path)
    df = df.set_index("StudyInstanceUID")
    cols = {}
    for k in LABELS:
        if k in df.columns:
            cols[k] = pd.to_numeric(df[k], errors="coerce")
    return pd.DataFrame(cols)


def unk_rate(path, scores):
    df = pd.read_csv(path).set_index("StudyInstanceUID")
    vcols = [f"{k}__verdict" for k in LABELS if f"{k}__verdict" in df.columns]
    if vcols:
        return float((df[vcols] == "UNK").to_numpy().mean())
    return float((scores.to_numpy() == 0.5).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", action="append", default=[], metavar="NAME=CSV")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gold = pd.read_csv(os.path.join(DATA, "gold58.csv")).set_index("StudyInstanceUID")[LABELS]
    readers = dict(PUBLIC)
    for spec in args.reader:
        name, path = spec.split("=", 1)
        readers[name] = path

    rows = []
    per_label = {}
    for name, path in readers.items():
        if not os.path.exists(path):
            print(f"skip {name}: {path} missing")
            continue
        s = load_scores(path)
        common = gold.index.intersection(s.index)
        aucs = {}
        for k in LABELS:
            if k not in s.columns:
                aucs[k] = np.nan
                continue
            g = gold.loc[common, k].to_numpy()
            p = s.loc[common, k].to_numpy(dtype=float)
            m = np.isfinite(p)
            if m.sum() < 5 or len(set(g[m])) < 2:
                aucs[k] = np.nan
            else:
                aucs[k] = roc_auc_score(g[m], p[m])
        per_label[name] = aucs
        vals = [v for v in aucs.values() if np.isfinite(v)]
        rows.append({"reader": name, "n_gold": len(common), "unk_rate": unk_rate(path, s),
                     "macro_auc": float(np.mean(vals)) if vals else np.nan,
                     "n_labels": len(vals)})

    summary = pd.DataFrame(rows).sort_values("macro_auc", ascending=False)
    table = pd.DataFrame(per_label).T[LABELS]
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(summary.to_string(index=False))
        print()
        print(table.to_string())
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("# Report readers vs gold-58\n\n")
            fh.write("Per-finding AUC of each reader's score against the 58 radiologist labels "
                     "(YES 1 / NO 0 / UNK 0.5 or the reader's probability). Information only.\n\n")
            fh.write(summary.to_markdown(index=False, floatfmt=".3f") + "\n\n")
            fh.write(table.to_markdown(floatfmt=".3f") + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
