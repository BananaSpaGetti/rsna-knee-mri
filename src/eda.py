"""EDA for RSNA Knee: gold subset, pseudo-label sets vs gold macro AUC, series
composition, and a crude report-language sniff.

Run: py src/eda.py
Writes docs/eda.md and prints "OK" plus the macro AUC comparison table.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
try:
    from sklearn.metrics import roc_auc_score
except ImportError:  # sklearn's compiled extension blocked (Smart App Control) or not installed
    from auc import roc_auc_score

from preprocess import LABELS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
DOCS = os.path.join(ROOT, "docs")

# (name, path, column prefix in the file, excluded-from-ranking reason or None)
LABEL_SETS = [
    ("stevenleehans_v2",
     os.path.join(DATA, "labels-stevenleehans", "llm_labels_v2.csv"), "", None),
    ("stevenleehans_v4_blend",
     os.path.join(DATA, "labels-stevenleehans", "llm_labels_v4_blend.csv"), "", None),
    ("pilkwang_v2",
     os.path.join(DATA, "labels-pilkwang", "report_labels_v2.csv"), "", None),
    ("barun2104_train_folds",
     os.path.join(DATA, "labels-barun2104", "train_folds.csv"), "",
     "gold copy: label columns are the 58 gold rows verbatim (0 diffs checked), "
     "not a pseudo-label set -- coverage is 58/4407 and any AUC vs gold is trivially 1.000"),
    ("barun2104_train_folds_with_pseudo",
     os.path.join(DATA, "labels-barun2104", "train_folds_with_pseudo.csv"), "pseudo_",
     "leaked: pseudo_* values on the 58 gold studies are identical to the gold labels "
     "(verified, 0 diffs) -- an is_manual flag confirms manual labels were written back "
     "into the pseudo columns, so AUC vs gold is not a fair estimate"),
    ("consensus_v1",
     os.path.join(DATA, "labels-consensus", "consensus_v1.csv"), "", None),
]


def load_gold(train: pd.DataFrame) -> pd.DataFrame:
    labeled = train.set_index("StudyInstanceUID")[LABELS]
    return labeled[labeled.notna().any(axis=1)]


def load_label_set(path: str, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path).set_index("StudyInstanceUID")
    cols = {prefix + l: l for l in LABELS}
    return df[list(cols)].rename(columns=cols)


def macro_auc(pred: pd.DataFrame, gold: pd.DataFrame) -> tuple[dict, list, list]:
    """Per-finding AUC on the gold studies present in `pred`. Returns
    (auc_by_label, scored_labels, skipped_labels)."""
    common = gold.index.intersection(pred.index)
    auc_by_label = {}
    scored, skipped = [], []
    for l in LABELS:
        y_true = gold.loc[common, l]
        y_score = pred.loc[common, l]
        mask = y_true.notna() & y_score.notna()
        y_true, y_score = y_true[mask], y_score[mask]
        if y_true.nunique() < 2:
            skipped.append(l)
            continue
        auc_by_label[l] = roc_auc_score(y_true, y_score)
        scored.append(l)
    return auc_by_label, scored, skipped


def fmt_table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def report_lang_stats(reports: pd.Series) -> tuple[int, float, int, float]:
    """Crude heuristic: fraction of non-ASCII characters per report.
    Returns (n_reports, mean_nonascii_frac, n_over_20pct, pct_over_20pct)."""
    fracs = []
    for text in reports:
        if not isinstance(text, str) or len(text) == 0:
            continue
        nonascii = sum(1 for c in text if ord(c) > 127)
        fracs.append(nonascii / len(text))
    fracs = np.asarray(fracs)
    n = len(fracs)
    over20 = int((fracs > 0.20).sum())
    return n, float(fracs.mean()) if n else 0.0, over20, (100.0 * over20 / n if n else 0.0)


def main() -> None:
    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    train_series = pd.read_csv(os.path.join(DATA, "train_series.csv"))
    all_ids = train["StudyInstanceUID"]
    n_studies = len(all_ids)

    # --- 1. gold subset ---
    gold = load_gold(train)
    n_gold = len(gold)
    gold_pos_counts = gold.sum().astype(int)

    # --- 2. label sets vs gold ---
    # each set is loaded and scored exactly once; everything below reads `sets`
    sets = {}
    ranking_rows = []
    excluded_rows = []
    for name, path, prefix, excluded_reason in LABEL_SETS:
        pred = load_label_set(path, prefix)
        # coverage = studies with at least one non-null label value (matches the
        # gold subset's own definition), not just a row present in the file
        coverage = int(pred.reindex(all_ids)[LABELS].notna().any(axis=1).sum())
        auc_by_label, scored, skipped = macro_auc(pred, gold)
        macro = float(np.mean(list(auc_by_label.values()))) if auc_by_label else float("nan")
        sets[name] = {"pred": pred, "auc": auc_by_label, "skipped": skipped,
                      "macro": macro, "eligible": excluded_reason is None}
        row = [name, f"{coverage}/{n_studies}", len(scored), f"{macro:.4f}"]
        if excluded_reason is None:
            ranking_rows.append(row + [""])
        else:
            excluded_rows.append(row + [excluded_reason])

    eligible_names = [n for n, s in sets.items() if s["eligible"]]
    # rank on the unrounded macro; NaN never wins
    best_name = max((n for n in eligible_names if not np.isnan(sets[n]["macro"])),
                    key=lambda n: sets[n]["macro"])
    winner = next(r for r in ranking_rows if r[0] == best_name)

    finding_auc_rows = [
        [l, int(gold_pos_counts[l])] +
        [f"{sets[n]['auc'].get(l, float('nan')):.4f}" for n in eligible_names]
        for l in LABELS
    ]

    skip_sets = {n: sets[n]["skipped"] for n in eligible_names}
    skip_consistent = len(set(tuple(sorted(v)) for v in skip_sets.values())) <= 1

    # --- 3. prevalence for the best label set ---
    prevalence = sets[best_name]["pred"][LABELS].mean()

    # --- 4. series composition ---
    per_study = train_series.groupby("StudyInstanceUID").size()
    series_min, series_med, series_max = int(per_study.min()), float(per_study.median()), int(per_study.max())
    plane_counts = train_series["Anatomical_Plane"].value_counts(dropna=False)
    fluid_counts = train_series["Fluid_Sensitive"].value_counts(dropna=False)
    fatsup_counts = train_series["Fat_Suppression"].value_counts(dropna=False)
    sag_fluid_studies = train_series[
        (train_series["Anatomical_Plane"] == "Sagittal") & (train_series["Fluid_Sensitive"] == 1)
    ]["StudyInstanceUID"].nunique()

    # --- 5. report language sniff ---
    n_rep, mean_frac, n_over20, pct_over20 = report_lang_stats(train["Report"])

    # --- write docs/eda.md ---
    os.makedirs(DOCS, exist_ok=True)
    lines = []
    lines.append("# RSNA Knee -- EDA\n")
    lines.append(f"Generated by `src/eda.py`. Full train set: {n_studies} studies.\n")

    lines.append("## 1. Gold subset\n")
    lines.append(f"Studies with non-null expert labels: **{n_gold}** / {n_studies}.\n")
    lines.append("Per-finding positive counts within gold:\n")
    lines.append(fmt_table(
        [[l, int(gold_pos_counts[l]), n_gold] for l in LABELS],
        ["Finding", "Positives", "Gold N"]))
    lines.append("")

    lines.append("## 2. Label sets vs gold (macro AUC)\n")
    lines.append(
        "Coverage = studies with at least one non-null label value in the file, out of "
        "the full train set. "
        "`n_scored` = findings with both classes present in the gold rows the file covers "
        "(all 12 findings have both classes in the full 58-row gold, so any n_scored < 12 "
        "below is caused by a set not covering all 58 gold studies).\n"
    )
    lines.append(fmt_table(
        [r[:4] for r in ranking_rows],
        ["Label set", "Coverage", "n_scored", "Macro AUC"]))
    lines.append("")
    lines.append(f"**Winner: {winner[0]}, macro AUC {winner[3]}** (among sets eligible for ranking).\n")
    lines.append(
        f"Skip-set (findings with a single class in the gold rows covered) identical across "
        f"eligible label sets: {skip_consistent}."
        + ("" if skip_consistent else " Differs -- see per-set skip lists below.") + "\n"
    )
    if not skip_consistent:
        for n, s in skip_sets.items():
            lines.append(f"- {n}: skipped {s if s else 'none'}")
        lines.append("")

    lines.append("### Excluded from ranking\n")
    lines.append(
        "These two rows share provenance with the barun2104 gold subset and would either "
        "trivially score 1.000 (exact copy of gold) or be a leaked AUC estimate (gold labels "
        "written back into the pseudo columns). Both are reported for transparency but are not "
        "eligible to be picked as the winner.\n"
    )
    lines.append(fmt_table(
        [[r[0], r[1], r[2], r[3], r[4]] for r in excluded_rows],
        ["Label set", "Coverage", "n_scored", "Macro AUC", "Reason excluded"]))
    lines.append("")

    lines.append("### Per-finding AUC, eligible label sets\n")
    lines.append(fmt_table(
        finding_auc_rows,
        ["Finding", "Gold positives"] + eligible_names))
    lines.append("")

    lines.append("## 3. Pseudo-label prevalence (best label set)\n")
    lines.append(f"Best label set: **{best_name}**. Mean of each column over all {n_studies} studies:\n")
    lines.append(fmt_table(
        [[l, f"{prevalence[l]:.4f}"] for l in LABELS],
        ["Finding", "Mean (prevalence proxy)"]))
    lines.append("")

    lines.append("## 4. Series composition\n")
    lines.append(f"Series per study: min {series_min}, median {series_med}, max {series_max}.\n")
    lines.append("Anatomical_Plane distribution:\n")
    lines.append(fmt_table(
        [[str(k), int(v)] for k, v in plane_counts.items()],
        ["Plane", "Count"]))
    lines.append("")
    lines.append("Fluid_Sensitive distribution:\n")
    lines.append(fmt_table(
        [[str(k), int(v)] for k, v in fluid_counts.items()],
        ["Fluid_Sensitive", "Count"]))
    lines.append("")
    lines.append("Fat_Suppression distribution:\n")
    lines.append(fmt_table(
        [[str(k), int(v)] for k, v in fatsup_counts.items()],
        ["Fat_Suppression", "Count"]))
    lines.append("")
    lines.append(
        f"Studies with at least one Sagittal fluid-sensitive series: "
        f"**{sag_fluid_studies}** / {train_series['StudyInstanceUID'].nunique()}.\n"
    )

    lines.append("## 5. Report language sniff (heuristic)\n")
    lines.append(
        "Crude heuristic: fraction of characters in each report with `ord(c) > 127` "
        "(non-ASCII, catches Thai/CJK/Cyrillic/etc. but not diacritics on Latin letters). "
        "Not a real language identifier -- a rough breakdown only.\n"
    )
    lines.append(fmt_table(
        [
            ["Reports with Report text", n_rep],
            ["Mean non-ASCII character fraction", f"{mean_frac:.4f}"],
            ["Reports with >20% non-ASCII chars", f"{n_over20} ({pct_over20:.1f}%)"],
        ],
        ["Metric", "Value"]))
    lines.append("")

    with open(os.path.join(DOCS, "eda.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("OK")
    print(fmt_table([r[:4] for r in ranking_rows], ["Label set", "Coverage", "n_scored", "Macro AUC"]))
    print(f"Winner: {winner[0]} macro AUC {winner[3]}")


if __name__ == "__main__":
    main()
