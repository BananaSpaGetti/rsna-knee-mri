"""Binary ROC AUC without scikit-learn.

Drop-in for sklearn.metrics.roc_auc_score on binary targets (the only way
this repo calls it): the Mann-Whitney U form with average ranks for ties,
which is exactly what sklearn's trapezoidal ROC integral evaluates to for a
binary problem. Exists because Windows Smart App Control (2026-09-30, this
desktop) started blocking sklearn's unsigned compiled extension
`sklearn.metrics.cluster._expected_mutual_info_fast`, which every
`from sklearn.metrics import roc_auc_score` pulls in; every caller now falls
back to this module when that import fails.

    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:  # blocked or not installed
        from auc import roc_auc_score
"""

import numpy as np
from scipy.stats import rankdata


def roc_auc_score(y_true, y_score):
    """AUC of `y_score` for binary `y_true` (0/1 or bool). Raises ValueError
    when only one class is present, like sklearn does."""
    t = np.asarray(y_true).astype(bool).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    if t.shape != s.shape:
        raise ValueError(f"shape mismatch {t.shape} vs {s.shape}")
    n_pos = int(t.sum())
    n_neg = int(t.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            "Only one class present in y_true. ROC AUC score is not defined "
            "in that case.")
    ranks = rankdata(s, method="average")
    u = ranks[t].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))
