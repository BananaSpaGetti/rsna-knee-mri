"""Independent recompute of O1's submission.csv from the downloaded per-arm outputs (visible-run gate).

  submission = pct rank per finding of the mean over arms of each arm's pct rank per finding

with pandas' average-tie percentile rank (rank / n), not src/blend_eval.py's numpy implementation. The mean is taken
over the integer (half-integer on ties) ranks, so it is exact and an exact tie between studies stays a tie: summing
float pct ranks splits it by one ulp in some summation orders (O6 gate, 2026-10-22: Medial OA ranks 2,1,3 / 2,1,3 /
3,2,1 tie at 7/3 and float sums gave 0.667 / 1.0 instead of 0.833 / 0.833). The kernel's own float mean
(blend_eval.rank_blend) can split such a tie either way too (O8a gate: Fracture tied at 3/2, kernel 0.333 / 0.667),
so a study inside an exact tie group is checked by order only: every pair of studies whose exact means differ must
keep that order in submission.csv, and every study outside a tie group must match exactly. Also checks that
submission.csv equals ours_leg.csv and that fusion_config.json ran only our arms. Prints the max abs difference
outside exact ties, the tie groups split by the kernel, and exits 1 on any order break, a diff above 1e-6 or any
other mismatch.

  py notebooks/o1/recompute_o1.py <out dir> [rs,t4,t5]
"""
import json
import os
import sys

import pandas as pd

if __name__ == "__main__":
    out = sys.argv[1]
    arms = (sys.argv[2] if len(sys.argv) > 2 else "rs,t4,t5").split(",")
    sub = pd.read_csv(os.path.join(out, "submission.csv"), dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    leg = pd.read_csv(os.path.join(out, "ours_leg.csv"), dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    cfg = json.load(open(os.path.join(out, "fusion_config.json")))
    ranks = []
    for a in arms:
        p = pd.read_csv(os.path.join(out, f"probs_{a}.csv"), dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
        assert list(p.index) == list(sub.index) and list(p.columns) == list(sub.columns), a
        ranks.append(p.rank(method="average"))
    total = sum(ranks)   # sums of half-integers: exact in float
    want = total.rank(method="average", pct=True)
    diff, order_breaks, split = 0.0, 0, 0
    for c in sub.columns:
        t, k, w = total[c].to_numpy(), sub[c].to_numpy(), want[c].to_numpy()
        tied = pd.Series(t).duplicated(keep=False).to_numpy()
        if (~tied).any():
            diff = max(diff, float(abs(w[~tied] - k[~tied]).max()))
        order_breaks += int(sum((t[i] < t[j]) != (k[i] < k[j]) for i in range(len(t)) for j in range(len(t))
                                if t[i] != t[j]))
        split += int(sum(len(set(k[t == v])) > 1 for v in set(t[tied])))
    leg_diff = float((leg - sub).abs().to_numpy().max())
    # arms_run is fusion_submit's _required_our_arm_ids order (sorted), so compare as sets of names
    ok_cfg = (cfg.get("ours_only") is True and cfg.get("ours_submit") is True
              and sorted(cfg.get("arms_run") or []) == sorted(arms))
    print(f"recompute max abs diff {diff:.3g} outside exact ties; order breaks {order_breaks}; exact-tie groups "
          f"split by the kernel {split}; submission vs ours_leg {leg_diff:.3g}; "
          f"arms_run {cfg.get('arms_run')} ours_only {cfg.get('ours_only')} ours_submit {cfg.get('ours_submit')}")
    sys.exit(0 if diff <= 1e-6 and order_breaks == 0 and leg_diff <= 1e-6 and ok_cfg else 1)
