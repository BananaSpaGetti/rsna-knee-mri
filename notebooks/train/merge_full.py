"""Merge N full-data runs (coatnet2_rsfull, one train seed each) into one weights dir that infer_submit loads like a
5-fold arm: run k's coatnet2_fold0.pt becomes coatnet2_fold{k}.pt (hard link), and run_config.json is run 0's with
n_folds = N, the seeds listed and use_tta off. Every run must share the label file, cache config and recipe; only train_seed and
the timing fields may differ.

  py notebooks/train/merge_full.py <out dir> <run dir 0> <run dir 1> ...
"""
import json
import os
import sys

# fields allowed to differ between the runs (train_seed by design, the rest are timing / gold readouts)
PER_RUN = {"train_seed", "epochs_done", "sec_per_epoch", "stopped_early", "gold58_auc", "tta_gold58_auc", "use_tta",
           "max_hours", "stage", "fold_auc_last", "fold_auc_swa"}


def merge(out_dir, run_dirs):
    cfgs = [json.load(open(os.path.join(d, "run_config.json"))) for d in run_dirs]
    base = cfgs[0]
    assert base["trained_on"] == "all" and base["n_folds"] == 1, (base["trained_on"], base["n_folds"])
    for d, c in zip(run_dirs, cfgs):
        diff = sorted(k for k in set(base) | set(c) if k not in PER_RUN and base.get(k) != c.get(k))
        assert not diff, (d, diff)
    seeds = [c["train_seed"] for c in cfgs]
    assert len(set(seeds)) == len(seeds), seeds
    os.makedirs(out_dir, exist_ok=True)
    family = base["family"]
    for k, d in enumerate(run_dirs):
        src, dst = os.path.join(d, f"{family}_fold0.pt"), os.path.join(out_dir, f"{family}_fold{k}.pt")
        assert os.path.isfile(src), src
        if not os.path.exists(dst):
            os.link(src, dst)
    # use_tta pinned off: infer_submit would otherwise turn hflip TTA on from run 0's own gold-58 pair, and every
    # 5-fold arm it is compared with (rs / t4 / t5 / t11 / t112) runs without TTA
    merged = dict(base, n_folds=len(run_dirs), train_seed=seeds, use_tta=False,
                  gold58_auc_per_seed=[c.get("gold58_auc") for c in cfgs],
                  sec_per_epoch=[c.get("sec_per_epoch") for c in cfgs],
                  merged_from=[os.path.basename(os.path.normpath(d)) for d in run_dirs])
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(merged, fh)
    return merged


if __name__ == "__main__":
    m = merge(sys.argv[1], sys.argv[2:])
    print(f"merged {m['n_folds']} runs, seeds {m['train_seed']}, label {m['label_file']}, "
          f"gold-58 per seed {m['gold58_auc_per_seed']}")
