# RSNA Knee: offline inference -> submission.csv. THE submission kernel: internet OFF, GPU on, <= 9 h.
# Attach: competition data, rsna-67knee-src, training fold-weights -- see docs/code-notes.md "inference pipeline overview".

import csv
import json
import os
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import numpy as np
import torch

_SRC_CANDIDATES = ("/kaggle/input/rsna-67knee-src",
                   "/kaggle/input/datasets/patimakornsrinamen/rsna-67knee-src",
                   # pre-rename slug, kept so the kernel runs against either name
                   "/kaggle/input/rsna-knee-src",
                   "/kaggle/input/datasets/patimakornsrinamen/rsna-knee-src")
for _src in _SRC_CANDIDATES:
    if os.path.exists(_src):
        sys.path.insert(0, _src)
        break
else:
    raise FileNotFoundError(f"rsna-67knee-src not mounted; tried {_SRC_CANDIDATES}")
from preprocess import (LABELS, build_study_tensor, find_by_marker, find_input,
                        read_series_table, recipe_kwargs)
from model import Model, POOLERS, predict_study
from blend_eval import rank_blend

COMP = find_input(
    "/kaggle/input/rsna-knee-abnormality-detection",
    "/kaggle/input/competitions/rsna-knee-abnormality-detection",
)
OUT = "/kaggle/working/submission.csv"
DECODE_WORKERS = 4
# weights come from a pinned dataset slug, never a kernel's latest output -- see docs/code-notes.md "pinned weights and version check"
EXPECTED_LABEL_FILE = "consensus_v1.csv"
EXPECTED_RECIPE_VERSION = {"dinov2s": "v4", "coatnet2": "v4", "convnexts": "v4"}
WEIGHTS_CANDIDATES = (
    "/kaggle/input/rsna-67knee-weights-v4b",
    "/kaggle/input/datasets/patimakornsrinamen/rsna-67knee-weights-v4b",
)
# dinov2s reads the WEIGHTS_CANDIDATES global so a monkeypatch still works -- see docs/code-notes.md "per-family weights candidates"
FAMILY_WEIGHTS = {
    "dinov2s": WEIGHTS_CANDIDATES,
    "coatnet2": (
        "/kaggle/input/rsna-67knee-weights-v6c",
        "/kaggle/input/datasets/patimakornsrinamen/rsna-67knee-weights-v6c",
    ),
    # round 10 pure-CNN arm (ConvNeXt-Small), trained locally; dataset pinned when the run lands
    "convnexts": (
        "/kaggle/input/rsna-67knee-weights-convnexts",
        "/kaggle/input/datasets/patimakornsrinamen/rsna-67knee-weights-convnexts",
    ),
}

# MAIN/EFFICIENCY presets trade fold+slice coverage for speed; see PRESETS below -- see docs/code-notes.md "MAIN and EFFICIENCY presets"
PRESET = "coatnet"  # "main" | "main_attn" | "efficiency" | "coatnet" | "fast" | "blend" - coatnet (CoAtNet-2 v6c alone, 0.912 LB) is the shipping default since round 6; main_attn (DINOv2-S v4b) scored 0.892 and the weighted blend 0.909

# TIMING_STUDIES>0 times training studies instead of scoring the test set -- see docs/code-notes.md "TIMING_STUDIES timing mode"
TIMING_STUDIES = 0

# TIMING_GOLD_SLICES sweeps slice budgets against gold58, ignoring TIMING_STUDIES -- see docs/code-notes.md "TIMING_GOLD_SLICES sweep mode"
TIMING_GOLD_SLICES = ""

# FAST_FOLDS/FAST_SLICES let this round's operating point move in one line -- see docs/code-notes.md "FAST preset constants"
FAST_FOLDS = 1
FAST_SLICES = None

# Optional per-preset key "window_stride" (absent/None = the training-time
# non-overlapping window grouping): an int makes model.study_windows score
# dense overlapping windows every `window_stride` slices at inference
# (stride 1 = +0.0071 paired OOF on v6c, ~3x the backbone work per study;
# see docs/methodology.md). Only meaningful for window-trained families.
PRESETS = {
    "main": {"families": ("dinov2s",), "use_tta": None, "max_folds": None,
             "max_slices": None, "pool": None},
    # MAIN_ATTN: the main preset with the tuned window_pool ignored - the
    # diagnostic that separates the pooling table from the pixels on the LB.
    "main_attn": {"families": ("dinov2s",), "use_tta": None,
                  "max_folds": None, "max_slices": None, "pool": "attn"},
    "efficiency": {"families": ("dinov2s",), "use_tta": False,
                   "max_folds": 1, "max_slices": 6, "pool": "attn"},
    # COATNET: arm 2 alone, same recipe and pooling stance as main_attn.
    "coatnet": {"families": ("coatnet2",), "use_tta": None,
                "max_folds": None, "max_slices": None, "pool": "attn"},
    # CONVNEXT: the round-10 ConvNeXt-Small arm alone, same stance as coatnet.
    "convnext": {"families": ("convnexts",), "use_tta": None,
                 "max_folds": None, "max_slices": None, "pool": "attn"},
    # FAST: CoAtNet-2 alone, no TTA, forced all-attn pooling -- see docs/code-notes.md "FAST preset definition"
    "fast": {"families": ("coatnet2",), "use_tta": False,
             "max_folds": FAST_FOLDS, "max_slices": FAST_SLICES,
             "pool": "attn"},
    # BLEND: rank-blend of both arms; allow_partial/allow_equal_weights stay False -- see docs/code-notes.md "BLEND preset definition"
    "blend": {"families": ("dinov2s", "coatnet2"), "use_tta": None,
              "max_folds": None, "max_slices": None, "pool": "attn",
              "weights_file": "blend_weights.json", "allow_partial": False,
              "allow_equal_weights": False},
}

# MAX_INFER_HOURS stops a multi-family run before it burns the whole session -- see docs/code-notes.md "MAX_INFER_HOURS budget"
MAX_INFER_HOURS = 8.5

# PRIOR_FALLBACK backs run_configs (round-2/3) that carry no prior of their own -- see docs/code-notes.md "PRIOR_FALLBACK prevalence"
PRIOR_FALLBACK = [0.2412, 0.1784, 0.4441, 0.2153, 0.3875, 0.3004,
                  0.462, 0.5571, 0.2665, 0.3159, 0.2467, 0.1619]

# decode-worker state, set by _init_worker in every process (pool or inline)
_WORKER_RECIPE = {}
_WORKER_MAX_SLICES = None
_WORKER_SLICE_ALIGN = 1
_WORKER_SERIES_DIR = "test_series"


def decode_study(args):
    """Worker: (uid, series_rows) -> (uid, volume, filled_slots); volume is
    None when decoding raised. filled_slots == 0 means nothing decoded."""
    uid, series_rows = args
    assert _WORKER_RECIPE, "decode worker used before _init_worker"
    try:
        vol, filled = build_study_tensor(
            os.path.join(COMP, _WORKER_SERIES_DIR, uid), series_rows,
            max_slices=_WORKER_MAX_SLICES, slice_align=_WORKER_SLICE_ALIGN,
            **_WORKER_RECIPE)
        return uid, vol, filled
    except Exception as e:
        print(f"WARN study {uid} decode failed ({type(e).__name__})",
              flush=True)
        return uid, None, 0


def _init_worker(recipe, max_slices, slice_align=1, series_dir="test_series"):
    global _WORKER_MAX_SLICES, _WORKER_SLICE_ALIGN, _WORKER_SERIES_DIR
    _WORKER_RECIPE.update(recipe)
    _WORKER_MAX_SLICES = max_slices
    _WORKER_SLICE_ALIGN = slice_align
    _WORKER_SERIES_DIR = series_dir


def pool_windows(probs, window_pool, p_attn):
    """probs: (n_models, n_windows, 12) NaN-padded per-window sigmoid probs
    (model.predict_study's win_probs). window_pool: {label: mode}. p_attn:
    (12,) from predict_study - the ensemble's own attention pooling. Returns
    (12,): per label, the pooled value for window_pool[label] (model.POOLERS:
    mean/max/top2, per model across windows then averaged across models -
    the same order tune_pooling.py uses per fold); an "attn" mode, or a label
    missing from window_pool, keeps p_attn unchanged. Pure function so it can
    be unit-tested against tune_pooling.py's pooling."""
    out = np.array(p_attn, dtype=np.float32).copy()
    if not window_pool:
        return out
    for c, name in enumerate(LABELS):
        mode = window_pool.get(name, "attn")
        if mode not in POOLERS:  # "attn", or an unknown mode: leave p_attn
            continue
        col = probs[:, :, c]  # (n_models, n_windows)
        per_model = POOLERS[mode](col, axis=1)
        out[c] = float(np.mean(per_model))
    return out


def decode_all(work_items, recipe, max_slices, workers=None, slice_align=1,
              series_dir="test_series"):
    """Yield (uid, vol, filled) for every work item, in input order.

    Runs a process pool with a bounded number of studies in flight (so a
    fast decoder cannot pile gigabytes of volumes ahead of the GPU) and, if a
    worker dies hard (segfault in a C decoder, OOM kill), finishes the rest
    inline instead of losing every prediction made so far. workers <= 1
    decodes inline from the start (local dry-runs). series_dir is the
    subdirectory of the competition mount holding each uid's DICOMs -
    "test_series" for a normal run, "train_series" for TIMING_STUDIES."""
    workers = DECODE_WORKERS if workers is None else workers
    if workers <= 1:
        _init_worker(recipe, max_slices, slice_align, series_dir)
        for w in work_items:
            yield decode_study(w)
        return
    ex = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(recipe, max_slices, slice_align,
                                      series_dir))
    pending = deque()
    submitted = done = 0
    try:
        while done < len(work_items):
            while submitted < len(work_items) and len(pending) < 4 * workers:
                pending.append(ex.submit(decode_study, work_items[submitted]))
                submitted += 1
            yield pending.popleft().result()
            done += 1
    except BrokenProcessPool:
        print(f"WARN decode pool died after {done} studies; finishing the "
              f"remaining {len(work_items) - done} inline", flush=True)
        _init_worker(recipe, max_slices, slice_align, series_dir)
        for w in work_items[done:]:
            yield decode_study(w)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def _expected_recipe_version(family):
    """Resolve EXPECTED_RECIPE_VERSION for one family. A bare string (or
    None) applies to every family - back-compat with a global override, e.g.
    a dry-run driver's `I.EXPECTED_RECIPE_VERSION = None`. A dict (the
    shipping default) is a per-family table keyed by run_config["family"];
    a family absent from it gets no version check."""
    if EXPECTED_RECIPE_VERSION is None or isinstance(EXPECTED_RECIPE_VERSION, str):
        return EXPECTED_RECIPE_VERSION
    return EXPECTED_RECIPE_VERSION.get(family)


def load_family(family, dev):
    """Resolve a family's weights dir, validate its run_config against this
    family, and load its fold models. Returns (models, cfg, weights_dir).

    dinov2s's candidates are read from the current WEIGHTS_CANDIDATES global
    (not FAMILY_WEIGHTS["dinov2s"]) so a dry-run driver that monkeypatches
    WEIGHTS_CANDIDATES after import still takes effect."""
    candidates = (WEIGHTS_CANDIDATES if family == "dinov2s"
                 else FAMILY_WEIGHTS[family])
    try:
        weights_dir = find_input(*candidates)
    except FileNotFoundError:
        # fallback for local dry-runs; on Kaggle the dataset must be attached
        weights_dir = find_by_marker(f"{family}_fold0.pt")
    cfg_path = os.path.join(weights_dir, "run_config.json")
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    print(f"WEIGHTS_DIR[{family}]: {weights_dir} | device: {dev} | cfg: {cfg}",
          flush=True)
    assert cfg["label_file"] == EXPECTED_LABEL_FILE, (
        f"{family} weights were trained on {cfg['label_file']}, expected "
        f"{EXPECTED_LABEL_FILE} - wrong weights attached")
    expected_version = _expected_recipe_version(family)
    if expected_version is not None:
        assert cfg.get("version", "v2") == expected_version, (
            f"{family} weights carry recipe {cfg.get('version', 'v2')}, "
            f"expected {expected_version} - wrong weights dataset attached")
    assert cfg.get("family", "dinov2s") == family, (
        f"weights at {weights_dir} carry family "
        f"{cfg.get('family', 'dinov2s')!r}, expected {family!r}")
    assert not cfg.get("smoke"), (
        f"weights at {weights_dir} are a smoke run (smoke={cfg.get('smoke')})")
    n_folds_expected = int(cfg.get("n_folds", 3))

    preset = PRESETS[PRESET]
    max_folds = preset["max_folds"] or n_folds_expected
    # "_last.pt" is the SWA companion file and must not count as a fold weight -- see docs/code-notes.md "fold weight file filter"
    weight_files = sorted(f for f in os.listdir(weights_dir)
                          if f.startswith(f"{family}_fold") and f.endswith(".pt")
                          and not f.endswith("_last.pt"))
    assert len(weight_files) == n_folds_expected, (
        f"expected {n_folds_expected} {family} fold weights in {weights_dir}, "
        f"found {len(weight_files)}")
    models = []
    for f in weight_files[:max_folds]:
        m = Model(pretrained=False, img_size=cfg["size"], arch=cfg.get("arch"),
                  n_slots=cfg.get("n_slots"),
                  spatial_pool=cfg.get("spatial_pool")).to(dev).eval()
        m.load_state_dict(torch.load(os.path.join(weights_dir, f),
                                     map_location=dev))
        models.append(m)
    print(f"{len(models)} {family} fold models loaded", flush=True)
    return models, cfg, weights_dir


def _resolve_use_tta(preset, cfg):
    """preset["use_tta"] overrides; None trusts the training run's own
    measurement (same rule for every family - see the PRESETS comment)."""
    use_tta = preset["use_tta"]
    if use_tta is None:
        if "use_tta" in cfg:  # decided in train on the unrounded AUCs
            use_tta = bool(cfg["use_tta"])
        else:  # round-2/3 run_config: only the rounded AUCs are recorded
            use_tta = (cfg.get("tta_gold58_auc") is not None
                       and cfg.get("gold58_auc") is not None
                       and cfg["tta_gold58_auc"] >= cfg["gold58_auc"])
    return use_tta


def score_study(models, dev, use_amp, use_tta, window, window_pool, prior,
                uid, vol, filled, stride=None):
    """One family's score for one study. Returns (probs (12,), failed: bool);
    failed marks a prior fallback (decode failure, non-finite prediction with
    no usable attention score, or an inference exception)."""
    if vol is None or filled == 0:
        # decode raised, or no slot decoded: a black image would get
        # an arbitrary constant score - use the prior instead
        return prior, True
    try:
        # one backbone pass yields both the attention score and per-window probs -- see docs/code-notes.md "one backbone pass per study"
        p_attn, wprobs, _ = predict_study(models, vol, dev, use_amp,
                                          tta=use_tta, window=window,
                                          stride=stride)
        non_finite = (not np.all(np.isfinite(p_attn))
                     or (wprobs is not None
                         and not np.all(np.isfinite(wprobs))))
        if non_finite:
            # never write nan into the CSV: fall back to the
            # attention-only score if it's usable, else the prior
            if np.all(np.isfinite(p_attn)):
                print(f"WARN study {uid} non-finite prediction; "
                      "using attention-only score", flush=True)
                return p_attn, False
            else:
                print(f"WARN study {uid} non-finite prediction; "
                      "using prior", flush=True)
                return prior, True
        elif window is None:
            return p_attn, False
        elif window_pool and any(m != "attn" for m in window_pool.values()):
            return pool_windows(wprobs, window_pool, p_attn), False
        else:
            return p_attn, False
    except Exception as e:
        print(f"WARN study {uid} inference failed "
              f"({type(e).__name__}); using prior", flush=True)
        return prior, True


def projected_hours(elapsed_s, done, total):
    """Pure: elapsed_s spent on `done` (of `total`) studies -> the projected
    total wall-clock hours at that same per-study rate. done must be > 0."""
    return elapsed_s / done * total / 3600.0


def check_projection(elapsed_s, done, total, n_families):
    """Print the projected total runtime and, for a multi-family preset
    (a blend, which pays for every family's forward pass), hard-fail if it
    would exceed MAX_INFER_HOURS - a single-family preset never raises here,
    same as before this check existed. MAX_INFER_HOURS is read as a module
    global so a driver that patches it after import still takes effect."""
    proj = projected_hours(elapsed_s, done, total)
    print(f"projected total: {proj:.2f} h", flush=True)
    if n_families > 1 and proj > MAX_INFER_HOURS:
        raise RuntimeError(
            f"projected {proj:.2f} h exceeds MAX_INFER_HOURS={MAX_INFER_HOURS} h")
    return proj


def _write_csv(path, uids, prob_rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["StudyInstanceUID"] + LABELS)
        w.writerows([[u] + [f"{p:.6f}" for p in row]
                    for u, row in zip(uids, prob_rows)])


def read_gold58(path):
    """gold58.csv -> (uids, y) with y (len(uids), 12) in LABELS order. Fails
    loudly, naming the missing column, if the uid column or any of the 12
    LABELS columns is absent."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        assert "StudyInstanceUID" in fields, (
            f"gold58.csv missing column 'StudyInstanceUID' (found {fields})")
        for name in LABELS:
            assert name in fields, (
                f"gold58.csv missing column {name!r} (found {fields})")
        uids, rows = [], []
        for row in reader:
            uids.append(row["StudyInstanceUID"])
            rows.append([float(row[name]) for name in LABELS])
    return uids, np.asarray(rows, dtype=np.float64)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = dev == "cuda"
    preset = PRESETS[PRESET]
    families = preset["families"]

    # a family load failure is a hard stop unless the preset sets allow_partial -- see docs/code-notes.md "family load failure handling"
    loaded = {}  # family -> (models, cfg, weights_dir, use_tta)
    for family in families:
        try:
            models, cfg, weights_dir = load_family(family, dev)
        except Exception as exc:
            if not preset.get("allow_partial"):
                raise
            print(f"WARN family {family} failed: {exc!r}", flush=True)
            continue
        loaded[family] = (models, cfg, weights_dir, _resolve_use_tta(preset, cfg))
    if not loaded:
        raise RuntimeError(
            f"no family loaded for preset {PRESET!r} (tried {families})")
    fam_names = list(loaded.keys())

    # blend_weights.json failures are hard errors; allow_equal_weights only WARNs -- see docs/code-notes.md "blend_weights.json contract"
    weights_file = preset.get("weights_file")
    blend_weights, blend_weights_matrix = None, None
    if weights_file:
        hits = [os.path.join(loaded[fam][2], weights_file) for fam in fam_names
               if os.path.isfile(os.path.join(loaded[fam][2], weights_file))]
        if not hits:
            if not preset.get("allow_equal_weights"):
                raise RuntimeError(
                    f"{weights_file} not found in any of "
                    f"{[loaded[fam][2] for fam in fam_names]}")
            print(f"WARN {weights_file} not found; using equal weights",
                  flush=True)
        elif len(hits) > 1:
            raise RuntimeError(
                f"{weights_file} found in more than one weights dir: {hits}")
        else:
            with open(hits[0]) as fh:
                bw_json = json.load(fh)
            json_families, want_families = set(bw_json["families"]), set(fam_names)
            if json_families != want_families:
                raise RuntimeError(
                    f"{weights_file} families {json_families} != loaded "
                    f"families {want_families}")
            try:
                blend_weights_matrix = np.array(
                    [[bw_json["weights"][label][fam] for label in LABELS]
                     for fam in fam_names])
            except KeyError as e:
                raise RuntimeError(
                    f"{weights_file} missing weight entry {e} (labels="
                    f"{LABELS}, families={fam_names})") from e
            blend_weights = {"path": hits[0], "families": fam_names,
                             "matrix": blend_weights_matrix.tolist()}

    # families sharing a (recipe, window) pair are grouped so decode_all runs once -- see docs/code-notes.md "recipe/window family grouping"
    recipe_by_family = {fam: recipe_kwargs(loaded[fam][1]) for fam in fam_names}
    window_by_family = {fam: loaded[fam][1].get("window") for fam in fam_names}
    groups = []  # [((recipe_items, window), [fam, ...])]
    group_index = {}
    for fam in fam_names:
        key = (tuple(sorted(recipe_by_family[fam].items())), window_by_family[fam])
        if key not in group_index:
            group_index[key] = len(groups)
            groups.append((key, []))
        groups[group_index[key]][1].append(fam)

    max_slices = preset["max_slices"]
    window_stride = preset.get("window_stride")  # None = training-time grouping
    slice_align_by_group = {}
    for key, fams in groups:
        _, group_window = key
        if max_slices is not None and group_window is not None:
            assert max_slices % group_window == 0, (
                f"preset max_slices={max_slices} is not a multiple of "
                f"{fams}'s window={group_window} - the decoded subset would "
                "not land on window-group boundaries")
        # slice_align keeps a truncated decode on window-group boundaries -- see docs/code-notes.md "slice_align boundary rule"
        slice_align_by_group[key] = group_window or 1

    # per-family window_pool / prior: each family's own tuned table / prior,
    # not shared, even across families that do share a recipe/window.
    window_pool_by_family, prior_by_family = {}, {}
    for fam in fam_names:
        cfg = loaded[fam][1]
        if preset["pool"] == "attn":
            # efficiency/coatnet/blend/main_attn force all-attn pooling -- see docs/code-notes.md "forced all-attn pooling presets"
            window_pool_by_family[fam] = {name: "attn" for name in LABELS}
        else:  # preset["pool"] is None (main): trust the run_config's table
            window_pool_by_family[fam] = cfg.get("window_pool")
        prior_by_family[fam] = cfg.get("prior", PRIOR_FALLBACK)

    infer_cfg = {
        "preset": PRESET, "families": fam_names, "max_slices": max_slices,
        "recipe_groups": [
            {"families": fams, "recipe": dict(key[0]), "window": key[1],
             "slice_align": slice_align_by_group[key]}
            for key, fams in groups
        ],
        "per_family": {
            fam: {"n_folds": len(loaded[fam][0]),
                  "arch": loaded[fam][1].get("arch"),
                  "use_tta": loaded[fam][3],
                  "weights_dir": loaded[fam][2],
                  "window_pool": window_pool_by_family[fam],
                  "recipe": recipe_by_family[fam],
                  "window": window_by_family[fam]}
            for fam in fam_names
        },
        "blend_weights": blend_weights,
        "projected_hours": None,  # filled in after the study loop
    }
    print(f"INFER_CFG: {infer_cfg}", flush=True)
    with open(os.path.join(os.path.dirname(OUT), "infer_config.json"),
              "w") as fh:
        json.dump(infer_cfg, fh)  # the effective config, next to the CSV

    # TIMING_GOLD_SLICES is checked before TIMING_STUDIES and ignores it entirely -- see docs/code-notes.md "TIMING_GOLD_SLICES precedence"
    if TIMING_GOLD_SLICES:
        # tune_pooling imported here so a src mount without it can't break other paths -- see docs/code-notes.md "deferred tune_pooling import"
        from tune_pooling import auc_per_label
        gold_slices = [int(s) for s in TIMING_GOLD_SLICES.split(",") if s.strip()]
        for budget in gold_slices:
            for key, fams in groups:
                group_window = key[1]
                assert group_window is None or budget % group_window == 0, (
                    f"TIMING_GOLD_SLICES budget={budget} is not a multiple "
                    f"of {fams}'s window={group_window} - the decoded "
                    "subset would not land on window-group boundaries")

        src_dir = find_input(*_SRC_CANDIDATES)
        gold_uids, gold_y = read_gold58(os.path.join(src_dir, "gold58.csv"))
        by_study = read_series_table(os.path.join(COMP, "train_series.csv"))
        work_items = [(u, by_study.get(u, [])) for u in gold_uids]
        n_studies = len(gold_uids)
        n_folds = sum(len(loaded[fam][0]) for fam in fam_names)
        print(f"{n_studies} gold studies (from gold58.csv)", flush=True)

        records = []
        for budget in gold_slices:
            decode_s = 0.0
            gpu_s = 0.0
            wt0 = time.time()
            probs_by_family = {fam: [None] * n_studies for fam in fam_names}
            for key, fams in groups:
                group_recipe, group_window = key[0], key[1]
                group_recipe = dict(group_recipe)
                group_slice_align = slice_align_by_group[key]
                decode_iter = iter(decode_all(
                    work_items, group_recipe, budget,
                    slice_align=group_slice_align, series_dir="train_series"))
                k = 0
                while True:
                    dt0 = time.time()
                    try:
                        uid, vol, filled = next(decode_iter)
                    except StopIteration:
                        break
                    decode_s += time.time() - dt0
                    assert uid == gold_uids[k], (
                        f"decode order drifted at index {k} ({uid!r} != "
                        f"{gold_uids[k]!r})")
                    for fam in fams:
                        models, _, _, use_tta = loaded[fam]
                        gt_start = time.time()
                        probs, _ = score_study(
                            models, dev, use_amp, use_tta, group_window,
                            window_pool_by_family[fam], prior_by_family[fam],
                            uid, vol, filled, stride=window_stride)
                        gpu_s += time.time() - gt_start
                        probs_by_family[fam][k] = np.asarray(probs, dtype=np.float64)
                    k += 1
            wall_s = time.time() - wt0
            if len(fam_names) == 1:
                final_probs = probs_by_family[fam_names[0]]
            else:
                mats = [np.stack(probs_by_family[fam]) for fam in fam_names]
                final_probs = list(rank_blend(mats, weights=blend_weights_matrix))
            scores = np.stack(final_probs)
            per_label = auc_per_label(gold_y, scores)
            # None (not NaN) when every label is single-class in this gold
            # set: json.dump would otherwise write a bare NaN, invalid JSON.
            macro_auc = (float(np.mean(list(per_label.values())))
                        if per_label else None)
            records.append({
                "max_slices": budget, "auc": macro_auc,
                "auc_per_label": per_label, "decode_s": decode_s,
                "gpu_s": gpu_s, "wall_s": wall_s, "n_studies": n_studies,
                "n_folds": n_folds,
            })

        print("GOLD_SWEEP results:", flush=True)
        print(f"{'max_slices':>10} {'auc':>8} {'decode_s':>9} {'gpu_s':>8} "
              f"{'wall_s':>8}", flush=True)
        for r in records:
            auc_str = f"{r['auc']:.4f}" if r["auc"] is not None else "n/a"
            print(f"{r['max_slices']:>10} {auc_str:>8} "
                  f"{r['decode_s']:>9.2f} {r['gpu_s']:>8.2f} "
                  f"{r['wall_s']:>8.2f}", flush=True)
        with open(os.path.join(os.path.dirname(OUT), "timing_gold.json"),
                  "w") as fh:
            json.dump(records, fh)
        return

    # TIMING_STUDIES times training studies since the visible test mount is tiny -- see docs/code-notes.md "TIMING_STUDIES uses training tree"
    timing = TIMING_STUDIES > 0
    if timing:
        by_study = read_series_table(os.path.join(COMP, "train_series.csv"))
        test_uids = sorted(by_study.keys())[:TIMING_STUDIES]
        series_dir = "train_series"
        print(f"{len(test_uids)} timing studies (from train_series.csv)",
              flush=True)
    else:
        by_study = read_series_table(os.path.join(COMP, "test_series.csv"))
        with open(os.path.join(COMP, "test.csv"), newline="",
                  encoding="utf-8") as fh:
            test_uids = [r["StudyInstanceUID"] for r in csv.DictReader(fh)]
        series_dir = "test_series"
        print(f"{len(test_uids)} test studies", flush=True)

    # per-study (uid, rows) tuples: the series table is NOT captured in the
    # callable, so nothing large is re-pickled per map call
    work_items = [(u, by_study.get(u, [])) for u in test_uids]
    out_uids = list(test_uids)
    probs_by_family = {fam: [None] * len(test_uids) for fam in fam_names}
    n_failed_by_family = {fam: 0 for fam in fam_names}
    proj_hours_by_group = {}
    decode_seconds = 0.0
    gpu_seconds = 0.0
    t0 = time.time()

    def _timed_decode(gen):
        # decode_seconds separates decode wait time from GPU time for the projection -- see docs/code-notes.md "timed decode bookkeeping"
        nonlocal decode_seconds
        it = iter(gen)
        while True:
            dt0 = time.time()
            try:
                item = next(it)
            except StopIteration:
                return
            decode_seconds += time.time() - dt0
            yield item

    for key, fams in groups:
        group_recipe, group_window = key[0], key[1]
        group_recipe = dict(group_recipe)
        group_slice_align = slice_align_by_group[key]
        gt0 = time.time()
        decoded = decode_all(work_items, group_recipe, max_slices,
                             slice_align=group_slice_align,
                             series_dir=series_dir)
        if timing:
            decoded = _timed_decode(decoded)
        for k, (uid, vol, filled) in enumerate(decoded):
            assert uid == test_uids[k], (
                f"decode order drifted at index {k} ({uid!r} != "
                f"{test_uids[k]!r})")
            for fam in fams:
                models, _, _, use_tta = loaded[fam]
                gt_start = time.time() if timing else None
                probs, failed = score_study(
                    models, dev, use_amp, use_tta, group_window,
                    window_pool_by_family[fam], prior_by_family[fam],
                    uid, vol, filled, stride=window_stride)
                if timing:
                    gpu_seconds += time.time() - gt_start
                if failed:
                    n_failed_by_family[fam] += 1
                probs_by_family[fam][k] = np.asarray(probs, dtype=np.float64)
            if (k + 1) % 100 == 0:
                rate = (k + 1) / (time.time() - gt0)
                print(f"[{','.join(fams)}] {k + 1}/{len(test_uids)} "
                      f"({rate:.2f} studies/s)", flush=True)
            if k + 1 == 200:
                proj_hours_by_group[key] = check_projection(
                    time.time() - gt0, k + 1, len(test_uids), len(fams))

    if timing:
        total_wall_s = time.time() - t0
        n_studies = len(test_uids)
        studies_per_s = n_studies / total_wall_s if total_wall_s > 0 else 0.0
        projected_hours_1300 = projected_hours(total_wall_s, n_studies, 1300)
        timing_block = {
            "preset": PRESET,
            "families": fam_names,
            "n_folds": {fam: len(loaded[fam][0]) for fam in fam_names},
            "max_slices": max_slices,
            "n_studies": n_studies,
            "total_wall_s": total_wall_s,
            "decode_s": decode_seconds,
            "gpu_s": gpu_seconds,
            "studies_per_s": studies_per_s,
            "projected_hours_1300": projected_hours_1300,
        }
        print(f"TIMING: preset={PRESET} families={fam_names} "
              f"n_folds={timing_block['n_folds']} max_slices={max_slices} "
              f"studies={n_studies} total_wall_s={total_wall_s:.2f} "
              f"decode_s={decode_seconds:.2f} gpu_s={gpu_seconds:.2f} "
              f"studies/s={studies_per_s:.3f} "
              f"projected_hours_1300={projected_hours_1300:.2f}", flush=True)
        with open(os.path.join(os.path.dirname(OUT), "timing.json"),
                  "w") as fh:
            json.dump(timing_block, fh)
        return

    infer_cfg["projected_hours"] = {
        ",".join(fams): proj_hours_by_group.get(key) for key, fams in groups
    }
    with open(os.path.join(os.path.dirname(OUT), "infer_config.json"),
              "w") as fh:
        json.dump(infer_cfg, fh)  # re-written now that projected_hours is known

    out_dir = os.path.dirname(OUT)
    for fam in fam_names:
        _write_csv(os.path.join(out_dir, f"probs_{fam}.csv"), out_uids,
                  probs_by_family[fam])

    if len(families) == 1:
        # a single-family preset (main/main_attn/efficiency/coatnet): write
        # the family's own scores unchanged, byte-identical to today's output.
        fam = fam_names[0]
        final_probs = probs_by_family[fam]
        n_failed = n_failed_by_family[fam]
    else:
        # a blend preset always rank_blends, even with one surviving family -- see docs/code-notes.md "blend rank_blend passthrough"
        mats = [np.stack(probs_by_family[fam]) for fam in fam_names]
        final_probs = list(rank_blend(mats, weights=blend_weights_matrix))
        n_failed = sum(n_failed_by_family.values())

    _write_csv(OUT, out_uids, final_probs)
    print(f"submission.csv written ({len(out_uids)} rows, {n_failed} "
          f"fallback, {(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
