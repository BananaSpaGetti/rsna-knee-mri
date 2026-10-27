# RSNA Knee: train a 2.5D arm on the pixel cache. Kaggle T4, internet ON (not the submission kernel).
# Attach: rsna-67knee-src, cache kernel outputs, rsna-knee-consensus-labels -- see docs/code-notes.md "training pipeline overview".

import csv
import json
import math
import os
import random
import time

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

print(f"timm {timm.__version__}", flush=True)
T_START = time.time()  # wall clock origin for the MAX_HOURS guard

SEED = 42
N_FOLDS = 5            # fold split, fixed across arms so both arms share folds
NUM_WORKERS = 2
WINDOW = 3            # cached slices per backbone input (RGB channels)
WINDOWS_PER_SLOT = 2   # windows sampled per filled slot per training step
LABEL_FILE = "consensus_v1.csv"  # 3-reader consensus; ablation used llm_labels_v4_blend
SPLIT_LABEL_FILE = "consensus_v1.csv"  # fold split is stratified on this file so every arm and every label variant shares one split

# ARM and SMOKE are the two switches a staged Kaggle copy flips before push -- see docs/code-notes.md "arm and SMOKE switches"
ARM = "dinov2"
# SWEEP_LR/SWEEP_EPOCHS drive the coatnet2 sweep; MAX_HOURS caps fold wall clock -- see docs/code-notes.md "sweep and MAX_HOURS constants"
SWEEP_LR = 1e-4
SWEEP_EPOCHS = 4
MAX_HOURS = 8.4
ARMS = {
    "dinov2": dict(arch=None, family="dinov2s", batch=12, epochs=20,
                   folds_to_train=(0, 1, 2, 3, 4), lr_backbone=3e-5,
                   lr_head=1e-3, grad_ckpt=False,
                   unfreeze=None, warmup_epochs=0, drop_path=None,
                   pos_weight=False, swa_last=0, full_data=False),
    "dinov2_full": dict(arch=None, family="dinov2s", batch=12, epochs=30,
                        folds_to_train=(0, 1, 2, 3, 4), lr_backbone=3e-5,
                        lr_head=1e-3, grad_ckpt=False,
                        unfreeze="all", warmup_epochs=1, drop_path=None,
                        pos_weight=False, swa_last=0, full_data=False),
    "coatnet2": dict(arch="coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k",
                     family="coatnet2", batch=12, epochs=20,
                     folds_to_train=(0, 1, 2, 3, 4), lr_backbone=5e-5,
                     lr_head=1e-3, grad_ckpt=True,
                     unfreeze="stem_stage0", warmup_epochs=1, drop_path=0.1,
                     pos_weight=False, swa_last=0, full_data=False),
    "coatnet2_sweep": dict(arch="coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k",
                           family="coatnet2", batch=12, epochs=SWEEP_EPOCHS,
                           folds_to_train=(0,), lr_backbone=SWEEP_LR,
                           lr_head=1e-3, grad_ckpt=True,
                           unfreeze="stem_stage0", warmup_epochs=1,
                           drop_path=0.1, pos_weight=False, swa_last=0,
                           full_data=False),
    # coatnet2_swa: the v6c recipe (v4 cache, 20 ep) plus SWA over the last 3 epochs - the one
    # knob never tried on the shipping arm (round 10 track 2)
    # coatnet2_rs: the v6c recipe with random-start training windows (any
    # non-empty 3-slice start, not only the N_GROUPS aligned ones) so the
    # training window distribution matches dense stride-1 inference
    # (model.study_windows stride=1), plus a second training seed so the run
    # doubles as a seed-ensemble partner for v6c (same fold split: the split
    # stays on SEED). Judged on paired dense OOF vs v6c (src/eval_windows.py).
    "coatnet2_rs": dict(arch="coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k",
                        family="coatnet2", batch=12, epochs=20,
                        folds_to_train=(0, 1, 2, 3, 4), lr_backbone=5e-5,
                        lr_head=1e-3, grad_ckpt=True,
                        unfreeze="stem_stage0", warmup_epochs=1, drop_path=0.1,
                        pos_weight=False, swa_last=0, full_data=False,
                        window_starts="random", train_seed=7),
    # coatnet2_rs2: coatnet2_rs with another training seed - a third CoAt
    # ensemble member, and rs vs rs2 paired OOF measures the seed-only noise
    # floor at the 5-fold level (rs beat v6c dense by +0.0068 +/- 0.0010).
    "coatnet2_rs2": dict(arch="coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k",
                         family="coatnet2", batch=12, epochs=20,
                         folds_to_train=(0, 1, 2, 3, 4), lr_backbone=5e-5,
                         lr_head=1e-3, grad_ckpt=True,
                         unfreeze="stem_stage0", warmup_epochs=1, drop_path=0.1,
                         pos_weight=False, swa_last=0, full_data=False,
                         window_starts="random", train_seed=11),
    # coatnet2_pa: coatnet2_rs unchanged (seed 7, random window starts) + spatial_pool="label_attn":
    # per-label attention over the 9x9 patch grid (LN'd tokens, cap-2 tanh, 0.85 attn + 0.15 GAP;
    # docs/competitors-code/mattiaangeli-d4-serving.md 2.2) replaces GAP before the shared 1024->256
    # projection; zero-init queries = the rs model at step 0. One factor vs coatnet2_rs: paired fold-0 dense OOF.
    # coatnet2_ua: coatnet2_rs with the whole backbone trainable (unfreeze="all" instead of
    # stem_stage0) - the public CoAtNet readers train everything (docs/competitors-code/
    # training-recipes.md). One factor vs coatnet2_rs: paired fold-0 dense OOF.
    "coatnet2_ua": dict(arch="coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k",
                        family="coatnet2", batch=12, epochs=20,
                        folds_to_train=(0, 1, 2, 3, 4), lr_backbone=5e-5,
                        lr_head=1e-3, grad_ckpt=True,
                        unfreeze="all", warmup_epochs=1, drop_path=0.1,
                        pos_weight=False, swa_last=0, full_data=False,
                        window_starts="random", train_seed=7),
    "coatnet2_pa": dict(arch="coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k",
                        family="coatnet2", batch=12, epochs=20,
                        folds_to_train=(0, 1, 2, 3, 4), lr_backbone=5e-5,
                        lr_head=1e-3, grad_ckpt=True,
                        unfreeze="stem_stage0", warmup_epochs=1, drop_path=0.1,
                        pos_weight=False, swa_last=0, full_data=False,
                        window_starts="random", train_seed=7, spatial_pool="label_attn"),
    # coatnet2_v4g: coatnet2_rs unchanged (seed 7, random window starts) on the v4g cache
    # (cardinal slice order + laterality canonicalisation + LUT/crop window; docs/specs/geometry-v4g.md).
    # One factor vs coatnet2_rs: paired fold-0 dense OOF.
    "coatnet2_v4g": dict(arch="coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k",
                         family="coatnet2", batch=12, epochs=20,
                         folds_to_train=(0, 1, 2, 3, 4), lr_backbone=5e-5,
                         lr_head=1e-3, grad_ckpt=True,
                         unfreeze="stem_stage0", warmup_epochs=1, drop_path=0.1,
                         pos_weight=False, swa_last=0, full_data=False,
                         window_starts="random", train_seed=7),
    "coatnet2_swa": dict(arch="coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k",
                         family="coatnet2", batch=12, epochs=20,
                         folds_to_train=(0, 1, 2, 3, 4), lr_backbone=5e-5,
                         lr_head=1e-3, grad_ckpt=True,
                         unfreeze="stem_stage0", warmup_epochs=1, drop_path=0.1,
                         pos_weight=False, swa_last=3, full_data=False),
    # coatnet2_384ck / coatnet2_384up: the v6c recipe with Raptor's 384-native checkpoint
    # (docs/raptor-diff.md item 1), fed the 288 px window as is / upsampled to 384 px
    "coatnet2_384ck": dict(arch="coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
                           family="coatnet2", batch=12, epochs=20,
                           folds_to_train=(0, 1, 2, 3, 4), lr_backbone=5e-5,
                           lr_head=1e-3, grad_ckpt=True,
                           unfreeze="stem_stage0", warmup_epochs=1, drop_path=0.1,
                           pos_weight=False, swa_last=0, full_data=False),
    "coatnet2_384up": dict(arch="coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k@384",
                           family="coatnet2", batch=12, epochs=20,
                           folds_to_train=(0, 1, 2, 3, 4), lr_backbone=5e-5,
                           lr_head=1e-3, grad_ckpt=True,
                           unfreeze="stem_stage0", warmup_epochs=1, drop_path=0.1,
                           pos_weight=False, swa_last=0, full_data=False),
    # coatnet2_e30: the v6c recipe on a 30-epoch cosine schedule (round 10 track 2, local)
    "coatnet2_e30": dict(arch="coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k",
                         family="coatnet2", batch=12, epochs=30,
                         folds_to_train=(0, 1, 2, 3, 4), lr_backbone=5e-5,
                         lr_head=1e-3, grad_ckpt=True,
                         unfreeze="stem_stage0", warmup_epochs=1, drop_path=0.1,
                         pos_weight=False, swa_last=0, full_data=False),
    # coatnet2_v5: v5 cache layout, pos_weight on, SWA-3, both decided a priori -- see docs/code-notes.md "coatnet2_v5 arm (cache v5)"
    "coatnet2_v5": dict(arch="coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k",
                        family="coatnet2", batch=12, epochs=16,
                        folds_to_train=(0, 1, 2, 3, 4), lr_backbone=5e-5,
                        lr_head=1e-3, grad_ckpt=True,
                        unfreeze="stem_stage0", warmup_epochs=1, drop_path=0.1,
                        pos_weight=True, swa_last=3, full_data=False),
    # convnexts: pure-CNN diversity arm (round 10), coatnet2 hyperparameters on ConvNeXt-Small
    # (in22k -> in1k @384 weights), v4 cache; trained locally on the desktop GPU
    "convnexts": dict(arch="convnext_small.fb_in22k_ft_in1k_384",
                      family="convnexts", batch=12, epochs=20,
                      folds_to_train=(0, 1, 2, 3, 4), lr_backbone=5e-5,
                      lr_head=1e-3, grad_ckpt=True,
                      unfreeze="stem_stage0", warmup_epochs=1, drop_path=0.1,
                      pos_weight=False, swa_last=0, full_data=False),
    # qwenv: Qwen3.5-2B vision tower (round 10 step A), full fine-tune, lower backbone lr for a
    # 300M-parameter pretrained ViT; local desktop only (needs transformers >= 5.17 + HF weights)
    # amp="bf16": the tower was pretrained in bf16 and learns faster under it (one-batch
    # test 2026-09-27: loss 0.66 -> 0.37 in 25 steps vs 0.44 in fp16); T4 kernels keep fp16.
    "qwenv": dict(arch="qwen3.5-2b-vision", family="qwenv", batch=12, epochs=20,
                  folds_to_train=(0, 1, 2, 3, 4), lr_backbone=SWEEP_LR,
                  lr_head=1e-3, grad_ckpt=True,
                  unfreeze="all", warmup_epochs=1, drop_path=None,
                  pos_weight=False, swa_last=0, full_data=False, amp="bf16"),
    # qwenvlm: round 10 step B, the full VLM with LoRA (leader recipe); local only, bf16
    "qwenvlm": dict(arch="qwen3.5-2b-vlm", family="qwenvlm", batch=12, epochs=12,
                    folds_to_train=(0, 1, 2, 3, 4), lr_backbone=1e-4,
                    lr_head=1e-3, grad_ckpt=True,
                    unfreeze="lora", warmup_epochs=1, drop_path=None,
                    pos_weight=False, swa_last=0, full_data=False, amp="bf16"),
    "qwenvlm_mi": dict(arch="qwen3.5-2b-vlm-meanimg", family="qwenvlm", batch=12, epochs=12,
                       folds_to_train=(0, 1, 2, 3, 4), lr_backbone=1e-4,
                       lr_head=1e-3, grad_ckpt=True,
                       unfreeze="lora", warmup_epochs=1, drop_path=None,
                       pos_weight=False, swa_last=0, full_data=False, amp="bf16"),
    # qwenvlm_mi2: the overnight bet after fold 0 of qwenvlm_mi under-fit (OOF 0.813, loss still
    # falling at the end): 3x LoRA lr, LoRA r 32, 20 epochs
    "qwenvlm_mi2": dict(arch="qwen3.5-2b-vlm-meanimg-r32", family="qwenvlm", batch=12, epochs=20,
                        folds_to_train=(0, 1, 2, 3, 4), lr_backbone=3e-4,
                        lr_head=1e-3, grad_ckpt=True,
                        unfreeze="lora", warmup_epochs=1, drop_path=None,
                        pos_weight=False, swa_last=0, full_data=False, amp="bf16"),
    # qwenvlm_mi24: qwenvlm_mi (the only VLM config that learned: fold 0 OOF 0.813 at 12 ep, loss
    # still falling) with a 24-epoch schedule; qwenvlm_mi2's lr 3e-4 collapsed at epoch 3
    "qwenvlm_mi24": dict(arch="qwen3.5-2b-vlm-meanimg", family="qwenvlm", batch=12, epochs=24,
                         folds_to_train=(0, 1, 2, 3, 4), lr_backbone=1e-4,
                         lr_head=1e-3, grad_ckpt=True,
                         unfreeze="lora", warmup_epochs=1, drop_path=None,
                         pos_weight=False, swa_last=0, full_data=False, amp="bf16"),
    "qwenvlm_sweep": dict(arch="qwen3.5-2b-vlm", family="qwenvlm", batch=12,
                          epochs=SWEEP_EPOCHS, folds_to_train=(0,), lr_backbone=SWEEP_LR,
                          lr_head=1e-3, grad_ckpt=True,
                          unfreeze="lora", warmup_epochs=1, drop_path=None,
                          pos_weight=False, swa_last=0, full_data=False, amp="bf16"),
    "qwenvp_sweep": dict(arch="qwen3.5-2b-vision-patch", family="qwenvp", batch=12,
                         epochs=SWEEP_EPOCHS, folds_to_train=(0,), lr_backbone=SWEEP_LR,
                         lr_head=1e-3, grad_ckpt=True,
                         unfreeze="all", warmup_epochs=1, drop_path=None,
                         pos_weight=False, swa_last=0, full_data=False, amp="bf16"),
    "qwenv_sweep": dict(arch="qwen3.5-2b-vision", family="qwenv", batch=12,
                        epochs=SWEEP_EPOCHS, folds_to_train=(0,), lr_backbone=SWEEP_LR,
                        lr_head=1e-3, grad_ckpt=True,
                        unfreeze="all", warmup_epochs=1, drop_path=None,
                        pos_weight=False, swa_last=0, full_data=False, amp="bf16"),
    # coatnet2_full: one model on all non-gold studies, no fold holdout, SWA-3 kept -- see docs/code-notes.md "coatnet2_full arm (round 7)"
    "coatnet2_full": dict(arch="coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k",
                          family="coatnet2", batch=12, epochs=16,
                          folds_to_train=(0, 1, 2, 3, 4), lr_backbone=5e-5,
                          lr_head=1e-3, grad_ckpt=True,
                          unfreeze="stem_stage0", warmup_epochs=1,
                          drop_path=0.1, pos_weight=False, swa_last=3,
                          full_data=True),
    # coatnet2_rsfull: coatnet2_rs unchanged (20 epochs, no SWA, random window starts) except full_data=True - one
    # model on every non-gold study. One factor vs the 5-fold arm on the same label file; five runs with different
    # train seeds stand in for the five folds (queue_full.sh stages one copy per seed, merge_full.py merges them).
    # No OOF exists, so the LB is the only read (2026-10-21).
    "coatnet2_rsfull": dict(arch="coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k",
                            family="coatnet2", batch=12, epochs=20,
                            folds_to_train=(0,), lr_backbone=5e-5,
                            lr_head=1e-3, grad_ckpt=True,
                            unfreeze="stem_stage0", warmup_epochs=1, drop_path=0.1,
                            pos_weight=False, swa_last=0, full_data=True,
                            window_starts="random",
                            train_seed=7),  # rsfull seed: queue_full.sh rewrites this line per run
    # convnexts_rsfull: the convnexts arm's hyperparameters with coatnet2_rsfull's two changes (full_data=True, random
    # window starts) - a second model class for a blend with an rsfull arm on the same label file (late round,
    # 2026-10-27). queue_full_cx.sh stages one copy per train seed; merge_full.py merges them like rsfull.
    "convnexts_rsfull": dict(arch="convnext_small.fb_in22k_ft_in1k_384",
                             family="convnexts", batch=12, epochs=20,
                             folds_to_train=(0,), lr_backbone=5e-5,
                             lr_head=1e-3, grad_ckpt=True,
                             unfreeze="stem_stage0", warmup_epochs=1, drop_path=0.1,
                             pos_weight=False, swa_last=0, full_data=True,
                             window_starts="random",
                             train_seed=7),  # cxfull seed: queue_full_cx.sh rewrites this line per run
}
_arm = ARMS[ARM]
ARCH = _arm["arch"]
FAMILY = _arm["family"]
BATCH_STUDIES = _arm["batch"]
EPOCHS = _arm["epochs"]
FOLDS_TO_TRAIN = _arm["folds_to_train"]
LR_BACKBONE = _arm["lr_backbone"]
LR_HEAD = _arm["lr_head"]
GRAD_CKPT = _arm["grad_ckpt"]
UNFREEZE = _arm["unfreeze"]
WARMUP_EPOCHS = _arm["warmup_epochs"]
WINDOW_STARTS = _arm.get("window_starts", "grouped")  # "grouped" (N_GROUPS aligned) | "random" (any start)
TRAIN_SEED = _arm.get("train_seed", SEED)  # torch/np/random seed only; the fold split stays on SEED
SPATIAL_POOL = _arm.get("spatial_pool", "gap")  # "gap" (GAP -> proj, every arm so far) | "label_attn" (model.py per-label patch attention)
assert WINDOW_STARTS in ("grouped", "random"), WINDOW_STARTS
DROP_PATH = _arm["drop_path"]
POS_WEIGHT = _arm["pos_weight"]  # label-prevalence-derived BCE pos_weight, off by default
SWA_LAST = _arm["swa_last"]      # >0: average the last N epoch checkpoints (0 = off)
FULL_DATA = _arm["full_data"]    # True: train on every non-gold study, no fold holdout
AMP = _arm.get("amp", "fp16")    # autocast dtype: "fp16" (+GradScaler, T4-safe) or "bf16"
AMP_DTYPE = torch.bfloat16 if AMP == "bf16" else torch.float16
SMOKE = 0  # >0: 1 fold, 1 epoch, this many train studies + 100 val studies
# Every new input shape makes cuDNN build convolution plans: ~6 s per shape on
# the heuristic path (~2 s for CoAtNet) and 11-17 s with benchmark mode, for
# ConvNeXt-Small fp16 on a 5090 / torch 2.11 (measured 2026-09-27). Batches
# have a variable window count, so collate() pads it up to a multiple of
# WINDOW_BUCKET (padding rows carry owner -1 and are ignored by the head),
# which leaves a handful of shapes per process. Benchmark mode stays off: the
# steady-state step is the same (0.19 vs 0.22 s) and its per-shape cost is higher.
WINDOW_BUCKET = 16
CUDNN_BENCHMARK = False
DUAL_GPU = False  # True: one child process per visible GPU, one fold each -- see docs/code-notes.md "dual-GPU fold launcher"

OUT = "/kaggle/working"
os.makedirs(OUT, exist_ok=True)  # a local staged copy points OUT at a fresh dir; a missing one killed a finished fold at torch.save

# child of run_dual(): pinned to one fold, writing into its own OUT/fold<k>/
if "RSNA_CHILD_FOLD" in os.environ:
    FOLDS_TO_TRAIN = (int(os.environ["RSNA_CHILD_FOLD"]),)
    OUT = os.path.join(OUT, f"fold{FOLDS_TO_TRAIN[0]}")
    os.makedirs(OUT, exist_ok=True)

import sys

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
from preprocess import LABELS, n_slots_for
from model import Model, find_input
import model as _model_mod  # for _model_mod.ARCH (the resolved name when ARCH is None);
                             # `model` itself is a per-fold local var in main()


CACHE_ROOT = "/kaggle/input"
CACHE_VERSION = None  # None: accept every cache part found (pre-v4g behaviour); "v4"/"v4g"/...: keep only parts whose cache_config.json version matches


def find_cache_parts(marker="cache_index.csv", root=None,
                     skip=("competitions", "rsna-knee-abnormality-detection"),
                     version=None):
    """Every input directory holding `marker` (deterministic, sorted walk,
    same pruning as preprocess.find_by_marker). A v4 cache ships as two
    Kaggle kernel outputs with disjoint global shard ranges; a v2/v3 cache
    (single directory) is the degenerate case. version (e.g. "v4g"), when
    given, keeps only the parts whose cache_config.json "version" matches."""
    root = root or CACHE_ROOT
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath == root:
            dirnames[:] = sorted(d for d in dirnames if d not in skip)
        else:
            dirnames.sort()
        if marker in filenames:
            found.append(dirpath)
    if not found:
        raise FileNotFoundError(f"{marker} not under {root} (skipped {skip})")
    # dedupe cache parts by real path so one mount doesn't look like two parts -- see docs/code-notes.md "cache part dedupe by real path"
    parts = sorted({os.path.realpath(d) for d in found})
    if version is not None:
        seen, keep = [], []
        for d in parts:
            try:
                with open(os.path.join(d, "cache_config.json")) as fh:
                    v = json.load(fh).get("version")
            except (OSError, ValueError):
                v = None
            seen.append((d, v))
            if v == version:
                keep.append(d)
        if not keep:
            raise FileNotFoundError(
                f"no cache part under {root} with version {version}; found {seen}")
        parts = keep
    return parts


CACHE_PARTS = find_cache_parts(version=CACHE_VERSION)
print(f"cache parts ({len(CACHE_PARTS)}): {CACHE_PARTS}", flush=True)
CACHE_CFG = None
for _part in CACHE_PARTS:
    with open(os.path.join(_part, "cache_config.json")) as _fh:
        _cfg = json.load(_fh)
    if CACHE_CFG is None:
        CACHE_CFG = _cfg
    else:
        assert _cfg == CACHE_CFG, (
            f"cache_config.json differs between {CACHE_PARTS[0]} and {_part}")
print("CACHE_CFG:", CACHE_CFG, flush=True)
N_SLICES = CACHE_CFG.get("n_slices", 9)
assert N_SLICES % WINDOW == 0, (
    f"n_slices={N_SLICES} not divisible by WINDOW={WINDOW}")
N_GROUPS = N_SLICES // WINDOW  # non-overlapping windows per slot
# n_slots is derived from slot_layout via n_slots_for, not a recipe key itself -- see docs/code-notes.md "n_slots derivation"
N_SLOTS_CFG = n_slots_for(CACHE_CFG)
# guard against a hand-edited cache_config where n_slots disagrees with slot_layout -- see docs/code-notes.md "n_slots consistency guard"
assert CACHE_CFG.get("n_slots", N_SLOTS_CFG) == N_SLOTS_CFG, (
    f"cache_config n_slots={CACHE_CFG.get('n_slots')} disagrees with "
    f"slot_layout={CACHE_CFG.get('slot_layout', 'plane6')} "
    f"({N_SLOTS_CFG} slots)")
LABELS_CSV = os.path.join(find_input(
    "/kaggle/input/rsna-knee-consensus-labels",
    "/kaggle/input/datasets/patimakornsrinamen/rsna-knee-consensus-labels",
    "/kaggle/input/rsna-knee-llm-report-labels",
    "/kaggle/input/datasets/stevenleehans/rsna-knee-llm-report-labels",
), LABEL_FILE)
assert os.path.exists(LABELS_CSV), (
    f"{LABELS_CSV} missing - expected the rsna-knee-consensus-labels dataset "
    f"(or rsna-knee-llm-report-labels) to carry {LABEL_FILE}")
SPLIT_LABELS_CSV = os.path.join(os.path.dirname(LABELS_CSV), SPLIT_LABEL_FILE)
assert os.path.exists(SPLIT_LABELS_CSV), (
    f"{SPLIT_LABELS_CSV} missing - expected the rsna-knee-consensus-labels dataset "
    f"(or rsna-knee-llm-report-labels) to carry {SPLIT_LABEL_FILE}")
SRC_DIR = _src  # bind explicitly; _src is the bootstrap loop variable
GOLD_CSV = os.path.join(SRC_DIR, "gold58.csv")

torch.manual_seed(TRAIN_SEED)
np.random.seed(TRAIN_SEED)
random.seed(TRAIN_SEED)

def _read_index_rows(part):
    with open(os.path.join(part, "cache_index.csv"), newline="") as fh:
        return list(csv.DictReader(fh))


def load_index():
    """({uid: (shard, row)}, {shard: part dir}) for studies with at least one
    filled slot, merged across every cache part (v4 ships as two kernel
    outputs with disjoint global shard ranges; a single-directory v2/v3
    cache is the degenerate case). A v2 index (no filled_slots column) keeps
    every study.

    Cache parts whose cache_index.csv rows are byte-for-byte identical are
    collapsed into one before merging - the same kernel output can be
    mounted under two input paths (slug and pre-rename slug), and that must
    not look like a second cache part.

    The malformed/truncated-row guard (missing StudyInstanceUID/shard/row,
    or a shard field that will not parse as int - a build that died
    mid-write leaves a partial last line) runs before int(r["shard"]) so a
    truncated last row is skipped and counted, never raising. The shard ->
    part map is built from shard_seen_in (the part whose index actually
    listed the shard), asserting cache_{shard:03d}.npy exists there - no
    separate existence scan over every part."""
    parts, parts_rows = [], []
    for p in CACHE_PARTS:
        rows = _read_index_rows(p)
        dup = next((q for q, qr in zip(parts, parts_rows) if qr == rows), None)
        if dup is not None:
            print(f"cache part {p} has an identical cache_index.csv to "
                  f"{dup}; collapsing into one part", flush=True)
            continue
        parts.append(p)
        parts_rows.append(rows)

    rows_out = {}
    seen_uids = set()
    shard_seen_in = {}  # shard -> part dir that first listed it (dup check)
    n_empty = n_bad = 0
    for part, rows in zip(parts, parts_rows):
        for r in rows:
            uid = r.get("StudyInstanceUID")
            shard_field = r.get("shard")
            row_field = r.get("row")
            if not uid or shard_field in (None, "") or row_field in (None, ""):
                n_bad += 1  # truncated row (build died mid-write)
                continue
            try:
                shard = int(shard_field)
            except (TypeError, ValueError):
                n_bad += 1  # malformed shard field
                continue
            prev = shard_seen_in.setdefault(shard, part)
            assert prev == part, (
                f"shard {shard} appears in both cache parts {prev} and {part}")
            assert uid not in seen_uids, (
                f"study {uid} appears in more than one cache part")
            seen_uids.add(uid)
            if "filled_slots" in r:
                fs = r["filled_slots"]
                if fs in (None, ""):
                    n_bad += 1  # truncated row (build died mid-write)
                    continue
                if int(float(fs)) == 0:
                    n_empty += 1  # black volume with real labels: never train on it
                    continue
            rows_out[uid] = (shard, int(row_field))

    shard_dir = {}
    for shard, part in shard_seen_in.items():
        path = os.path.join(part, f"cache_{shard:03d}.npy")
        assert os.path.exists(path), (
            f"shard {shard} listed in {part}'s cache_index.csv but {path} "
            "does not exist")
        shard_dir[shard] = part
    print(f"index: {len(parts)} cache part(s), {len(rows_out)} studies kept, "
          f"{n_empty} all-zero dropped, {n_bad} malformed rows skipped", flush=True)
    return rows_out, shard_dir


def load_labels(path=None):
    """{uid: label vector}, {uid: weight vector}. Weight columns are named
    w_<label>; an older label file without all 12 of them trains as if every
    cell weighed 1.0 (unweighted BCE, unchanged behaviour)."""
    if path is None:
        path = LABELS_CSV
    lab, wgt = {}, {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        wcols = [f"w_{c}" for c in LABELS]
        has_weights = reader.fieldnames is not None and all(
            c in reader.fieldnames for c in wcols)
        for r in reader:
            lab[r["StudyInstanceUID"]] = np.array(
                [float(r[c]) for c in LABELS], dtype=np.float32)
            if has_weights:
                wgt[r["StudyInstanceUID"]] = np.array(
                    [float(r[c]) for c in wcols], dtype=np.float32)
            else:
                wgt[r["StudyInstanceUID"]] = np.ones(len(LABELS), dtype=np.float32)
    print(f"labels: {path} "
          f"({'weighted' if has_weights else 'unweighted, no w_ columns found'})",
          flush=True)
    return lab, wgt


def make_fold_of(y_split):
    """fold_of: per-study fold index, stratified on y_split's positive-label
    count (capped at 6) and shuffled deterministically per stratum."""
    strat = np.minimum((y_split >= 0.5).sum(axis=1), 6)
    rng = np.random.default_rng(SEED)
    fold_of = np.empty(len(y_split), dtype=int)
    for s in np.unique(strat):
        idx = np.flatnonzero(strat == s)
        rng.shuffle(idx)
        fold_of[idx] = np.arange(idx.size) % N_FOLDS
    return fold_of


class Shards:
    """Memory-maps cache_XXX.npy on demand, resolving each shard number to
    whichever cache part directory holds it (the map load_index() returns)."""

    def __init__(self, shard_dir):
        self.shard_dir = shard_dir
        self._maps = {}

    def get(self, shard, row):
        if shard not in self._maps:
            self._maps[shard] = np.load(
                os.path.join(self.shard_dir[shard], f"cache_{shard:03d}.npy"),
                mmap_mode="r")
        return self._maps[shard][row]


class StudyDataset(Dataset):
    def __init__(self, uids, index, labels, weights, shards, train):
        self.uids = uids
        self.index = index
        self.labels = labels
        self.weights = weights
        self.shards = shards
        self.train = train

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        uid = self.uids[i]
        shard, row = self.index[uid]
        vol = np.asarray(self.shards.get(shard, row))  # (N_SLOTS_CFG,N_SLICES,size,size) uint8
        windows, slots = [], []
        for s in range(N_SLOTS_CFG):
            groups = [g for g in range(N_GROUPS)
                     if vol[s, g * WINDOW:(g + 1) * WINDOW].any()]
            if not groups:
                continue
            if self.train and WINDOW_STARTS == "random":
                # any non-empty WINDOW-slice start, not only the aligned
                # groups: the same window population dense stride-1
                # inference scores (minus the zero-padded edge windows)
                starts = [i for i in range(vol.shape[1] - WINDOW + 1)
                          if vol[s, i:i + WINDOW].any()]
                k = min(WINDOWS_PER_SLOT, len(starts))
                chosen = np.random.choice(starts, size=k, replace=False)
            elif self.train:
                k = min(WINDOWS_PER_SLOT, len(groups))
                chosen = np.random.choice(groups, size=k, replace=False) * WINDOW
            else:
                chosen = np.asarray(groups) * WINDOW
            for i in chosen:
                windows.append(vol[s, i:i + WINDOW])
                slots.append(s)
        if not windows:  # fully-empty study: one zero window, slot 0
            windows, slots = [np.zeros_like(vol[0, :WINDOW])], [0]
        x = torch.from_numpy(np.stack(windows)).float() / 255.0  # (N,3,size,size)
        return (x, torch.tensor(slots), torch.from_numpy(self.labels[uid]),
                torch.from_numpy(self.weights[uid]))


def collate(batch):
    xs, slots, ys, ws, owner = [], [], [], [], []
    for i, (x, s, y, w) in enumerate(batch):
        xs.append(x)
        slots.append(s)
        ys.append(y)
        ws.append(w)
        owner.append(torch.full((len(s),), i, dtype=torch.long))
    n_win = sum(len(s) for s in slots)
    n_pad = (-n_win) % WINDOW_BUCKET  # see WINDOW_BUCKET: fixed shapes for cuDNN
    if n_pad:
        xs.append(torch.zeros((n_pad,) + tuple(xs[0].shape[1:]), dtype=xs[0].dtype))
        slots.append(torch.zeros(n_pad, dtype=slots[0].dtype))
        owner.append(torch.full((n_pad,), -1, dtype=torch.long))
    return (torch.cat(xs), torch.cat(slots), torch.cat(owner),
            torch.stack(ys), torch.stack(ws), len(batch))


def macro_auc(y_true, y_prob):
    """Per-label AUCs keyed by name; single-class labels are omitted from
    both the dict and the macro mean (so names never misalign)."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:  # sklearn's compiled extension blocked (Smart App Control) or not installed
        from auc import roc_auc_score
    aucs = {}
    for c, name in enumerate(LABELS):
        t = (y_true[:, c] >= 0.5).astype(int)
        if t.min() == t.max():
            continue
        aucs[name] = roc_auc_score(t, y_prob[:, c])
    return float(np.mean(list(aucs.values()))), aucs


def pos_weight_from_labels(y):
    """Per-label BCE pos_weight from a (N,12) label matrix: clip((1-p)/p, 1,
    10), where p is the positive prevalence (labels binarised at 0.5). Pure
    (no torch), so it can be unit-tested against a toy label matrix. Used
    when the arm's `pos_weight` flag is on (coatnet2_v5)."""
    p = (y >= 0.5).mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = (1.0 - p) / p
    return np.clip(raw, 1.0, 10.0).astype(np.float32)


def seed_worker(worker_id):
    # torch reseeds itself per worker; numpy needs this or both workers
    # draw the identical augmentation stream
    np.random.seed(torch.initial_seed() % 2**31 + worker_id)


def load_gold():
    gold = {}
    with open(GOLD_CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            gold[r["StudyInstanceUID"]] = np.array(
                [float(r[c]) for c in LABELS], dtype=np.float32)
    return gold


def predict_studies(models, uids, index, shards, dev, use_amp, tta=False,
                    window=None):
    """Ensemble probs over `uids` via the shared model.predict_volume - the
    exact code path infer_submit.py ships, so the TTA verdict is honest."""
    from model import predict_volume
    out = np.zeros((len(uids), len(LABELS)), dtype=np.float32)
    for i, u in enumerate(uids):
        shard, row = index[u]
        vol = np.asarray(shards.get(shard, row))
        out[i] = predict_volume(models, vol, dev, use_amp, tta=tta, window=window)
    return out


def gold_window_probs(models, uids, index, shards, dev, use_amp, window, tta):
    """Per-window, per-fold sigmoid probs for `uids`, each window scored
    ALONE (model.predict_study's win_probs: owner=arange, no cross-window
    attention), plus the study-level per-fold probs predict_study's p_attn
    actually ships, for the next task's window-pooling analysis. Calls
    model.predict_study once per fold per study (one backbone pass) rather
    than reimplementing it, so this cannot drift from what infer_submit.py
    does.

    Returns (probs, slots, study_probs): probs is (n_uids, n_folds,
    max_windows, 12) NaN-padded, slots is (n_uids, max_windows) -1-padded,
    study_probs is (n_uids, n_folds, 12)."""
    from model import predict_study
    n_folds, n_labels = len(models), len(LABELS)
    per_study, per_slots, per_study_probs, max_w = [], [], [], 0
    for u in uids:
        shard, row = index[u]
        vol = np.asarray(shards.get(shard, row))
        sp = np.zeros((n_folds, n_labels), dtype=np.float32)
        per_fold_windows = []
        s = None
        for fi, m in enumerate(models):
            p_attn, wp, s = predict_study([m], vol, dev, use_amp, tta=tta,
                                          window=window)
            sp[fi] = p_attn
            per_fold_windows.append(wp[0])  # (n_windows, 12): the one model
        p = np.stack(per_fold_windows)  # (n_folds, n_windows, 12)
        max_w = max(max_w, p.shape[1])
        per_study.append(p)
        per_slots.append(s)
        per_study_probs.append(sp)
    probs_arr = np.full((len(uids), n_folds, max_w, n_labels), np.nan, dtype=np.float32)
    slots_arr = np.full((len(uids), max_w), -1, dtype=np.int64)
    for i, (p, s) in enumerate(zip(per_study, per_slots)):
        probs_arr[i, :, :p.shape[1], :] = p
        slots_arr[i, :s.shape[0]] = s
    study_probs_arr = np.stack(per_study_probs)
    return probs_arr, slots_arr, study_probs_arr


def make_scheduler(opt, steps, warm_steps):
    """`warm_steps == 0` (today's every-arm default) is the plain cosine
    schedule, unchanged. Otherwise a single LambdaLR ramps linearly from 0 to
    the group's base lr over `warm_steps` steps, then cosine-decays over the
    remaining steps - one schedule object so both param groups share the
    same warmup/decay shape, just scaled by their own base lr."""
    if warm_steps == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    def f(s):
        if s < warm_steps:
            return (s + 1) / warm_steps
        return 0.5 * (1 + math.cos(
            math.pi * (s - warm_steps) / max(1, steps - warm_steps)))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def main():
    global EPOCHS  # SMOKE mode reassigns it to 1 below
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    index, shard_dir = load_index()
    labels, weights = load_labels()
    gold = load_gold()
    # gold-58 never trains and never enters a fold: it is the honest holdout
    gold_uids = sorted(u for u in gold if u in index)
    gold_missing = sorted(u for u in gold if u not in index)
    if gold_missing:  # the holdout metric silently shrinking is worse than noise
        print(f"WARN {len(gold_missing)} gold studies absent from the cache "
              f"index (all-zero or unbuilt): {gold_missing}", flush=True)
    uids = sorted(u for u in index if u in labels and u not in gold)
    print(f"{len(uids)} train studies (gold {len(gold_uids)} held out), "
          f"labels: {LABEL_FILE}", flush=True)

    y_all = np.stack([labels[u] for u in uids])
    # pos_weight (coatnet2_v5 only; every other arm trains with POS_WEIGHT
    # False, so pos_weight_t stays None and the BCE call below is unchanged)
    pos_weight_vals = pos_weight_from_labels(y_all) if POS_WEIGHT else None
    pos_weight_t = (torch.from_numpy(pos_weight_vals).to(dev)
                    if pos_weight_vals is not None else None)
    if SPLIT_LABELS_CSV == LABELS_CSV:
        y_split = y_all
    else:
        split_labels, _ = load_labels(SPLIT_LABELS_CSV)
        y_split = np.stack([split_labels[u] for u in uids])
    fold_of = make_fold_of(y_split)

    shards = Shards(shard_dir)
    oof = np.zeros_like(y_all)
    oof_written = np.zeros(len(y_all), dtype=bool)
    fold_models = []
    epochs_done, stopped_early, sec_per_epoch = {}, {}, {}
    fold_aucs_last, fold_aucs_swa = {}, {}
    # SMOKE: 1 fold, 1 epoch, a capped train/val set - a throughput probe,
    # not a real run. N_FOLDS (the split) is never touched.
    folds = FOLDS_TO_TRAIN[:1] if SMOKE else FOLDS_TO_TRAIN
    if SMOKE:
        EPOCHS = 1
    # full_data (coatnet2_full): one model, no fold holdout - the loop below
    # runs exactly once, as fold 0, regardless of the arm's folds_to_train.
    if FULL_DATA:
        folds = (0,)
    for fold in folds:
        if FULL_DATA:
            tr, va = uids, []  # every non-gold study trains; no validation split
        else:
            tr = [u for u, f in zip(uids, fold_of) if f != fold]
            va = [u for u, f in zip(uids, fold_of) if f == fold]
        if SMOKE:
            tr = tr[:SMOKE]
            va = va[:100]
        model = Model(img_size=CACHE_CFG["size"], arch=ARCH,
                     grad_ckpt=GRAD_CKPT, unfreeze=UNFREEZE,
                     drop_path=DROP_PATH, n_slots=N_SLOTS_CFG,
                     spatial_pool=SPATIAL_POOL).to(dev)
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        # split optimizer params by name prefix, not attribute, so none go untrained -- see docs/code-notes.md "optimizer param groups by prefix"
        backbone_params = [p for n, p in model.named_parameters()
                           if p.requires_grad and n.startswith("backbone.")]
        head_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith("backbone.")]
        n_trainable = sum(1 for _, p in model.named_parameters() if p.requires_grad)
        assert len(backbone_params) + len(head_params) == n_trainable, (
            f"optimizer param groups ({len(backbone_params)} backbone + "
            f"{len(head_params)} head) do not cover all {n_trainable} "
            "trainable parameters")
        print(f"optimizer groups: {len(backbone_params)} backbone params "
              f"(lr={LR_BACKBONE}), {len(head_params)} head params "
              f"(lr={LR_HEAD}), {n_trainable} trainable total", flush=True)
        params = [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head_params, "lr": LR_HEAD},
        ]
        opt = torch.optim.AdamW(params, weight_decay=1e-4)
        steps_per_epoch = math.ceil(len(tr) / BATCH_STUDIES)
        steps = steps_per_epoch * EPOCHS
        warm_steps = round(WARMUP_EPOCHS * steps_per_epoch)
        sched = make_scheduler(opt, steps, warm_steps)
        use_amp = dev == "cuda"  # autocast on CPU is ~9x slower, not faster
        torch.backends.cudnn.benchmark = CUDNN_BENCHMARK and dev == "cuda"
        scaler = torch.amp.GradScaler(dev, enabled=use_amp and AMP == "fp16")

        dl_tr = DataLoader(StudyDataset(tr, index, labels, weights, shards, True),
                           batch_size=BATCH_STUDIES, shuffle=True,
                           num_workers=NUM_WORKERS, collate_fn=collate,
                           worker_init_fn=seed_worker, drop_last=True)
        dl_va = (None if FULL_DATA else
                DataLoader(StudyDataset(va, index, labels, weights, shards, False),
                          batch_size=4, shuffle=False,
                          num_workers=NUM_WORKERS, collate_fn=collate))

        first_step = True  # gradient sanity check, first optimizer step only
        ep_secs = []
        stopped_this_fold = False
        swa_states = []  # last SWA_LAST epoch state_dicts (CPU), this fold only
        for ep in range(EPOCHS):
            model.train()
            tot = 0.0
            ep_t0 = time.time()
            for x, slots, owner, y, w, n in dl_tr:
                x, slots, owner, y, w = (
                    t.to(dev) for t in (x, slots, owner, y, w))
                with torch.amp.autocast(dev, dtype=AMP_DTYPE, enabled=use_amp):
                    logits = model(x, slots, owner, n)
                    loss = (F.binary_cross_entropy_with_logits(
                        logits, y, pos_weight=pos_weight_t, reduction="none")
                            * w).sum() / w.sum().clamp_min(1)
                opt.zero_grad()
                scaler.scale(loss).backward()
                if first_step:
                    n_missing = sum(1 for p in backbone_params if p.grad is None)
                    assert n_missing == 0, (
                        f"no gradient reached {n_missing} trainable backbone "
                        f"params (unfreeze={UNFREEZE!r}) - "
                        "checkpointing/freezing trap")
                    first_step = False
                scaler.step(opt)
                scaler.update()
                sched.step()
                tot += loss.item()
            ep_sec = time.time() - ep_t0
            ep_secs.append(ep_sec)
            print(f"fold {fold} ep {ep}: loss {tot/len(dl_tr):.4f} "
                  f"({ep_sec:.0f}s)", flush=True)
            if SWA_LAST > 0:
                swa_states.append({k: v.detach().cpu().clone()
                                   for k, v in model.state_dict().items()})
                if len(swa_states) > SWA_LAST:
                    swa_states.pop(0)
            if SMOKE:
                dt = ep_sec
                peak = (torch.cuda.max_memory_allocated() / 2**30
                        if dev == "cuda" else 0.0)
                print(f"SMOKE: {len(tr)} studies in {dt:.0f}s = "
                      f"{len(tr)/dt:.2f} studies/s, peak {peak:.2f} GiB",
                      flush=True)
            if (ep + 1 < EPOCHS and
                    (time.time() - T_START) + 1.15 * ep_sec > MAX_HOURS * 3600):
                print(f"MAX_HOURS {MAX_HOURS} reached after epoch {ep}: "
                      f"stopping fold {fold} early", flush=True)
                stopped_this_fold = True
                break
        epochs_done[fold] = ep + 1
        stopped_early[fold] = stopped_this_fold
        sec_per_epoch[fold] = float(np.mean(ep_secs)) if ep_secs else 0.0

        def eval_va():
            model.eval()
            out = []
            with torch.no_grad():
                for x, slots, owner, y, w, n in dl_va:
                    x, slots, owner = (t.to(dev) for t in (x, slots, owner))
                    with torch.amp.autocast(dev, dtype=AMP_DTYPE, enabled=use_amp):
                        p = torch.sigmoid(model(x, slots, owner, n))
                    out.append(p.float().cpu().numpy())
            return np.concatenate(out)

        if FULL_DATA:
            # no validation split for a full-data run: no eval_va() call, no
            # per-fold AUC - fold_auc_last/fold_auc_swa carry null below
            fold_auc_last = None
            last_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            probs = eval_va()
            # va_idx matches probs even under SMOKE's va[:100] truncation -- see docs/code-notes.md "OOF index alignment under SMOKE"
            va_idx = np.flatnonzero(fold_of == fold)[:len(va)]
            fold_auc_last, _ = macro_auc(y_all[va_idx], probs)
            print(f"fold {fold} macro AUC vs LLM labels (last): "
                  f"{fold_auc_last:.4f}", flush=True)
            last_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        # SWA: average the last SWA_LAST epoch checkpoints, decided a priori -- see docs/code-notes.md "SWA checkpoint averaging"
        if SWA_LAST > 0 and swa_states:
            swa_state = {k: torch.stack([sd[k].float() for sd in swa_states])
                        .mean(dim=0).to(last_state[k].dtype)
                        for k in last_state}
            if FULL_DATA:
                fold_auc_swa = None
            else:
                model.load_state_dict(swa_state)
                model.to(dev)
                probs_swa = eval_va()
                fold_auc_swa, _ = macro_auc(y_all[va_idx], probs_swa)
                print(f"fold {fold} macro AUC vs LLM labels (SWA-{SWA_LAST}): "
                      f"{fold_auc_swa:.4f}", flush=True)
        else:
            # a full-data run never evaluates, so it has no `probs` (coatnet2_rsfull is the first full_data arm
            # without SWA)
            swa_state, fold_auc_swa = last_state, fold_auc_last
            probs_swa = None if FULL_DATA else probs

        if not FULL_DATA:
            oof[va_idx] = probs_swa
            oof_written[va_idx] = True
        fold_aucs_last[fold] = (None if FULL_DATA
                                else round(float(fold_auc_last), 4))
        fold_aucs_swa[fold] = (None if FULL_DATA
                               else round(float(fold_auc_swa), 4))
        # the SWA file is what ships (decided a priori); `_last` is kept
        # alongside for inspection, never loaded by infer_submit.py
        torch.save(swa_state, f"{OUT}/{FAMILY}_fold{fold}.pt")
        if SWA_LAST > 0:
            # only write the _last checkpoint when it differs from the SWA one -- see docs/code-notes.md "SWA _last checkpoint file"
            torch.save(last_state, f"{OUT}/{FAMILY}_fold{fold}_last.pt")
        model.load_state_dict(swa_state)
        model.to(dev)
        model.eval()
        fold_models.append(model)

    # OOF scoring uses oof_written, not fold membership, so SMOKE truncation is safe -- see docs/code-notes.md "OOF row masking"
    if FULL_DATA:
        # every study trained (no fold holdout), so there is no OOF row to
        # score - printing a number computed from zero rows would be a lie
        print("OOF macro AUC vs LLM labels: skipped (full-data run, "
              "no validation folds)", flush=True)
        auc, per = None, {}
    else:
        trained_mask = oof_written
        auc, per = macro_auc(y_all[trained_mask], oof[trained_mask])
        print(f"OOF macro AUC vs LLM labels: {auc:.4f} "
              f"({len(per)}/{len(LABELS)} labels scored)", flush=True)
        for name, a in per.items():
            print(f"  {name}: {a:.4f}", flush=True)

    # run_config.json is written early, then re-saved as gold results arrive -- see docs/code-notes.md "run_config.json write order"
    w_all = np.stack([weights[u] for u in uids])  # same population as `prior`
    weight_mean = {name: round(float(v), 4)
                   for name, v in zip(LABELS, w_all.mean(axis=0))}
    # full_data reports n_folds=1 so infer_submit's fold-count assertion passes -- see docs/code-notes.md "full_data n_folds_cfg contract"
    n_folds_cfg = 1 if FULL_DATA else len(FOLDS_TO_TRAIN)
    folds_trained_cfg = [] if FULL_DATA else list(FOLDS_TO_TRAIN)
    run_cfg = {**CACHE_CFG, "n_slices": N_SLICES, "n_slots": N_SLOTS_CFG,
               "label_file": LABEL_FILE, "split_label_file": SPLIT_LABEL_FILE,
               "epochs": EPOCHS, "n_folds": n_folds_cfg,
               "split_folds": N_FOLDS, "folds_trained": folds_trained_cfg,
               "trained_on": "all" if FULL_DATA else "folds",
               "window": WINDOW, "windows_per_slot": WINDOWS_PER_SLOT,
               "window_starts": WINDOW_STARTS, "train_seed": TRAIN_SEED,
               "spatial_pool": SPATIAL_POOL,
               "arch": ARCH if ARCH is not None else _model_mod.ARCH,
               "family": FAMILY, "batch_studies": BATCH_STUDIES,
               "lr_backbone": LR_BACKBONE, "lr_head": LR_HEAD,
               "grad_ckpt": GRAD_CKPT, "smoke": SMOKE,
               "arm": ARM, "unfreeze": UNFREEZE, "warmup_epochs": WARMUP_EPOCHS,
               "drop_path": DROP_PATH, "sweep_lr": SWEEP_LR,
               "sweep_epochs": SWEEP_EPOCHS, "max_hours": MAX_HOURS, "amp": AMP,
               "epochs_done": epochs_done, "sec_per_epoch": sec_per_epoch,
               "stopped_early": stopped_early,
               "pos_weight": POS_WEIGHT, "swa_last": SWA_LAST,
               "fold_auc_last": fold_aucs_last, "fold_auc_swa": fold_aucs_swa,
               "prior": [round(float(v), 4) for v in y_all.mean(axis=0)],
               "oof_llm_auc": None if FULL_DATA else round(auc, 4),
               "gold_n": len(gold_uids), "gold58_auc": None,
               "tta_gold58_auc": None, "use_tta": False,
               "stage": "B", "weight_mean": weight_mean}

    def save_run_config():
        with open(f"{OUT}/run_config.json", "w") as fh:
            json.dump(run_cfg, fh)

    np.save(f"{OUT}/oof.npy", oof)
    with open(f"{OUT}/fold_of.json", "w") as fh:
        json.dump({u: int(f) for u, f in zip(uids, fold_of)}, fh)
    save_run_config()

    # the honest number: N_FOLDS-fold ensemble vs the 58 expert-labeled studies
    use_amp = dev == "cuda"
    t0 = time.time() if SMOKE else None
    gold_probs = predict_studies(fold_models, gold_uids, index, shards,
                                 dev, use_amp, window=WINDOW)
    if SMOKE:
        print(f"SMOKE gold pass: {len(gold_uids)} studies in "
              f"{time.time()-t0:.1f}s", flush=True)
    gold_y = np.stack([gold[u] for u in gold_uids])
    gauc, gper = macro_auc(gold_y, gold_probs)
    print(f"GOLD-{len(gold_uids)} macro AUC (ensemble): {gauc:.4f} "
          f"({len(gper)}/{len(LABELS)} labels scored)", flush=True)
    for name, a in gper.items():
        print(f"  {name}: {a:.4f}", flush=True)
    run_cfg["gold58_auc"] = round(gauc, 4)
    save_run_config()

    # hflip TTA is decided by measurement, not assumed: infer reads use_tta
    tta_probs = predict_studies(fold_models, gold_uids, index, shards,
                                dev, use_amp, tta=True, window=WINDOW)
    tauc, _ = macro_auc(gold_y, tta_probs)
    use_tta = bool(tauc >= gauc)
    print(f"GOLD-{len(gold_uids)} macro AUC with hflip TTA: {tauc:.4f} "
          f"({'use' if use_tta else 'skip'} TTA at inference)", flush=True)
    run_cfg["tta_gold58_auc"] = round(tauc, 4)
    run_cfg["use_tta"] = use_tta
    save_run_config()

    # gold window probs are scored with the TTA verdict just decided, so they match -- see docs/code-notes.md "gold window-pooling probs"
    gw_probs, gw_slots, gw_study_probs = gold_window_probs(
        fold_models, gold_uids, index, shards, dev, use_amp, WINDOW, use_tta)
    np.savez(f"{OUT}/gold_windows.npz", probs=gw_probs, slots=gw_slots,
             study_probs=gw_study_probs, uids=np.array(gold_uids), y=gold_y,
             tta=use_tta)


def run_dual():
    """Dual-GPU launcher: one child process per visible GPU, each running
    main() for a single fold of FOLDS_TO_TRAIN into OUT/fold<k>/ (merge the
    fold dirs afterwards with src/merge_fold_runs.py). Fewer than two GPUs
    falls back to main() in-process. Under SMOKE every GPU still gets one
    fold so the probe measures the concurrent footprint. Host RSS per child
    is reported once a minute; the peak is printed at the end."""
    import subprocess
    import threading
    n_slots = int(os.environ.get("RSNA_DUAL_SLOTS", torch.cuda.device_count()))
    if n_slots < 2:
        print(f"DUAL_GPU: {n_slots} GPU visible, running main() in-process",
              flush=True)
        main()
        return
    script = os.path.abspath(globals().get("__file__") or sys.argv[0])
    if not os.path.isfile(script):
        raise FileNotFoundError(f"DUAL_GPU: cannot relaunch {script!r}")
    folds = list(FOLDS_TO_TRAIN)
    if SMOKE:
        folds = folds[:n_slots]
    print(f"DUAL_GPU: {n_slots} slots, folds {folds}, script {script}",
          flush=True)
    try:
        import psutil
    except ImportError:  # local dry-runs without psutil still exercise the launcher
        psutil = None

    running = {}   # slot -> (fold, Popen)
    pending = list(folds)
    failed = []
    rss_peak = {}

    def pump(fold, proc):
        for line in proc.stdout:
            print(f"[fold {fold}] {line}", end="", flush=True)

    def launch(slot, fold):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(slot),
                   RSNA_CHILD_FOLD=str(fold), PYTHONUNBUFFERED="1")
        proc = subprocess.Popen([sys.executable, script], env=env,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        threading.Thread(target=pump, args=(fold, proc), daemon=True).start()
        running[slot] = (fold, proc)
        print(f"DUAL_GPU: fold {fold} -> GPU {slot} (pid {proc.pid})", flush=True)

    def report():
        parts, total = [], 0
        for slot, (fold, proc) in running.items():
            try:
                p = psutil.Process(proc.pid)
                rss = p.memory_info().rss + sum(
                    c.memory_info().rss for c in p.children(recursive=True))
            except psutil.Error:
                continue
            rss_peak[fold] = max(rss_peak.get(fold, 0), rss)
            total += rss
            parts.append(f"fold{fold}={rss / 2**30:.1f}")
        vm = psutil.virtual_memory()
        print(f"DUAL_GPU rss GiB: {' '.join(parts)} total={total / 2**30:.1f} "
              f"host used={vm.used / 2**30:.1f} avail={vm.available / 2**30:.1f}",
              flush=True)

    last_report = time.time()
    while pending or running:
        for slot in range(n_slots):
            if slot not in running and pending:
                launch(slot, pending.pop(0))
        time.sleep(5)
        for slot, (fold, proc) in list(running.items()):
            rc = proc.poll()
            if rc is not None:
                del running[slot]
                print(f"DUAL_GPU: fold {fold} finished rc={rc} at "
                      f"{(time.time() - T_START) / 60:.1f} min", flush=True)
                if rc != 0:
                    failed.append(fold)
        if psutil is not None and time.time() - last_report >= 60:
            report()
            last_report = time.time()
    if rss_peak:
        print("DUAL_GPU peak rss GiB: " + " ".join(
            f"fold{f}={v / 2**30:.1f}" for f, v in sorted(rss_peak.items())),
            flush=True)
    if failed:
        raise SystemExit(f"DUAL_GPU: folds {failed} failed")


if __name__ == "__main__":
    if DUAL_GPU and "RSNA_CHILD_FOLD" not in os.environ:
        run_dual()
    else:
        main()
