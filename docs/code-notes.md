# Kernel code notes

These are the design notes moved out of the kernel files on 2026-09-14, so the code pages stay readable. Each `### ` heading below is referenced from a one-line comment in the corresponding source file.

## notebooks/train/train_dinov2.py

### train_dinov2.py — training pipeline overview
Anchor: import csv

RSNA Knee: train DINOv2-S 2.5D on the pixel cache. Kaggle GPU (T4), internet ON
(downloads timm weights; this is NOT the submission kernel).

Attach: rsna-67knee-src, the cache kernel outputs (v4 ships as two parts,
rsna-knee-build-cache + rsna-knee-build-cache-b, merged by shard number), and
patimakornsrinamen/rsna-knee-consensus-labels. The competition is NOT attached.

Per study the cache holds (6 slots, n_slices slices, size, size) uint8 (v4:
15 slices). Each slot's slices are grouped into non-overlapping WINDOW-sized
windows, one 3-channel backbone input per window. Training samples
WINDOWS_PER_SLOT random non-empty windows per filled slot per step; eval uses
every non-empty window. Window embeddings from a shared ViT-S/14 encoder are
projected, a learned slot embedding added, then each of the 12 labels attends
separately over the study's windows before a per-label linear readout.
Labels: consensus_v1.csv soft probabilities, per-cell weighted BCE loss.
CV: 5 folds, stratified by rounded count of positive findings.

### train_dinov2.py — arm and SMOKE switches
Anchor: ARM = "dinov2"

Per-arm recipe. `arch` is the timm name to pass to Model (None picks
model.py's DINOv2 default); `family` names the checkpoint files so the two
arms' weights never collide in OUT. Selected arm and SMOKE (studies-per-fold
cap for a throughput probe) are the two switches a staged Kaggle copy flips
before push; the committed values here are the shipping default.

### train_dinov2.py — sweep and MAX_HOURS constants
Anchor: SWEEP_LR = 1e-4

Round 6 additions: SWEEP_LR/SWEEP_EPOCHS parameterize the coatnet2 optimizer
sweep arm; MAX_HOURS bounds a fold's wall clock (Kaggle session cap) - the
guard in main() checks it after every epoch. Defined before ARMS so the
sweep entry below can reference them.

### train_dinov2.py — coatnet2_v5 arm (cache v5)
Anchor: "coatnet2_v5": dict(arch="coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k",

Cache v5 (5-slot raptor5 layout, 18/14/12/8/12 slices/slot): otherwise
the round-6 coatnet2 winner unchanged (lr 5e-5, warmup 1, stem+stage0
frozen); pos_weight turns on the label-prevalence-derived BCE weight,
swa_last=3 averages the last 3 epoch checkpoints for the shipped file
(both decided a priori, not selected on gold-58).

### train_dinov2.py — coatnet2_full arm (round 7)
Anchor: "coatnet2_full": dict(arch="coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k",

Round 7: one model trained on ALL 4,349 non-gold studies (no fold
holdout), copying what the Raptor arm's author does - 25% more data
than any single fold, one forward pass at inference instead of five.
Otherwise the round-6 coatnet2 recipe unchanged; SWA-3 checkpoint
selection stays a priori (never best-epoch-by-gold-58); pos_weight
off (that combination was never validated for this arm).

### train_dinov2.py — cache part dedupe by real path
Anchor: return sorted({os.path.realpath(d) for d in found})

dedupe by real path: Kaggle mount layouts can expose the same kernel
output under more than one input path (e.g. slug and pre-rename slug);
that must not look like two cache parts to the merge below.

### train_dinov2.py — n_slots derivation
Anchor: N_SLOTS_CFG = n_slots_for(CACHE_CFG)

n_slots is not itself a recipe key (see src/preprocess._RECIPE_KEYS); it is
derived from slot_layout via n_slots_for so a v5 (raptor5, 5 slots) cache
config drives the right embedding/loop size without a new recipe field.

### train_dinov2.py — n_slots consistency guard
Anchor: assert CACHE_CFG.get("n_slots", N_SLOTS_CFG) == N_SLOTS_CFG, (

the cache also serialises n_slots; slot_layout is the source of truth,
so a config where the two disagree is a hand-edit that would otherwise
train the wrong slot-embedding size in silence

### train_dinov2.py — optimizer param groups by prefix
Anchor: backbone_params = [p for n, p in model.named_parameters()

Named by prefix, not by attribute, so a head rewrite (new module
names) cannot silently drop parameters into neither group and
leave them untrained while the run still looks healthy.

### train_dinov2.py — OOF index alignment under SMOKE
Anchor: va_idx = np.flatnonzero(fold_of == fold)[:len(va)]

same order va was built in; sliced to len(va) so SMOKE's va[:100]
truncation (fewer rows than the fold actually holds) still lines
up with `probs` instead of raising a shape mismatch on the write

### train_dinov2.py — SWA checkpoint averaging
Anchor: if SWA_LAST > 0 and swa_states:

SWA: average of the last SWA_LAST epoch checkpoints, decided a
priori (not selected on gold-58). SWA_LAST == 0 (every arm but
coatnet2_v5/coatnet2_full) takes the else branch untouched: swa_state
is last_state and probs_swa is probs (no reload, no second forward
pass), so oof / the saved checkpoint / fold_models stay
byte-identical to before this feature existed. A full-data run
still averages the epoch checkpoints but has no validation set to
re-evaluate them against.

### train_dinov2.py — SWA _last checkpoint file
Anchor: torch.save(last_state, f"{OUT}/{FAMILY}_fold{fold}_last.pt")

only worth a second 293 MB file when it differs from the SWA
one; with SWA_LAST == 0 last_state IS swa_state, so writing it
again would just duplicate every fold checkpoint in the output

### train_dinov2.py — OOF row masking
Anchor: if FULL_DATA:

restricted to studies whose oof row was actually written this run.
Masking on fold membership (FOLDS_TO_TRAIN or even just `folds`) is not
enough under SMOKE: `va` is also truncated to 100 rows within a trained
fold, so most of that fold's rows stay at oof's zero-init too.

### train_dinov2.py — run_config.json write order
Anchor: w_all = np.stack([weights[u] for u in uids])  # same population as `prior`

run_config.json: the recipe (from cache_config) plus everything infer
needs to trust these weights. Written BEFORE the gold eval so the
weights stay usable even if that block raises, then re-saved as the
gold numbers arrive. `prior` is the fallback row for undecodable test
studies; `use_tta` is the one decision infer reads (decided on the
unrounded AUCs, so the log and the shipped behaviour cannot disagree).

### train_dinov2.py — full_data n_folds_cfg contract
Anchor: n_folds_cfg = 1 if FULL_DATA else len(FOLDS_TO_TRAIN)

full_data (coatnet2_full): one model trained on every non-gold study,
no fold holdout - n_folds/folds_trained read as "1 fold, none of them
a CV split" so infer_submit.load_family's `len(weight_files) ==
cfg["n_folds"]` assertion (fold0.pt only) is satisfied unchanged.

### train_dinov2.py — gold window-pooling probs
Anchor: gw_probs, gw_slots, gw_study_probs = gold_window_probs(

per-window, per-fold gold-58 probs for the next task's window-pooling
analysis - scored with the TTA verdict just decided above, so these
numbers match what inference will actually ship

### train_dinov2.py — dual-GPU fold launcher
Anchor: DUAL_GPU = False

A Kaggle `NvidiaTeslaT4` session exposes two T4s, and the weekly GPU quota is
charged per session hour, not per GPU. With `DUAL_GPU = True` the script does
not train in-process: `run_dual()` relaunches itself once per visible GPU
(`CUDA_VISIBLE_DEVICES=<slot>`, `RSNA_CHILD_FOLD=<k>`), each child pinned to a
single fold of `FOLDS_TO_TRAIN` and writing into `OUT/fold<k>/`. Folds beyond the
slot count queue until a slot frees. The per-fold directories are merged with
`src/merge_fold_runs.py`, exactly like the one-fold-per-kernel outputs of rounds
5–7. Fewer than two GPUs (a P100 session, a local dry-run) falls back to
`main()` in-process, so the flag is safe to leave on. Under `SMOKE` every slot
still receives one fold, so the probe measures the concurrent host-RAM and GPU
footprint; the launcher reports each child's RSS (with DataLoader workers) every
minute and the peak at exit. Dry-run drivers redirect the relaunch target by
setting the module's `__file__` (see the session scratchpad `dryrun_dual.py`).
Decided 2026-09-15 after the resource audit in `docs/team/resources.md`.

## notebooks/infer/infer_submit.py

### infer_submit.py — inference pipeline overview
Anchor: import csv

RSNA Knee: offline inference -> submission.csv. This IS the submission kernel:
internet OFF, GPU on, <= 9 h.

Attach: competition data, rsna-67knee-src, and the training kernel's output
(fold weights + run_config.json) via kernel_sources. The preprocessing recipe
(img size, crop, slice sampling) is read from run_config.json so inference
can never silently diverge from what the weights were trained on. DICOM
decode runs in a 4-worker process pool that prefetches ahead of the GPU.
A study that fails preprocessing gets the training prevalence prior.

Stage B weights (run_config carries "window"): per label, run_config's
"window_pool" table (from tune_pooling.py, tuned on gold-58) picks between
the model's own attention over windows and a window-pooled score.

### infer_submit.py — pinned weights and version check
Anchor: EXPECTED_LABEL_FILE = "consensus_v1.csv"

the weights this kernel must ship: pinned dataset, never a kernel's
"latest output" (the train kernel's latest output is whatever ran last).
Both the label file and the recipe version are asserted: the round-2 and
round-3 weights share the label file, so only `version` tells them apart.
EXPECTED_RECIPE_VERSION is per-family (a family's own cache version - v4
for the arms trained on the v4 cache, v5 once an arm ships off the v5
cache) so a blend can mix families trained on different cache versions
without the check assuming they all match; see _expected_recipe_version.
None disables the version check (local dry-runs on a v2-era run_config).

### infer_submit.py — per-family weights candidates
Anchor: FAMILY_WEIGHTS = {

Per-family weights-dir candidates, keyed by run_config["family"]. dinov2s's
entry is never read - load_family() substitutes the current WEIGHTS_CANDIDATES
global instead, so a dry-run driver that monkeypatches WEIGHTS_CANDIDATES
after import (the existing single-family rigs) still takes effect.

### infer_submit.py — MAIN and EFFICIENCY presets
Anchor: PRESET = "coatnet"  # "main" | "main_attn" | "efficiency" | "coatnet" | "fast" | "blend" - coatnet (CoAtNet-2 v6c alone, 0.912 LB) is the shipping default since round 6; main_attn (DINOv2-S v4b) scored 0.892 and the weighted blend 0.909

Presets. MAIN: every fold, all sampled slices, TTA if the training run
measured it as helping on gold-58, and the run_config's tuned window_pool
table. EFFICIENCY: one fold, the middle 6 of the 15 cached slices per slot
(decoded, not just dropped; 6 = two 3-slice windows, so the cap must stay a
multiple of the window; slice_align keeps that middle-6 decode aligned to
the same window-group boundaries a full 15-slice decode would use, so the
windows scored here match the ones training pooled over), no TTA, and
forced all-attn pooling - the tuned table was fit on 5 folds x 5 windows
and does not transfer to 1 fold x 2 windows, so trusting it here would
apply a pooling mode measured on a study shape this preset never builds.

### infer_submit.py — TIMING_STUDIES timing mode
Anchor: TIMING_STUDIES = 0

TIMING_STUDIES > 0: score that many TRAINING studies (train_series.csv /
train_series/<uid>, first N uids in sorted order) instead of the test set,
write no submission.csv and no probs_<fam>.csv, and instead print a timing
block and write /kaggle/working/timing.json - see main(). 0 (the shipping
value) leaves every other path untouched.

### infer_submit.py — TIMING_GOLD_SLICES sweep mode
Anchor: TIMING_GOLD_SLICES = ""

TIMING_GOLD_SLICES: comma-separated list of max_slices values (e.g.
"3,6,9,12,15"). Non-empty: TIMING_STUDIES is ignored and main() instead
reads gold58.csv (the mounted rsna-knee-src dataset), decodes those
studies from the TRAINING tree (same series_dir plumbing TIMING_STUDIES
uses), and for EACH budget in the list scores them with the loaded
family/families and reports gold macro AUC (tune_pooling.auc_per_label)
alongside decode/GPU/wall seconds for that budget - see the
TIMING_GOLD_SLICES branch in main(). Writes no submission.csv and no
probs_<fam>.csv; writes /kaggle/working/timing_gold.json instead. This
measures the thing that cannot be computed offline: the per-label
attention head re-pools over whatever windows survive a slice cut, so a
reduced-slice score is NOT the mean of the stored per-window probs.
"" (the shipping value) leaves every other path, including
TIMING_STUDIES, byte-identical.

### infer_submit.py — FAST preset constants
Anchor: FAST_FOLDS = 1

FAST preset's fold/slice cap, split out as module constants so the
operating point this round settles on is a one-line edit here rather than
a PRESETS edit.

### infer_submit.py — FAST preset definition
Anchor: "fast": {"families": ("coatnet2",), "use_tta": False,

FAST: the efficiency-track operating point. CoAtNet-2 alone (the
cheaper family - no DINOv2 pass), no TTA, forced all-attn pooling
(same reasoning as EFFICIENCY above), FAST_FOLDS/FAST_SLICES pulled in
from the module constants above so this round's timing sweep can move
the operating point without touching this dict.

### infer_submit.py — BLEND preset definition
Anchor: "blend": {"families": ("dinov2s", "coatnet2"), "use_tta": None,

BLEND: both arms, per-label rank-blended (blend_eval.rank_blend) with
weights from weights_file (blend_weights.json, produced offline from
each family's OOF AUC - never gold-58). allow_partial/allow_equal_weights
stay False: a family that fails to load, or a weights file that is
missing/mismatched, must fail loudly rather than silently degrade to
a single arm or an equal-weight blend.

### infer_submit.py — MAX_INFER_HOURS budget
Anchor: MAX_INFER_HOURS = 8.5

hard runtime budget for a multi-family preset (see check_projection):
projecting past this from the studies seen so far stops the run before it
burns the whole 9 h session on a blend that will not finish anyway.

### infer_submit.py — PRIOR_FALLBACK prevalence
Anchor: PRIOR_FALLBACK = [0.2412, 0.1784, 0.4441, 0.2153, 0.3875, 0.3004,

per-label training prevalence (consensus_v1.csv, the train-set positive
rate per label): the fallback row for undecodable studies when the
weights' run_config carries no `prior` (round-2/3 weights); newer training
runs write their own (see train_dinov2.py's run_cfg["prior"]).

### infer_submit.py — fold weight file filter
Anchor: weight_files = sorted(f for f in os.listdir(weights_dir)

"_last.pt" (train_dinov2.py's SWA companion file, same dir, same
{family}_fold prefix) must not count as a fold weight here - the SWA
file (no suffix) is what ships.

### infer_submit.py — one backbone pass per study
Anchor: p_attn, wprobs, _ = predict_study(models, vol, dev, use_amp,

one backbone pass per study: p_attn is the ensemble's own
attention over every window/slice; wprobs (window=<int>
weights only) is the per-window scores window_pool needs.

### infer_submit.py — family load failure handling
Anchor: loaded = {}  # family -> (models, cfg, weights_dir, use_tta)

load every family the preset needs. preset["allow_partial"] decides what
a load failure means: unset/False (every preset today) re-raises - a
family that cannot be loaded is a hard stop, even for a blend, since a
silent single-arm fallback is exactly what round 6 must not do. Only a
preset that opts in with allow_partial=True gets the WARN-and-continue
(the other arm can still ship a submission).

### infer_submit.py — blend_weights.json contract
Anchor: weights_file = preset.get("weights_file")

blend_weights.json: required for a preset with weights_file (only
"blend" today); build the per-label, per-family weight matrix rank_blend
needs, in fam_names order. Every failure mode here (missing file,
ambiguous file, family mismatch, a label/family absent from the JSON)
is a hard RuntimeError - allow_equal_weights is the only opt-out, and
it still WARNs rather than staying silent.

### infer_submit.py — recipe/window family grouping
Anchor: recipe_by_family = {fam: recipe_kwargs(loaded[fam][1]) for fam in fam_names}

Families sharing an identical (recipe, window) pair are grouped so
decode_all still runs once per group - one decode per study, shared by
every family in that group's predict_study, exactly as before when
every loaded family trained on the same cache. A genuinely mixed blend
(e.g. a v4-cache family alongside a v5-cache one) pays for a second
decode pass over the test set, one per distinct recipe.

### infer_submit.py — slice_align boundary rule
Anchor: slice_align_by_group[key] = group_window or 1

slice_align keeps a truncated (max_slices) decode on the same
window-group boundaries a full decode would use; window=None
(stage A) has no window groups, so align is a no-op (1).

### infer_submit.py — forced all-attn pooling presets
Anchor: window_pool_by_family[fam] = {name: "attn" for name in LABELS}

efficiency/coatnet/blend/main_attn: force all-attn - the tuned
window_pool table was fit on 5 folds x 5 windows and does not
transfer to another study shape (see the PRESETS comment above)

### infer_submit.py — TIMING_GOLD_SLICES precedence
Anchor: if TIMING_GOLD_SLICES:

TIMING_GOLD_SLICES: gold-58 slice sweep, checked before TIMING_STUDIES
since it ignores TIMING_STUDIES entirely and returns before any
submission.csv / probs_<fam>.csv would be written.

### infer_submit.py — deferred tune_pooling import
Anchor: from tune_pooling import auc_per_label

imported here, not at module top, so a src mount that lacks
tune_pooling.py (this mode is new; the module may not be listed
in every src dataset version yet) cannot break every other path.

### infer_submit.py — TIMING_STUDIES uses training tree
Anchor: timing = TIMING_STUDIES > 0

TIMING_STUDIES > 0: time the pipeline over TRAINING studies (the
visible test mount holds only ~3 studies - nothing timed there means
anything). series_dir switches with the source so decode_study reads
the matching tree.

### infer_submit.py — timed decode bookkeeping
Anchor: nonlocal decode_seconds

timing mode only: add to decode_seconds the wall time spent
waiting on each item, so the projection can tell decode cost
apart from GPU cost (score_study, timed at the call site below).

### infer_submit.py — blend rank_blend passthrough
Anchor: mats = [np.stack(probs_by_family[fam]) for fam in fam_names]

blend preset: rank_blend over whichever families loaded (WARN'd
families above already dropped out of fam_names) - even a single
surviving family still goes through rank_blend, not a passthrough,
since the preset requested a blend. blend_weights_matrix is None
when weights_file was absent under allow_equal_weights (rank_blend
treats None as equal weights) or when the preset has no weights_file.

## notebooks/cache/build_cache.py

### build_cache.py — cache build overview
Anchor: import csv

RSNA Knee: build the uint8 pixel cache on Kaggle CPU.

Reads the competition's train_series DICOMs and writes, per shard of studies:
  cache_XXX.npy  — (n, n_slots, max(slices_per_slot), SIZE, SIZE) uint8
  (v4: 6 slots, n_slices=15, SIZE=280, ~29 GiB over all 18 shards;
   v5: 5 slots (raptor5 layout), slices_per_slot=(18,14,12,8,12), SIZE=336,
   ~29.7 GiB over all 18 shards - see the module docstring in
   src/preprocess.py for the slot layout and per-slot slice counts)
plus cache_index.csv mapping StudyInstanceUID -> (shard, row, filled_slots)
and cache_config.json carrying the full recipe dict (RECIPE_V5 from
src/preprocess.py, plus n_slots) - every key is consumed downstream via
recipe_kwargs()/n_slots_for().

The cache exceeds the ~20 GB kernel-output cap, so it is built in several
kernels: shards are numbered over the FULL sorted study list, and each kernel
writes only the shards in SHARD_RANGE. A v5 shard is 250 x 5 x 18 x 336 x 336
bytes = 2.54 GB (every slot is zero-padded to the 18-slice maximum, so a
study costs 90 slices, not the 64 real ones), so 9 shards per kernel (22.9
GB) overflowed the disk on 2026-09-13 and the split is three kernels of 6
shards (15.2 GB): parts 0-5, 6-11, 12-17. The v4 build (1.18 GB per shard)
used two kernels, 0-8 and 9-17. Every part dumps an identical
cache_config.json; the training kernel mounts all parts and concatenates the
cache_index.csv files.

Attach: competition data + the "rsna-knee-src" utility dataset (src/preprocess.py).
Set TINY=True for a 3-study smoke run before burning a full session.

### build_cache.py — SHARD_RANGE shard slice
Anchor: SHARD_RANGE = (0, 6)

shards [lo, hi) this kernel builds; v5 parts = (0, 6), (6, 12), (12, 18)
(slugs rsna-knee-build-cache-v5 / -v5b / -v5c); the v4 build used (0, 9) and
(9, 18). Shard numbers are global (over the full sorted study list) so the
parts never collide and the index rows carry the right shard id.

### build_cache.py — RECIPE dict contract
Anchor: RECIPE = RECIPE_V5

the recipe is a single dict from src/preprocess.py; it is dumped to
cache_config.json (plus n_slots) and consumed downstream via
recipe_kwargs()/n_slots_for()

### build_cache.py — N_SL matches n_slices
Anchor: assert N_SL == RECIPE["n_slices"], (

`is None`, not truthiness, to match build_study_tensor's own rule. The
training script reads the slice-axis length from n_slices, so a recipe
whose deepest slot disagrees with it would build a cache with slices no
window ever covers.

### build_cache.py — TINY smoke prefix handling
Anchor: studies = studies_all[:lo * SHARD_SIZE] + studies_all[

keep the first 3 studies OF THE SELECTED SHARD RANGE, not of the
global list - studies[:3] always lands in shard 0, so a part-B
smoke (SHARD_RANGE=(9,18)) built nothing. The prefix up to
lo*SHARD_SIZE is kept (just UIDs, no decode work) so shard_no below
still lands on the real global shard number (e.g. 9).

### build_cache.py — cache_config write order
Anchor: with open(os.path.join(OUT, "cache_config.json"), "w") as fh:

config first: it is the invariant that makes any surviving shards
interpretable after a mid-build crash. n_slots is derived (not a
_RECIPE_KEYS entry) but consumers read it straight off cache_config.json
(see train_dinov2.find_cache_parts / cache_check.py) so it is added here.
