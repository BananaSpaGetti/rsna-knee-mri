# RSNA Knee 0.912 — CoAtNet-2 2.5D with per-label attention, readable inference edition

This is the readable, output-preserving notebook edition of our submission script kernel
`rsna-67knee-infer-submit`: the code cell below is that script verbatim, with `PRESET = "coatnet"`.
It predicts the twelve knee-MRI findings scored by macro ROC-AUC. The notebook is private to
the team until the competition closes.

This arm alone scores **0.912** on the public leaderboard. Our 0.929–0.940 rows come from the
public fusion recipe (Raptor CoAtNet + residual-gated CoAtNet), not from this kernel.

## Model stack

| Stage | What | Notes |
|---|---|---|
| Pixel cache (v4) | DICOM → uint8, 6 series slots × 15 slices per study, 280 px, isotropic 150 mm centre crop | built once on a CPU kernel; the recipe travels in `run_config.json` and is asserted here |
| 2.5D windows | 3 consecutive slices → one 3-channel image; every non-empty window of a study is scored | training samples 2 windows per slot per step |
| Backbone | timm `coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k` at 288 (280 + in-model padding), stem + stage 0 frozen, gradient checkpointing | 5 folds × 20 epochs, 12 studies per batch, lr 5e-5 backbone / 1e-3 head, 1-epoch warm-up then cosine, drop-path 0.1 |
| Head | slot embedding added to each window vector, then one attention query per label over the study's windows and a linear readout | 12 logits per study — a torn meniscus lives in a handful of slices, and attention keeps it from averaging out |
| Ensemble | mean of the 5 fold probabilities; horizontal-flip TTA off (it did not help on the radiologist-labelled holdout) | `probs_coatnet2.csv` → `submission.csv` |

Training labels are the consensus of three LLM readers over the radiology reports
(`consensus_v1.csv`, soft targets, per-cell weighted BCE). Out-of-fold macro AUC against that
consensus: 0.8639. On the 58 radiologist-labelled studies we hold out and never train on:
0.8933.

## Runtime

Measured on 300 training studies on one T4: **366.9 s wall = 364.1 s GPU + 2.8 s waiting for
decode** (99.2 % GPU-bound), projecting to about 0.44 h for the hidden test set. DICOM decoding
runs in a 4-process pool ahead of the GPU, and a study that fails to decode receives the
training prevalence prior instead of crashing the run. The `fast` preset (one fold) scores
0.898 in roughly 0.09 h for the efficiency track.

| Preset | Families | Public LB | Runtime |
|---|---|---|---|
| `coatnet` (this notebook) | CoAtNet-2, 5 folds | **0.912** | ~0.44 h |
| `blend` | DINOv2-S + CoAtNet-2, per-label rank blend weighted by OOF AUC | 0.909 | ~0.9 h |
| `fast` | CoAtNet-2, 1 fold | 0.898 | ~0.09 h |

## How it is built

- `src/` (mounted as the `rsna-knee-src` dataset): `preprocess.py` holds the recipe and the
  DICOM → tensor path used by both the cache build and this kernel; `model.py` the backbone
  and head; `blend_eval.py` the rank blend.
- Weights come from the pinned dataset `rsna-67knee-weights-v6c`, never from a kernel's latest
  output. `load_family()` asserts the label file, recipe version and fold count before anything
  is scored, so weights and pixels cannot silently drift apart.
- One switch, `PRESET`, decides what ships. Per-run values are flipped in a staged copy of the
  script, never in the committed file, so every leaderboard number stays reproducible.
- The 58-study gold holdout is a referee, not a teacher: nothing selected on it ships without a
  leaderboard check, and no public pseudo-labels are used anywhere.

Credits: RSNA and the competition hosts for the data; `timm` for the CoAtNet weights; the
`rsna-knee-llm-report-labels` dataset by stevenleehans for the report-derived labels our
consensus builds on.
