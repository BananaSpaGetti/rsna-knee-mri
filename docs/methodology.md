# Methodology: noise floors and how we decide (2026-09-29)

Measured with `weights-v6c` (5-fold CoAtNet-2) predictions; bootstrap 2000x on gold-58, 300 random
subsamples of the 4,349 out-of-fold training studies for the LB-sized rows.

| Measurement | std of macro AUC | a difference is real only above |
|---|---|---|
| gold-58 (58 radiologist-labelled studies) | 0.0164 (95% CI of 0.882 = [0.847, 0.913]) | ~0.03 |
| public LB if it scores ~300 of the 1,300 hidden studies | 0.0085 | ~0.017 |
| public LB if it scores ~650 | 0.0054 | ~0.011 |
| whole hidden set, 1,300 studies | 0.0038 | ~0.008 |
| paired comparison of two arms on the SAME studies (OOF, n=1,300) | 0.0017 | ~0.004 |

Consequences:
- gold-58 cannot rank arms whose true difference is below ~0.03. Every "0.882 vs 0.878" reading in the
  round-10 log is noise. gold-58 stays a sanity check, not a selector.
- OOF against the LLM consensus is precise (paired std 0.002) but measures agreement with the reports,
  not with radiologists: pseudo_v1 beats v6c by +0.020 +/- 0.002 on OOF and by -0.001 on the LB.
  Gains against the consensus labels do not transfer one-to-one to the radiologist-labelled test set.
- The LB is the only radiologist-aligned signal with a usable sample size, at 5 submissions/day and
  std ~0.004-0.008. Experiments must therefore have an expected effect above ~0.01, be few, and be
  designed to separate one factor per submission (e.g. remove the transformer branch first, then
  change our arms' weights), not to tune continuous weights.
- Run-to-run training noise on one fold is ~0.005 OOF (v6c recipe re-run locally: 0.864 vs 0.870).

## Addendum 2026-09-30 — what the rules found, and paired LB reads

- Model/preprocess changes judged on paired OOF (n = 4,349 train studies, bootstrap std ~0.001 for
  the macro difference): dense stride-1 inference windows +0.0071 +/- 0.0007 on v6c (+0.0046 on rs,
  +0.0020 on pseudo1); random-start training windows + seed (coatnet2_rs) +0.0068 +/- 0.0010 over v6c,
  both dense; v6c+rs rank ensemble +0.0016 +/- 0.0005 over rs alone. All labels moved the same way.
- Those effects are invisible on the public LB: E1 (v6c grouped) 0.935, E2 (v6c dense) 0.935,
  E3 (+ pseudo1 dense) 0.935 - a +0.003-0.005 change to one 0.375-weight arm is under the 0.0085 LB
  std. The LB is a runtime/format check for such changes; the OOF decides.
- Paired LB reads: two submissions that differ in ONE factor are scored on the same ~300 public
  studies, so the std of their difference is far below the 0.0085 of a single score (the paired
  OOF std is ~0.45x the unpaired one at the same n; expect roughly 0.004 here). A label-policy
  effect >= 0.01 on the radiologist labels should therefore show as a paired fold-0 difference;
  smaller ones still will not, and the raw scores must never be compared across days of different
  models.
- Seed-only noise at the 5-fold level, measured directly (coatnet2_rs vs coatnet2_rs2, same recipe,
  seeds 7 vs 11): rs2 - rs = -0.0029 +/- 0.0010 paired dense OOF (0.8749 vs 0.8778). So a one-factor
  5-fold comparison must clear ~0.006 (2x) before the seed is ruled out; the rs-vs-v6c +0.0068 is at
  that edge (it also changed the seed). Seed ensembles: rs+rs2 +0.0025 +/- 0.0005 over rs; adding v6c
  to rs+rs2 +0.0006 (a third seed of the same recipe buys little; a different recipe buys more).
- Gold-58 as a label-set selector was a mistake twice over (std 0.016, and it is 58 studies of
  4,407 with the same report-derived prior); public write-ups reach the same verdict ("a guard,
  not a tuner").

## Addendum 2026-10-01 — the label lever, measured three ways

- Paired public-LB gate (rs-recipe fold 0, stride 3, CPU kernels, one factor = the label file):
  control consensus_v1 0.900, uteacher 0.900, teach4 0.907. The three local reads pointed the same
  way before the LB did: fold-0 dense OOF vs consensus +0.0065 +/- 0.0020 (teach4) vs +0.0005
  (uteacher); 5-fold dense +0.0039 +/- 0.0009; gold-58 5-fold 0.9038 vs 0.8954. uignore was
  -0.026 on fold-0 OOF and never went to the LB.
- Reading: replacing a report-derived column wholesale with the image teacher's OOF (w = 1) on
  the findings where images beat reports moves the radiologist-labelled score; filling only the
  silent cells (uteacher) does not, because the silent cells were already a soft negative that the
  model mostly agreed with. The OOF-vs-consensus metric under-states teach4 (it is scored against
  the very labels it was told to distrust on four columns) and still showed it, so a variant that
  changes labels must be read on all three: paired dense OOF, gold-58 direction, paired LB.
- Kaggle mechanics learned: the weekly 30 h GPU quota ends every T4 kernel until Saturday; CPU
  kernels still run and still score (hidden rerun ~6.5 h for a single fold at stride 3), which is
  enough for one-factor single-fold gates.
- Next screens (fold 0, local, paired dense OOF, 0.004 single-fold bar): t6 = teach4 minus
  Contusion plus Lateral OA / Baker's / Fracture, teacher = mean of rs / rs2 / t4 dense OOF (the
  findings where that ensemble beats consensus_v1 on gold-58 by > 0.02; Contusion was the one
  teach4 column that got worse, -0.005 OOF / -0.035 gold); t4b = the teach4 columns with that
  stronger teacher.
- Geometry contract (v4g cache: positive-cardinal slice order, right knees mirrored to left on
  coronal/axial and sagittal order reversed, modality LUT, 1-99 percentile window on the crop per
  series; one factor vs the rs recipe on the byte-identical v4 pipeline otherwise): fold 0 read
  +0.0013 +/- 0.0025 with the side-dependent labels up, the 5-fold read -0.0020 +/- 0.0010 (dense
  0.8758 vs 0.8778; gold-58 0.8962 vs 0.8954). A single-fold read of +0.001 with std 0.0025 is
  not a read; the five-fold one is, and it says the public lineage's gain over us is not in the
  geometry contract. The same week the per-label attention head (-0.002), the fully trainable
  backbone (-0.002) and the stronger label teacher (t4b -0.004, t6 -0.002) all read flat or
  negative at fold 0, so of the single-arm differences catalogued in docs/competitors-code/ only
  the label lever has moved anything measurable so far.
- The label lever extends: t5 (teach4's four columns plus Baker's and Lateral OA from the same rs
  teacher) is +0.0034 +/- 0.0009 paired dense OOF over teach4 (0.8852 vs 0.8817) and +0.0074 over
  rs, with gold-58 flat (0.9037 vs 0.9038 - 58 studies cannot resolve 0.003). The two replaced
  columns rose against the very consensus they replaced (Baker's +0.017, Lateral OA +0.011), which
  is the signature teach4 showed: where the image model out-reads the reports, its OOF is the
  better training target even by the reports' own yardstick. A single t5 (0.8852) now beats the
  t4+t42 seed pair (0.8839); t4+t5 = 0.8866. The bracket continues with t11 (all findings except
  Fracture) to find where distillation stops paying.
- Distillation read, corrected: the student's OOF AUC against the consensus labels is leaky when
  the labels are the teacher's OOF. The teacher that produced fold j's labels was trained on
  fold k, so the student trained on fold j and scored on fold k has indirect access to fold k's
  consensus. The leak grows with the number of distilled columns: t4 (4 columns) +0.0039, t5 (6)
  +0.0074, t11 (11) +0.0132 over rs on OOF, while gold-58 (labels the teacher never saw) read
  +0.008, +0.008, -0.005. teach4 is still real (paired LB +0.007), so part of the OOF gain is
  genuine; the rest is leak, and the split is unknown without nested teachers. Rule from here:
  rank distilled variants on gold-58 direction and the paired LB, never on OOF-vs-consensus
  alone; "all 12 columns up on OOF" is the leak's signature, not a result.
