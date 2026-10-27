# Raptor (LB 0.924) vs our CoAtNet-2 pipeline (LB 0.912): code-level differences (2026-09-29)

Source: notebooks/fusion/raptor_infer.py (vendored public inference) vs src/preprocess.py,
src/model.py, notebooks/train/train_dinov2.py. Our v5 arm already copied the geometry
(336 px, 140 mm crop, span 0.02-0.98, raptor5 slots, 64-slice span, pos_weight, SWA-3, 16 ep)
and LOST 0.012 (LB 0.900), so geometry is ruled out. Not yet copied, ranked by plausibility:

1. **Backbone checkpoint.** Raptor: `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` (fine-tuned at 384 px).
   Ours: `coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k` fed 280/336 px through timm's img_size kwarg.
   Experiment: same v4 recipe with the 384-native checkpoint at 288 px, then upsampled to 384.
2. **Window density.** Raptor scores K_EVAL = 62 overlapping 3-slice windows per study (about one
   per slice); we form non-overlapping 3-slice groups (about 12 windows per study). Experiment:
   stride-1 windows at inference with the existing v6c checkpoints (paired OOF).
3. **Intensity.** Raptor applies the DICOM modality LUT (RescaleSlope/Intercept), inverts
   MONOCHROME1, then clips to the 2nd/98th percentile computed once per slot over all its slices.
   We do none of the LUT/photometric handling and window each slice on its own 1st/99th percentile.
   Needs a cache rebuild on Kaggle (DICOMs are not local).
4. **Axial slot.** Raptor fills the axial slot with any axial series (fluid=-1); our raptor5 layout
   only accepts fluid-sensitive axial, so some studies have an empty axial slot.
5. Minor: attention head is a Tanh MLP (theirs) vs a dot-product query (ours); series tie-break by
   CSV order (theirs) vs most slices (ours); their "0.924 single" is itself a 4-sub-arm blend.
