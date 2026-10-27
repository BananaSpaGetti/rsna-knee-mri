# Spec: cache recipe v4g (cardinal slice order, laterality canonicalisation, modality-LUT + crop windowing)

Repo: `C:\Users\Windows-11\Claude\Projects\Kaggle\rsna-67knee\`. Written 2026-10-01 (planner pass; saved verbatim by the main session).

## Approach and why

One-factor experiment: v4g = RECIPE_V4 plus the three geometry-contract changes (a)(b)(c) from the public 0.943 lineage's `cnx_dicom_geometry.py` (`C:\kaggle\staging\competitors\lineage\rsna-knee-coatnet-d4-depthzone-swa3-b2\cnx_dicom_geometry.py`, analysed in `docs/competitors-code/mattiaangeli-d4-serving.md` 1.2-1.5). Everything else - slot table, 15 slices from the central 80 % band, 280 px, 150 mm isotropic crop, uint8, shard format, `cache_index.csv`, `cache_config.json` fields - stays byte-compatible with v4. Each change is its own boolean recipe key so (i) every existing config (v2-v5 caches, every `run_config.json`) decodes byte-identically through the same module, and (ii) the switches travel cache_config -> run_config -> `infer_submit.recipe_kwargs` automatically, so inference applies the same transforms to test DICOMs without touching the model code.

Rejected alternative: port cnx's full D4 in-plane canonicalisation and its fail-closed crop. Both are extra factors; the caller asked for exactly three. Also rejected: building the cache locally - the raw DICOM tree is not on this machine (see Blocker), so the build runs on Kaggle CPU kernels exactly as v4 did.

## Constraints that are fixed

- `build_study_tensor` output shape `(6, 15, 280, 280)` uint8, `filled_slots` semantics, shard layout `cache_XXX.npy` of 250 studies, global shard numbering 0-17 split (0,9)/(9,18), `cache_index.csv` columns, `cache_config.json` = recipe dict + `n_slots`.
- With `cardinal_order=canonical_side=crop_window=False` the module must be byte-identical to today's output (guards in Task 10). Do not edit any existing function body except where this spec says; add branches, do not restructure.
- Laterality and flip rules are copied from cnx verbatim (quoted in Task 3). Do not "improve" them.
- `src/model.py`, `notebooks/infer/infer_submit.py`, `notebooks/train/train_dinov2.py` training loop: unchanged except the cache-selection knob and the new arm row in Task 7/8.
- Do not put anything under `C:\kaggle\input\` until `C:\kaggle\staging\queue_night.log` contains "screens done" (the running queue's eval walks that root).

## Out of scope

D4 in-plane rotation/mirror canonicalisation; fail-closed crop; per-slice vs per-series window ablation; 130/140 mm crops; 384 backbone; depth/stride metadata; multi-fold training; LB submission (only noted in Task 8 how `EXPECTED_RECIPE_VERSION` would change); refactoring v4's double header read; `eval_windows.py` changes (its `--cache-root` flag already suffices).

## Blocker and data layout (verified 2026-10-01)

- `C:\kaggle\input\rsna-knee-abnormality-detection\` = 3-study fake rig (`train_series.csv` 15 rows, studies `1.2.0.study..1.2.2.study`, 5 series x 25 files, 41 MB). Every file: IPP `[0,0,0]`, IOP `[1,0,0,0,1,0]`, no Laterality/ImageLaterality, MONOCHROME2, no RescaleSlope/Intercept, no NumberOfFrames, PixelSpacing `[1,1]`, 160x160, 16-bit unsigned. The real set (~570 GB, 4,407 studies, `train_series/<StudyUID>/<SeriesUID>/*.dcm`, `train_series.csv` columns `StudyInstanceUID,SeriesInstanceUID,Anatomical_Plane,Fluid_Sensitive,Fat_Suppression`) exists only on Kaggle.
- Local v4 cache: `C:\kaggle\input\rsna-knee-build-cache\cache_000..008.npy` + `rsna-knee-build-cache-b\cache_009..017.npy`, each shard `(250,6,15,280,280)` = 1,764,000,128 bytes (last 1,107,792,128), `cache_config.json` = `{"version": "v4", "size": 280, "crop_mm": 150.0, "unique_slices": true, "isotropic": true, "guarantee_k": true, "expand_frames": true, "strict_geometry": true, "n_slices": 15}` (no `n_slots` key - the v4 build predates it; the training script tolerates both).
- Kaggle build kernels: `notebooks/cache/kernel-metadata.json` (`patimakornsrinamen/rsna-knee-build-cache`, CPU, `dataset_sources: ["patimakornsrinamen/rsna-knee-src"]`, `competition_sources: ["rsna-knee-abnormality-detection"]`); part B staged at `C:\kaggle\staging\cache-b\` (older script, `SHARD_RANGE = (9, 18)`). The kernels import `preprocess.py` from the **`rsna-knee-src` dataset**, staged at `C:\kaggle\staging\rsna-src\` (`dataset-metadata.json` id `patimakornsrinamen/rsna-knee-src`); push with `py -m kaggle datasets version -p C:/kaggle/staging/rsna-src -m "<msg>"`. Kaggle CLI 2.2.4 is installed.
- Local toolchain: Python 3.13.15, pydicom 3.0.2, numpy 2.5.2, cv2 5.0.0, 32 logical CPUs. Hotspot ~180 Mbps down.
- Baseline run for the paired judgement: `C:\kaggle\staging\weights-rs\` (arm `coatnet2_rs`, all 5 folds, `fold_auc_last` fold 0 = 0.8743, `oof_stride1_eval.npy`/`oof_stride3_eval.npy` present, stride-1 macro 0.8778 all folds).

## Tasks

### Task 1 - recipe keys and RECIPE_V4G (`src/preprocess.py` lines 83-130)

Change:
- `RECIPE_V2` (line 83): append `"cardinal_order": False, "canonical_side": False, "crop_window": False`.
- After `RECIPE_V5` (line 101-103) add:
  ```python
  RECIPE_V4G = {**RECIPE_V4, "version": "v4g", "cardinal_order": True,
               "canonical_side": True, "crop_window": True}
  ```
- `_RECIPE_KEYS` (line 104): append `"cardinal_order", "canonical_side", "crop_window"`.
- Module docstring: add a "v4g" paragraph (3 lines: the three keys, each gated, None/False reproduces v4).
- `recipe_kwargs` needs no change: missing keys fall back to `RECIPE_V2[k]` = False. Note in its docstring that the three v4g keys default False for every pre-v4g config.

Consumers (no edits needed, verify by reading): `build_cache.py` dumps `{**RECIPE, "n_slots"}`; `train_dinov2.py` line 849 copies `CACHE_CFG` into `run_config.json`; `infer_submit.py` line 426 `recipe_kwargs(run_config)` -> `build_study_tensor(**recipe)` (line 122), so test-time decode gets the same three switches. `infer_submit._expected_recipe_version` compares `run_config["version"]` with `EXPECTED_RECIPE_VERSION["coatnet2"] == "v4"`; a v4g family would need that entry set to `"v4g"` in a staged infer copy (Task 8 note). Family grouping (line 431) keys on the full recipe tuple; every family gains the same three False keys, so existing groupings are unchanged.

Verification: `py -c "import sys; sys.path.insert(0,'rsna-67knee/src'); import preprocess as p; k=p.recipe_kwargs({'version':'v4','size':280}); print(k['cardinal_order'], k['canonical_side'], k['crop_window']); print(p.recipe_kwargs(p.RECIPE_V4G))"` prints `False False False` then a dict with the three True.

Depends on: nothing.

### Task 2 - header pass and cardinal ordering (`src/preprocess.py`, new functions after `_frame_order`, line ~494)

Add module constants:
```python
CARDINAL_NORMAL = {"Sagittal": (1.0, 0.0, 0.0), "Coronal": (0.0, 1.0, 0.0),
                   "Axial": (0.0, 0.0, 1.0)}   # cnx CARDINAL_NORMAL
POSITION_TOLERANCE_MM = 1e-3                      # cnx POSITION_TOLERANCE_MM
```

`_normalize_laterality(value) -> str | None`: `str(value).strip().upper()` in `{"L","LEFT"}` -> `"L"`, in `{"R","RIGHT"}` -> `"R"`, else None (cnx `normalize_laterality`).

`_read_slice_headers(files: list[str]) -> list[dict]`: one `pydicom.dcmread(p, stop_before_pixels=True)` per file (wrap in try/except; an unreadable header yields a record with all geometry None, like v4 keeps the file). Record keys: `path`, `inst` (int InstanceNumber or 0), `sloc` (float SliceLocation or None), `ipp` (np.float64 (3,) or None; non-finite -> None), `iop` ((6,) or None; fallback `SharedFunctionalGroupsSequence[0].PlaneOrientationSequence[0].ImageOrientationPatient`), `n_frames` (max(1, NumberOfFrames)), `frame_ipp` (list of (3,)/None per frame from `PerFrameFunctionalGroupsSequence[i].PlanePositionSequence[0].ImagePositionPatient`, only when `n_frames > 1`; list of None if the sequence is missing or its length != n_frames), `lat` = `_resolve_lat(ds)` where: `a = _normalize_laterality(getattr(ds,"Laterality",None))`, `b = _normalize_laterality(getattr(ds,"ImageLaterality",None))`; if both non-None and `a != b` -> `"conflict"` (cnx raises for that file; we neutralise it) else `a or b`; if still None try `SharedFunctionalGroupsSequence[0].FrameAnatomySequence[0].FrameLaterality` (multi-frame). Also `photometric` (str upper or ""), `slope`/`intercept` (floats; root `RescaleSlope/RescaleIntercept`, fallback `SharedFunctionalGroupsSequence[0].PixelValueTransformationSequence[0]`; default 1.0/0.0; non-finite or slope == 0 -> defaults, counted in `bad_rescale`).

`order_slices_cardinal(records, plane) -> (refs, receipt)`:
1. Expand to slice entries `(path, frame, ipp, iop, inst, name, frame_idx)`: `n_frames == 1` -> one entry with `frame=None, ipp=rec.ipp`; `n_frames > 1` -> one entry per frame with `frame=f, ipp=rec.frame_ipp[f]`, `iop=rec.iop` (shared), `frame_idx=f`.
2. `n`: from the first entry with a 6-vector IOP: `r = iop[:3]/|iop[:3]|`, `c = iop[3:]/|iop[3:]|`, `n = (r x c)/|r x c|`. If no entry has IOP, or no entry has IPP -> `refs = series_slice_refs(files, expand_frames=True, strict_geometry=True)` (the v4 path), `receipt.method = "v4_fallback_no_geometry"`, return.
3. Cardinalise: `e = CARDINAL_NORMAL[plane]` (plane from the CSV `Anatomical_Plane`; plane not in the table -> v4 fallback, method `"v4_fallback_unknown_plane"`). If `argmax(|n|) != argmax(e)` -> v4 fallback, method `"v4_fallback_plane_mismatch"` (cnx raises). If `dot(n, e) < 0`: `n = -n`, `receipt.flipped_normal = True` (this is the whole of change (a): sagittal stacks then always run patient-right -> patient-left, coronal anterior -> posterior, axial inferior -> superior).
4. `p_i = dot(ipp_i, n)` for entries with IPP, None otherwise. Sort key `(p is None, p or 0.0, inst, name, frame_idx)` - identical to v4's strict key, only `p` differs by the sign.
5. Duplicate rejection: over positioned entries sorted by `p`, `delta = diff(p)`; if `len(p) > 1 and any(delta <= POSITION_TOLERANCE_MM)` -> `receipt.method = "rejected_duplicate_position"`, `receipt.min_delta = min(delta)`, return `refs = []`. Caller semantics (Task 5): a rejected series is removed from the slot candidates.
6. Otherwise `method = "ipp_projection_cardinalized"`, `refs = [(path, frame)]` in sorted order.
Receipt (plain dict): `method, flipped_normal, normal (3 floats), n_entries, n_positioned, min_delta, ipp_x_median` (median of `ipp[0]` over positioned entries, None if none), `lat_values` (sorted set of `"L"/"R"` over records), `n_lat_conflict`, `photometric_values` (set), `n_bad_rescale`, `n_multiframe_files`.

Relationship to v4 you can assert in tests: for a fully-positioned single-frame series, v4g order == v4 order when `flipped_normal` is False and == reversed v4 order when True (ties excluded by step 5).

Verification: a synthetic test (Task 10b) with IOP `[0,1,0,0,0,-1]` (sagittal, raw normal = -x) must give `flipped_normal=True` and reversed order; duplicates `p` must give `rejected_duplicate_position`.

Depends on: Task 1 (constants only). Independent of Tasks 3-4.

### Task 3 - study side decision (`src/preprocess.py`, new function next to Task 2)

`decide_study_side(series_receipts: dict[uid -> (plane, receipt)]) -> (side, source, tags, sag_x_median)`, a transcription of cnx `decide_study_side` + `resolve_study_side`:
```python
tags = sorted({v for _, r in series_receipts.values() for v in r["lat_values"] if v in {"L", "R"}})  # pooled over EVERY series of the study (all CSV rows with a directory), as cnx pools laterality_values across all records
sag_x = [r["ipp_x_median"] for plane, r in series_receipts.values()
         if plane == "Sagittal" and r["ipp_x_median"] is not None]
sag_med = median(sag_x) if sag_x else None
if len(tags) == 1: side, source = tags[0], "dicom_laterality_tag"
elif len(tags) > 1: side, source = None, "conflicting_laterality_tags"      # never flip
elif sag_med is not None: side, source = ("L" if sag_med >= 0 else "R"), "sagittal_ipp_x"
else: side, source = None, "unknown_no_flip"
```
Pixel rule, cnx verbatim: `flip_for_plane(plane, side) = str(plane) != "Sagittal" and side == "R"` -> mirror columns (`img[:, ::-1]`) as the final pixel operation. Sagittal reversal: NOT in `cnx_dicom_geometry.py`; it is the Global96 stack builder (`raptor_global_stack_smart336.py`, verified copy, our doc section 2.1): `if canonical_sagittal and series_is_sagittal and series_is_right: chosen = chosen[::-1]; depths = 1 - depths[::-1]`. Our transcription: for `side == "R"` and plane `"Sagittal"`, `picks = picks[::-1]` after `sample_indices` (same slices, reversed row order; zero padding of short series stays at the end). Combined with (a) this makes every sagittal slot run lateral -> medial and every coronal/axial frame look like a left knee.

Add a module-level helper `flip_for_plane(plane, side)` with exactly the cnx body so the smoke test and `build_study_tensor` share it.

Verification: unit asserts in Task 10b: `({"L"}) -> ("L","dicom_laterality_tag")`, `({"L","R"}) -> (None, "conflicting...")`, no tags + sag medians `[-12.0, -9.0] -> ("R","sagittal_ipp_x")`, nothing -> `(None,"unknown_no_flip")`; `flip_for_plane("Coronal","R") is True`, `("Sagittal","R") is False`, `("Axial","L") is False`.

Depends on: Task 2 receipts.

### Task 4 - modality LUT, MONOCHROME1, per-series crop window (`src/preprocess.py`, new functions near `load_slice`, line ~421)

`load_crop_float(path, crop_mm, frame, pixels_cache, isotropic=True, strict_geometry=True) -> np.ndarray float32`:
1. `ds, pixels = _read_pixels(path, pixels_cache)`; frame/RGB selection exactly as `load_slice` lines 435-446 (including the strict 2-D-with-frame ValueError).
2. `x = pixels.astype(np.float32)`; `slope, intercept` from the header rule in Task 2 (`_read_slice_headers`'s `_rescale(ds, frame)`: root -> `SharedFunctionalGroupsSequence[0].PixelValueTransformationSequence[0]` -> `PerFrameFunctionalGroupsSequence[frame].PixelValueTransformationSequence[0]` when `frame is not None`; defaults 1.0/0.0; slope 0 or non-finite -> defaults). `x = x * slope + intercept` (cnx `image * slope + intercept`).
3. If `str(getattr(ds,"PhotometricInterpretation","")).strip().upper() == "MONOCHROME1"`: `x = (float(x.min()) + float(x.max())) - x` (cnx, computed on the full frame before crop).
4. `return center_crop_mm(x, ds, crop_mm, isotropic=isotropic, strict_geometry=strict_geometry)` - the existing function only slices, so it works on float arrays unchanged; crop geometry is therefore identical to v4.

`window_series_crops(crops: list[np.ndarray], size: int, flip: bool) -> list[np.ndarray uint8]` (crops = the selected slices of ONE series, in cache row order):
```
sample = concatenate(c[::4, ::4].ravel() for c in crops)          # 4x4 subsample, as cnx `crop[::4, ::4]`, pooled over the series' selected slices (caller's per-series rule; cnx itself is per slice)
lo, hi = np.percentile(sample, [1, 99])
for c in crops:
    y = clip((c - lo) / max(float(hi - lo), 1e-6), 0, 1)             # cnx
    img = clip(cv2.resize(y, (size, size), interpolation=INTER_AREA), 0, 1)   # float32 resize, as cnx
    if flip: img = img[:, ::-1]                                         # laterality mirror = final pixel op
    out.append((img * 255).astype(np.uint8))                           # truncation, as cnx
```
Empty `crops` -> empty list. Note the deliberate difference from v4's `window_to_uint8` + uint8 `resize_uint8`: v4 windows per slice on the full frame then resizes uint8; v4g scales after crop, resizes float, truncates. This is the whole of change (c).

Verification (Task 10b): a synthetic MONOCHROME1 frame with slope 2 / intercept -100 produces, after `window_series_crops`, the same uint8 as the hand-computed formula (assert exact equality on a 16x16 array); a MONOCHROME2 frame with slope 1 / intercept 0 and `crop_window` pixels compared to v4's path must differ only by the window/resize order (no assert, just print max abs diff to show the branch is live).

Depends on: Task 1. Independent of Tasks 2-3.

### Task 5 - wire the switches into `build_study_tensor` (`src/preprocess.py` lines 532-639)

Signature: add `cardinal_order: bool = False, canonical_side: bool = False, crop_window: bool = False, receipts: dict | None = None` (receipts is diagnostic, not a recipe key; when a dict is passed it is filled with `{"side", "side_source", "tags", "sag_x_median", "series": {uid: {"plane":..., **receipt, "flip":bool, "reversed":bool}}}`).

Pre-pick loop (lines 580-599), inside the `for row in series_rows` after `files` is known:
- `if cardinal_order or canonical_side:` `records = _read_slice_headers(files)` (one header pass; used by both switches).
- `if cardinal_order:` `refs, rc = order_slices_cardinal(records, row["Anatomical_Plane"])`; if `rc["method"].startswith("rejected")`: record it in `receipts["series"]` and `continue` (series excluded from `pick_slot_series`; the next-best same-slot series wins; no candidate -> slot stays zero, `filled` unaffected - identical to a series whose files all fail today). Else `r["_refs"] = refs; r["n_slices"] = len(refs)`.
- `elif` (cardinal_order False): existing v4 code unchanged. If `canonical_side` only, still build `rc` from `records` via a light receipt (`lat_values`, `ipp_x_median` from root IPP medians) so the side decision works with v4 ordering.
- Keep `r["_plane"] = row["Anatomical_Plane"]`, `r["_receipt"] = rc`.

Before the slot loop (after line 601): `side = None`; `if canonical_side: side, source, tags, sag_med = decide_study_side({uid: (r["_plane"], r["_receipt"]) for r in rows})`.

Slot loop (lines 605-638):
- after `picks` are computed (line 616) and the `max_slices` subset (line 624): `if canonical_side and side == "R" and r["_plane"] == "Sagittal": picks = picks[::-1]` (Task 3). Place this AFTER the max_slices subset so the efficiency preset still truncates the same centred slices.
- `flip = canonical_side and flip_for_plane(r["_plane"], side)`.
- `if crop_window:` collect `crops = []` with `load_crop_float(path, crop_mm, frame, decoded if frame is not None else None, isotropic, strict_geometry)` per pick inside the same try/except (a failed slice appends `None` and stays zero); then `for j, u8 in zip(indices_of_non_None, window_series_crops([c for c in crops if c is not None], size, flip)): out[slot, row_start + j] = u8`. `any_ok` as before.
- `else:` existing `load_slice` path, then `if flip: out[slot, row_start + j] = out[slot, row_start + j][:, ::-1]` (so (b) works without (c)).
- When `crop_mm is None` (v1 defaults) `load_crop_float` returns the full frame; fine.

Docstring: add one paragraph per switch pointing at Tasks 2-4's functions.

Verification: `py -c` build of the 3 rig studies with `**recipe_kwargs(RECIPE_V4)` -> md5 equals the pre-change module's (Task 10a); with `RECIPE_V4G` the rig returns `filled == 0` for all 3 studies and `receipts["series"][*]["method"] == "rejected_duplicate_position"` (the rig's IPPs are all zero) - prints that, proving the rejection path.

Depends on: Tasks 1-4.

### Task 6 - cache build script (`notebooks/cache/build_cache.py`)

- Line 28-29 import: add `RECIPE_V4G`.
- Line 43-44: `RECIPE = RECIPE_V4G` and keep the commented switches (`# RECIPE = RECIPE_V5`, `# RECIPE = RECIPE_V4`).
- Line 40: `SHARD_RANGE = (0, 9)` for part A; part B staged copy `(9, 18)` (sed in staging, as `C:\kaggle\staging\cache-b` did). Shard size 250, WORKERS 4 unchanged. v4g shard bytes = v4 (1.764 GB), 9 per kernel = 15.9 GB < 20 GB cap.
- Header comment: mention v4g.

Staging/push (after Task 9's src push):
```
mkdir C:/kaggle/staging/cache-v4g C:/kaggle/staging/cache-v4g-b
cp rsna-67knee/notebooks/cache/build_cache.py C:/kaggle/staging/cache-v4g/
cp rsna-67knee/notebooks/cache/build_cache.py C:/kaggle/staging/cache-v4g-b/ && sed -i 's/^SHARD_RANGE = (0, 9)$/SHARD_RANGE = (9, 18)/' C:/kaggle/staging/cache-v4g-b/build_cache.py
# kernel-metadata.json: copy notebooks/cache/kernel-metadata.json, set id/title to patimakornsrinamen/rsna-knee-build-cache-v4g (and -v4g-b)
py -m kaggle kernels push -p C:/kaggle/staging/cache-v4g ; py -m kaggle kernels push -p C:/kaggle/staging/cache-v4g-b
py -m kaggle kernels status patimakornsrinamen/rsna-knee-build-cache-v4g
```
Download (NOT into `C:\kaggle\input` yet - see Constraints):
```
py -m kaggle kernels output patimakornsrinamen/rsna-knee-build-cache-v4g   -p C:/kaggle/staging/cache-v4g-dl/a
py -m kaggle kernels output patimakornsrinamen/rsna-knee-build-cache-v4g-b -p C:/kaggle/staging/cache-v4g-dl/b
```
then, once `grep -q "screens done" C:/kaggle/staging/queue_night.log`: `mkdir C:/kaggle/input/rsna-knee-build-cache-v4g && mv C:/kaggle/staging/cache-v4g-dl/a C:/kaggle/staging/cache-v4g-dl/b C:/kaggle/input/rsna-knee-build-cache-v4g/`. Final layout: `C:\kaggle\input\rsna-knee-build-cache-v4g\a\{cache_000..008.npy, cache_config.json, cache_index.csv}` and `\b\{cache_009..017.npy, ...}` - the two "parts" the walkers find (`find_cache_parts` recurses, so a parent dir with two part subdirs is one root with two parts).

Local build command (only meaningful if the real `train_series` tree is ever placed at `C:\kaggle\input\rsna-knee-abnormality-detection\`; against the rig it builds 3 all-zero studies and serves only as a mechanical check of the script):
```
cp rsna-67knee/notebooks/cache/build_cache.py C:/kaggle/staging/cache-v4g-local/build_cache.py
sed -i 's#^OUT = "/kaggle/working"#OUT = "C:/kaggle/input/rsna-knee-build-cache-v4g/a"#; s/^WORKERS = 4/WORKERS = 24/; s/^SHARD_RANGE = (0, 9)/SHARD_RANGE = (0, 18)/' C:/kaggle/staging/cache-v4g-local/build_cache.py
mkdir -p C:/kaggle/input/rsna-knee-build-cache-v4g/a && cd C:/kaggle/staging/cache-v4g-local && py build_cache.py
```
(`/kaggle/input/...` resolves to `C:\kaggle\input\...` on this machine - how every local run works today; `find_input` picks `rsna-knee-src`, i.e. the local src mirror must be refreshed: `cp rsna-67knee/src/preprocess.py C:/kaggle/input/rsna-knee-src/`.)

Verification: on Kaggle the log prints `recipe={... 'version': 'v4g', 'cardinal_order': True, ...}` and `DONE 2250 studies in shards [0, 9) (N all-zero)`; expect N small (v4 had 0). After download: `py -c "import numpy as np, json; a=np.load('C:/kaggle/input/rsna-knee-build-cache-v4g/a/cache_000.npy', mmap_mode='r'); print(a.shape, a.dtype); print(json.load(open('C:/kaggle/input/rsna-knee-build-cache-v4g/a/cache_config.json')))"` -> `(250, 6, 15, 280, 280) uint8` and `version v4g, n_slots 6`; `wc -l` of the two `cache_index.csv` = 2251 and 2158 (same as v4).

Depends on: Tasks 1-5, 9 (src dataset version pushed) and Task 10c (gate kernel passed).

### Task 7 - cache selection in the train script (`notebooks/train/train_dinov2.py` lines 243-263) and `tools/stage_train.py`

Change `train_dinov2.py`:
- New module constants right above `find_cache_parts`: `CACHE_ROOT = "/kaggle/input"` and `CACHE_VERSION = None  # None: accept every cache part found (pre-v4g behaviour); "v4"/"v4g"/...: keep only parts whose cache_config.json version matches`.
- `find_cache_parts(marker="cache_index.csv", root=None, skip=(...), version=None)`: `root = root or CACHE_ROOT`; after the dedupe, if `version` is not None, keep only `d` where `json.load(open(d/cache_config.json)).get("version") == version`; raise `FileNotFoundError(f"no cache part under {root} with version {version}; found {[(d, v)...]}")` when none survive.
- Line 263: `CACHE_PARTS = find_cache_parts(version=CACHE_VERSION)`.
- The existing `assert _cfg == CACHE_CFG` (line 272) stays as the second guard: it is what makes mixing fail fast today; with the filter, a v4 run started while a v4g dir exists under the root no longer dies.

`tools/stage_train.py`:
- `CACHES["v4g"] = [f"{DATA_OWNER}/rsna-knee-build-cache-v4g", f"{DATA_OWNER}/rsna-knee-build-cache-v4g-b"]`.
- `ARM_CACHE["coatnet2_v4g"] = "v4g"`.
- After the `DUAL_GPU` rewrite: `src = rewrite(src, r'^CACHE_VERSION = .*$', f'CACHE_VERSION = "{cache}"  # staged by tools/stage_train.py', "CACHE_VERSION")` so every staged copy (Kaggle or local) pins its cache version.

Verification: `py rsna-67knee/tools/stage_train.py --user patimakornsrinamen --arm coatnet2_v4g --folds 0 --out C:/kaggle/staging/local-coatnet-v4g-f0` prints `cache=v4g` and `grep -n "^CACHE_VERSION" C:/kaggle/staging/local-coatnet-v4g-f0/train_dinov2.py` shows `"v4g"`. Negative test: run the staged script with `CACHE_VERSION="v4g"` before the cache lands -> must raise the new FileNotFoundError naming the found `(dir, version)` pairs, not the config-mismatch assert.

Depends on: nothing in the preprocess tasks (can be done in parallel with Tasks 1-5); Task 8 depends on it.

### Task 8 - the arm (`notebooks/train/train_dinov2.py` ARMS, after `coatnet2_rs2` line 82)

```python
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
```
The fold split stays on `SEED` and `consensus_v1.csv`, so fold 0 is the same study set as `weights-rs` fold 0 (paired comparison valid).

Launch (after the cache is under `C:\kaggle\input` and the GPU is free):
```
py rsna-67knee/tools/stage_train.py --user patimakornsrinamen --arm coatnet2_v4g --folds 0 --out C:/kaggle/staging/local-coatnet-v4g-f0
sed -i 's/^NUM_WORKERS = 2$/NUM_WORKERS = 0/; s#^OUT = "/kaggle/working"#OUT = "C:/kaggle/working-coatnet-v4g-f0"#; s/^MAX_HOURS = 8.4$/MAX_HOURS = 24.0/' C:/kaggle/staging/local-coatnet-v4g-f0/train_dinov2.py
cp rsna-67knee/src/preprocess.py rsna-67knee/src/model.py C:/kaggle/input/rsna-knee-src/     # local src mirror = what the script imports
cd C:/kaggle/staging/local-coatnet-v4g-f0 && HF_HUB_OFFLINE=1 py train_dinov2.py > train.log 2>&1
```
Expected first log lines: `cache parts (2): ['C:\\kaggle\\input\\rsna-knee-build-cache-v4g\\a', '...\\b']` and `CACHE_CFG: {... 'version': 'v4g' ...}` - two parts, none from `rsna-knee-build-cache{,-b}`. ~5.2 min/epoch, ~1.7 h.

Judge:
```
cd rsna-67knee && py src/eval_windows.py --folds 0 --weights C:/kaggle/working-coatnet-v4g-f0 --labels C:/kaggle/input/rsna-knee-consensus-labels/consensus_v1.csv --strides 3,1 --cache-root C:/kaggle/input/rsna-knee-build-cache-v4g --out C:/kaggle/working-coatnet-v4g-f0/eval_windows.md
py - <<'EOF'
import json, numpy as np, sys; sys.path.insert(0, "src")
from eval_windows import bootstrap_macro_diff, load_labels_csv; from preprocess import LABELS
f = json.load(open("C:/kaggle/staging/weights-rs/fold_of.json")); uids = list(f); idx = [i for i, u in enumerate(uids) if f[u] == 0]
y = np.stack([load_labels_csv("C:/kaggle/input/rsna-knee-consensus-labels/consensus_v1.csv")[u] for u in uids])[idx]
base = np.load("C:/kaggle/staging/weights-rs/oof_stride1_eval.npy")[idx]; chal = np.load("C:/kaggle/working-coatnet-v4g-f0/oof_stride1_eval.npy")[idx]
print("fold0 dense macro diff v4g - rs:", bootstrap_macro_diff(y, base, chal))
EOF
```
(`fold_of.json` of the v4g run must equal weights-rs's - assert it in the snippet.) Decision rule is the caller's; seed-only noise at fold level is documented in `docs/methodology.md`.

LB path (not part of this experiment): a staged `infer_submit.py` copy with `EXPECTED_RECIPE_VERSION["coatnet2"] = "v4g"` and weights whose `run_config.json` carries `version: "v4g"` will decode test studies with the three switches automatically (Task 1). The decode cost rises by one header pass per file; measure with `TIMING_STUDIES` before any submission.

Depends on: Tasks 6, 7.

### Task 9 - push the new `preprocess.py` to the src dataset (needed by every Kaggle kernel)

```
cp rsna-67knee/src/preprocess.py C:/kaggle/staging/rsna-src/preprocess.py
py -m kaggle datasets version -p C:/kaggle/staging/rsna-src -m "preprocess v4g: cardinal order, canonical side, crop window (all off by default)"
```
Backward compatibility is guaranteed by Task 1 (every existing config -> switches False -> byte-identical), so the infer kernel and any queued train kernel keep working on the new version. Verify with `py -m kaggle datasets status patimakornsrinamen/rsna-knee-src` -> `ready`.

Depends on: Tasks 1-5 and the local guards of Task 10a/10b passing.

### Task 10 - guards

**10a. Local byte-identity guard (old module vs new module, switches off).** Before editing `src/preprocess.py`: `cp rsna-67knee/src/preprocess.py "$SCRATCH/preprocess_old.py"` (scratchpad). After Tasks 1-5:
```
py - <<'EOF'
import sys, os, hashlib, importlib.util
def load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
old = load("pre_old", os.environ["SCRATCH"] + "/preprocess_old.py"); new = load("pre_new", "rsna-67knee/src/preprocess.py")
root = "C:/kaggle/input/rsna-knee-abnormality-detection"; by = new.read_series_table(root + "/train_series.csv")
for recipe in ("RECIPE_V2", "RECIPE_V3", "RECIPE_V4", "RECIPE_V5"):
    for uid in sorted(by):
        a, fa = old.build_study_tensor(f"{root}/train_series/{uid}", by[uid], **old.recipe_kwargs(getattr(old, recipe)))
        b, fb = new.build_study_tensor(f"{root}/train_series/{uid}", by[uid], **new.recipe_kwargs(getattr(new, recipe)))
        ha, hb = hashlib.md5(a.tobytes()).hexdigest(), hashlib.md5(b.tobytes()).hexdigest()
        print(recipe, uid, fa == fb, ha == hb, ha); assert fa == fb and ha == hb
print("BYTE-IDENTICAL OK")
EOF
```
Passing = every line `True True` and `BYTE-IDENTICAL OK`. Also run the same with `new.recipe_kwargs({"version": "v4", "size": 280, "crop_mm": 150.0, "unique_slices": True, "isotropic": True, "guarantee_k": True, "expand_frames": True, "strict_geometry": True, "n_slices": 15})` (the literal local `cache_config.json`) to prove a config lacking the new keys decodes identically.

**10b. Synthetic unit test (`src/test_geometry_v4g.py`, new, runs with `py -m pytest rsna-67knee/src/test_geometry_v4g.py -q` or plain `py`).** Build in-memory pydicom `Dataset`s (`pydicom.dataset.FileDataset` with `file_meta`, 16x16 uint16 PixelData, PixelSpacing [1,1], InstanceNumber, IOP, IPP) and write them to a temp series dir. Cases: (1) sagittal IOP `[0,1,0,0,0,-1]` 5 slices with IPP x = 10..50 -> `order_slices_cardinal` returns `flipped_normal=True` and order reversed vs `sort_series_files(strict_geometry=True)`; (2) IOP `[0,1,0,0,0,1]` (r=[0,1,0], c=[0,0,1], r x c = [1,0,0]) -> `flipped_normal=False`, order identical; (3) two slices with equal IPP -> `rejected_duplicate_position`; (4) no IPP -> `v4_fallback_no_geometry` and refs == v4 refs; (5) `decide_study_side` cases from Task 3; (6) `flip_for_plane` cases; (7) MONOCHROME1 + slope/intercept exactness from Task 4; (8) a 3-series study (sag/cor/ax, Laterality "R") through `build_study_tensor(**recipe_kwargs(RECIPE_V4G), receipts=rc)`: `rc["side"] == "R"`, coronal slot row 0 equals `np.fliplr` of the same study built with `canonical_side=False` (other switches on), sagittal slot rows equal the unflipped build's rows reversed over the filled range; (9) multi-frame: one Enhanced file with 4 frames whose per-frame IPP decreases along the raw normal -> cardinal order ascending, `n_multiframe_files == 1`. Passing = all asserts hold.

**10c. Kaggle gate kernel (`notebooks/cache/gate_v4g.py`, new; staged at `C:\kaggle\staging\gate-v4g\`, metadata = copy of `notebooks/cache/kernel-metadata.json` with id `patimakornsrinamen/rsna-knee-gate-v4g`).** Same imports as `build_cache.py`. Part 1 (byte identity on real data): build `sorted(by_study)[0:3]` with `**recipe_kwargs(RECIPE_V4)` (new module) into `tiny_v4.npy` of shape `(3,6,15,280,280)` and print its md5 and per-study `filled`. Part 2 (smoke, 20 studies = `sorted(by_study)[0:20]`): for each study call `build_study_tensor(..., **recipe_kwargs(RECIPE_V4G), receipts=rc)` and print one study line `uid[-8:] side=<L|R|None> source=<...> tags=<...> sag_x_med=<float|None> filled=<n>` and one line per series `  <plane> fluid=<0|1> files=<n> method=<...> flipped_normal=<bool> flip=<bool> reversed=<bool> photometric=<...> slope_range=<...> n_multiframe=<n> n_lat_conflict=<n> min_delta=<...>`; then a summary: counts of side sources, of `flipped_normal` per plane, of methods, of MONOCHROME1 series, of series with slope != 1 or intercept != 0, of multi-frame files, of rejected series. Also print, for the 20 studies, `filled` under v4 vs v4g (any drop = rejection). Push with `py -m kaggle kernels push -p C:/kaggle/staging/gate-v4g`; fetch with `py -m kaggle kernels output patimakornsrinamen/rsna-knee-gate-v4g -p C:/kaggle/staging/gate-v4g-out` (output is ~21 MB + log, safe).
Local comparison:
```
py -c "import numpy as np, hashlib; a=np.load('C:/kaggle/input/rsna-knee-build-cache/cache_000.npy', mmap_mode='r'); b=np.load('C:/kaggle/staging/gate-v4g-out/tiny_v4.npy'); print(np.array_equal(np.asarray(a[:3]), b), hashlib.md5(np.ascontiguousarray(a[:3]).tobytes()).hexdigest())"
```
Passing = `True` and the md5 equals the one printed in the kernel log. This is the real-data proof that the switched-off module reproduces the shipped v4 cache bit for bit (same pydicom/numpy/cv2 images on Kaggle as the v4 build). Gate for Task 6: this passes AND the smoke shows at least one `flipped_normal=True` sagittal series or one `side=R` study (otherwise the switches are inert on this data and the experiment is moot - stop and report).

Depends on: 10a/10b before Task 9; 10c after Task 9, before Task 6.

## Risks and how (a)-(c) apply to them

- **Multi-frame DICOMs** (v4 expands frames; count in the real data unknown - the gate kernel prints it). (a): each frame is a slice entry with its own `PerFrameFunctionalGroupsSequence[i].PlanePositionSequence[0].ImagePositionPatient` projected on the cardinalised shared normal (`SharedFunctionalGroupsSequence[0].PlaneOrientationSequence[0]` when the root IOP is absent); frames with no per-frame IPP sort after positioned entries by `(inst, name, frame_idx)`, as v4's positionless files do. Multi-file multi-frame series are sorted globally by projection (v4 sorted files by root position and frames inside files; for single-file series the two agree up to the sign). (b): tags from root `Laterality`/`ImageLaterality`, else `Shared...FrameAnatomySequence[0].FrameLaterality`; `ipp_x_median` over per-frame IPP x. (c): slope/intercept from root, else `Shared...PixelValueTransformationSequence[0]`, else the frame's own `PerFrame...PixelValueTransformationSequence[0]`. cnx itself rejects such files (root IPP/IOP required); we keep v4's expansion.
- **Series with no IPP** (or no IOP): (a) cannot cardinalise -> v4 order (`v4_fallback_no_geometry`), counted in the receipt; it contributes nothing to the sagittal-x fallback; (b)/(c) still apply.
- **Mixed laterality tags**: two different values across the study -> `conflicting_laterality_tags`, no flip, no reversal (cnx). A single file whose `Laterality` and `ImageLaterality` disagree contributes no tag (cnx would reject that file's series); counted as `n_lat_conflict`.
- **Duplicate positions** reject the series (cnx: GeometryError on `delta <= 1e-3 mm`). Ours drops the series from slot candidates so a second same-slot series can fill the slot; a slot with no surviving series stays zero. If the gate shows many rejections (v4 had 0 all-zero studies), that is a data fact to report before building; an alternative is `rejected -> v4 order` (one-line change in Task 5), but it must be decided, not silently mixed.
- **Sign of the sagittal-x fallback**: cnx's `x >= 0 -> "L"` (DICOM +x = patient left) is copied verbatim; a different competitor uses image-centre x with a +/-20 mm dead zone. With tags absent and the knee near isocentre the sign can be weak; the smoke prints the medians so the caller can see how far from 0 they sit.
- **No in-plane canonicalisation** (out of scope): the mirror assumes the usual IOP in-plane sign; a mirrored/rotated acquisition would be flipped the wrong way. cnx fixes that with its D4 step; v4g does not.
- **Window/resize order change in (c)** makes v4g pixels differ from v4 even on a left knee with standard geometry - intended, it is factor (c).
- **Local root walk**: any local run or `eval_windows` without `--cache-root` that starts while both v4 and v4g live under `C:\kaggle\input` fails fast at the config assert unless `CACHE_VERSION` is set (Task 7). Not a silent-mix risk; a scheduling one (see Constraints).
- **Kaggle concurrency**: unverified whether two CPU kernels run in parallel; if serialised, add ~35 min.

## Unverified (stated plainly)

- Prevalence in the real data of: Laterality/ImageLaterality tags, MONOCHROME1, RescaleSlope != 1 / Intercept != 0, Enhanced multi-frame files, negative-normal sagittal stacks, duplicate IPPs. The gate kernel measures all of them on 20 studies; the build log's all-zero count covers rejections globally.
- That `/kaggle/input` -> `C:\kaggle\input` resolution holds for `build_cache.py` (it holds for `train_dinov2.py`, per memory and the running local jobs).
- Kaggle CPU session availability right now (GPU quota is spent; CPU kernels were used for the stride-3 gates this week).
