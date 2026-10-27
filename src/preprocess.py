"""Shared preprocessing for the RSNA Knee pixel cache.

One study -> uint8 tensor of shape (6 slots, n_slices slices, size, size)
(n_slices defaults to 9; v4 raises it to 15). The active recipe travels as
cache_config.json / run_config.json (v2: size=280, crop_mm=150,
unique_slices=True); the module-level defaults below reproduce the retired
v1 cache (224, no crop, duplicated slices).

Slots are (Anatomical_Plane x Fluid_Sensitive). Two slot layouts exist,
selected by the recipe's slot_layout key (n_slots(layout) gives the count):
"plane6" (default, v2-v4):
    0 Sagittal fluid   1 Sagittal non-fluid
    2 Coronal  fluid   3 Coronal  non-fluid
    4 Axial    fluid   5 Axial    non-fluid
"raptor5" (v5, the public Raptor recipe's order; axial non-fluid has no slot):
    0 Sagittal fluid   1 Sagittal non-fluid
    2 Coronal  fluid   3 Coronal  non-fluid
    4 Axial    fluid

Within a slot the series with the most slices wins (ties: fat-suppressed first,
then lower SeriesInstanceUID for determinism). Missing slots stay zero.

Slices are sorted anatomically by projecting ImagePositionPatient onto the slice
normal from ImageOrientationPatient (filename order is random in this dataset),
with fallbacks SliceLocation -> InstanceNumber -> filename. n_slices slices are
sampled evenly from the middle 80% of the stack; v3 widens that span until
n_slices distinct indices exist (series shorter than n_slices contribute every
slice, remaining rows stay zero). Multi-frame files with few siblings are
expanded frame-by-frame. Each slice is windowed to its own 1st..99th intensity
percentile computed on the FULL frame, then center-cropped (crop_mm) and
resized. Window-before-crop is the recorded v2 recipe order: reordering
changes pixels and requires a cache rebuild plus retrain.

v4 (strict_geometry=True) fixes seven geometry/sampling defects found by
audit (frame expansion gated by file count, frame order, position-sort
fallback, empty-spacing fallback, crop lower bound, round-half-to-even
sampling) - see the `strict_geometry` branches below. Every fix is gated
behind the flag because stage A trains on this module in v3 mode against an
already-built cache; a v3-mode output must stay byte-identical to the
pre-v4 module or the cache and inference silently disagree with the weights.

v5 adds two more recipe knobs, both gated the same way (None reproduces v2-v4
exactly): slot_layout switches the slot table (see above); slices_per_slot
gives each slot its own slice count (a tuple/list indexed by slot, None means
every slot uses n_slices) so build_study_tensor's output is
(n_slots, max(slices_per_slot), size, size) with per-slot zero padding beyond
each slot's own count; span replaces the centre-fraction sampling with a
direct (lo, hi) fraction of the ordered stack (sample_indices' `span` branch),
still honoring unique/guarantee_k/strict_geometry.

v4g = v4 plus three boolean keys (cardinal_order, canonical_side, crop_window),
each gated: cardinal slice order (order_slices_cardinal), study-level laterality
canonicalisation (decide_study_side, flip_for_plane) and modality-LUT + per-series
crop windowing (load_crop_float, window_series_crops). False/None reproduces v4
(and v2-v5) byte for byte; see docs/specs/geometry-v4g.md.
"""

from __future__ import annotations

import os

import numpy as np
import pydicom

N_SLOTS = 6
N_SLICES = 9
IMG_SIZE = 224
CENTER_FRACTION = 0.8  # middle portion of the stack that slices are sampled from
# series with at most this many files are probed for NumberOfFrames (v3
# multi-frame expansion); larger series are assumed single-frame per file
FRAME_PROBE_MAX_FILES = 3

PLANES = ("Sagittal", "Coronal", "Axial")

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

# Recipes. v2 = the 0.867 cache (280 px, 150 mm crop, unique sampling); v3 adds
# the isotropic crop clamp, a full-9 sampling guarantee, and multi-frame
# expansion. v4 raises n_slices to 15 and turns on strict_geometry (the seven
# fixes below). v5 switches to the 5-slot Raptor layout, per-slot slice counts,
# and a (lo, hi) span in place of the centre band. The cache kernel dumps the
# active recipe to cache_config.json, the train kernel carries it into
# run_config.json, and consumers pass it back via recipe_kwargs(). Keys not in
# _RECIPE_KEYS (e.g. version) are informational.
RECIPE_V2 = {
    "version": "v2",
    "size": 280,
    "crop_mm": 150.0,
    "unique_slices": True,
    "isotropic": False,
    "guarantee_k": False,
    "expand_frames": False,
    "strict_geometry": False,
    "n_slices": 9,
    "slot_layout": "plane6",
    "slices_per_slot": None,
    "span": None,
    "cardinal_order": False,
    "canonical_side": False,
    "crop_window": False,
}
RECIPE_V3 = {**RECIPE_V2, "version": "v3", "isotropic": True,
             "guarantee_k": True, "expand_frames": True}
RECIPE_V4 = {**RECIPE_V3, "version": "v4", "n_slices": 15,
             "strict_geometry": True}
RECIPE_V5 = {**RECIPE_V4, "version": "v5", "slot_layout": "raptor5",
             "slices_per_slot": (18, 14, 12, 8, 12), "n_slices": 18,
             "span": (0.02, 0.98), "crop_mm": 140.0, "size": 336}
RECIPE_V4G = {**RECIPE_V4, "version": "v4g", "cardinal_order": True,
              "canonical_side": True, "crop_window": True}
_RECIPE_KEYS = ("size", "crop_mm", "unique_slices", "isotropic",
                "guarantee_k", "expand_frames", "strict_geometry", "n_slices",
                "slot_layout", "slices_per_slot", "span", "cardinal_order",
                "canonical_side", "crop_window")


def recipe_kwargs(cfg: dict) -> dict:
    """Pick the build_study_tensor recipe keys out of a config dict (which may
    also carry label_file, epochs, AUCs...). Keys a v2-era config does not
    carry fall back to RECIPE_V2, never to the v1 function defaults (the three
    v4g keys cardinal_order/canonical_side/crop_window default to False for
    every pre-v4g config).

    slices_per_slot round-trips a JSON-dumped cache_config.json as a list
    (JSON has no tuple type); normalized back to a tuple here so downstream
    indexing (slices_per_slot[slot]) matches the recipe dict literals.

    Round-2/3-era run_config.json files (see the dry-run fake mount) carry an
    unrelated informational int field of the same name (e.g.
    "slices_per_slot": 1) predating this recipe key; any value that is not a
    list/tuple is coerced to None here instead of crashing on tuple(int)."""
    picked = {k: cfg[k] for k in _RECIPE_KEYS if k in cfg}
    kw = {**{k: RECIPE_V2[k] for k in _RECIPE_KEYS}, **picked}
    sps = kw["slices_per_slot"]
    kw["slices_per_slot"] = tuple(sps) if isinstance(sps, (list, tuple)) else None
    # span round-trips as a list too; a tuple keeps the recipe dict hashable
    # (infer groups families by tuple(sorted(recipe.items())))
    span = kw["span"]
    kw["span"] = tuple(span) if isinstance(span, (list, tuple)) else None
    return kw


def find_input(*candidates: str) -> str:
    """First existing path among old/new Kaggle mount layouts."""
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(candidates)


def find_by_marker(marker: str, root: str = "/kaggle/input",
                   skip: tuple = ("competitions",
                                  "rsna-knee-abnormality-detection")) -> str:
    """Locate the input dir containing `marker`, whatever the mount layout.

    Deterministic (sorted walk); prunes `skip` top-level dirs so a marker
    search never descends into the multi-hundred-GB competition mount.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath == root:
            dirnames[:] = sorted(d for d in dirnames if d not in skip)
        else:
            dirnames.sort()
        if marker in filenames:
            return dirpath
    raise FileNotFoundError(f"{marker} not under {root} (skipped {skip})")


def read_series_table(path: str) -> dict:
    """test_series.csv / train_series.csv -> {StudyInstanceUID: [row dicts]}."""
    import csv
    by_study: dict = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by_study.setdefault(row["StudyInstanceUID"], []).append(row)
    return by_study


def _flag(value) -> int:
    """CSV cell -> 0/1, tolerating blanks and float strings ("1.0")."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


# raptor5: sag-fluid, sag-nonfluid, cor-fluid, cor-nonfluid, ax-fluid.
# Axial non-fluid has no slot in this layout (mapping.get -> None).
_RAPTOR5_SLOTS = {
    ("Sagittal", 1): 0, ("Sagittal", 0): 1,
    ("Coronal", 1): 2, ("Coronal", 0): 3,
    ("Axial", 1): 4,
}


def n_slots(layout: str = "plane6") -> int:
    """Slot count for a slot_layout name."""
    return {"plane6": 6, "raptor5": 5}[layout]


def n_slots_for(cfg: dict) -> int:
    """Slot count for a recipe/config dict (defaults to plane6, like RECIPE_V2)."""
    return n_slots(cfg.get("slot_layout", "plane6"))


def slot_index(plane: str, fluid_sensitive: int,
               layout: str = "plane6") -> int | None:
    """Map a series' plane/fluid flags to a cache slot, or None if unmapped
    (unknown plane in "plane6"; axial non-fluid in "raptor5")."""
    if layout == "raptor5":
        return _RAPTOR5_SLOTS.get((plane, 1 if fluid_sensitive else 0))
    try:
        p = PLANES.index(plane)
    except ValueError:
        return None
    return p * 2 + (0 if fluid_sensitive else 1)


def pick_slot_series(series_rows, layout: str = "plane6"):
    """From train_series/test_series rows of one study, choose one series per slot.

    series_rows: iterable of dicts with keys SeriesInstanceUID, Anatomical_Plane,
    Fluid_Sensitive, Fat_Suppression, and n_slices (caller counts the files).
    Returns {slot: SeriesInstanceUID}.
    """
    chosen = {}
    for row in series_rows:
        s = slot_index(row["Anatomical_Plane"], _flag(row["Fluid_Sensitive"]),
                       layout)
        if s is None:
            continue
        # rank: most slices, then fat-suppressed, then lowest UID (determinism)
        key = (int(row["n_slices"]), _flag(row["Fat_Suppression"]),
               row["SeriesInstanceUID"])
        best = chosen.get(s)
        if best is None or (key[0], key[1]) > (best[0], best[1]) or (
                (key[0], key[1]) == (best[0], best[1]) and key[2] < best[2]):
            chosen[s] = key
    return {s: k[2] for s, k in chosen.items()}


def _sort_key_position(ds) -> float | None:
    ipp = getattr(ds, "ImagePositionPatient", None)
    iop = getattr(ds, "ImageOrientationPatient", None)
    if ipp is None or iop is None or len(iop) != 6:
        return None
    row = np.asarray(iop[:3], dtype=float)
    col = np.asarray(iop[3:], dtype=float)
    normal = np.cross(row, col)
    return float(np.dot(np.asarray(ipp, dtype=float), normal))


def sort_series_files(paths: list[str], strict_geometry: bool = False) -> list[str]:
    """Anatomical sort. Reads headers only (no pixels).

    strict_geometry=False (v2/v3): position order is used only if EVERY file
    has a position; one file missing it drops position ordering for the whole
    series. strict_geometry=True (v4): sort key (pos is None, pos or 0.0,
    inst, name) so positioned files keep position order and only the
    positionless ones fall back to instance order, at the end."""
    keyed = []
    for p in paths:
        pos, inst = None, 0
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True)
            pos = _sort_key_position(ds)
            if pos is None:
                loc = getattr(ds, "SliceLocation", None)
                pos = float(loc) if loc not in (None, "") else None
            n = getattr(ds, "InstanceNumber", None)
            inst = int(n) if n not in (None, "") else 0
        except Exception:
            pass  # unreadable header: keep the file, fall back to name order
        keyed.append((p, pos, inst))
    if strict_geometry:
        keyed.sort(key=lambda k: (k[1] is None, k[1] or 0.0, k[2],
                                  os.path.basename(k[0])))
    elif all(k[1] is not None for k in keyed):
        keyed.sort(key=lambda k: (k[1], k[2], os.path.basename(k[0])))
    else:
        keyed.sort(key=lambda k: (k[2], os.path.basename(k[0])))
    return [k[0] for k in keyed]


def sample_indices(n: int, k: int = N_SLICES, center: float = CENTER_FRACTION,
                   unique: bool = False, guarantee_k: bool = False,
                   strict_geometry: bool = False, span: tuple | None = None):
    """k evenly spaced indices from the middle `center` fraction of n slices.

    unique=False reproduces the v1 cache (short series repeat slices).
    unique=True (v2) deduplicates, returning fewer than k for short series so
    the missing cache rows stay zero instead of over-weighting one slice.
    guarantee_k=True (v3) widens the sampled span toward the full stack until
    k distinct indices exist (always possible when n >= k); for n < k every
    slice is returned. Series with n >= 11 are unaffected (v2 == v3).
    strict_geometry=True (v4) rounds half-up (np.floor(x + 0.5)) instead of
    np.round's round-half-to-even, so a true centre index (e.g. n=10, k=9,
    idx=4.5) is not always dropped in the same direction.
    span=(lo, hi) (v5) replaces the centre-fraction pick with k indices
    linearly spaced over [lo*n, hi*n] of the ordered stack (the public
    Raptor recipe's span, e.g. (0.02, 0.98)); unique/guarantee_k widen this
    span outward (toward (0, 1)) instead of widening `center`, and
    strict_geometry still selects the rounding rule. span=None (default)
    reproduces the v2-v4 centre-fraction behavior exactly.
    """
    if n <= 0:
        return []
    unique = unique or guarantee_k  # a full-k guarantee only makes sense on distinct picks
    if guarantee_k and n < k:
        return list(range(n))

    def round_idx(idx):
        if strict_geometry:
            return np.floor(idx + 0.5).astype(int)
        return np.round(idx).astype(int)

    if span is not None:
        def pick_span(lo, hi):
            idx = np.linspace(lo * n, hi * n, k)
            return np.clip(round_idx(idx), 0, n - 1)

        lo0, hi0 = span
        idx = pick_span(lo0, hi0)
        if unique:
            idx = np.unique(idx)
            if guarantee_k:
                lo, hi = lo0, hi0
                while idx.size < k and (lo > 0.0 or hi < 1.0):
                    lo = max(0.0, lo - 0.05)
                    hi = min(1.0, hi + 0.05)
                    idx = np.unique(pick_span(lo, hi))
        return idx.tolist()

    def pick(c):
        lo = (1.0 - c) / 2.0
        idx = np.linspace(lo * (n - 1), (1.0 - lo) * (n - 1), k)
        return np.clip(round_idx(idx), 0, n - 1)

    idx = pick(center)
    if unique:
        idx = np.unique(idx)
        if guarantee_k:
            c = center
            while idx.size < k and c < 1.0:
                c = min(1.0, c + 0.05)
                idx = np.unique(pick(c))
    return idx.tolist()


def window_to_uint8(pixels: np.ndarray) -> np.ndarray:
    """Per-slice robust windowing: 1st..99th percentile -> 0..255."""
    x = pixels.astype(np.float32)
    lo, hi = np.percentile(x, (1.0, 99.0))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.uint8)
    x = (x - lo) / (hi - lo)
    return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def resize_uint8(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    import cv2
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def pixel_spacing(ds, strict_geometry: bool = False):
    """(row_mm, col_mm) or None. Enhanced multi-frame SOPs keep PixelSpacing
    under SharedFunctionalGroupsSequence, not at the dataset root.

    strict_geometry=True (v4) also falls back on a present-but-empty root
    PixelSpacing (`[]`), not just a missing one."""
    spacing = getattr(ds, "PixelSpacing", None)
    missing = spacing is None if not strict_geometry else not spacing
    if missing:
        try:
            spacing = (ds.SharedFunctionalGroupsSequence[0]
                       .PixelMeasuresSequence[0].PixelSpacing)
        except Exception:
            return None
    try:
        row_sp, col_sp = float(spacing[0]), float(spacing[1])
    except (TypeError, ValueError, IndexError):
        return None
    if row_sp <= 0 or col_sp <= 0:
        return None
    return row_sp, col_sp


def center_crop_mm(img: np.ndarray, ds, crop_mm: float,
                   isotropic: bool = False,
                   strict_geometry: bool = False) -> np.ndarray:
    """Center-crop `img` to a crop_mm x crop_mm physical window using
    PixelSpacing; returns img unchanged when spacing is missing/invalid.

    isotropic=False (v2) clamps each axis to the image independently, which
    squashes rectangular acquisitions anisotropically after the square resize.
    isotropic=True (v3) uses one physical extent for both axes, so mm/px is
    equal on both axes for every study.
    strict_geometry=True (v4) also clamps ch/cw to at least 1 px, so an
    absurd PixelSpacing cannot compute a 0-px crop."""
    sp = pixel_spacing(ds, strict_geometry=strict_geometry)
    if sp is None:
        return img
    row_sp, col_sp = sp
    h, w = img.shape[:2]
    if isotropic:
        c_mm = min(h * row_sp, w * col_sp, crop_mm)
        ch = min(h, int(round(c_mm / row_sp)))
        cw = min(w, int(round(c_mm / col_sp)))
    else:
        ch = min(h, int(round(crop_mm / row_sp)))
        cw = min(w, int(round(crop_mm / col_sp)))
    if strict_geometry:
        ch, cw = max(1, ch), max(1, cw)
    top = (h - ch) // 2
    left = (w - cw) // 2
    return img[top:top + ch, left:left + cw]


def _read_pixels(path: str, cache: dict | None = None):
    """(dataset, pixel_array) for `path`; memoised in `cache` when given so a
    multi-frame file is decoded once, not once per sampled frame."""
    if cache is not None and path in cache:
        return cache[path]
    ds = pydicom.dcmread(path)
    hit = (ds, ds.pixel_array)
    if cache is not None:
        cache[path] = hit
    return hit


def load_slice(path: str, size: int = IMG_SIZE,
               crop_mm: float | None = None, frame: int | None = None,
               isotropic: bool = False,
               pixels_cache: dict | None = None,
               strict_geometry: bool = False) -> np.ndarray:
    """One DICOM (optionally one frame of a multi-frame file) -> uint8 image.
    pixels_cache: optional {path: (ds, pixel_array)} shared across the frames
    of one file (see _read_pixels).

    strict_geometry=True (v4): a `frame` index against a 2-D pixel array (no
    frame axis to index) raises instead of silently returning the same image
    for every requested frame; build_study_tensor's per-slice try/except then
    leaves that slice blank."""
    ds, pixels = _read_pixels(path, pixels_cache)
    if pixels.ndim == 4:  # RGB multi-frame (frames, H, W, 3)
        pixels = pixels.mean(axis=3)
    if pixels.ndim == 3:
        if pixels.shape[-1] == 3:  # RGB secondary capture
            pixels = pixels.mean(axis=2)
        elif frame is not None:  # multi-frame, expanded by the caller
            pixels = pixels[frame]
        else:  # multi-frame, v2 behavior: middle frame
            pixels = pixels[pixels.shape[0] // 2]
    elif frame is not None and strict_geometry:
        raise ValueError(f"frame={frame} requested but {path} decoded to a "
                         f"2-D pixel array (no frame axis)")
    img = window_to_uint8(pixels)
    if crop_mm is not None:
        img = center_crop_mm(img, ds, crop_mm, isotropic=isotropic,
                             strict_geometry=strict_geometry)
    return resize_uint8(img, size)


def load_crop_float(path: str, crop_mm: float | None, frame: int | None = None,
                    pixels_cache: dict | None = None, isotropic: bool = True,
                    strict_geometry: bool = True) -> np.ndarray:
    """v4g change (c), first half: one DICOM frame -> float32 crop with the
    modality LUT applied (x * RescaleSlope + RescaleIntercept) and MONOCHROME1
    inverted (min + max - x on the full frame), cropped with the unchanged
    center_crop_mm. Windowing happens later per series (window_series_crops)."""
    ds, pixels = _read_pixels(path, pixels_cache)
    if pixels.ndim == 4:  # RGB multi-frame (frames, H, W, 3)
        pixels = pixels.mean(axis=3)
    if pixels.ndim == 3:
        if pixels.shape[-1] == 3:  # RGB secondary capture
            pixels = pixels.mean(axis=2)
        elif frame is not None:  # multi-frame, expanded by the caller
            pixels = pixels[frame]
        else:  # multi-frame, v2 behavior: middle frame
            pixels = pixels[pixels.shape[0] // 2]
    elif frame is not None and strict_geometry:
        raise ValueError(f"frame={frame} requested but {path} decoded to a "
                         f"2-D pixel array (no frame axis)")
    x = pixels.astype(np.float32)
    slope, intercept, _ = _rescale(ds, frame)
    x = x * slope + intercept
    photometric = str(getattr(ds, "PhotometricInterpretation", "")).strip().upper()
    if photometric == "MONOCHROME1":
        x = (float(x.min()) + float(x.max())) - x
    if crop_mm is None:
        return x
    return center_crop_mm(x, ds, crop_mm, isotropic=isotropic,
                          strict_geometry=strict_geometry)


def window_series_crops(crops: list, size: int, flip: bool) -> list:
    """v4g change (c), second half: window the selected float crops of ONE
    series with a shared 1st..99th percentile of their 4x4-subsampled pixels,
    resize in float32 (INTER_AREA), optionally mirror columns (laterality), and
    truncate to uint8. Returns one uint8 (size, size) image per crop."""
    import cv2
    if not crops:
        return []
    sample = np.concatenate([c[::4, ::4].ravel() for c in crops])
    lo, hi = np.percentile(sample, [1, 99])
    denom = max(float(hi - lo), 1e-6)
    out = []
    for c in crops:
        y = np.clip((c - lo) / denom, 0, 1).astype(np.float32)
        img = np.clip(cv2.resize(y, (size, size),
                                 interpolation=cv2.INTER_AREA), 0, 1)
        if flip:
            img = img[:, ::-1]
        out.append((img * 255).astype(np.uint8))
    return out


def _frame_count(path: str) -> int:
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        return max(1, int(getattr(ds, "NumberOfFrames", 1) or 1))
    except Exception:
        return 1


def _frame_order(path: str, n: int, strict_geometry: bool = False) -> list[int]:
    """Frame indices for one multi-frame file, in anatomical order.

    strict_geometry=False: storage order (range(n)), the v3 assumption.
    strict_geometry=True (v4): project each frame's ImagePositionPatient
    (from PerFrameFunctionalGroupsSequence) onto the shared
    ImageOrientationPatient normal, exactly as _sort_key_position, and sort
    frame indices by that; falls back to storage order if any of the
    per-frame geometry is missing."""
    if not strict_geometry:
        return list(range(n))
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        iop = getattr(ds, "ImageOrientationPatient", None)
        if iop is None:
            iop = (ds.SharedFunctionalGroupsSequence[0]
                   .PlaneOrientationSequence[0].ImageOrientationPatient)
        if iop is None or len(iop) != 6:
            return list(range(n))
        row = np.asarray(iop[:3], dtype=float)
        col = np.asarray(iop[3:], dtype=float)
        normal = np.cross(row, col)
        pfg = ds.PerFrameFunctionalGroupsSequence
        if len(pfg) != n:
            return list(range(n))
        positions = [
            float(np.dot(np.asarray(
                grp.PlanePositionSequence[0].ImagePositionPatient,
                dtype=float), normal))
            for grp in pfg]
        return [i for i, _ in sorted(enumerate(positions), key=lambda t: t[1])]
    except Exception:
        return list(range(n))  # missing/unreadable per-frame geometry


CARDINAL_NORMAL = {"Sagittal": (1.0, 0.0, 0.0), "Coronal": (0.0, 1.0, 0.0),
                   "Axial": (0.0, 0.0, 1.0)}   # cnx CARDINAL_NORMAL
POSITION_TOLERANCE_MM = 1e-3                      # cnx POSITION_TOLERANCE_MM


def _normalize_laterality(value) -> str | None:
    if value is None:
        return None
    v = str(value).strip().upper()
    if v in ("L", "LEFT"):
        return "L"
    if v in ("R", "RIGHT"):
        return "R"
    return None


def _vec(value, n: int):
    """value -> finite float64 (n,) array, else None."""
    try:
        arr = np.asarray([float(x) for x in value], dtype=np.float64)
    except Exception:
        return None
    if arr.shape != (n,) or not np.all(np.isfinite(arr)):
        return None
    return arr


def _rescale(ds, frame: int | None = None):
    """(slope, intercept, bad) for a dataset: first of root -> Shared
    PixelValueTransformationSequence -> (frame given) the frame's own
    PixelValueTransformationSequence that carries RescaleSlope/Intercept;
    defaults 1.0/0.0; a non-finite or zero slope (or non-finite intercept)
    yields the defaults with bad=True."""
    sources = [ds]
    try:
        sources.append(ds.SharedFunctionalGroupsSequence[0]
                       .PixelValueTransformationSequence[0])
    except Exception:
        pass
    if frame is not None:
        try:
            sources.append(ds.PerFrameFunctionalGroupsSequence[frame]
                           .PixelValueTransformationSequence[0])
        except Exception:
            pass
    for src in sources:
        sl = getattr(src, "RescaleSlope", None)
        ic = getattr(src, "RescaleIntercept", None)
        if sl in (None, "") and ic in (None, ""):
            continue
        try:
            slope = float(sl) if sl not in (None, "") else 1.0
        except (TypeError, ValueError):
            slope = float("nan")
        try:
            intercept = float(ic) if ic not in (None, "") else 0.0
        except (TypeError, ValueError):
            intercept = float("nan")
        if (not np.isfinite(slope) or slope == 0.0
                or not np.isfinite(intercept)):
            return 1.0, 0.0, True
        return slope, intercept, False
    return 1.0, 0.0, False


def _read_slice_headers(files: list[str]) -> list[dict]:
    """One header-only read per file -> geometry/laterality/rescale records
    (see order_slices_cardinal). An unreadable header yields a record with
    every geometry field None, so the file is kept like v4 keeps it."""
    records = []
    for p in files:
        rec = {"path": p, "inst": 0, "sloc": None, "ipp": None, "iop": None,
               "n_frames": 1, "frame_ipp": [], "lat": None, "photometric": "",
               "slope": 1.0, "intercept": 0.0, "bad_rescale": False}
        records.append(rec)
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True)
        except Exception:
            continue
        try:
            n = getattr(ds, "InstanceNumber", None)
            rec["inst"] = int(n) if n not in (None, "") else 0
        except Exception:
            pass
        try:
            loc = getattr(ds, "SliceLocation", None)
            if loc not in (None, ""):
                f = float(loc)
                rec["sloc"] = f if np.isfinite(f) else None
        except Exception:
            pass
        rec["ipp"] = _vec(getattr(ds, "ImagePositionPatient", None), 3)
        iop = _vec(getattr(ds, "ImageOrientationPatient", None), 6)
        if iop is None:
            try:
                iop = _vec(ds.SharedFunctionalGroupsSequence[0]
                           .PlaneOrientationSequence[0]
                           .ImageOrientationPatient, 6)
            except Exception:
                iop = None
        rec["iop"] = iop
        try:
            nf = max(1, int(getattr(ds, "NumberOfFrames", 1) or 1))
        except Exception:
            nf = 1
        rec["n_frames"] = nf
        if nf > 1:
            fipp = [None] * nf
            try:
                pfg = ds.PerFrameFunctionalGroupsSequence
                if len(pfg) == nf:
                    for i, grp in enumerate(pfg):
                        try:
                            fipp[i] = _vec(grp.PlanePositionSequence[0]
                                           .ImagePositionPatient, 3)
                        except Exception:
                            fipp[i] = None
            except Exception:
                pass
            rec["frame_ipp"] = fipp
        a = _normalize_laterality(getattr(ds, "Laterality", None))
        b = _normalize_laterality(getattr(ds, "ImageLaterality", None))
        if a is not None and b is not None and a != b:
            lat = "conflict"
        else:
            lat = a or b
        if lat is None:
            try:
                lat = _normalize_laterality(
                    ds.SharedFunctionalGroupsSequence[0]
                    .FrameAnatomySequence[0].FrameLaterality)
            except Exception:
                lat = None
        rec["lat"] = lat
        rec["photometric"] = str(
            getattr(ds, "PhotometricInterpretation", "") or "").strip().upper()
        rec["slope"], rec["intercept"], rec["bad_rescale"] = _rescale(ds, None)
    return records


def _expand_entries(records: list[dict]) -> list[tuple]:
    """Slice entries (path, frame, ipp, iop, inst, name, frame_idx): one per
    single-frame file, one per frame of a multi-frame file."""
    entries = []
    for rec in records:
        name = os.path.basename(rec["path"])
        if rec["n_frames"] > 1:
            for f in range(rec["n_frames"]):
                fi = rec["frame_ipp"][f] if f < len(rec["frame_ipp"]) else None
                entries.append((rec["path"], f, fi, rec["iop"], rec["inst"],
                                name, f))
        else:
            entries.append((rec["path"], None, rec["ipp"], rec["iop"],
                            rec["inst"], name, 0))
    return entries


def _receipt_base(records: list[dict], entries: list[tuple]) -> dict:
    xs = [float(e[2][0]) for e in entries if e[2] is not None]
    return {
        "method": None, "flipped_normal": False, "normal": None,
        "n_entries": len(entries),
        "n_positioned": sum(1 for e in entries if e[2] is not None),
        "min_delta": None,
        "ipp_x_median": float(np.median(xs)) if xs else None,
        "lat_values": sorted({r["lat"] for r in records
                              if r["lat"] in ("L", "R")}),
        "n_lat_conflict": sum(1 for r in records if r["lat"] == "conflict"),
        "photometric_values": sorted({r["photometric"] for r in records
                                      if r["photometric"]}),
        "n_bad_rescale": sum(1 for r in records if r["bad_rescale"]),
        "n_multiframe_files": sum(1 for r in records if r["n_frames"] > 1),
    }


def _light_receipt(records: list[dict]) -> dict:
    """Receipt without ordering (canonical_side on, cardinal_order off)."""
    rc = _receipt_base(records, _expand_entries(records))
    rc["method"] = "v4_order"
    return rc


def order_slices_cardinal(records: list[dict], plane: str):
    """(refs, receipt) for one series: [(path, frame)] ordered by position
    projected on the cardinalised slice normal (v4g change (a)). Sagittal
    stacks run patient-right -> left, coronal anterior -> posterior, axial
    inferior -> superior. Falls back to the v4 order when geometry is missing
    or the plane disagrees with the normal; a series with duplicate positions
    is rejected (refs == [], method "rejected_duplicate_position")."""
    entries = _expand_entries(records)
    rc = _receipt_base(records, entries)

    def v4_fallback(method):
        rc["method"] = method
        refs = series_slice_refs([r["path"] for r in records],
                                 expand_frames=True, strict_geometry=True)
        return refs, rc

    n = None
    for e in entries:
        if e[3] is not None:
            r = e[3][:3]
            c = e[3][3:]
            nr, nc = np.linalg.norm(r), np.linalg.norm(c)
            if nr > 0 and nc > 0:
                cr = np.cross(r / nr, c / nc)
                ncr = np.linalg.norm(cr)
                if ncr > 0:
                    n = cr / ncr
            break
    if n is None or rc["n_positioned"] == 0:
        return v4_fallback("v4_fallback_no_geometry")
    ev = CARDINAL_NORMAL.get(plane)
    if ev is None:
        return v4_fallback("v4_fallback_unknown_plane")
    ev = np.asarray(ev, dtype=np.float64)
    if int(np.argmax(np.abs(n))) != int(np.argmax(ev)):
        return v4_fallback("v4_fallback_plane_mismatch")
    if float(np.dot(n, ev)) < 0:
        n = -n
        rc["flipped_normal"] = True
    rc["normal"] = [float(v) for v in n]
    keyed = []
    for e in entries:
        p = float(np.dot(e[2], n)) if e[2] is not None else None
        keyed.append((p, e))
    keyed.sort(key=lambda k: (k[0] is None, k[0] or 0.0, k[1][4], k[1][5],
                              k[1][6]))
    ps = sorted(k[0] for k in keyed if k[0] is not None)
    if len(ps) > 1:
        delta = np.diff(np.asarray(ps))
        rc["min_delta"] = float(delta.min())
        if bool(np.any(delta <= POSITION_TOLERANCE_MM)):
            rc["method"] = "rejected_duplicate_position"
            return [], rc
    rc["method"] = "ipp_projection_cardinalized"
    return [(k[1][0], k[1][1]) for k in keyed], rc


def flip_for_plane(plane, side) -> bool:
    """cnx flip rule: mirror columns of every non-sagittal image of a right
    knee (sagittal handling is the row reversal in build_study_tensor)."""
    return str(plane) != "Sagittal" and side == "R"


def decide_study_side(series_receipts: dict):
    """series_receipts {uid: (plane, receipt)} -> (side, source, tags,
    sag_x_median). Laterality tags are pooled over every series of the study;
    one distinct tag decides, several disagree -> no flip; no tag -> the median
    sagittal IPP x (x >= 0 -> "L"); nothing -> no flip (cnx decide_study_side /
    resolve_study_side)."""
    tags = sorted({v for _, r in series_receipts.values()
                   for v in r["lat_values"] if v in {"L", "R"}})
    sag_x = [r["ipp_x_median"] for plane, r in series_receipts.values()
             if plane == "Sagittal" and r["ipp_x_median"] is not None]
    sag_med = float(np.median(sag_x)) if sag_x else None
    if len(tags) == 1:
        side, source = tags[0], "dicom_laterality_tag"
    elif len(tags) > 1:
        side, source = None, "conflicting_laterality_tags"  # never flip
    elif sag_med is not None:
        side, source = ("L" if sag_med >= 0 else "R"), "sagittal_ipp_x"
    else:
        side, source = None, "unknown_no_flip"
    return side, source, tags, sag_med


def series_slice_refs(files: list[str], expand_frames: bool = False,
                      strict_geometry: bool = False):
    """Ordered slice references [(path, frame)] for one series.

    expand_frames=True (v3): a single-file multi-frame series contributes each
    of its frames as a slice (frames are stored in acquisition order, so no
    header sort is needed inside the file). Only series with at most
    FRAME_PROBE_MAX_FILES files are probed for NumberOfFrames to keep header
    reads off the common path.
    strict_geometry=True (v4): every file is probed for NumberOfFrames
    regardless of series size (the FRAME_PROBE_MAX_FILES shortcut only
    applies in v3 mode), and multi-frame files are frame-ordered by
    projected position instead of storage order (see _frame_order)."""
    files = sort_series_files(files, strict_geometry=strict_geometry)
    if not expand_frames:
        return [(p, None) for p in files]
    if strict_geometry:
        refs = []
        for p in files:
            n = _frame_count(p)
            if n > 1:
                order = _frame_order(p, n, strict_geometry=True)
                refs.extend((p, f) for f in order)
            else:
                refs.append((p, None))
        return refs
    if len(files) <= FRAME_PROBE_MAX_FILES:
        refs = []
        for p in files:
            n = _frame_count(p)
            refs.extend((p, f) if n > 1 else (p, None) for f in range(n))
        return refs
    return [(p, None) for p in files]


def build_study_tensor(series_dir: str, series_rows, size: int = IMG_SIZE,
                       crop_mm: float | None = None,
                       unique_slices: bool = False,
                       isotropic: bool = False,
                       guarantee_k: bool = False,
                       expand_frames: bool = False,
                       max_slices: int | None = None,
                       strict_geometry: bool = False,
                       n_slices: int = N_SLICES,
                       slice_align: int = 1,
                       slot_layout: str = "plane6",
                       slices_per_slot: tuple | None = None,
                       span: tuple | None = None,
                       cardinal_order: bool = False,
                       canonical_side: bool = False,
                       crop_window: bool = False,
                       receipts: dict | None = None):
    """series_dir: <root>/<StudyInstanceUID>; series_rows as for pick_slot_series
    (n_slices is recounted from disk). Returns (tensor, filled_slots): the
    (n_slots(slot_layout), max(slices_per_slot or [n_slices]), size, size)
    uint8 volume and how many slots decoded at least one non-black slice
    (0 means the study is unusable).

    Defaults reproduce the v1 cache; pass **recipe_kwargs(cfg) for v2/v3/v4/v5
    (see RECIPE_V2 / RECIPE_V3 / RECIPE_V4 / RECIPE_V5). max_slices caps how
    many of the sampled slices are decoded per slot (centered subset) - the
    efficiency-track knob; it never changes which slices the full run would
    have picked. slice_align (decode-time only, like max_slices - not a
    recipe key) rounds the centered subset's start row down to a multiple of
    itself, so a truncated slot's rows land on the same window-group
    boundaries (see model.study_windows) a full, untruncated decode would
    have put them on - the efficiency preset's windows then match the ones
    training actually pooled over. slice_align=1 (default) keeps the
    pre-existing placement at rows 0..max_slices-1 exactly, byte-identical to
    before this knob existed; slice_align>1 places the subset at rows
    start..start+max_slices-1 instead (the rest of the slot stays zero).
    strict_geometry=True (v4) turns on seven geometry/sampling fixes
    throughout the helpers below (see each helper's docstring); it must stay
    False by default so stage A's v3 cache and inference are unaffected.
    slot_layout (v5) picks the slot table (see the module docstring) via
    pick_slot_series/slot_index; default "plane6" is byte-identical to
    pre-v5. slices_per_slot (v5), a tuple/list indexed by slot (a JSON list
    from cache_config.json is normalized to a tuple), gives each slot its own
    k for sample_indices instead of the shared n_slices; None (default) uses
    n_slices for every slot, unchanged from pre-v5. span (v5) is passed
    through to sample_indices, replacing the centre-fraction pick; None
    (default) is unchanged from pre-v5.
    cardinal_order (v4g, a): slices of each series are ordered by position on
    the cardinalised normal (order_slices_cardinal); a series with duplicate
    positions is dropped from the slot candidates. canonical_side (v4g, b):
    one study side is decided from laterality tags / sagittal IPP x
    (decide_study_side); right-knee non-sagittal slots are column-mirrored
    (flip_for_plane) and right-knee sagittal slots have their rows reversed.
    crop_window (v4g, c): slices are decoded by load_crop_float and windowed
    per series by window_series_crops instead of load_slice. All three default
    False (byte-identical to v4). receipts, when a dict, is filled with the
    side decision and per-series diagnostics (not a recipe key)."""
    if slices_per_slot is not None:
        slices_per_slot = tuple(slices_per_slot)
    n_slots_val = n_slots(slot_layout)
    max_k = n_slices if slices_per_slot is None else max(slices_per_slot)
    rows = []
    all_rc = {}  # uid -> (plane, receipt), every series with files (side decision)
    series_rc = {}
    for row in series_rows:
        d = os.path.join(series_dir, row["SeriesInstanceUID"])
        if not os.path.isdir(d):
            continue
        files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".dcm")]
        if not files:
            continue
        rc, refs_card = None, None
        if cardinal_order or canonical_side:
            records = _read_slice_headers(files)
            if cardinal_order:
                refs_card, rc = order_slices_cardinal(
                    records, row["Anatomical_Plane"])
            else:
                rc = _light_receipt(records)
            suid = row["SeriesInstanceUID"]
            all_rc[suid] = (row["Anatomical_Plane"], rc)
            series_rc[suid] = {"plane": row["Anatomical_Plane"], **rc,
                               "flip": False, "reversed": False}
            if rc["method"].startswith("rejected"):
                continue  # dropped from the slot candidates
        r = dict(row)
        r["_files"] = files
        r["_plane"] = row["Anatomical_Plane"]
        r["_receipt"] = rc
        # slice refs are needed up front only when frame expansion can change
        # the slice count: v3 only for series at or under FRAME_PROBE_MAX_FILES,
        # v4 (strict_geometry) always, since every file is probed there
        if cardinal_order:
            r["_refs"] = refs_card
            r["n_slices"] = len(refs_card)
        elif expand_frames and (strict_geometry or len(files) <= FRAME_PROBE_MAX_FILES):
            r["_refs"] = series_slice_refs(files, expand_frames=True,
                                           strict_geometry=strict_geometry)
            r["n_slices"] = len(r["_refs"])
        else:
            r["_refs"] = None
            r["n_slices"] = len(files)
        rows.append(r)
    chosen = pick_slot_series(rows, layout=slot_layout)
    by_uid = {r["SeriesInstanceUID"]: r for r in rows}
    side, side_source, side_tags, sag_med = None, None, [], None
    if canonical_side:
        side, side_source, side_tags, sag_med = decide_study_side(all_rc)
    if receipts is not None:
        receipts.update({"side": side, "side_source": side_source,
                         "tags": side_tags, "sag_x_median": sag_med,
                         "series": series_rc})

    out = np.zeros((n_slots_val, max_k, size, size), dtype=np.uint8)
    filled = 0
    for slot, uid in chosen.items():
        r = by_uid[uid]
        # lazy path: series above the probe threshold are never expanded in
        # v3 mode, so passing expand_frames through is a no-op that keeps
        # one rule (v4 always takes the eager path above)
        refs = r["_refs"] or series_slice_refs(
            r["_files"], expand_frames=expand_frames,
            strict_geometry=strict_geometry)
        k = n_slices if slices_per_slot is None else slices_per_slot[slot]
        picks = sample_indices(len(refs), k=k, unique=unique_slices,
                               guarantee_k=guarantee_k,
                               strict_geometry=strict_geometry, span=span)
        row_start = 0
        if max_slices is not None and len(picks) > max_slices:
            if slice_align > 1:
                start = ((len(picks) - max_slices) // 2) // slice_align * slice_align
                row_start = start
            else:
                start = (len(picks) - max_slices) // 2
            picks = picks[start:start + max_slices]
        reverse = bool(canonical_side and side == "R"
                       and r["_plane"] == "Sagittal")
        if reverse:
            picks = picks[::-1]
        flip = bool(canonical_side and flip_for_plane(r["_plane"], side))
        if uid in series_rc:
            series_rc[uid]["flip"] = flip
            series_rc[uid]["reversed"] = reverse
        any_ok = False
        decoded = {}  # per-slot memo so a multi-frame file is decoded once
        if crop_window:
            crops = []
            for j, i in enumerate(picks):
                path, frame = refs[i]
                try:
                    crops.append((j, load_crop_float(
                        path, crop_mm, frame,
                        decoded if frame is not None else None,
                        isotropic, strict_geometry)))
                except Exception:
                    pass  # unreadable slice stays zero; study is still usable
            if crops:
                try:
                    imgs = window_series_crops([c for _, c in crops], size,
                                               flip)
                    for (j, _), u8 in zip(crops, imgs):
                        out[slot, row_start + j] = u8
                        any_ok = any_ok or bool(u8.any())
                except Exception:
                    pass
        else:
            for j, i in enumerate(picks):
                path, frame = refs[i]
                try:
                    out[slot, row_start + j] = load_slice(
                        path, size=size, crop_mm=crop_mm, frame=frame,
                        isotropic=isotropic,
                        pixels_cache=decoded if frame is not None else None,
                        strict_geometry=strict_geometry)
                    if flip:
                        out[slot, row_start + j] = out[slot, row_start + j][:, ::-1]
                    any_ok = any_ok or bool(out[slot, row_start + j].any())
                except Exception:
                    pass  # unreadable slice stays zero; study is still usable
        filled += int(any_ok)
    return out, filled
