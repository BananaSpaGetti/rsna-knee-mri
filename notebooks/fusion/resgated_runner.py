# notebooks/fusion/resgated_runner.py
#
# Standalone extraction of the subprocess launch that `_coat_substitute()`
# makes in the public kernel "the-bee-s-knees-final-rsna-push" (Kaggle
# kernel prvsiyan/the-bee-s-knees-final-rsna-push, pulled to
# C:/kaggle/staging/competitors/the-bee-s-knees-final-rsna-push/ on
# 2026-09-09; source authenticated by the pulled notebook's own SHA-256
# 6b03919bbca5fe61e37c5ea6546609a771cd7db6503121f3e2dacdd8c9b6b052).
#
# Original residual-gated CoAtNet checkpoints and runtime module:
# Mattia Angeli, dataset `mattiaangeli/rsna-knee-coat-resgated-ep10-top3`
# (`coatnet_resgated_ep10_top3_inference.py` + manifest +
# `coat_resgated_gold_e0{4,6,8}.pt`), CC0-1.0. Offline opencv wheel:
# `mattiaangeli/opencv-python-headless-4120088-x86`
# (opencv-python-headless 4.12.0.88, Apache-2.0). Assembled/reproduced (per
# the source kernel's licence table) from Mattia Angeli's V27 notebook,
# Apache-2.0.
#
# `_coat_substitute()` in the source cell runs the released module as a
# subprocess and then rank-blends its output into `/kaggle/working/_raptor.csv`
# at private_alpha=0.40. This file reproduces only the subprocess launch
# (manifest/wheel SHA-256 verification kept, exactly as the source does it)
# and writes the resgated arm's own per-study probabilities straight to
# `_resgated.csv`, in whatever column order the released module's own
# `run_submission()` emits (that module is the one that owns
# `sample_submission` column ordering; this runner does not reindex or
# validate it against `_raptor.csv` -- that identity check, and all
# cross-arm fusion, happens in `fusion_submit.py`, a later task).
#
# Inputs are located by marker file (manifest / wheel filename) walked
# recursively under /kaggle/input, which resolves both the flat
# /kaggle/input/<name>/... mount layout and the nested
# /kaggle/input/datasets/<owner>/<name>/... layout, matching the pattern
# `src/preprocess.py:find_by_marker` uses elsewhere in this repo.
#
# Argument/env contract (all optional; defaults reproduce the source
# kernel's hardcoded behaviour):
#   RESGATED_ARTIFACT_ROOT       directory holding
#                                 coat_resgated_ep10_top3_manifest.json and
#                                 coatnet_resgated_ep10_top3_inference.py
#                                 (default: discovered under /kaggle/input by
#                                 manifest SHA-256)
#   RESGATED_OUTPUT_CSV          output path (default:
#                                 /kaggle/working/_resgated.csv)
#   RESGATED_GPU_BATCH_STUDIES   passed through to run_submission()
#                                 (default: 2, the source kernel's value)
#   RESGATED_BACKBONE_MICRO_IMAGES  passed through to run_submission()
#                                 (default: 8, the source kernel's value)
# No other parameterisation is added: the manifest/wheel SHA-256 checks and
# the epoch 4/6/8 checkpoint contract are exactly as in the source kernel
# and are not overridable.
#
# Run standalone: `python resgated_runner.py` (writes _resgated.csv and
# exits; manifest/wheel absent locally -> RuntimeError naming the missing
# dataset).
# Import-safe: `import resgated_runner` defines everything below without
# running anything.

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

MANIFEST_NAME = "coat_resgated_ep10_top3_manifest.json"
MANIFEST_SHA256 = "98511a8fdeb9da0e6e70c78d013ff636e1476f31c80b5dc134d294b18c3f284e"
WHEEL_GLOB = "opencv_python_headless-4.12.0.88-*.whl"
WHEEL_SHA256 = "236c8df54a90f4d02076e6f9c1cc763d794542e886c576a6fee46ec8ff75a7a9"
# the two competition mount layouts, exactly as src/preprocess.py:find_by_marker
# prunes them. "train_series"/"test_series" are NOT top-level under
# /kaggle/input - they live inside the competition dir - so listing them
# here pruned nothing and left the flat layout's ~570 GB DICOM tree in
# the rglob below.
SKIP_TOP_DIRS = ("competitions", "rsna-knee-abnormality-detection")
DEFAULT_OUTPUT_CSV = "/kaggle/working/_resgated.csv"
DEFAULT_GPU_BATCH_STUDIES = 2
DEFAULT_BACKBONE_MICRO_IMAGES = 8


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_by_marker_and_hash(pattern, expected_sha256, root="/kaggle/input"):
    """Locate a file matching `pattern` under `root`, whatever the mount
    layout, and verify its SHA-256. Returns the first verified match found
    in a deterministic (sorted) walk, or None."""
    root_path = Path(root)
    if not root_path.is_dir():
        return None
    for base in sorted(root_path.iterdir()):
        if base.name in SKIP_TOP_DIRS:
            continue
        for candidate in sorted(base.rglob(pattern)):
            if candidate.is_file() and _sha256(candidate) == expected_sha256:
                return candidate
    return None


def find_artifact_root():
    """Directory holding the manifest (and the released inference module
    next to it), verified by SHA-256."""
    override = os.environ.get("RESGATED_ARTIFACT_ROOT")
    if override:
        manifest = Path(override) / MANIFEST_NAME
        if not manifest.is_file() or _sha256(manifest) != MANIFEST_SHA256:
            raise FileNotFoundError(
                f"RESGATED_ARTIFACT_ROOT={override!r} does not contain a "
                f"hash-verified {MANIFEST_NAME}")
        return Path(override)
    manifest = _find_by_marker_and_hash(MANIFEST_NAME, MANIFEST_SHA256)
    if manifest is None:
        raise FileNotFoundError(
            f"{MANIFEST_NAME} absent or hash mismatch under /kaggle/input "
            "(expected dataset mattiaangeli/rsna-knee-coat-resgated-ep10-top3)")
    return manifest.parent


def find_opencv_wheel():
    wheel = _find_by_marker_and_hash(WHEEL_GLOB, WHEEL_SHA256)
    if wheel is None:
        raise FileNotFoundError(
            f"{WHEEL_GLOB} absent or hash mismatch under /kaggle/input "
            "(expected dataset mattiaangeli/opencv-python-headless-4120088-x86)")
    return wheel


def install_opencv_wheel(wheel_path, target_dir):
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--quiet",
         "--target", str(target_dir), str(wheel_path)],
        check=True)


def run(output_csv=None, artifact_root=None):
    """Locate the released resgated module, install its pinned opencv wheel
    offline, run it in a subprocess, and write its per-study probabilities
    to `output_csv` (sample_submission column order). No blending."""
    out = Path(output_csv or os.environ.get("RESGATED_OUTPUT_CSV", DEFAULT_OUTPUT_CSV))
    art = Path(artifact_root) if artifact_root else find_artifact_root()
    wheel = find_opencv_wheel()

    env_dir = Path("/kaggle/working/_resgated_env")
    install_opencv_wheel(wheel, env_dir)

    gpu_batch_studies = int(os.environ.get(
        "RESGATED_GPU_BATCH_STUDIES", DEFAULT_GPU_BATCH_STUDIES))
    backbone_micro_images = int(os.environ.get(
        "RESGATED_BACKBONE_MICRO_IMAGES", DEFAULT_BACKBONE_MICRO_IMAGES))
    receipt_path = out.with_name(out.stem + "_receipt.json")

    child = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(env_dir)!r})\n"
        f"sys.path.insert(0, {str(art)!r})\n"
        "import cv2; assert cv2.__version__ == '4.12.0', cv2.__version__\n"
        "import torch; assert torch.cuda.device_count() == 2\n"
        "import coatnet_resgated_ep10_top3_inference as rt\n"
        "assert rt.base.cv2.__version__ == '4.12.0'\n"
        "from pathlib import Path\n"
        "r = rt.run_submission(competition_root=rt.base.find_competition_root(),\n"
        f"    artifact_root=Path({str(art)!r}), output_path=Path({str(out)!r}),\n"
        f"    gpu_batch_studies={gpu_batch_studies}, "
        f"backbone_micro_images={backbone_micro_images})\n"
        "assert r['status'] == rt.SUBMISSION_STATUS\n"
        "assert r['models'] == 3\n"
        "assert [i['epoch'] for i in r['checkpoints']] == [4, 6, 8]\n"
        "assert r['fallback_studies'] == 0, r['failures']\n"
        f"Path({str(receipt_path)!r}).write_text(json.dumps(r, indent=2))\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{env_dir}{os.pathsep}{art}{os.pathsep}" + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", child], env=env,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"resgated child failed: {proc.stderr[-700:]}")
    return out


if __name__ == "__main__":
    written = run()
    print(f"wrote {written}")
