"""O1: our own arms only, as a gradable submission (owner, 2026-10-21: "do it ourselves, don't copy").

rs + t4 + t5 (CoAtNet-2 on cache v4, 5 folds each, trained by us), dense stride-1 windows, equal-weight per-label
rank blend. The kernel is notebooks/fusion/fusion_submit.py with five lines flipped in the staged copy (OURS_ONLY,
OURS_SUBMIT, BLEND preset, BLEND coat, BLEND window_stride); kernel-metadata.json mounts only our src and weights
datasets. No public checkpoint, no public notebook cell, no Raptor or resgated subprocess.

Not ours, said plainly: the training labels start from consensus_v1 (the mean of three public LLM report readers;
t4 / t5 replace some columns with our own rs out-of-fold predictions), and the backbone starts from timm's ImageNet
weights.

Runtime estimate (2026-10-21): t4 alone, stride 3, 650 studies = 914.6 s (F1 probe v3, 2026-10-13); x2 for the
hidden test size, x~3 for stride 1, x3 arms -> ~4.6 h, under fusion_submit's 8.5 h fail-fast.

O2 = the same with rs swapped for t11 (labels: every finding except Fracture from rs's dense OOF), one change against
O1, so the LB reads the t11 label set directly:
  py notebooks/o1/build_o1.py C:/kaggle/staging/o2 --kernel-id patimakornsrinamen/rsna-67knee-o2 --arms t4,t5,t11

O3 (only if O2 reads the t11 labels as a help) = O2 with t4 swapped for t112, the t11 seed replicate:
  py notebooks/o1/build_o1.py C:/kaggle/staging/o3 --kernel-id patimakornsrinamen/rsna-67knee-o3 --arms t5,t11,t112

O1b = O1 with t5 swapped for t52 (same labels, train seed 11): the noise of a one-arm seed swap on the public score.
O5 = O1 with t5 swapped for t5f (same labels, trained on every non-gold study):
  py notebooks/o1/build_o1.py C:/kaggle/staging/o1b --kernel-id patimakornsrinamen/rsna-67knee-o1b --arms rs,t4,t52
  py notebooks/o1/build_o1.py C:/kaggle/staging/o5 --kernel-id patimakornsrinamen/rsna-67knee-o5 --arms rs,t4,t5f
O6 = O5 with t4 swapped for t4f (teach4 labels, full data), only if O5 reads full data as a gain:
  py notebooks/o1/build_o1.py C:/kaggle/staging/o6 --kernel-id patimakornsrinamen/rsna-67knee-o6 --arms rs,t4f,t5f
O7 = O6 with rs swapped for rsf (consensus_v1 labels, full data), the all-full-data trio, same condition:
  py notebooks/o1/build_o1.py C:/kaggle/staging/o7 --kernel-id patimakornsrinamen/rsna-67knee-o7 --arms rsf,t4f,t5f
  (O7 = rsf,t4,t5f instead if O6 reads t4f as a loss: O6 - O5 <= -0.002)
O8a / O8b = O7 plus one fold-trained arm (rs / t5), a width probe on top of the all-full-data trio:
  py notebooks/o1/build_o1.py C:/kaggle/staging/o8a --kernel-id patimakornsrinamen/rsna-67knee-o8a --arms rsf,t4f,t5f,rs
  py notebooks/o1/build_o1.py C:/kaggle/staging/o8b --kernel-id patimakornsrinamen/rsna-67knee-o8b --arms rsf,t4f,t5f,t5

  py notebooks/o1/build_o1.py <out dir> [--kernel-id patimakornsrinamen/rsna-67knee-o1] [--arms rs,t4,t5] [--stride 1]
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FUSION = os.path.join(HERE, "..", "fusion", "fusion_submit.py")
# the image our own leg ran on (F1 probe v3, 2026-10-13, returncode 0)
DOCKER_IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
WEIGHTS = {"rs": "rsna-67knee-weights-rs", "t4": "rsna-67knee-weights-t4", "t5": "rsna-67knee-weights-t5",
           "rs2": "rsna-67knee-weights-rs2", "v6c": "rsna-67knee-weights-v6c", "t11": "rsna-67knee-weights-t11",
           "t112": "rsna-67knee-weights-t112", "t42": "rsna-67knee-weights-t42", "t52": "rsna-67knee-weights-t52",
           "t5f": "rsna-67knee-weights-t5f", "t4f": "rsna-67knee-weights-t4f", "rsf": "rsna-67knee-weights-rsf",
           "l7f": "rsna-67knee-weights-l7f", "c7f": "rsna-67knee-weights-c7f"}

BLEND_DEFAULT = '''BLEND = {  # repo default = the round-7 shipped recipe; staged copies flip this dict per submission
    "preset": "full",
    "coat": {"raptor": 0.375, "v6c": 0.375, "resgated": 0.25},
    "transformer": "v4b",
    "w_coat": 0.60,
    "w_coat_lateral_meniscus": 1.00,
    "public_coat": {"raptor": 0.60, "resgated": 0.40},
    "window_stride": None,
}'''


def _replace_once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def staged_source(arms, stride):
    src = open(FUSION, encoding="utf-8").read()
    src = _replace_once(src, 'OURS_ONLY = os.environ.get("RSNA_OURS_ONLY") == "1"\n',
                        'OURS_ONLY = True  # O1 staged copy\n')
    src = _replace_once(src, "OURS_SUBMIT = False\n", "OURS_SUBMIT = True  # O1 staged copy\n")
    coat = "{" + ", ".join(f'"{a}": 1.0' for a in arms) + "}"
    blend = (BLEND_DEFAULT
             .replace('"preset": "full",', '"preset": "coat_only",  # O1')
             .replace('"coat": {"raptor": 0.375, "v6c": 0.375, "resgated": 0.25},', f'"coat": {coat},  # O1')
             .replace('"window_stride": None,', f'"window_stride": {stride},  # O1'))
    return _replace_once(src, BLEND_DEFAULT, blend)


def build(out_dir, kernel_id="patimakornsrinamen/rsna-67knee-o1", arms=("rs", "t4", "t5"), stride=1):
    assert arms and all(a in WEIGHTS for a in arms), arms
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "fusion_submit.py")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(staged_source(arms, stride))
    slug = kernel_id.split("/", 1)[1]
    meta = {
        "id": kernel_id, "title": slug, "code_file": "fusion_submit.py", "language": "python",
        "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "machine_shape": "NvidiaTeslaT4",
        "enable_internet": "false", "docker_image": DOCKER_IMAGE,
        "dataset_sources": ["patimakornsrinamen/rsna-knee-src"] + [f"patimakornsrinamen/{WEIGHTS[a]}" for a in arms],
        "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [],
    }
    with open(os.path.join(out_dir, "kernel-metadata.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, indent=2)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--kernel-id", default="patimakornsrinamen/rsna-67knee-o1")
    ap.add_argument("--arms", default="rs,t4,t5")
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    print(build(a.out_dir, a.kernel_id, tuple(a.arms.split(",")), a.stride))
