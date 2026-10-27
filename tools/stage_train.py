"""Stage a training kernel for Kaggle without editing the committed script.

Copies notebooks/train/train_dinov2.py into a staging directory, rewrites the
per-run switches (ARM, FOLDS_TO_TRAIN, SMOKE, DUAL_GPU, MAX_HOURS) in that copy
only, and writes a kernel-metadata.json whose id belongs to the account that
will push it. The committed files never change, so every number in README.md
stays reproducible, and a teammate can run a fold subset from their own
account (their own weekly GPU quota) with one command.

Examples (run from the repo root, any OS):

    py rsna-67knee/tools/stage_train.py --user patimakornsrinamen \\
        --arm coatnet2_v5 --folds 1,2 --dual --out C:/kaggle/staging/r9-v5-f12

    py rsna-67knee/tools/stage_train.py --user somefriend --arm coatnet2_v5 \\
        --folds 3,4 --dual --smoke 120 --out C:/kaggle/staging/smoke-f34

Then: py -m kaggle kernels push -p <out>. The script prints the push, status
and output commands for the chosen slug, plus the one-line record to paste
into the round's .plans log.
"""
import argparse
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TRAIN = os.path.join(REPO, "notebooks", "train", "train_dinov2.py")

# Every input the training kernel mounts lives on the team lead's account; a
# teammate only needs Viewer access on each of these (see docs/team/run-on-your-account.md).
DATA_OWNER = "patimakornsrinamen"
DATASETS = [f"{DATA_OWNER}/rsna-knee-src",
            f"{DATA_OWNER}/rsna-knee-consensus-labels",
            "stevenleehans/rsna-knee-llm-report-labels"]
CACHES = {
    "v4": [f"{DATA_OWNER}/rsna-knee-build-cache",
           f"{DATA_OWNER}/rsna-knee-build-cache-b"],
    "v5": [f"{DATA_OWNER}/rsna-knee-build-cache-v5",
           f"{DATA_OWNER}/rsna-knee-build-cache-v5b",
           f"{DATA_OWNER}/rsna-knee-build-cache-v5c"],
    "v4g": [f"{DATA_OWNER}/rsna-knee-build-cache-v4g",
            f"{DATA_OWNER}/rsna-knee-build-cache-v4g-b"],
}
# Which cache each arm was designed for; the training script asserts the
# recipe version it finds, so a wrong mount fails fast rather than silently.
ARM_CACHE = {"dinov2": "v4", "dinov2_full": "v4", "coatnet2": "v4", "coatnet2_swa": "v4", "coatnet2_e30": "v4", "coatnet2_rs": "v4", "coatnet2_rs2": "v4", "coatnet2_pa": "v4", "coatnet2_ua": "v4", "coatnet2_v4g": "v4g", "coatnet2_384ck": "v4", "coatnet2_384up": "v4",
             "coatnet2_sweep": "v4", "coatnet2_v5": "v5", "coatnet2_full": "v4",
             "convnexts": "v4", "qwenv": "v4", "qwenv_sweep": "v4", "qwenvp_sweep": "v4",
             "qwenvlm": "v4", "qwenvlm_sweep": "v4", "qwenvlm_mi": "v4", "qwenvlm_mi2": "v4", "qwenvlm_mi24": "v4"}


def rewrite(src: str, pattern: str, replacement: str, what: str) -> str:
    """Replace exactly one line matching `pattern` (anchored, multiline)."""
    hits = re.findall(pattern, src, flags=re.M)
    if len(hits) != 1:
        raise SystemExit(f"stage_train: expected exactly one {what} line, found {len(hits)}")
    return re.sub(pattern, replacement, src, count=1, flags=re.M)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--user", required=True, help="Kaggle username that will push (owns the kernel)")
    ap.add_argument("--arm", required=True, choices=sorted(ARM_CACHE), help="row of the ARMS table")
    ap.add_argument("--folds", required=True, help="comma-separated fold ids, e.g. 1,2")
    ap.add_argument("--out", required=True, help="staging directory to create (must not be inside the repo)")
    ap.add_argument("--slug", default="rsna-67knee-train-dinov2s", help="kernel slug under --user")
    ap.add_argument("--dual", action="store_true", help="set DUAL_GPU = True (one fold per T4)")
    ap.add_argument("--smoke", type=int, default=0, help="SMOKE study cap (0 = real run)")
    ap.add_argument("--max-hours", type=float, default=None, help="override MAX_HOURS")
    ap.add_argument("--label-file", default=None,
                    help="override LABEL_FILE (a file inside the consensus-labels dataset)")
    ap.add_argument("--cache", choices=sorted(CACHES), default=None,
                    help="cache kernels to mount (default: the arm's own version)")
    args = ap.parse_args()

    folds = tuple(int(f) for f in args.folds.split(",") if f.strip() != "")
    if not folds or any(f < 0 or f > 4 for f in folds) or len(set(folds)) != len(folds):
        raise SystemExit(f"stage_train: bad --folds {args.folds!r} (want distinct ids in 0..4)")
    out = os.path.abspath(args.out)
    if os.path.commonpath([out, REPO]) == REPO:
        raise SystemExit("stage_train: --out must be outside the repository")
    cache = args.cache or ARM_CACHE[args.arm]

    src = open(TRAIN, encoding="utf-8").read()
    src = rewrite(src, r'^ARM = "[^"]+"$', f'ARM = "{args.arm}"', "ARM")
    src = rewrite(src, r"^FOLDS_TO_TRAIN = .*$",
                  f"FOLDS_TO_TRAIN = {folds!r}  # staged by tools/stage_train.py", "FOLDS_TO_TRAIN")
    src = rewrite(src, r"^SMOKE = \d+(  #.*)?$", f"SMOKE = {args.smoke}\\1", "SMOKE")
    src = rewrite(src, r"^DUAL_GPU = (True|False)(  #.*)?$",
                  f"DUAL_GPU = {args.dual}\\2", "DUAL_GPU")
    src = rewrite(src, r'^CACHE_VERSION = .*$',
                  f'CACHE_VERSION = "{cache}"  # staged by tools/stage_train.py', "CACHE_VERSION")
    if args.label_file is not None:
        src = rewrite(src, r'^LABEL_FILE = "[^"]+"(  #.*)?$',
                      f'LABEL_FILE = "{args.label_file}"\\1', "LABEL_FILE")
    if args.max_hours is not None:
        src = rewrite(src, r"^MAX_HOURS = [0-9.]+$", f"MAX_HOURS = {args.max_hours}", "MAX_HOURS")

    # a bad rewrite must fail here, not two minutes into a GPU session
    if any(ord(ch) < 32 and ch not in "\n\t" for ch in src):
        raise SystemExit("stage_train: control character in the rewritten script")
    compile(src, "train_dinov2.py", "exec")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "train_dinov2.py"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(src)
    meta = {
        "id": f"{args.user}/{args.slug}",
        "title": args.slug,
        "code_file": "train_dinov2.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": "true",
        "dataset_sources": DATASETS,
        "competition_sources": [],
        "kernel_sources": CACHES[cache],
    }
    with open(os.path.join(out, "kernel-metadata.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    for stale in ("__pycache__",):
        shutil.rmtree(os.path.join(out, stale), ignore_errors=True)

    ref = meta["id"]
    print(f"staged {out}")
    print(f"  ARM={args.arm} FOLDS_TO_TRAIN={folds} SMOKE={args.smoke} DUAL_GPU={args.dual}"
          f"{'' if args.label_file is None else ' LABEL_FILE=' + args.label_file}"
          f"{'' if args.max_hours is None else f' MAX_HOURS={args.max_hours}'} cache={cache}")
    print(f"  mounts: {', '.join(DATASETS + CACHES[cache])}")
    print("next:")
    print(f"  {sys.executable} -m kaggle kernels push -p {out}")
    print(f"  {sys.executable} -m kaggle kernels status {ref}")
    print(f"  {sys.executable} -m kaggle kernels output {ref} -p {out}-out")
    print("plans log line:")
    print(f"  {ref} v? = {args.arm} folds {list(folds)}"
          f"{' SMOKE ' + str(args.smoke) if args.smoke else ''}"
          f"{' DUAL_GPU' if args.dual else ''}, staged {out}")


if __name__ == "__main__":
    main()
