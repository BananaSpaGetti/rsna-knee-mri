# notebooks/fusion/bundle.py
#
# Regenerates the three "# BEGIN BUNDLED <name>" / "# END BUNDLED <name>"
# blocks in fusion_submit.py (the _RAPTOR_SRC, _RESGATED_SRC, and _INFER_SRC
# string-literal assignments) from the three standalone scripts they
# package. A Kaggle script kernel is a single code file, so
# raptor_infer.py, resgated_runner.py, and ../infer/infer_submit.py -- none
# of which lives in the rsna-knee-src dataset -- are embedded here as
# `repr()`-encoded string literals that fusion_submit.py writes out to
# /kaggle/working/ and runs/imports at runtime.
#
# Run this after editing any of the three source files. With no source
# changes, re-running it leaves fusion_submit.py byte-identical
# (idempotent) -- `git diff --stat` shows nothing.
#
# Usage: py notebooks/fusion/bundle.py

import os

HERE = os.path.dirname(os.path.abspath(__file__))
FUSION_PATH = os.path.join(HERE, "fusion_submit.py")

# (sentinel name, source file, variable name)
BUNDLES = [
    ("RAPTOR", os.path.join(HERE, "raptor_infer.py"), "_RAPTOR_SRC"),
    ("RESGATED", os.path.join(HERE, "resgated_runner.py"), "_RESGATED_SRC"),
    ("INFER_SUBMIT", os.path.join(HERE, "..", "infer", "infer_submit.py"),
     "_INFER_SRC"),
]


def _replace_block(text, name, var_name, source_path):
    begin, end = f"# BEGIN BUNDLED {name}\n", f"# END BUNDLED {name}\n"
    b = text.index(begin) + len(begin)
    e = text.index(end, b)
    with open(source_path, "r", newline="") as fh:
        source_text = fh.read()
    new_block = f"{var_name} = {source_text!r}\n"
    return text[:b] + new_block + text[e:]


def main():
    with open(FUSION_PATH, "r", newline="") as fh:
        text = fh.read()
    for name, source_path, var_name in BUNDLES:
        text = _replace_block(text, name, var_name, source_path)
    with open(FUSION_PATH, "w", newline="") as fh:
        fh.write(text)
    print(f"regenerated {len(BUNDLES)} bundled block(s) in {FUSION_PATH}")


if __name__ == "__main__":
    main()
