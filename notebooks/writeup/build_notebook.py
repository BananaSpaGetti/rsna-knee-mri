"""Build the readable notebook edition of the inference kernel.

Kaggle renders a notebook's leading markdown cells as the page you see when
you open it, the way public write-ups do. Our shipping kernel is a script
(one .py file, no markdown), so this tool assembles a separate notebook from
two sources without duplicating code in the repository:

  header.md              the write-up (markdown cells, split on "## " headings)
  ../infer/infer_submit.py   the submission script, verbatim, as one code cell

It writes <out>/infer_readable.ipynb plus a kernel-metadata.json for the
private notebook slug, ready for `kaggle kernels push -p <out>`. The output
directory must be outside the repository; the generated .ipynb is never
committed - rebuild it after editing either source.

Usage: py notebooks/writeup/build_notebook.py --out C:/kaggle/staging/infer-readable
"""
import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
HEADER = os.path.join(HERE, "header.md")
SCRIPT = os.path.join(REPO, "notebooks", "infer", "infer_submit.py")
INFER_META = os.path.join(REPO, "notebooks", "infer", "kernel-metadata.json")

# Kaggle requires the title to slugify to the id (lowercase, non-alphanumerics -> "-").
TITLE = "RSNA Knee CoAtNet-2 Readable Inference"
SLUG = "rsna-knee-coatnet-2-readable-inference"


def markdown_cells(text: str):
    """One cell per top-level section: the title block, then each '## ' heading."""
    parts = re.split(r"(?m)^(?=## )", text.strip() + "\n")
    return [p.strip() + "\n" for p in parts if p.strip()]


_cell_ids = iter(range(1, 1000))


def cell(kind: str, source: str) -> dict:
    lines = source.splitlines(keepends=True)
    # nbformat 4.5 requires a per-cell id; Kaggle warns when it is missing.
    c = {"cell_type": kind, "id": f"cell-{next(_cell_ids)}", "metadata": {}, "source": lines}
    if kind == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="staging directory (outside the repo)")
    ap.add_argument("--user", default="patimakornsrinamen", help="Kaggle account that pushes")
    ap.add_argument("--slug", default=SLUG)
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    if os.path.commonpath([out, REPO]) == REPO:
        raise SystemExit("build_notebook: --out must be outside the repository")

    header = open(HEADER, encoding="utf-8").read()
    script = open(SCRIPT, encoding="utf-8").read()
    cells = [cell("markdown", md) for md in markdown_cells(header)]
    cells.append(cell("markdown", "## The kernel\n\nThe submission script, unchanged. "
                      "Design notes for every block live in the repository's "
                      "`docs/code-notes.md`.\n"))
    cells.append(cell("code", script))
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }
    os.makedirs(out, exist_ok=True)
    nb_path = os.path.join(out, "infer_readable.ipynb")
    with open(nb_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
        fh.write("\n")

    infer_meta = json.load(open(INFER_META, encoding="utf-8"))
    meta = dict(infer_meta)
    meta.update({
        "id": f"{args.user}/{args.slug}",
        "title": TITLE,
        "code_file": "infer_readable.ipynb",
        "kernel_type": "notebook",
        "is_private": "true",
    })
    with open(os.path.join(out, "kernel-metadata.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    print(f"wrote {nb_path}: {len(cells) - 1} markdown cells + 1 code cell "
          f"({script.count(chr(10))} lines of code)")
    print(f"metadata id {meta['id']} ({meta['kernel_type']}, private, "
          f"gpu={meta['enable_gpu']}, internet={meta['enable_internet']})")
    print(f"push: py -m kaggle kernels push -p {out}")


if __name__ == "__main__":
    main()
