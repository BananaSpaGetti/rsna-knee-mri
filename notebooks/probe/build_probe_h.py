"""Timing probe for a heliosli-based notebook (H7, F3, F3 + reader, ...): the notebook run unchanged on N sampled
TRAIN studies posing as the test set, with a timestamp before every code cell, so each stage's wall time (Raptor,
the R7 compact, the S6 compact, the blend, the reader leg) is measured instead of bounded from hidden-run totals.

Reuses notebooks/f1/build_probe.py's SETUP (fake competition root in /tmp with per-file symlinks, clocks, a
once-a-minute GPU / CPU sampler), STAMP and FINISH cells. Only three lines of the base notebook point at the
competition data, and each is rewritten to the fake root (counts asserted):
  * Raptor's root candidates "/kaggle/input/competitions/rsna-knee-abnormality-detection" and
    "/kaggle/input/rsna-knee-abnormality-detection" (in the %%writefile /tmp/raptor_run.py cell),
  * the compact asset cell's `DATA = find_mount(COMPETITION, 'test.csv')` (the S6 stage, the blend and the reader
    leg all read DATA).
Everything else, deadlines included, is left as is. Output: /kaggle/working/probe/{timeline.jsonl, summary.json,
clocks.json, resources.csv}.

--reader-only (base = a notebook built by notebooks/reader/build_reader.py): only the reader leg's cells (from its
markdown header on) run, after a prelude that defines the globals the leg reads (NB_T0, WORK, DATA = the fake root,
TMP, and find_mount copied from the base's asset cell) and puts the fake root's sample_submission in place of the
stack's submission.csv. Measures the reader alone at N studies.

  py notebooks/probe/build_probe_h.py --base C:/kaggle/staging/f3r/rsna-67knee-f3r.ipynb \\
      --out C:/kaggle/staging/f3r-probe --kernel-id patimakornsrinamen/rsna-67knee-f3r-probe [--n 650]
"""
import argparse
import ast
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "f1"))
from build_probe import FAKE, FINISH, OUT, SETUP, STAMP, _code_cell, _src  # noqa: E402

SLUG = "rsna-knee-abnormality-detection"
REWRITES = (  # (old, new, expected count over the base notebook's code cells)
    (f'"/kaggle/input/competitions/{SLUG}"', repr(FAKE).replace("'", '"'), 1),
    (f'"/kaggle/input/{SLUG}"', repr(FAKE).replace("'", '"'), 1),
    ("DATA = find_mount(COMPETITION, 'test.csv')", f"DATA = Path({FAKE!r})", 1),
)
ALLOWED_SLUG_LINES = (f"COMPETITION = '{SLUG}'",)
READER_HEAD = "## Reader leg:"
PRELUDE = '''# PROBE reader-only prelude: the globals the reader leg reads, and the fake root's sample_submission (0.5
# everywhere) as the stack's submission.csv.
import shutil, time
from pathlib import Path
NB_T0 = time.time()
WORK = Path('/kaggle/working')
INPUT = Path('/kaggle/input')
DATA = Path(__FAKE__)
TMP = (Path('/kaggle/temp') if Path('/kaggle/temp').is_dir() else Path('/tmp')) / 'probe'
TMP.mkdir(parents=True, exist_ok=True)
__FIND_MOUNT__
shutil.copy(DATA / 'sample_submission.csv', WORK / 'submission.csv')
'''


def reader_only_cells(nb):
    """The prelude and the reader leg's cells (markdown header on) of a build_reader.py notebook."""
    heads = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "markdown" and _src(c).startswith(READER_HEAD)]
    assert len(heads) == 1, f"{len(heads)} reader-leg headers"
    asset = [_src(c) for c in nb["cells"] if c["cell_type"] == "code" and "def find_mount(slug, required):" in _src(c)]
    assert len(asset) >= 1, "no find_mount in the base"
    s = asset[0]
    start = s.index("def find_mount(slug, required):")
    tail = "    return found[0]\n"
    end = s.index(tail, start) + len(tail)
    prelude = PRELUDE.replace("__FAKE__", repr(FAKE)).replace("__FIND_MOUNT__", s[start:end])
    ast.parse(prelude)
    return [_code_cell(prelude)] + nb["cells"][heads[0]:]


def build(base_ipynb, out_dir, kernel_id, n, reader_only=False):
    nb = json.load(open(base_ipynb, encoding="utf-8"))
    meta = json.load(open(os.path.join(os.path.dirname(base_ipynb), "kernel-metadata.json"), encoding="utf-8"))
    if reader_only:
        nb["cells"] = reader_only_cells(nb)
    counts = {old: 0 for old, _, _ in REWRITES}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        s = _src(cell)
        for old, new, _ in REWRITES:
            counts[old] += s.count(old)
            s = s.replace(old, new)
        cell["source"] = s
    for old, _, want in REWRITES:
        assert counts[old] == (0 if reader_only else want), f"{old}: {counts[old]} hits, expected {want}"
    left = [ln.strip()[:120] for c in nb["cells"] if c["cell_type"] == "code" for ln in _src(c).split("\n")
            if SLUG in ln and ln.strip() not in ALLOWED_SLUG_LINES]
    assert not left, f"competition slug still referenced: {left}"

    cells = [_code_cell(SETUP.replace("__N__", str(n)).replace("__FAKE__", FAKE).replace("__OUT__", OUT))]
    for k, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            cells.append(_code_cell(STAMP.replace("__OUT__", OUT).replace("__K__", str(k))))
        cells.append(cell)
    cells.append(_code_cell(FINISH.replace("__OUT__", OUT)))
    for k, cell in enumerate(cells):
        if cell["cell_type"] == "code" and not _src(cell).startswith("%%"):
            body = "\n".join(ln for ln in _src(cell).split("\n") if not ln.lstrip().startswith(("!", "%")))
            try:
                ast.parse(body)
            except SyntaxError as err:
                raise SystemExit(f"probe cell {k} does not parse: {err}")
    nb["cells"] = cells

    meta.pop("id_no", None)
    meta["id"] = kernel_id
    meta["title"] = kernel_id.split("/")[1]
    meta["code_file"] = kernel_id.split("/")[1] + ".ipynb"
    meta["is_private"] = True
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, meta["code_file"]), "w", encoding="utf-8") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "kernel-metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {out_dir} ({len(cells)} cells, N = {n}{', reader only' if reader_only else ''}); rewrites: "
          + ", ".join(f"{old[:50]!r} x{c}" for old, c in counts.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="heliosli-based notebook (.ipynb); kernel-metadata.json beside it")
    ap.add_argument("--out", required=True, help="staging dir for the probe kernel")
    ap.add_argument("--kernel-id", required=True, help="Kaggle kernel id for the probe")
    ap.add_argument("--n", type=int, default=650, help="train studies posing as the test set")
    ap.add_argument("--reader-only", action="store_true", help="run only the reader leg (see the docstring)")
    a = ap.parse_args()
    build(a.base, a.out, a.kernel_id, a.n, reader_only=a.reader_only)
