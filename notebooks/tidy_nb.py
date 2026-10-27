"""Make a built submission notebook easier to read on Kaggle without changing what it runs.

Adds a short markdown heading before each known stage (Kaggle's Table of Contents lists them, so a reader can jump
between stages) and hides the input of long code cells in Kaggle's viewer (cell metadata `_kg_hide-input`, the flag
Kaggle's own "hide input" sets; the cell still runs, and "show hidden code" expands it).

Strict: code cells keep their sources byte for byte and their order (asserted), each heading rule matches at most one
cell, and existing markdown cells are left alone. Headings this tool added earlier (metadata `rsna67_tidy`) are
removed first, so running it twice gives the same notebook.

With --about <slug>, an "About this notebook" cell from tidy_about.json goes on top (what it is, its result).

  py notebooks/tidy_nb.py C:/kaggle/staging/h7r/rsna-67knee-h7r.ipynb [--hide-lines 40] [--about rsna-67knee-h7r]
"""
import argparse
import json
import os

# (first non-empty line of a code cell starts with, heading, one-line note)
RULES = (
    ("%%writefile /tmp/raptor_run.py", "## Raptor: model script",
     "heliosli's public Raptor (CoAtNet) source, written to /tmp; the next cell runs it."),
    ("# Stage 1: public Raptor model", "## Raptor: run", "Runs the script above as its own process."),
    ("ARM = 'E_lrprior'", "## Compact model: assets", "Finds heliosli's compact bundle; sets paths and the deadline."),
    ("deadline = T0 + DEADLINE_HOURS * 3600", "## Compact model: run",
     "ConvNeXt-small, 4 folds; the bundle is the one named in the assets cell (R7, or R6 in G1)."),
    ("# F3 stage 2b", "## Compact model S6: run", "The second compact bundle, a third member."),
    ("%%writefile /tmp/raptor224_run.py", "## Raptor-224 (c08): model script",
     "nartaa's public 224-pixel Raptor checkpoint (0.945 alone), source pinned by sha256; the next cell runs it."),
    ("# Stage 1b [d957]", "## Raptor-224 (c08): run", "Runs the script above (heliosli's version of this stage)."),
    ("# G2 stage 2b", "## Second compact model (R7): assets and run",
     "heliosli's R7 bundle on this notebook's own compact code; only the bundle, dirs and outputs differ."),
    ("# Stage 2c (rsna-67knee", "## Raptor-224 (c08): run",
     "Runs the script above; the blend mixes it into Raptor's rank at 0.35."),
    ("# Stage 3: per-finding rank blend", "## Blend", "Per-finding rank blend; weights fixed before any run."),
    ("# F3 stage 3:", "## Blend", "Per-finding rank blend of three members; weights fixed before any run."),
    ("%%writefile /kaggle/working/own_src/preprocess.py", "### Reader source (goodpjw2008, as published)",
     "Three files written to own_src, pinned by sha256."),
    ("# Reader leg (rsna-67knee", "### Reader run and blend",
     "Runs the reader, then rank-blends it into the submission with fixed per-finding weights."),
    ("# F1 cell 1/3", "## Our t4 leg: script", "Our own CoAtNet fusion script, kept as a string."),
    ("# F1 cell 2/3", "## Our t4 leg: run", "Runs our own models as a separate process."),
    ("# F1 cell 3/3", "## Our t4 leg: blend", "One global rank weight: rank((1 - W) rank(public) + W rank(ours))."),
    ("# PROBE setup", "## Probe setup",
     "Not a submission: builds a fake test set from N training studies and starts the clocks."),
    ("# PROBE leg-only", "## Probe: our leg only", "No public stack in this version; times our leg alone."),
    ("# PROBE reader-only prelude", "## Probe: reader only", "Sets what the reader needs; times the reader alone."),
    ("# PROBE finish", "## Probe finish", "Stops the clocks and writes probe/summary.json (seconds per cell)."),
)
# one-line timing stamps a probe puts before every cell: input hidden, no heading
STAMP_PREFIX = 'open("/kaggle/working/probe/timeline.jsonl"'
TAG = "rsna67_tidy"


def _src(cell):
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def _first_line(s):
    return next((ln for ln in s.split("\n") if ln.strip()), "")


def tidy(nb, hide_lines=40, about=None):
    cells = [c for c in nb["cells"] if not (c.get("metadata") or {}).get(TAG)]
    code_before = [_src(c) for c in cells if c["cell_type"] == "code"]
    hits = {rule[0]: 0 for rule in RULES}
    out = []
    if about:
        out.append({"cell_type": "markdown", "metadata": {TAG: True}, "source": "## About this notebook\n\n" + about})
    for c in cells:
        if c["cell_type"] == "code":
            first = _first_line(_src(c))
            for prefix, head, note in RULES:
                if first.startswith(prefix):
                    hits[prefix] += 1
                    out.append({"cell_type": "markdown", "metadata": {TAG: True}, "source": f"{head}\n\n{note}"})
                    break
            meta = dict(c.get("metadata") or {})
            if len(_src(c).split("\n")) > hide_lines or first.startswith(STAMP_PREFIX):
                meta["_kg_hide-input"] = True
            else:
                meta.pop("_kg_hide-input", None)
            c = dict(c, metadata=meta)
        out.append(c)
    over = {p: n for p, n in hits.items() if n > 1}
    assert not over, f"heading rules matched more than one cell: {over}"
    assert [_src(c) for c in out if c["cell_type"] == "code"] == code_before, "code cells changed"
    nb = dict(nb, cells=out)
    added = sum(1 for c in out if (c.get("metadata") or {}).get(TAG))
    hidden = sum(1 for c in out if (c.get("metadata") or {}).get("_kg_hide-input"))
    return nb, added, hidden


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ipynb", help="built notebook, rewritten in place")
    ap.add_argument("--hide-lines", type=int, default=40, help="hide the input of code cells longer than this")
    ap.add_argument("--about", help="kernel slug to look up in tidy_about.json (an 'About this notebook' cell on top)")
    a = ap.parse_args()
    about = None
    if a.about:
        about = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tidy_about.json"),
                               encoding="utf-8"))[a.about]
    nb = json.load(open(a.ipynb, encoding="utf-8"))
    before = len(nb["cells"])
    nb, added, hidden = tidy(nb, a.hide_lines, about)
    again, added2, hidden2 = tidy(nb, a.hide_lines, about)
    assert again == nb and (added2, hidden2) == (added, hidden), "not idempotent"
    with open(a.ipynb, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)
    print(f"{a.ipynb}: {before} -> {len(nb['cells'])} cells, {added} headings, {hidden} code cells hidden")
