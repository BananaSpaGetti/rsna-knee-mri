"""Draw the README flow diagrams (light + dark PNG) without Mermaid."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

THEMES = {
    "light": dict(bg="#ffffff", box="#f3f2ef", edge="#c9c8c2", ink="#0b0b0b", ink2="#52514e",
                  accent="#2a78d6", accent_fill="#e3eefb", arrow="#8a8983"),
    "dark": dict(bg="#0d1117", box="#1c2128", edge="#3a3f47", ink="#f0efec", ink2="#c3c2b7",
                 accent="#3987e5", accent_fill="#16263b", arrow="#8a8983"),
}
BW, BH = 2.3, 0.95

def draw(name, size, nodes, edges, out, note=None):
    for mode, t in THEMES.items():
        W, H = size
        fig = plt.figure(figsize=(W, H), dpi=200)
        fig.patch.set_facecolor(t["bg"])
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
        for key, (x, y, title, sub, hi) in nodes.items():
            ax.add_patch(FancyBboxPatch((x - BW / 2, y - BH / 2), BW, BH,
                         boxstyle="round,pad=0,rounding_size=0.14",
                         fc=t["accent_fill"] if hi else t["box"],
                         ec=t["accent"] if hi else t["edge"], lw=1.6 if hi else 1.0))
            ax.text(x, y + (0.15 if sub else 0), title, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=t["ink"])
            if sub:
                ax.text(x, y - 0.2, sub, ha="center", va="center", fontsize=8.5, color=t["ink2"])
        for a, b, label, style in edges:
            xa, ya = nodes[a][:2]; xb, yb = nodes[b][:2]
            if abs(ya - yb) < 1e-6:
                p0 = (xa + BW / 2 * (1 if xb > xa else -1), ya); p1 = (xb - BW / 2 * (1 if xb > xa else -1), yb)
                rad = 0
            elif abs(xa - xb) < 1e-6:
                p0 = (xa, ya - BH / 2 * (1 if ya > yb else -1)); p1 = (xb, yb + BH / 2 * (1 if ya > yb else -1))
                rad = 0
            else:
                p0 = (xa + BW / 2, ya); p1 = (xb - BW / 2, yb); rad = 0
            ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13,
                         color=t["arrow"], lw=1.4, linestyle="--" if style == "dash" else "-",
                         connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
            if label:
                ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + 0.14, label, ha="center",
                        va="bottom", fontsize=8.5, color=t["ink2"], style="italic")
        if note:
            ax.text(0.25, 0.22, note, ha="left", va="center", fontsize=8.5, color=t["ink2"])
        fig.savefig(f"{out}/{name}-{mode}.png", facecolor=t["bg"])
        plt.close(fig)

which, out = sys.argv[1], sys.argv[2]
if which == "rsna":
    nodes = {
        "dcm": (1.6, 4.3, "Raw DICOM", "570 GB, random slice order", False),
        "rep": (1.6, 2.7, "Radiology reports", "train only, multilingual", False),
        "cache": (4.6, 4.3, "Pixel cache", "sorted slices, uint8", False),
        "lab": (4.6, 2.7, "Soft labels", "LLM reads of reports", True),
        "train": (7.6, 3.5, "2.5D training", "CoAtNet-2 / ConvNeXt", False),
        "w": (10.6, 3.5, "Weights", "5 seeds, full data", False),
        "kern": (10.6, 1.25, "Offline kernel", "rank blend per label", False),
        "gate": (7.6, 1.25, "Recompute gate", "must match exactly", False),
        "sub": (4.6, 1.25, "submission.csv", "", False),
    }
    edges = [("dcm", "cache", "", ""), ("rep", "lab", "", ""), ("cache", "train", "", ""),
             ("lab", "train", "", ""), ("train", "w", "", ""), ("w", "kern", "", ""),
             ("kern", "gate", "", ""), ("gate", "sub", "pass", "")]
    draw("pipeline", (12.2, 5.1), nodes, edges, out,
         note="Highlighted: the labels, the step that moved the private score most (0.907 -> 0.946).")
else:
    top, bot = 3.0, 1.2
    nodes = {
        "pre": (1.6, top, "1  Preflight", "", False),
        "ori": (4.6, top, "2  Orient", "subagents read the repo", False),
        "int": (7.6, top, "3  Interview", "you answer", True),
        "pla": (10.6, top, "4  Plan", ".plans/<slug>.md", False),
        "app": (10.6, bot, "5  Approve", "you sign off", True),
        "exe": (7.6, bot, "6  Execute", "script: 1 executor per task", False),
        "rev": (4.6, bot, "7  Review", "up to 3 fix rounds", False),
        "clo": (1.6, bot, "8  Close out", "is this what you asked for?", True),
    }
    edges = [("pre", "ori", "", ""), ("ori", "int", "", ""), ("int", "pla", "", ""),
             ("pla", "app", "", ""), ("app", "exe", "", ""), ("exe", "rev", "", ""),
             ("rev", "clo", "", "")]
    draw("anvil-flow", (12.2, 4.3), nodes, edges, out,
         note="Highlighted: the three places it stops and waits for you. A task that fails its verify is retried once at high effort.")
print("ok", which)
