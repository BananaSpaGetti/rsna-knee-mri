"""Dot plot of private AUC per entry for the rsna-knee-mri README (light + dark PNG)."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
# (label, private AUC, highlight) -- sources: Kaggle submissions API, read 2026-10-23 / 10-27
ROWS = [
    ("O1  own CoAtNet-2, 5-fold", 0.903, False),
    ("O7  own, refit on full data", 0.907, False),
    ("Team final  public-notebook blend", 0.942, False),
    ("L2  L1 + flip TTA", 0.944, False),
    ("L1  own model, better labels", 0.946, True),
]
THEMES = {
    "light": dict(bg="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df", mute="#8a8983", accent="#2a78d6"),
    "dark": dict(bg="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#33332f", mute="#8a8983", accent="#3987e5"),
}

for mode, t in THEMES.items():
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=200)
    fig.patch.set_facecolor(t["bg"]); ax.set_facecolor(t["bg"])
    y = list(range(len(ROWS)))
    for yi, (lab, v, hi) in zip(y, ROWS):
        c = t["accent"] if hi else t["mute"]
        ax.axhline(yi, color=t["grid"], lw=0.8, zorder=1)
        ax.scatter([v], [yi], s=90, color=c, edgecolor=t["bg"], linewidth=2, zorder=3)
        ax.text(v + 0.0012, yi, f"{v:.3f}", va="center", ha="left", fontsize=10,
                color=t["ink"], fontweight="bold" if hi else "normal")
    ax.set_yticks(y, [r[0] for r in ROWS], fontsize=10, color=t["ink"])
    ax.set_xlim(0.895, 0.956)
    ax.tick_params(axis="x", colors=t["ink2"], labelsize=9, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=t["grid"], lw=0.8); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Private leaderboard macro AUC", color=t["ink2"], fontsize=9)
    fig.suptitle("Better labels moved the score more than any model change",
                 x=0.02, ha="left", fontsize=12.5, fontweight="bold", color=t["ink"])
    fig.text(0.02, 0.015, "RSNA Knee Abnormality Detection, Kaggle 2026. L1/L2 are late submissions (scored, not ranked).",
             fontsize=7.5, color=t["ink2"])
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(f"{OUT}/private-auc-{mode}.png", facecolor=t["bg"])
    plt.close(fig)
print("ok")
