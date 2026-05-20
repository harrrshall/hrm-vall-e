"""
Render the HRM-vs-baseline architecture diagram to a PNG.

Standalone matplotlib — no browser, no Excalidraw dependency. Dark,
spacious, big focal numbers. Produces assets/architecture.png, the
image to attach to the launch tweet.

    python scripts/generate_diagram.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "assets" / "architecture.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ---- palette (dark "nerd" theme) ----
BG        = "#0d1117"
CARD      = "#161b22"
EDGE      = "#2b313b"
INK       = "#e6edf3"
MUTED     = "#8b949e"
AMBER     = "#f0a93b"   # baseline accent
AMBER_DK  = "#2a2113"
VIOLET    = "#a371f7"   # HRM accent
VIOLET_DK = "#241b3a"
GREEN     = "#3fb950"
MONO      = "monospace"


def rbox(ax, x, y, w, h, *, fc, ec, lw=1.6, rs=18, z=2, alpha=1.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rs}",
        fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha))


def label(ax, x, y, text, *, fs, color=INK, ha="center", va="center",
          weight="normal", mono=False):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color,
            weight=weight, family=(MONO if mono else "sans-serif"), zorder=6)


def main():
    fig, ax = plt.subplots(figsize=(13.4, 8.4))
    fig.patch.set_facecolor(BG)
    ax.set_xlim(0, 1600)
    ax.set_ylim(0, 1000)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((-200, -200), 2000, 1400, boxstyle="square,pad=0",
                                fc=BG, ec="none", zorder=0))

    # ---- title (two-tone) ----
    label(ax, 762, 938, "HRM", fs=43, color=VIOLET, weight="bold", ha="right")
    label(ax, 800, 936, "vs", fs=26, color=MUTED, ha="center")
    label(ax, 838, 938, "Transformer", fs=43, color=AMBER, weight="bold", ha="left")
    label(ax, 800, 884,
          "an audio language model  —  text prompt in, speech tokens out",
          fs=15, color=MUTED, mono=True)

    # ================= LEFT CARD : BASELINE =================
    lx, lw_ = 96, 648
    lcx = lx + lw_ / 2
    rbox(ax, lx, 150, lw_, 690, fc=CARD, ec=EDGE, lw=1.6, rs=26)
    rbox(ax, lx + 36, 766, 188, 48, fc=AMBER_DK, ec=AMBER, lw=1.4, rs=14)
    label(ax, lx + 36 + 94, 790, "BASELINE", fs=15, color=AMBER,
          weight="bold", mono=True)

    for yy in (660, 592, 524):
        rbox(ax, lcx - 200, yy, 270, 52, fc=AMBER_DK, ec=AMBER, lw=1.5, rs=12)
        label(ax, lcx - 65, yy + 26, "block", fs=15, color=INK, mono=True)
    label(ax, lcx + 150, 618, "x8", fs=46, color=AMBER, weight="bold", mono=True)
    label(ax, lcx + 150, 566, "stacked", fs=13, color=MUTED, mono=True)

    label(ax, lcx, 462, "8 distinct blocks, each used once",
          fs=14.5, color=MUTED, mono=True)
    ax.plot([lx + 60, lx + lw_ - 60], [430, 430], color=EDGE, lw=1.2, zorder=1)
    label(ax, lcx, 392, "EFFECTIVE  DEPTH", fs=14, color=MUTED, mono=True)
    label(ax, lcx, 288, "8", fs=128, color=AMBER, weight="bold", mono=True)
    label(ax, lcx, 196, "8 blocks  x  1 pass", fs=13.5, color=MUTED, mono=True)

    # ================= RIGHT CARD : HRM =================
    rx, rw_ = 856, 648
    rcx = rx + rw_ / 2
    rbox(ax, rx, 150, rw_, 690, fc=CARD, ec=EDGE, lw=1.6, rs=26)
    rbox(ax, rx + 36, 766, 130, 48, fc=VIOLET_DK, ec=VIOLET, lw=1.4, rs=14)
    label(ax, rx + 36 + 65, 790, "HRM", fs=15, color=VIOLET,
          weight="bold", mono=True)

    rbox(ax, rcx - 175, 648, 350, 66, fc=VIOLET_DK, ec=VIOLET, lw=1.6, rs=14)
    label(ax, rcx, 681, "H  .  planner", fs=16, color=INK, mono=True)
    rbox(ax, rcx - 175, 520, 350, 66, fc=VIOLET_DK, ec=VIOLET, lw=1.6, rs=14)
    label(ax, rcx, 553, "L  .  worker", fs=16, color=INK, mono=True)

    # recurrence loop (two curved arrows)
    ax.add_patch(FancyArrowPatch((rcx - 90, 648), (rcx - 90, 586),
                                 connectionstyle="arc3,rad=0.5",
                                 arrowstyle="-|>", mutation_scale=20,
                                 color=VIOLET, lw=2.2, zorder=4))
    ax.add_patch(FancyArrowPatch((rcx + 90, 586), (rcx + 90, 648),
                                 connectionstyle="arc3,rad=0.5",
                                 arrowstyle="-|>", mutation_scale=20,
                                 color=VIOLET, lw=2.2, zorder=4))
    label(ax, rcx, 617, "loop x6", fs=13.5, color=VIOLET, weight="bold", mono=True)
    ax.add_patch(FancyArrowPatch((rcx - 230, 681), (rcx - 175, 681),
                                 arrowstyle="-|>", mutation_scale=18,
                                 color=MUTED, lw=2.0, zorder=4))
    label(ax, rcx - 238, 681, "input", fs=12.5, color=MUTED, ha="right", mono=True)

    label(ax, rcx, 462, "4 + 4 shared blocks, reused every loop",
          fs=14.5, color=MUTED, mono=True)
    ax.plot([rx + 60, rx + rw_ - 60], [430, 430], color=EDGE, lw=1.2, zorder=1)
    label(ax, rcx, 392, "EFFECTIVE  DEPTH", fs=14, color=MUTED, mono=True)
    label(ax, rcx, 288, "32", fs=128, color=VIOLET, weight="bold", mono=True)
    label(ax, rcx, 196, "8 blocks  x  4 passes", fs=13.5, color=MUTED, mono=True)

    # ---- bridge: same params ----
    rbox(ax, 690, 408, 220, 46, fc=BG, ec=MUTED, lw=1.4, rs=14, z=5)
    label(ax, 800, 431, "= 15.1M params", fs=13.5, color=INK, mono=True)

    # ---- thesis ----
    label(ax, 800, 92,
          "Same parameters. If recurrence beats stacking,",
          fs=18, color=INK)
    label(ax, 800, 60,
          "speech models get deeper for free.",
          fs=18, color=VIOLET, weight="bold")
    label(ax, 800, 24, "github.com/harrrshall/hrm-vall-e",
          fs=12, color=MUTED, mono=True)

    fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor=BG)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
