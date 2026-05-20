"""
Render the HRM-vs-baseline architecture diagram to a PNG.

Standalone matplotlib — no browser, no Excalidraw dependency. The design
is honest to the architecture: both backbones have exactly 8 transformer
blocks; the ONLY change is that HRM loops them instead of stacking them.
So the picture is two identical 8-block stacks — one plain, one with a
recurrence loop — and the depth readout (8 vs 32) tells the rest.

    python scripts/generate_diagram.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "assets" / "architecture.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ---- palette: dark, disciplined (grey = control, violet = subject) ----
BG       = "#0a0d14"
GREY_F   = "#242a36"
GREY_E   = "#4a5468"
GREY_T   = "#9aa3b3"
VIO      = "#a371f7"
VIO_F    = "#5e42a3"
DIMLINE  = "#5a6273"
WHITE    = "#f4f6fa"
MUTED    = "#737c8d"
MONO     = "monospace"


def sp(s):
    """letter-space a string for an engineered, technical look"""
    return " ".join(s)


def rrect(ax, x, y, w, h, *, fc, ec, lw, rs=7, z=2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rs}",
        fc=fc, ec=ec, lw=lw, zorder=z))


def txt(ax, x, y, s, *, fs, color=WHITE, ha="center", va="center",
        weight="normal", mono=False, rot=0, z=6):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, weight=weight,
            family=(MONO if mono else "sans-serif"), rotation=rot, zorder=z)


def stack(ax, cx, y0, *, fc, ec, lw):
    """draw a vertical stack of 8 transformer blocks centered at cx"""
    bw, bh, gap = 210, 40, 14
    for i in range(8):
        rrect(ax, cx - bw / 2, y0 + i * (bh + gap), bw, bh,
              fc=fc, ec=ec, lw=lw)
    return bw, y0, y0 + 8 * bh + 7 * gap   # width, bottom, top


def main():
    fig, ax = plt.subplots(figsize=(13.4, 8.38))
    fig.patch.set_facecolor(BG)
    ax.set_xlim(0, 1600)
    ax.set_ylim(0, 1000)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((-300, -300), 2200, 1600,
                 boxstyle="square,pad=0", fc=BG, ec="none", zorder=0))

    LX, RX, Y0 = 450, 1150, 300

    # ---- header ----
    txt(ax, 800, 958, sp("HRM  x  TTS"), fs=12.5, color=VIO,
        mono=True, weight="bold")
    txt(ax, 815, 902, "Same parameters.", fs=30, color=WHITE,
        weight="bold", ha="right")
    txt(ax, 831, 902, "4x the depth.", fs=30, color=VIO,
        weight="bold", ha="left")
    txt(ax, 800, 856,
        "The only architectural change: looping the backbone, not stacking it.",
        fs=14, color=MUTED)

    # ---- column headers ----
    txt(ax, LX, 772, sp("TRANSFORMER"), fs=17, color=GREY_T,
        mono=True, weight="bold")
    txt(ax, LX, 748, "stack 8 blocks, use once", fs=12.5,
        color=MUTED, mono=True)
    txt(ax, RX, 772, sp("HRM"), fs=17, color=VIO, mono=True, weight="bold")
    txt(ax, RX, 748, "4 H + 4 L blocks, looped", fs=12.5,
        color=MUTED, mono=True)

    # ---- the two stacks (identical blocks; the loop is the difference) ----
    bw, ybot, ytop = stack(ax, LX, Y0, fc=GREY_F, ec=GREY_E, lw=1.4)
    stack(ax, RX, Y0, fc=VIO_F, ec=VIO, lw=1.7)

    # recurrence loop on the HRM stack — output feeds back to input
    redge = RX + bw / 2
    ax.add_patch(FancyArrowPatch((redge + 6, ytop - 24), (redge + 6, ybot + 22),
                 connectionstyle="arc3,rad=-0.55", arrowstyle="-|>",
                 mutation_scale=26, color=VIO, lw=3.0, zorder=5))
    txt(ax, redge + 132, (ybot + ytop) / 2, sp("LOOP"), fs=13, color=VIO,
        mono=True, weight="bold", rot=90)

    # ---- "same params" dimension line between the stacks ----
    x1, x2, ym = LX + bw / 2 + 30, RX - bw / 2 - 30, (ybot + ytop) / 2
    ax.plot([x1, x2], [ym, ym], color=DIMLINE, lw=1.3, zorder=3)
    for xx in (x1, x2):
        ax.plot([xx, xx], [ym - 11, ym + 11], color=DIMLINE, lw=1.3, zorder=3)
    txt(ax, 800, ym + 31, "15.1M parameters", fs=14, color=WHITE, mono=True)
    txt(ax, 800, ym - 29, sp("identical"), fs=11, color=MUTED, mono=True)

    # ---- depth readouts ----
    for cx, num, col in [(LX, "8", GREY_T), (RX, "32", VIO)]:
        txt(ax, cx, 278, sp("DEPTH"), fs=11.5, color=MUTED, mono=True)
        txt(ax, cx, 230, num, fs=46, color=col, mono=True, weight="bold")
        txt(ax, cx, 184, "block applications", fs=11, color=MUTED, mono=True)

    # ---- footer ----
    txt(ax, 946, 126, "If the loop wins, speech models get deeper — ",
        fs=17, color=WHITE, ha="right")
    txt(ax, 952, 126, "for free.", fs=17, color=VIO, weight="bold", ha="left")
    txt(ax, 800, 80,
        "tested on LibriTTS-R speech   ·   github.com/harrrshall/hrm-vall-e",
        fs=11.5, color=MUTED, mono=True)

    fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor=BG)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
