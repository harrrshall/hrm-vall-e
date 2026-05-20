"""
Render the HRM-vs-baseline architecture diagram to a PNG.

Standalone matplotlib — no browser, no Excalidraw dependency. Produces
assets/architecture.png, the image to attach to the launch tweet.

    python scripts/generate_diagram.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = Path(__file__).resolve().parents[1] / "assets" / "architecture.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# palette
INK, GREY, WHITE = "#1e1e1e", "#5b6472", "#ffffff"
ORANGE, ORANGE_BG, ORANGE_FILL = "#e08a00", "#fdebd3", "#ffd8a8"
PURPLE, PURPLE_BG, PURPLE_FILL = "#7c3aed", "#ece3ff", "#d0bfff"
BLUE, BLUE_BG, BLUE_FILL = "#2f7fd8", "#e4ecfb", "#a5d8ff"
GREEN, GREEN_FILL = "#1f9d4d", "#b2f2bb"
YELLOW, YELLOW_FILL = "#e0a800", "#fff3bf"


def box(ax, x, y, w, h, *, fc, ec, text="", fs=13, tc=INK, weight="normal",
        lw=1.8, rs=14, alpha=1.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rs}",
        fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=2))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, weight=weight, zorder=3)


def arrow(ax, p0, p1, *, color=INK, lw=2.2, style="-|>", ms=22):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                                 color=color, lw=lw, zorder=4,
                                 shrinkA=0, shrinkB=0))


def main():
    fig, ax = plt.subplots(figsize=(13.2, 8.2))
    ax.set_xlim(0, 1600)
    ax.set_ylim(0, 1000)
    ax.axis("off")

    # ---- title ----
    ax.text(800, 958, "HRM  vs  Transformer  —  same parameters, 4x the compute",
            ha="center", va="center", fontsize=21, weight="bold", color=INK)
    ax.text(800, 922, "Can hierarchical recurrence beat plain stacking on audio tokens?",
            ha="center", va="center", fontsize=13.5, color=GREY)

    # ---- left: baseline ----
    box(ax, 70, 432, 640, 458, fc=ORANGE_BG, ec=ORANGE, alpha=0.55, lw=1.4, rs=22)
    box(ax, 130, 818, 520, 52, fc=ORANGE_FILL, ec=ORANGE,
        text="BASELINE  —  stacked transformer", fs=14.5, weight="bold")
    arrow(ax, (210, 512), (210, 800), color=BLUE, lw=2.4)
    blocks = [("Block 8", 738), ("Block 7", 678)]
    for label, y in blocks:
        box(ax, 270, y, 320, 48, fc=BLUE_FILL, ec=BLUE, text=label, fs=14)
    ax.text(430, 643, ". . .", ha="center", va="center", fontsize=22, color=GREY)
    for label, y in [("Block 2", 568), ("Block 1", 508)]:
        box(ax, 270, y, 320, 48, fc=BLUE_FILL, ec=BLUE, text=label, fs=14)
    box(ax, 210, 452, 360, 44, fc=ORANGE, ec=ORANGE,
        text="effective depth = 8", fs=14, tc=WHITE, weight="bold")

    # ---- right: HRM ----
    box(ax, 890, 432, 640, 458, fc=PURPLE_BG, ec=PURPLE, alpha=0.55, lw=1.4, rs=22)
    box(ax, 950, 818, 520, 52, fc=PURPLE_FILL, ec=PURPLE,
        text="HRM  —  recurrent backbone", fs=14.5, weight="bold")
    box(ax, 1050, 686, 380, 82, fc=PURPLE_FILL, ec=PURPLE,
        text="H-module : 4 blocks\n(slow planner)", fs=13.5)
    box(ax, 1050, 556, 380, 82, fc=PURPLE_FILL, ec=PURPLE,
        text="L-module : 4 blocks\n(fast worker)", fs=13.5)
    arrow(ax, (1120, 686), (1120, 638), color=PURPLE)      # H -> L
    arrow(ax, (1360, 638), (1360, 686), color=PURPLE)      # L -> H
    ax.text(1240, 662, "loop x6", ha="center", va="center", fontsize=12.5,
            color=PURPLE, weight="bold")
    arrow(ax, (965, 727), (1050, 727), color=PURPLE)
    ax.text(965, 752, "input inject", ha="center", va="bottom", fontsize=11.5,
            color=PURPLE)
    box(ax, 1070, 452, 380, 44, fc=PURPLE, ec=PURPLE,
        text="effective depth = 32", fs=14, tc=WHITE, weight="bold")

    # ---- center VS ----
    ax.add_patch(Circle((800, 660), 46, fc=YELLOW_FILL, ec=YELLOW, lw=2.2,
                        zorder=5))
    ax.text(800, 660, "VS", ha="center", va="center", fontsize=18,
            weight="bold", color=INK, zorder=6)

    # ---- bottom: shared task ----
    box(ax, 70, 150, 1460, 250, fc=BLUE_BG, ec=BLUE, alpha=0.55, lw=1.4, rs=24)
    ax.text(800, 360, "THE SHARED TASK  —  autoregressive audio language model",
            ha="center", va="center", fontsize=14.5, weight="bold", color=INK)
    box(ax, 210, 250, 470, 70, fc=BLUE_FILL, ec=BLUE,
        text="[BOS]  text prompt  [SEP]", fs=14.5)
    arrow(ax, (690, 285), (910, 285), color=INK, lw=2.4)
    ax.text(800, 305, "predict", ha="center", va="bottom", fontsize=12, color=GREY)
    box(ax, 920, 250, 470, 70, fc=GREEN_FILL, ec=GREEN,
        text="EnCodec speech tokens  [EOS]", fs=14.5)
    ax.text(800, 195, "PrefixLM: bidirectional text in  ->  causal speech tokens out",
            ha="center", va="center", fontsize=12, color=GREY)

    # ---- thesis banner ----
    box(ax, 70, 56, 1460, 60, fc=YELLOW_FILL, ec=YELLOW, lw=1.6, rs=18,
        text="If HRM wins: deeper, smarter speech models at NO extra parameters.",
        fs=15, weight="bold")

    fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
