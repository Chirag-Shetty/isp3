"""
ppt_architecture.py
-------------------
Generates a compact, PPT-ready architecture diagram.
Wide (16:9), clean, light background, single horizontal flow.
Run: python ppt_architecture.py
Output: results/diagrams/ppt_architecture.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

os.makedirs("results/diagrams", exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
BG      = "#FFFFFF"
TEXT    = "#1a1a2e"
GREY    = "#6B7280"

COLORS = {
    "input"  : ("#2563EB", "#EFF6FF"),   # blue
    "proj"   : ("#7C3AED", "#F5F3FF"),   # purple
    "trans"  : ("#059669", "#ECFDF5"),   # green
    "cnn"    : ("#D97706", "#FFFBEB"),   # amber
    "lstm"   : ("#DC2626", "#FEF2F2"),   # red
    "pool"   : ("#7C3AED", "#F5F3FF"),   # purple
    "out"    : ("#1D4ED8", "#EFF6FF"),   # blue
    "fall"   : ("#DC2626", "#FEF2F2"),   # red result
    "nofall" : ("#16A34A", "#F0FDF4"),   # green result
}

fig = plt.figure(figsize=(16, 5), facecolor=BG)
ax  = fig.add_axes([0.01, 0.05, 0.98, 0.88])
ax.set_facecolor(BG)
ax.set_xlim(0, 16); ax.set_ylim(0, 5)
ax.axis("off")

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(8, 4.72, "Transformer-CNN-LSTM  |  Radar Fall Detection Architecture",
        ha="center", va="center", fontsize=13, fontweight="bold", color=TEXT)

# ── Helper: draw a block ──────────────────────────────────────────────────────
def block(ax, x, y, w, h, label, sublabel, color_pair, fontsize=9):
    edge_c, face_c = color_pair
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.08",
                         facecolor=face_c, edgecolor=edge_c,
                         linewidth=2, zorder=3)
    ax.add_patch(box)
    ax.text(x, y + 0.13, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=edge_c, zorder=4)
    ax.text(x, y - 0.22, sublabel, ha="center", va="center",
            fontsize=7, color=GREY, zorder=4)

def arrow(ax, x1, x2, y=2.3, col="#9CA3AF"):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=col,
                                lw=1.8, mutation_scale=14),
                zorder=2)

# ── Block positions (horizontal, centered at y=2.3) ───────────────────────────
Y = 2.3
BW = 1.55   # block width
BH = 1.15   # block height
GAP = 0.28  # gap between blocks

blocks = [
    # (x_center, label, sublabel, color_key)
    (0.95,  "RADAR\nINPUT",          "40 frames\n× 20 features",   "input"),
    (2.90,  "INPUT\nPROJECTION",     "Linear\n20 → 64 dims",       "proj"),
    (4.85,  "TRANSFORMER\nENCODER",  "2L × 4-Head\nSelf-Attention", "trans"),
    (6.80,  "1D CNN\nBLOCK",         "Conv 64→128\n+ MaxPool",      "cnn"),
    (8.75,  "Bi-LSTM",               "2L, h=128\nBidirectional",    "lstm"),
    (10.70, "ATTENTION\nPOOLING",    "Focus on\nfall frames",       "pool"),
    (12.65, "CLASSIFIER\nHEAD",      "FC 256→128→2\n+ Softmax",     "out"),
]

for x, lbl, sub, ckey in blocks:
    block(ax, x, Y, BW, BH, lbl, sub, COLORS[ckey])

# Arrows between blocks
xs = [b[0] for b in blocks]
for i in range(len(xs) - 1):
    arrow(ax, xs[i] + BW/2, xs[i+1] - BW/2, Y)

# ── Output split after classifier ─────────────────────────────────────────────
# Arrow down-left to NO FALL
ax.annotate("", xy=(11.9, 0.82), xytext=(13.42, Y - BH/2),
            arrowprops=dict(arrowstyle="-|>", color=COLORS["nofall"][0],
                            lw=1.8, mutation_scale=13))
# Arrow down-right to FALL
ax.annotate("", xy=(15.2, 0.82), xytext=(13.88, Y - BH/2),
            arrowprops=dict(arrowstyle="-|>", color=COLORS["fall"][0],
                            lw=1.8, mutation_scale=13))

# NO FALL box
nf = FancyBboxPatch((10.85, 0.12), 2.0, 0.68,
                    boxstyle="round,pad=0.07",
                    facecolor=COLORS["nofall"][1],
                    edgecolor=COLORS["nofall"][0], linewidth=2, zorder=3)
ax.add_patch(nf)
ax.text(11.85, 0.60, "✓  NO FALL", ha="center", fontsize=9.5,
        fontweight="bold", color=COLORS["nofall"][0], zorder=4)
ax.text(11.85, 0.28, "P = 3.2%", ha="center", fontsize=8,
        color=GREY, zorder=4)

# FALL box
fb = FancyBboxPatch((14.25, 0.12), 1.65, 0.68,
                    boxstyle="round,pad=0.07",
                    facecolor=COLORS["fall"][1],
                    edgecolor=COLORS["fall"][0], linewidth=2.5, zorder=3)
ax.add_patch(fb)
ax.text(15.12, 0.60, "⚠  FALL", ha="center", fontsize=9.5,
        fontweight="bold", color=COLORS["fall"][0], zorder=4)
ax.text(15.12, 0.28, "P = 96.8%", ha="center", fontsize=8,
        color=GREY, zorder=4)

# ── Bottom annotation strip ───────────────────────────────────────────────────
ax.axhline(4.35, xmin=0.01, xmax=0.99, color="#E5E7EB", lw=1)

notes = [
    (0.95,  "Radar Point Cloud\n18 fps streaming"),
    (2.90,  "Embed radar features\ninto 64-dim space"),
    (4.85,  "Global context\nacross all 40 frames"),
    (6.80,  "Local sharp pattern\ndetection (3-frame)"),
    (8.75,  "Sequential memory\nfwd + bwd"),
    (10.70, "Weighted sum\nover timesteps"),
    (12.65, "Binary output\nFall / No Fall"),
]
for x, note in notes:
    ax.text(x, 4.12, note, ha="center", va="center",
            fontsize=6.5, color=GREY, linespacing=1.4)

# ── Metrics footer ────────────────────────────────────────────────────────────
metrics = [
    "Accuracy: 97.85%",
    "Fall Recall: 95.56%",
    "AUC-ROC: 0.9934",
    "Inference: 18.3 ms",
    "Params: 1.48 M",
]
for i, m in enumerate(metrics):
    xm = 1.5 + i * 2.6
    ax.text(xm, 4.60, m, ha="center", fontsize=7.5,
            color="#374151", fontweight="bold")

plt.savefig("results/diagrams/ppt_architecture.png",
            dpi=220, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved: results/diagrams/ppt_architecture.png")
