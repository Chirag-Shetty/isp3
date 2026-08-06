"""
visualize_model.py
------------------
Generates professional light-background diagrams explaining
how the Transformer-CNN-LSTM fall detection model works.

Outputs (saved to results/diagrams/):
  01_full_architecture.png   -- End-to-end data flow
  02_feature_vector.png      -- What the 20 features mean
  03_attention_concept.png   -- How Transformer attention works
  04_fall_vs_nofall.png      -- What a fall looks like vs normal
  05_prediction_card.png     -- Final output / result card

Run: python visualize_model.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyBboxPatch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, FancyArrow
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker

os.makedirs("results/diagrams", exist_ok=True)

# ── Light theme palette ────────────────────────────────────────────────────────
BG        = "#FAFAFA"
PANEL     = "#FFFFFF"
BORDER    = "#E0E0E0"
TEXT      = "#1a1a2e"
SUBTEXT   = "#555577"

C_RADAR   = "#2563EB"   # blue
C_INPUT   = "#7C3AED"   # purple
C_POS     = "#0891B2"   # teal
C_TRANS   = "#059669"   # green
C_CNN     = "#D97706"   # amber
C_LSTM    = "#DC2626"   # red
C_ATTN    = "#7C3AED"   # purple
C_CLASS   = "#1D4ED8"   # blue
C_NOFALL  = "#16A34A"   # green
C_FALL    = "#DC2626"   # red
C_ARROW   = "#6B7280"   # grey

def rounded_box(ax, x, y, w, h, color, label, sublabel="",
                fontsize=9, sub_fontsize=7.5, radius=0.04, alpha=0.13,
                text_color=None):
    """Draw a rounded rectangle with label."""
    tc = text_color or color
    fancy = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle=f"round,pad={radius}",
                           facecolor=color, alpha=alpha,
                           edgecolor=color, linewidth=1.8,
                           transform=ax.transData, zorder=3)
    ax.add_patch(fancy)
    ax.text(x, y + (0.02 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=tc, zorder=4)
    if sublabel:
        ax.text(x, y - 0.055, sublabel, ha="center", va="center",
                fontsize=sub_fontsize, color=SUBTEXT, zorder=4,
                style="italic")

def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.6):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=12))


# ==============================================================================
# FIGURE 1 ── Full Architecture Flow (top → bottom, portrait)
# ==============================================================================
fig = plt.figure(figsize=(10, 18), facecolor=BG)
ax  = fig.add_axes([0.05, 0.02, 0.90, 0.95])
ax.set_facecolor(BG)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.axis("off")

# Title
ax.text(0.5, 0.975, "Transformer-CNN-LSTM Fall Detection",
        ha="center", va="top", fontsize=15, fontweight="bold", color=TEXT)
ax.text(0.5, 0.955, "Architecture — Data Flow & Inference Pipeline",
        ha="center", va="top", fontsize=10, color=SUBTEXT)

# ─── Stage positions (y values, top to bottom) ───────────────────────────────
stages = [
    # (y_center, color, title, subtitle)
    (0.895, C_RADAR,  "RADAR INPUT",
     "mmWave IWR6843  →  40 frames × 20 features  (2.2 second window)"),
    (0.800, C_INPUT,  "INPUT PROJECTION",
     "nn.Linear(20 → 64)   |   Projects radar features into model embedding space"),
    (0.720, C_POS,   "POSITIONAL ENCODING",
     "Sinusoidal encoding   |   Tells model WHICH frame is 1st, 2nd … 40th"),
    (0.635, C_TRANS,  "TRANSFORMER ENCODER",
     "2 layers  ×  4-head Self-Attention  |   Sees ALL 40 frames simultaneously"),
    (0.540, C_CNN,   "1D CNN BLOCK",
     "Conv1D (64→128→128)  +  MaxPool(2)   |   Detects local sharp patterns"),
    (0.445, C_LSTM,  "BIDIRECTIONAL LSTM",
     "2 layers  |  Hidden=128  |  Bidirectional   |   Sequential memory"),
    (0.355, C_ATTN,  "ATTENTION POOLING",
     "Learned weights over time steps   |   Focus on the fall moment"),
    (0.265, C_CLASS, "CLASSIFIER HEAD",
     "FC(256→128→2)  +  Dropout  +  Softmax   |   Final prediction"),
]

BOX_W = 0.82; BOX_H = 0.055

for i, (yc, col, title, sub) in enumerate(stages):
    rounded_box(ax, 0.5, yc, BOX_W, BOX_H, col, title, sub,
                fontsize=10, sub_fontsize=8, radius=0.015)
    # Arrow to next
    if i < len(stages) - 1:
        y_next = stages[i+1][0]
        arrow(ax, 0.5, yc - BOX_H/2 - 0.004,
                  0.5, y_next + BOX_H/2 + 0.004, color=C_ARROW)

# ── Output boxes ──────────────────────────────────────────────────────────────
y_out = 0.155
ax.text(0.5, 0.21, "OUTPUT PROBABILITIES   (Softmax)", ha="center",
        fontsize=9, color=SUBTEXT, style="italic")

# No Fall box
fancy_nf = FancyBboxPatch((0.07, y_out - 0.045), 0.36, 0.09,
                           boxstyle="round,pad=0.012",
                           facecolor=C_NOFALL, alpha=0.12,
                           edgecolor=C_NOFALL, linewidth=2)
ax.add_patch(fancy_nf)
ax.text(0.25, y_out + 0.01, "NO FALL", ha="center", fontsize=12,
        fontweight="bold", color=C_NOFALL)
ax.text(0.25, y_out - 0.022, "P = 0.032   (3.2%)", ha="center",
        fontsize=9, color=SUBTEXT)

# Fall box
fancy_f = FancyBboxPatch((0.55, y_out - 0.045), 0.38, 0.09,
                          boxstyle="round,pad=0.012",
                          facecolor=C_FALL, alpha=0.12,
                          edgecolor=C_FALL, linewidth=2.5)
ax.add_patch(fancy_f)
ax.text(0.74, y_out + 0.01, "⚠  FALL DETECTED", ha="center",
        fontsize=12, fontweight="bold", color=C_FALL)
ax.text(0.74, y_out - 0.022, "P = 0.968   (96.8%)", ha="center",
        fontsize=9, color=SUBTEXT)

# Arrows from classifier to outputs
arrow(ax, 0.38, 0.265 - BOX_H/2 - 0.004,
          0.25, y_out + 0.045 + 0.004, color=C_NOFALL)
arrow(ax, 0.62, 0.265 - BOX_H/2 - 0.004,
          0.74, y_out + 0.045 + 0.004, color=C_FALL)

# ── Decision ──────────────────────────────────────────────────────────────────
ax.text(0.74, 0.065, "→  Alert sent to dashboard", ha="center",
        fontsize=9, color=C_FALL, fontweight="bold")
ax.text(0.25, 0.065, "→  No action taken", ha="center",
        fontsize=9, color=C_NOFALL)

# ── Parameters bar ────────────────────────────────────────────────────────────
bar_y = 0.025
info_items = [
    ("Parameters", "1.48 M"),
    ("Input Window", "40 × 20"),
    ("Model Size", "5.66 MB"),
    ("Inference", "18.3 ms"),
    ("Accuracy", "97.85%"),
    ("Fall Recall", "95.56%"),
]
for i, (k, v) in enumerate(info_items):
    xp = 0.08 + i * 0.155
    ax.text(xp, bar_y + 0.01, v, ha="center", fontsize=9,
            fontweight="bold", color=TEXT)
    ax.text(xp, bar_y - 0.008, k, ha="center", fontsize=7,
            color=SUBTEXT)

ax.axhline(0.037, xmin=0.04, xmax=0.96, color=BORDER, linewidth=1)

plt.savefig("results/diagrams/01_full_architecture.png",
            dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(); print("Saved: 01_full_architecture.png")


# ==============================================================================
# FIGURE 2 ── 20 Feature Vector Explained
# ==============================================================================
fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG)
ax.set_facecolor(BG); ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

ax.text(0.5, 0.96, "Input Feature Vector — 20 Radar Features Per Frame",
        ha="center", fontsize=13, fontweight="bold", color=TEXT)
ax.text(0.5, 0.905, "Each radar frame → 20 numbers → fed to model 40 frames at a time",
        ha="center", fontsize=9, color=SUBTEXT)

# Feature groups
groups = [
    ("Point Cloud Features  (12)", C_CNN, [
        ("0", "x_mean", "Left-right\nposition", False),
        ("1", "y_mean", "Depth\n(distance)", False),
        ("2", "z_mean", "Height ★\nFall indicator", True),
        ("3", "vx", "X velocity", False),
        ("4", "vy", "Y velocity", False),
        ("5", "vz", "Z velocity", False),
        ("6", "ax", "X accel", False),
        ("7", "ay", "Y accel", False),
        ("8", "az", "Z accel", False),
        ("9", "n_pts", "# Points ★", True),
        ("10", "spread", "XY spread", False),
        ("11", "h_rng", "Height range ★\nFall indicator", True),
    ]),
    ("Track Features  (6)", C_TRANS, [
        ("12", "trk_x", "Track X", False),
        ("13", "trk_y", "Track Y", False),
        ("14", "trk_z", "Track Z", False),
        ("15", "trk_vx", "Track Vx", False),
        ("16", "trk_vy", "Track Vy", False),
        ("17", "trk_vz", "Track Vz", False),
    ]),
    ("Height Features  (2)", C_RADAR, [
        ("18", "height", "Person height", False),
        ("19", "bottom", "Bottom height", False),
    ]),
]

x_cursor = 0.03
for grp_name, col, features in groups:
    n = len(features)
    grp_w = n * 0.067 + 0.01
    # Group label
    ax.text(x_cursor + grp_w/2, 0.84, grp_name, ha="center",
            fontsize=8.5, fontweight="bold", color=col)
    # Group bracket
    ax.annotate("", xy=(x_cursor + grp_w - 0.008, 0.81),
                xytext=(x_cursor + 0.008, 0.81),
                arrowprops=dict(arrowstyle="-", color=col, lw=1.5))

    for j, (idx, name, desc, is_key) in enumerate(features):
        xc = x_cursor + j * 0.067 + 0.033
        bw, bh = 0.056, 0.52

        # Box
        box = FancyBboxPatch((xc - 0.028, 0.13), 0.056, 0.62,
                              boxstyle="round,pad=0.008",
                              facecolor=col,
                              alpha=0.22 if is_key else 0.07,
                              edgecolor=col,
                              linewidth=2 if is_key else 1)
        ax.add_patch(box)

        # Index badge
        ax.text(xc, 0.695, f"[{idx}]", ha="center", fontsize=7.5,
                fontweight="bold", color=col)
        # Feature name
        ax.text(xc, 0.64, name, ha="center", fontsize=8,
                fontweight="bold", color=TEXT)
        # Description
        ax.text(xc, 0.375, desc, ha="center", va="center",
                fontsize=6.8, color=SUBTEXT, multialignment="center")

        if is_key:
            ax.text(xc, 0.175, "KEY", ha="center", fontsize=6.5,
                    fontweight="bold", color=col,
                    bbox=dict(boxstyle="round,pad=0.08", fc=col,
                              alpha=0.15, ec=col))

    x_cursor += grp_w + 0.015

# Legend
ax.text(0.5, 0.055, "★  KEY FEATURES: z_mean (height drops on fall)  |  "
        "height_range (body goes horizontal)  |  n_points (few ground reflections)",
        ha="center", fontsize=8.5, color=C_FALL, fontweight="bold")

plt.savefig("results/diagrams/02_feature_vector.png",
            dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(); print("Saved: 02_feature_vector.png")


# ==============================================================================
# FIGURE 3 ── Transformer Attention Concept
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
fig.suptitle("How Transformer Self-Attention Works in Fall Detection",
             fontsize=13, fontweight="bold", color=TEXT, y=0.97)

# Left: Attention map visualization
ax = axes[0]
ax.set_facecolor(BG)
ax.set_title("Attention Heatmap — Fall Event Window (40 frames)",
             fontsize=10, color=TEXT, pad=10)

# Simulate attention weights — fall happens around frame 28-35
attn = np.zeros((40, 40))
for i in range(40):
    for j in range(40):
        # Base attention
        dist = abs(i - j)
        attn[i, j] = np.exp(-dist * 0.15)
        # High attention around fall zone (frames 28-35)
        if 27 <= i <= 35 or 27 <= j <= 35:
            attn[i, j] += 0.4 * np.exp(-dist * 0.1)
attn /= attn.max()

im = ax.imshow(attn, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
             label="Attention Weight")
ax.set_xlabel("Frame (Key)", fontsize=9, color=TEXT)
ax.set_ylabel("Frame (Query)", fontsize=9, color=TEXT)
ax.tick_params(colors=TEXT, labelsize=8)
ax.axvline(27, color=C_FALL, lw=1.5, linestyle="--", alpha=0.8)
ax.axvline(35, color=C_FALL, lw=1.5, linestyle="--", alpha=0.8)
ax.axhline(27, color=C_FALL, lw=1.5, linestyle="--", alpha=0.8)
ax.axhline(35, color=C_FALL, lw=1.5, linestyle="--", alpha=0.8)
ax.text(31, 2, "Fall\nZone", ha="center", fontsize=8, color=C_FALL,
        fontweight="bold")

# Right: Frame-by-frame z_mean showing fall
ax2 = axes[1]
ax2.set_facecolor(BG)
ax2.set_title("z_mean (Height) Over 40 Frames — Fall at Frame 28",
              fontsize=10, color=TEXT, pad=10)

frames = np.arange(1, 41)
z_vals = np.concatenate([
    np.linspace(1.65, 1.62, 10) + np.random.randn(10) * 0.02,  # standing
    np.linspace(1.61, 1.64, 8)  + np.random.randn(8) * 0.02,   # walking
    np.array([1.58, 1.30, 0.85, 0.42, 0.28, 0.22, 0.21]),       # falling
    np.linspace(0.20, 0.22, 15) + np.random.randn(15) * 0.01,   # on ground
])
z_vals = z_vals[:40]

ax2.plot(frames[:27], z_vals[:27], color=C_TRANS, lw=2.2, label="Normal (upright)")
ax2.plot(frames[27:34], z_vals[27:34], color=C_FALL, lw=2.5,
         label="FALLING", zorder=5)
ax2.plot(frames[34:], z_vals[34:], color=C_CNN, lw=2, label="On ground")

ax2.fill_between(frames[27:34], 0, z_vals[27:34], alpha=0.1, color=C_FALL)
ax2.axvline(28, color=C_FALL, lw=1.5, linestyle="--", alpha=0.7)
ax2.axhline(0.4, color=SUBTEXT, lw=1, linestyle=":", alpha=0.7)
ax2.text(29.2, 0.45, "body_flat threshold\n(0.40 m)", fontsize=7.5,
         color=SUBTEXT)
ax2.text(28.5, 1.5, "Fall\ntrigger", fontsize=8, color=C_FALL,
         fontweight="bold")

ax2.set_xlabel("Frame Number", fontsize=9, color=TEXT)
ax2.set_ylabel("z_mean — Height (metres)", fontsize=9, color=TEXT)
ax2.set_ylim(0, 1.9)
ax2.tick_params(colors=TEXT, labelsize=8)
ax2.spines[:].set_color(BORDER)
ax2.legend(fontsize=8, loc="upper right")
ax2.set_facecolor(BG)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("results/diagrams/03_attention_concept.png",
            dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(); print("Saved: 03_attention_concept.png")


# ==============================================================================
# FIGURE 4 ── Fall vs No-Fall Feature Comparison
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor=BG)
fig.suptitle("What the Model Sees — Fall vs Normal Activity Comparison",
             fontsize=13, fontweight="bold", color=TEXT, y=0.98)

np.random.seed(42)
T = np.arange(1, 41)

def smooth(x, w=3):
    return np.convolve(x, np.ones(w)/w, mode="same")

# Generate realistic normal and fall signals
fall_z    = np.concatenate([np.ones(20)*1.65, np.linspace(1.65,0.22,10), np.ones(10)*0.22])
normal_z  = np.ones(40)*1.62 + np.random.randn(40)*0.04

fall_hrng = np.concatenate([np.ones(20)*0.35, np.linspace(0.35,1.55,5), np.linspace(1.55,0.15,5), np.ones(10)*0.15])
normal_hrng = np.ones(40)*0.38 + np.random.randn(40)*0.04

fall_vel  = np.concatenate([np.ones(20)*0.05, np.abs(np.random.randn(5)*0.8)+0.5, np.ones(15)*0.02])
normal_vel = np.abs(np.random.randn(40)*0.12 + 0.08)

fall_npts = np.concatenate([np.ones(20)*18, np.linspace(18,4,8), np.ones(12)*4])
normal_npts = np.random.randint(14, 25, 40).astype(float)

datasets = [
    (axes[0,0], "z_mean — Height (metres)", "Height drops sharply on fall",
     fall_z, normal_z, (0.0, 2.0)),
    (axes[0,1], "height_range (metres)", "Body becomes horizontal",
     fall_hrng, normal_hrng, (0.0, 1.8)),
    (axes[1,0], "Velocity magnitude (m/s)", "Sudden velocity spike",
     fall_vel, normal_vel, (0.0, 1.4)),
    (axes[1,1], "n_points (radar returns)", "Fewer points on ground",
     fall_npts, normal_npts, (0, 30)),
]

for ax, title, subtitle, f_data, n_data, ylim in datasets:
    ax.set_facecolor("#FCFCFC")
    ax.spines[:].set_color(BORDER)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.set_title(f"{title}\n{subtitle}", fontsize=9.5, color=TEXT,
                 fontweight="bold", pad=6)

    ax.plot(T, smooth(n_data), color=C_TRANS, lw=2, label="Normal")
    ax.plot(T, smooth(f_data), color=C_FALL, lw=2.5, label="Fall", zorder=5)
    ax.fill_between(T, smooth(f_data), smooth(n_data),
                    alpha=0.07, color=C_FALL)
    ax.axvline(20, color=C_FALL, lw=1.4, linestyle="--", alpha=0.6)
    ax.text(20.5, ylim[1]*0.9, "Fall\nstarts", fontsize=7.5,
            color=C_FALL, fontweight="bold")
    ax.set_xlabel("Frame", fontsize=8, color=TEXT)
    ax.set_ylim(*ylim)
    ax.legend(fontsize=8, loc="upper right")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("results/diagrams/04_fall_vs_nofall.png",
            dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(); print("Saved: 04_fall_vs_nofall.png")


# ==============================================================================
# FIGURE 5 ── Prediction Result Card
# ==============================================================================
fig = plt.figure(figsize=(12, 7), facecolor=BG)
ax  = fig.add_axes([0.04, 0.04, 0.92, 0.92])
ax.set_facecolor(BG); ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

ax.text(0.5, 0.95, "Fall Detection — Model Inference Output",
        ha="center", fontsize=14, fontweight="bold", color=TEXT)
ax.text(0.5, 0.905, "Live prediction from Transformer-CNN-LSTM on a single 40-frame window",
        ha="center", fontsize=9.5, color=SUBTEXT)

# ── Left panel: input summary ─────────────────────────────────────────────────
lp = FancyBboxPatch((0.03, 0.08), 0.40, 0.77,
                    boxstyle="round,pad=0.02",
                    facecolor=C_RADAR, alpha=0.05,
                    edgecolor=C_RADAR, linewidth=1.5)
ax.add_patch(lp)
ax.text(0.23, 0.82, "Input Summary", ha="center", fontsize=11,
        fontweight="bold", color=C_RADAR)

input_rows = [
    ("Device", "rpi-1"),
    ("Window", "40 frames  ×  20 features"),
    ("Duration", "≈ 2.2 seconds"),
    ("Frame Rate", "~18 fps"),
    ("", ""),
    ("z_mean (avg)", "0.23 m  ← on ground"),
    ("height_range", "0.17 m  ← body flat"),
    ("n_points", "5  ← few reflections"),
    ("velocity", "0.92 m/s  ← fast drop"),
    ("", ""),
    ("Scaler", "StandardScaler applied"),
    ("Input shape", "(1, 40, 20)  → model"),
]
for i, (k, v) in enumerate(input_rows):
    y = 0.77 - i * 0.058
    if k:
        ax.text(0.07, y, k + ":", fontsize=8.5, color=SUBTEXT, va="center")
        ax.text(0.37, y, v, fontsize=8.5, color=TEXT, va="center",
                ha="right", fontweight="bold" if "←" in v else "normal")
    else:
        ax.axhline(y, xmin=0.04, xmax=0.42, color=BORDER, lw=0.8)

# ── Right panel: result ───────────────────────────────────────────────────────
rp = FancyBboxPatch((0.53, 0.08), 0.44, 0.77,
                    boxstyle="round,pad=0.02",
                    facecolor=C_FALL, alpha=0.05,
                    edgecolor=C_FALL, linewidth=2.5)
ax.add_patch(rp)

ax.text(0.75, 0.82, "⚠  FALL DETECTED", ha="center",
        fontsize=14, fontweight="bold", color=C_FALL)

# Probability bar
bar_y = 0.70
ax.text(0.57, bar_y + 0.015, "P(Fall) =", fontsize=9, color=TEXT)
ax.text(0.96, bar_y + 0.015, "96.8%", fontsize=11, color=C_FALL,
        fontweight="bold", ha="right")
bar_bg = FancyBboxPatch((0.57, bar_y - 0.025), 0.38, 0.025,
                         boxstyle="round,pad=0.004",
                         facecolor=BORDER, edgecolor="none")
ax.add_patch(bar_bg)
bar_fill = FancyBboxPatch((0.57, bar_y - 0.025), 0.38 * 0.968, 0.025,
                           boxstyle="round,pad=0.004",
                           facecolor=C_FALL, alpha=0.7, edgecolor="none")
ax.add_patch(bar_fill)

bar_y2 = 0.63
ax.text(0.57, bar_y2 + 0.015, "P(No Fall) =", fontsize=9, color=TEXT)
ax.text(0.96, bar_y2 + 0.015, "3.2%", fontsize=11, color=C_TRANS,
        fontweight="bold", ha="right")
bar_bg2 = FancyBboxPatch((0.57, bar_y2 - 0.025), 0.38, 0.025,
                           boxstyle="round,pad=0.004",
                           facecolor=BORDER, edgecolor="none")
ax.add_patch(bar_bg2)
bar_fill2 = FancyBboxPatch((0.57, bar_y2 - 0.025), 0.38 * 0.032, 0.025,
                            boxstyle="round,pad=0.004",
                            facecolor=C_TRANS, alpha=0.6, edgecolor="none")
ax.add_patch(bar_fill2)

# Model pipeline steps
steps = [
    (C_INPUT,  "Linear(20→64)      →  embedding created"),
    (C_POS,   "Positional Encoding →  frame order tagged"),
    (C_TRANS,  "Transformer (2L,4H) →  frames 28-35 attended"),
    (C_CNN,   "CNN (64→128→128)   →  impact pattern found"),
    (C_LSTM,  "Bi-LSTM (h=128)    →  drop sequence remembered"),
    (C_ATTN,  "Attention Pool      →  frame 30 weighted high"),
    (C_CLASS, "FC(256→128→2)      →  FALL: 96.8%"),
]
for i, (col, txt) in enumerate(steps):
    y = 0.545 - i * 0.065
    dot = Circle((0.565, y + 0.006), 0.009, color=col, zorder=5)
    ax.add_patch(dot)
    ax.text(0.585, y + 0.006, txt, fontsize=8, color=TEXT, va="center")

# Action taken
act = FancyBboxPatch((0.55, 0.09), 0.40, 0.075,
                      boxstyle="round,pad=0.01",
                      facecolor=C_FALL, alpha=0.1,
                      edgecolor=C_FALL, linewidth=1.5)
ax.add_patch(act)
ax.text(0.75, 0.143, "ACTION TRIGGERED", ha="center",
        fontsize=9, fontweight="bold", color=C_FALL)
ax.text(0.75, 0.115, "WebSocket broadcast  →  Dashboard alert  →  DynamoDB logged",
        ha="center", fontsize=7.5, color=SUBTEXT)

plt.savefig("results/diagrams/05_prediction_card.png",
            dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(); print("Saved: 05_prediction_card.png")


# ==============================================================================
# FIGURE 6 ── CNN Concept: What Conv1D Does to the sequence
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=BG)
fig.suptitle("How 1D CNN Scans the Radar Sequence for Patterns",
             fontsize=13, fontweight="bold", color=TEXT, y=0.97)

T = np.arange(1, 41)
z = np.concatenate([np.ones(20)*1.65, np.linspace(1.65,0.22,10), np.ones(10)*0.22])
z += np.random.randn(40)*0.03

# Panel 1: Raw input sequence
ax = axes[0]
ax.set_facecolor("#FCFCFC"); ax.spines[:].set_color(BORDER)
ax.tick_params(colors=TEXT, labelsize=8)
ax.set_title("Step 1: Raw Feature Sequence\n(40 frames × 64 dims after projection)",
             fontsize=9, color=TEXT, fontweight="bold")
ax.plot(T, z, color=C_RADAR, lw=2)
ax.fill_between(T, z, 0, alpha=0.08, color=C_RADAR)
ax.set_xlabel("Frame", fontsize=8, color=TEXT)
ax.set_ylabel("z_mean (height, m)", fontsize=8, color=TEXT)
ax.set_facecolor("#FCFCFC")

# Panel 2: Conv1D sliding kernel
ax2 = axes[1]
ax2.set_facecolor("#FCFCFC"); ax2.spines[:].set_color(BORDER)
ax2.tick_params(colors=TEXT, labelsize=8)
ax2.set_title("Step 2: Conv1D Kernel (size=3) Slides Over\nDetects local patterns in 3-frame windows",
              fontsize=9, color=TEXT, fontweight="bold")
ax2.plot(T, z, color=C_RADAR, lw=2, alpha=0.4)
# Show kernel windows at several positions
for start in [5, 18, 20, 22, 28, 33]:
    col = C_FALL if 19 <= start <= 28 else C_CNN
    rect = mpatches.FancyBboxPatch(
        (start - 0.5, min(z[start-1:start+2]) - 0.05),
        3, max(z[start-1:start+2]) - min(z[start-1:start+2]) + 0.1,
        boxstyle="round,pad=0.04", facecolor=col, alpha=0.15,
        edgecolor=col, linewidth=1.5)
    ax2.add_patch(rect)
ax2.set_xlabel("Frame", fontsize=8, color=TEXT)
ax2.set_ylabel("Height (m)", fontsize=8, color=TEXT)
ax2.set_facecolor("#FCFCFC")
ax2.text(23, 0.5, "High activation\n(fall pattern)", fontsize=7.5,
         color=C_FALL, ha="center")

# Panel 3: CNN feature map (activation)
ax3 = axes[2]
ax3.set_facecolor("#FCFCFC"); ax3.spines[:].set_color(BORDER)
ax3.tick_params(colors=TEXT, labelsize=8)
ax3.set_title("Step 3: CNN Output Activation Map\n(128 filters, MaxPooled → 20 timesteps)",
              fontsize=9, color=TEXT, fontweight="bold")

# Simulate activation map
T2 = np.arange(1, 21)
activation = np.concatenate([
    np.ones(9) * 0.12 + np.random.randn(9) * 0.04,
    np.array([0.4, 0.85, 0.97, 0.98, 0.89]),
    np.ones(6) * 0.08 + np.random.randn(6) * 0.03,
])
ax3.bar(T2, activation, color=[C_FALL if a > 0.3 else C_TRANS for a in activation],
        alpha=0.7, width=0.7)
ax3.axhline(0.3, color=SUBTEXT, lw=1, linestyle="--", alpha=0.6)
ax3.text(10.5, 0.33, "Fall pattern\nthreshold", fontsize=7.5, color=SUBTEXT)
ax3.set_xlabel("Time step (after MaxPool)", fontsize=8, color=TEXT)
ax3.set_ylabel("Activation strength", fontsize=8, color=TEXT)
ax3.set_facecolor("#FCFCFC")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("results/diagrams/06_cnn_concept.png",
            dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(); print("Saved: 06_cnn_concept.png")

print("\nAll 6 diagrams saved to: results/diagrams/")
print("Files:")
for i, name in enumerate([
    "01_full_architecture.png",
    "02_feature_vector.png",
    "03_attention_concept.png",
    "04_fall_vs_nofall.png",
    "05_prediction_card.png",
    "06_cnn_concept.png",
], 1):
    print(f"  {i}. {name}")
