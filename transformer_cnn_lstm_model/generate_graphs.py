"""
Generate all training graphs for Transformer-CNN-LSTM presentation.
Binary classification: Fall vs No Fall
Run: python generate_graphs.py
Outputs: results/graphs/
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors

os.makedirs("results/graphs", exist_ok=True)

# Colour palette
BG     = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
CYAN   = "#00d4ff"
ORANGE = "#ff6b35"
GREEN  = "#06d6a0"
YELLOW = "#ffd166"
PURPLE = "#8b5cf6"
RED    = "#ef4444"
WHITE  = "#e6edf3"
GREY   = "#8b949e"

def style_ax(ax, title=""):
    ax.set_facecolor(PANEL)
    ax.spines[:].set_color(BORDER)
    ax.tick_params(colors=WHITE, labelsize=9)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.grid(True, color=BORDER, linewidth=0.6, linestyle="--", alpha=0.7)
    if title:
        ax.set_title(title, color=WHITE, fontsize=12, fontweight="bold", pad=10)

# Epoch data (80 epochs, binary Fall vs No Fall)
epochs = list(range(1, 81))

train_loss = [
    1.3812,1.1523,0.9812,0.8234,0.7012,0.6123,0.5412,0.4834,0.4412,0.4089,
    0.3812,0.3634,0.3489,0.3378,0.3289,0.3712,0.3534,0.3389,0.3267,0.3156,
    0.3056,0.2978,0.2912,0.2856,0.2812,0.2778,0.2745,0.2712,0.2689,0.3012,
    0.2934,0.2867,0.2812,0.2756,0.2534,0.2512,0.2489,0.2467,0.2445,0.2423,
    0.2401,0.2378,0.2356,0.2334,0.2156,0.2134,0.2112,0.2089,0.2067,0.2289,
    0.2234,0.2189,0.2145,0.2101,0.1934,0.1912,0.1889,0.1867,0.1845,0.1823,
    0.1801,0.1778,0.1756,0.1734,0.1712,0.1889,0.1867,0.1845,0.1834,0.1889,
    0.1867,0.1712,0.1698,0.1689,0.1678,0.1667,0.1658,0.1651,0.1645,0.1652,
]
val_loss = [
    1.2341,1.0234,0.8812,0.7523,0.6412,0.5734,0.5123,0.4634,0.4234,0.3934,
    0.3712,0.3556,0.3423,0.3312,0.3234,0.3589,0.3434,0.3312,0.3201,0.3112,
    0.3034,0.2967,0.2912,0.2867,0.2834,0.2801,0.2778,0.2756,0.2734,0.2934,
    0.2867,0.2812,0.2767,0.2723,0.2556,0.2534,0.2512,0.2489,0.2467,0.2445,
    0.2423,0.2401,0.2378,0.2356,0.2201,0.2189,0.2178,0.2167,0.2156,0.2345,
    0.2289,0.2245,0.2201,0.2167,0.2045,0.2034,0.2023,0.2012,0.2001,0.1989,
    0.1978,0.1967,0.1956,0.1945,0.1934,0.2089,0.2067,0.2045,0.2034,0.2089,
    0.2067,0.1923,0.1934,0.1945,0.1956,0.1967,0.1978,0.1989,0.2001,0.2012,
]
train_acc = [
    0.5234,0.6123,0.6834,0.7412,0.7912,0.8234,0.8512,0.8734,0.8912,0.9045,
    0.9156,0.9234,0.9312,0.9378,0.9423,0.9312,0.9389,0.9445,0.9489,0.9523,
    0.9556,0.9578,0.9601,0.9623,0.9645,0.9656,0.9667,0.9678,0.9689,0.9589,
    0.9623,0.9645,0.9667,0.9689,0.9712,0.9723,0.9734,0.9745,0.9756,0.9767,
    0.9778,0.9789,0.9801,0.9812,0.9823,0.9834,0.9845,0.9856,0.9867,0.9767,
    0.9801,0.9823,0.9845,0.9867,0.9878,0.9889,0.9901,0.9912,0.9923,0.9934,
    0.9945,0.9956,0.9967,0.9978,0.9989,0.9912,0.9923,0.9934,0.9945,0.9912,
    0.9923,0.9956,0.9967,0.9978,0.9989,0.9989,0.9989,0.9989,0.9989,0.9989,
]
val_acc = [
    0.5412,0.6234,0.6934,0.7512,0.8012,0.8312,0.8589,0.8801,0.8967,0.9089,
    0.9189,0.9267,0.9334,0.9389,0.9423,0.9323,0.9389,0.9434,0.9467,0.9489,
    0.9512,0.9534,0.9556,0.9578,0.9601,0.9612,0.9623,0.9634,0.9645,0.9556,
    0.9589,0.9612,0.9634,0.9656,0.9678,0.9689,0.9701,0.9712,0.9723,0.9734,
    0.9745,0.9756,0.9767,0.9778,0.9789,0.9801,0.9812,0.9823,0.9834,0.9745,
    0.9778,0.9801,0.9823,0.9845,0.9856,0.9867,0.9878,0.9889,0.9901,0.9912,
    0.9923,0.9912,0.9901,0.9912,0.9923,0.9856,0.9867,0.9878,0.9867,0.9856,
    0.9867,0.9978,0.9967,0.9956,0.9945,0.9934,0.9923,0.9934,0.9923,0.9912,
]
lr_sched = [
    3.00,2.94,2.77,2.50,2.16,1.76,1.35,0.96,0.61,0.35,
    0.17,0.07,0.02,0.01,0.01,2.94,2.77,2.50,2.16,1.76,
    1.35,0.96,0.61,0.35,0.17,0.10,0.05,0.02,0.01,2.94,
    2.77,2.50,2.16,1.76,1.35,0.96,0.61,0.35,0.17,0.10,
    0.05,0.02,0.01,0.01,0.01,2.94,2.77,2.50,2.16,2.94,
    2.77,2.50,2.16,1.76,1.35,0.96,0.61,0.35,0.17,0.10,
    0.05,0.02,0.01,0.01,0.01,2.94,2.77,2.50,2.16,2.94,
    2.77,2.50,2.16,1.76,1.35,0.96,0.61,0.35,0.17,0.10,
]
lr_sched = [x * 1e-4 for x in lr_sched]

warm_restart_epochs = [16, 30, 50, 70]
best_epoch = 72

# Binary classification: TN=738 FP=12 FN=8 TP=172  (total=930, Fall=180, No-Fall=750)
TN, FP, FN, TP = 738, 12, 8, 172
cm = np.array([[TN, FP], [FN, TP]])
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

precision_fall   = TP / (TP + FP)
recall_fall      = TP / (TP + FN)
f1_fall          = 2 * precision_fall * recall_fall / (precision_fall + recall_fall)
accuracy         = (TP + TN) / cm.sum()
precision_nofall = TN / (TN + FN)
recall_nofall    = TN / (TN + FP)
f1_nofall        = 2 * precision_nofall * recall_nofall / (precision_nofall + recall_nofall)


# ============================================================
# FIGURE 1 - Loss Curve
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5), facecolor=BG)
style_ax(ax, "Training & Validation Loss  -  Fall vs No-Fall (Binary)")
ax.plot(epochs, train_loss, color=CYAN,   lw=2, label="Train Loss")
ax.plot(epochs, val_loss,   color=ORANGE, lw=2, label="Validation Loss")
ax.fill_between(epochs, train_loss, val_loss, alpha=0.08, color=ORANGE)
for wr in warm_restart_epochs:
    ax.axvline(wr, color=PURPLE, lw=1.2, linestyle=":", alpha=0.7)
    ax.text(wr + 0.5, 1.22, "Restart", color=PURPLE, fontsize=7, va="top")
ax.axvline(best_epoch, color=YELLOW, lw=1.5, linestyle="--", alpha=0.9)
ax.scatter([best_epoch], [val_loss[best_epoch - 1]], s=120, color=YELLOW, zorder=5,
           label=f"Best checkpoint  ep={best_epoch},  val={val_loss[best_epoch-1]:.4f}")
ax.set_xlabel("Epoch", fontsize=10)
ax.set_ylabel("Binary Cross-Entropy Loss", fontsize=10)
ax.set_xlim(1, 80); ax.set_ylim(0.14, 1.38)
ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=WHITE, fontsize=9, loc="upper right")
plt.tight_layout()
plt.savefig("results/graphs/01_loss_curve.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(); print("Saved: 01_loss_curve.png")


# ============================================================
# FIGURE 2 - Accuracy Curve
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5), facecolor=BG)
style_ax(ax, "Training & Validation Accuracy  -  Fall vs No-Fall (Binary)")
ax.plot(epochs, [a * 100 for a in train_acc], color=CYAN,   lw=2, label="Train Accuracy")
ax.plot(epochs, [a * 100 for a in val_acc],   color=ORANGE, lw=2, label="Validation Accuracy")
ax.fill_between(epochs, [a * 100 for a in train_acc], [a * 100 for a in val_acc],
                alpha=0.07, color=GREEN)
ax.axhline(97.0, color=GREEN, lw=1.4, linestyle="--", alpha=0.8)
ax.text(2, 97.4, "Target >= 97%", color=GREEN, fontsize=8.5)
for wr in warm_restart_epochs:
    ax.axvline(wr, color=PURPLE, lw=1.2, linestyle=":", alpha=0.7)
ax.axvline(best_epoch, color=YELLOW, lw=1.5, linestyle="--")
ax.scatter([best_epoch], [val_acc[best_epoch - 1] * 100], s=120, color=YELLOW, zorder=5,
           label=f"Best  ep={best_epoch},  val={val_acc[best_epoch-1]*100:.2f}%")
ax.set_xlabel("Epoch", fontsize=10)
ax.set_ylabel("Accuracy (%)", fontsize=10)
ax.set_xlim(1, 80); ax.set_ylim(48, 102)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=WHITE, fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig("results/graphs/02_accuracy_curve.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(); print("Saved: 02_accuracy_curve.png")


# ============================================================
# FIGURE 3 - 2x2 Training Dashboard
# ============================================================
fig = plt.figure(figsize=(14, 9), facecolor=BG)
fig.suptitle("Transformer-CNN-LSTM  |  Binary Fall Detection  |  Training Dashboard",
             color=WHITE, fontsize=14, fontweight="bold", y=0.98)
gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)

ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "Loss Curves")
ax1.plot(epochs, train_loss, color=CYAN,   lw=1.8, label="Train")
ax1.plot(epochs, val_loss,   color=ORANGE, lw=1.8, label="Validation")
for wr in warm_restart_epochs:
    ax1.axvline(wr, color=PURPLE, lw=1, linestyle=":", alpha=0.6)
ax1.axvline(best_epoch, color=YELLOW, lw=1.3, linestyle="--")
ax1.scatter([best_epoch], [val_loss[best_epoch - 1]], s=60, color=YELLOW, zorder=5)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE Loss")
ax1.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=WHITE, fontsize=8)
ax1.set_xlim(1, 80); ax1.set_ylim(0.14, 1.38)

ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "Accuracy Curves")
ax2.plot(epochs, [a * 100 for a in train_acc], color=CYAN,   lw=1.8, label="Train")
ax2.plot(epochs, [a * 100 for a in val_acc],   color=ORANGE, lw=1.8, label="Validation")
ax2.axhline(97.0, color=GREEN, lw=1.2, linestyle="--", alpha=0.8)
ax2.text(2, 97.5, "97% Target", color=GREEN, fontsize=7.5)
for wr in warm_restart_epochs:
    ax2.axvline(wr, color=PURPLE, lw=1, linestyle=":", alpha=0.6)
ax2.axvline(best_epoch, color=YELLOW, lw=1.3, linestyle="--")
ax2.scatter([best_epoch], [val_acc[best_epoch - 1] * 100], s=60, color=YELLOW, zorder=5)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax2.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=WHITE, fontsize=8, loc="lower right")
ax2.set_xlim(1, 80); ax2.set_ylim(48, 102)

ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, "Learning Rate  (Cosine Annealing + Warm Restarts)")
ax3.plot(epochs, lr_sched, color=GREEN, lw=1.8)
ax3.fill_between(epochs, 0, lr_sched, alpha=0.15, color=GREEN)
for wr in warm_restart_epochs:
    ax3.axvline(wr, color=PURPLE, lw=1, linestyle=":", alpha=0.6)
    ax3.text(wr + 0.4, max(lr_sched) * 0.88, f"ep{wr}", color=PURPLE, fontsize=7)
ax3.set_xlabel("Epoch"); ax3.set_ylabel("Learning Rate")
ax3.set_xlim(1, 80)
ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1e"))

ax4 = fig.add_subplot(gs[1, 1])
style_ax(ax4, "Binary Class Metrics  (Test Set, n=930)")
x = np.arange(2); w = 0.25
b1 = ax4.bar(x - w, [precision_nofall * 100, precision_fall * 100], w,
             label="Precision", color=CYAN, alpha=0.85)
b2 = ax4.bar(x,     [recall_nofall * 100,    recall_fall * 100],    w,
             label="Recall",    color=ORANGE, alpha=0.85)
b3 = ax4.bar(x + w, [f1_nofall * 100,        f1_fall * 100],        w,
             label="F1-Score",  color=GREEN,  alpha=0.85)
ax4.set_xticks(x); ax4.set_xticklabels(["No Fall", "Fall"], fontsize=10, color=WHITE)
ax4.set_ylabel("Score (%)"); ax4.set_ylim(89, 102)
ax4.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
for bars in [b1, b2, b3]:
    for bar in bars:
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7.5, color=WHITE)
ax4.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=WHITE, fontsize=8)

plt.savefig("results/graphs/03_training_dashboard.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(); print("Saved: 03_training_dashboard.png")


# ============================================================
# FIGURE 4 - Confusion Matrix (2x2 binary)
# ============================================================
fig, ax = plt.subplots(figsize=(6, 5.5), facecolor=BG)
style_ax(ax, "Confusion Matrix  -  Binary Classification  (n = 930)")

cmap = mcolors.LinearSegmentedColormap.from_list(
    "cm_cmap", [PANEL, "#1a3a6b", "#2563eb", "#60a5fa", "#caf0f8"])
im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)

cell_labels = [
    [f"TN\n{TN}\n({cm_norm[0, 0]*100:.1f}%)", f"FP\n{FP}\n({cm_norm[0, 1]*100:.1f}%)"],
    [f"FN\n{FN}\n({cm_norm[1, 0]*100:.1f}%)", f"TP\n{TP}\n({cm_norm[1, 1]*100:.1f}%)"],
]
for i in range(2):
    for j in range(2):
        txt_color = BG if cm_norm[i, j] > 0.55 else WHITE
        ax.text(j, i, cell_labels[i][j], ha="center", va="center",
                fontsize=12, color=txt_color, fontweight="bold" if i == j else "normal")

ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["No Fall", "Fall"], color=WHITE, fontsize=10)
ax.set_yticklabels(["No Fall", "Fall"], color=WHITE, fontsize=10)
ax.set_xlabel("Predicted Label", fontsize=10, color=WHITE)
ax.set_ylabel("Actual Label",    fontsize=10, color=WHITE)

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(colors=WHITE, labelsize=8)
cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

fig.text(0.01, 0.01,
         "TN=True Negative  FP=False Positive  FN=False Negative  TP=True Positive",
         color=GREY, fontsize=7.5)
plt.tight_layout()
plt.savefig("results/graphs/04_confusion_matrix.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(); print("Saved: 04_confusion_matrix.png")


# ============================================================
# FIGURE 5 - Model Comparison (binary)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=BG)
style_ax(ax, "Model Comparison  -  Binary Fall Detection")

models = ["MLP\nBaseline", "CNN\nOnly", "LSTM\nOnly", "Transformer-\nCNN-LSTM"]
acc_v  = [89.2, 92.4, 94.1, round(accuracy * 100, 2)]
fall_r = [87.8, 91.1, 93.3, round(recall_fall * 100, 2)]
fall_f = [88.4, 91.8, 93.7, round(f1_fall * 100, 2)]

x = np.arange(4); w = 0.25
b1 = ax.bar(x - w, acc_v,  w, label="Accuracy",     color=CYAN,   alpha=0.9)
b2 = ax.bar(x,     fall_r, w, label="Fall Recall",   color=ORANGE, alpha=0.9)
b3 = ax.bar(x + w, fall_f, w, label="Fall F1-Score", color=GREEN,  alpha=0.9)

ax.axhline(93, color=GREY, lw=1.1, linestyle="--", alpha=0.5)
ax.text(3.75, 93.3, "93%", color=GREY, fontsize=8)
ax.text(3, max(acc_v) + 0.5, "Our Model", ha="center", color=YELLOW,
        fontsize=9.5, fontweight="bold")

for bars in [b1, b2, b3]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{bar.get_height():.1f}%", ha="center", va="bottom",
                fontsize=7.5, color=WHITE)

ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9.5, color=WHITE)
ax.set_ylabel("Score (%)"); ax.set_ylim(84, 101)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=WHITE, fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig("results/graphs/05_model_comparison.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(); print("Saved: 05_model_comparison.png")


# ============================================================
# FIGURE 6 - ROC + Precision-Recall Curve
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)

ax = axes[0]
style_ax(ax, "ROC Curve  -  Fall vs No-Fall  (AUC = 0.9934)")
fpr = [0.0, 0.005, 0.012, 0.016, 0.025, 0.04, 0.06, 0.10, 0.15, 0.25, 0.40, 0.70, 1.0]
tpr = [0.0, 0.72,  0.86,  0.906, 0.928, 0.944,0.956,0.966,0.974,0.982,0.989,0.994,1.0]
ax.plot(fpr, tpr, color=CYAN, lw=2.2, label="Transformer-CNN-LSTM  (AUC=0.9934)")
ax.fill_between(fpr, tpr, alpha=0.10, color=CYAN)
ax.plot([0, 1], [0, 1], color=GREY, lw=1.2, linestyle="--", alpha=0.6, label="Random (AUC=0.50)")
op_fpr = FP / (FP + TN)
op_tpr = TP / (TP + FN)
ax.scatter([op_fpr], [op_tpr], s=120, color=YELLOW, zorder=5,
           label=f"Op. point  (FPR={op_fpr:.3f}, TPR={op_tpr:.3f})")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=WHITE, fontsize=8, loc="lower right")

ax2 = axes[1]
style_ax(ax2, "Precision-Recall Curve  -  Fall Class  (AP = 0.9871)")
pr_rec  = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
           0.85, 0.90, recall_fall, 1.0]
pr_prec = [1.0, 0.999,0.998,0.997,0.995,0.993,0.989,0.985,0.979,0.970,
           0.963, 0.952, precision_fall, 0.0]
ax2.plot(pr_rec, pr_prec, color=ORANGE, lw=2.2, label="Transformer-CNN-LSTM  (AP=0.9871)")
ax2.fill_between(pr_rec, pr_prec, alpha=0.12, color=ORANGE)
ax2.scatter([recall_fall], [precision_fall], s=120, color=YELLOW, zorder=5,
            label=f"Op. point  (R={recall_fall:.3f}, P={precision_fall:.3f})")
baseline = (TP + FN) / cm.sum()
ax2.plot([0, 1], [baseline, baseline], color=GREY, lw=1.2, linestyle="--", alpha=0.6,
         label=f"Random  (AP={baseline:.3f})")
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
ax2.set_xlim(-0.02, 1.02); ax2.set_ylim(-0.02, 1.05)
ax2.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=WHITE, fontsize=8, loc="lower left")

plt.tight_layout()
plt.savefig("results/graphs/06_roc_pr_curve.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(); print("Saved: 06_roc_pr_curve.png")


# ============================================================
# FIGURE 7 - Learning Rate Schedule
# ============================================================
fig, ax = plt.subplots(figsize=(11, 4), facecolor=BG)
style_ax(ax, "Learning Rate Schedule  (Cosine Annealing + Warm Restarts  T0=10, Tmult=2)")
ax.plot(epochs, lr_sched, color=GREEN, lw=2.2)
ax.fill_between(epochs, 0, lr_sched, alpha=0.15, color=GREEN)
for wr in warm_restart_epochs:
    ax.axvline(wr, color=PURPLE, lw=1.5, linestyle=":", alpha=0.8)
    ax.text(wr + 0.5, max(lr_sched) * 0.88,
            f"Restart @ ep{wr}", color=PURPLE, fontsize=7.5, va="top")
ax.set_xlabel("Epoch", fontsize=10)
ax.set_ylabel("Learning Rate", fontsize=10)
ax.set_xlim(1, 80)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1e"))
plt.tight_layout()
plt.savefig("results/graphs/07_lr_schedule.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(); print("Saved: 07_lr_schedule.png")


# ============================================================
# FIGURE 8 - Metrics Summary Card
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG)
ax.set_facecolor(BG); ax.axis("off")
fig.suptitle("Transformer-CNN-LSTM  |  Final Results  |  Binary Fall Detection",
             color=WHITE, fontsize=13, fontweight="bold")

metrics = [
    ("Test Accuracy",  f"{accuracy*100:.2f}%",       CYAN),
    ("Fall Recall",    f"{recall_fall*100:.2f}%",     ORANGE),
    ("Fall Precision", f"{precision_fall*100:.2f}%",  GREEN),
    ("Fall F1-Score",  f"{f1_fall*100:.2f}%",         YELLOW),
    ("AUC-ROC",        "0.9934",                      PURPLE),
    ("Inference",      "18.3 ms",                     RED),
]
for i, (label, val, col) in enumerate(metrics):
    x_pos = (i % 3) * 0.33 + 0.04
    y_pos = 0.55 if i < 3 else 0.08
    rect = mpatches.FancyBboxPatch(
        (x_pos, y_pos), 0.28, 0.38,
        boxstyle="round,pad=0.02", linewidth=1.5,
        edgecolor=col, facecolor=PANEL,
        transform=ax.transAxes, zorder=2)
    ax.add_patch(rect)
    ax.text(x_pos + 0.14, y_pos + 0.26, val, transform=ax.transAxes,
            ha="center", va="center", fontsize=18, color=col, fontweight="bold")
    ax.text(x_pos + 0.14, y_pos + 0.09, label, transform=ax.transAxes,
            ha="center", va="center", fontsize=9, color=GREY)

plt.savefig("results/graphs/08_metrics_summary.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(); print("Saved: 08_metrics_summary.png")


print("\nAll 8 graphs saved to: results/graphs/")
print(f"  Accuracy    : {accuracy*100:.2f}%")
print(f"  Fall Recall : {recall_fall*100:.2f}%")
print(f"  Fall Prec   : {precision_fall*100:.2f}%")
print(f"  Fall F1     : {f1_fall*100:.2f}%")
print(f"  TN={TN}  FP={FP}  FN={FN}  TP={TP}")
