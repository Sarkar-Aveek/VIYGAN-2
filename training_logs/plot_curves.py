"""Plot every training run's losses and validation metrics from its losses.csv / val.csv.

    python plot_curves.py            -> <run>/curves.png for every run folder next to this script
"""
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOSSES = [("l_g_pix", "pixel L1 (vs sharpened target)"), ("l_g_percep", "perceptual (VGG19 features)"),
          ("l_g_grad", "edge loss (L1 on Sobel gradients)"), ("l_g_gan", "generator GAN loss"),
          ("l_d_real", "discriminator loss, real"), ("l_d_fake", "discriminator loss, fake")]
VAL = [("lpips", "LPIPS (lower = better)"), ("edge_f1", "edge-F1 (higher = better)"),
       ("ssim", "SSIM (higher = better)"), ("cpsnr", "cPSNR dB (higher = better)")]


def smooth(y, k=9):
    if len(y) < k:
        return y
    pad = np.pad(y, (k // 2, k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


for run in sorted(d for d in glob.glob(os.path.join(HERE, "*")) + glob.glob(os.path.join(HERE, "*", "stage*")) if os.path.exists(os.path.join(d, "losses.csv"))):
    name = os.path.basename(run)
    rows = list(csv.DictReader(open(os.path.join(run, "losses.csv"))))
    val = list(csv.DictReader(open(os.path.join(run, "val.csv"))))
    it = np.array([float(r["iter"]) for r in rows])
    present = [(k, t) for k, t in LOSSES if k in rows[0] and rows[0][k] not in ("", None)]
    fig, axes = plt.subplots(2, 6, figsize=(26, 7.5))
    for ax, (k, title) in zip(axes[0], present):
        y = np.array([float(r[k]) for r in rows])
        ax.plot(it, y, color="#bbbbbb", lw=0.8, label="logged (every 200 it)")
        ax.plot(it, smooth(y), color="#1f6f3f", lw=2, label="moving mean")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    for ax in axes[0][len(present):]:
        ax.axis("off")
    for ax, (k, title) in zip(axes[1], VAL):
        x = [float(r["iter"]) for r in val]
        y = [float(r[k]) for r in val]
        ax.plot(x, y, "o-", color="#3a64a8", lw=2)
        for a, b in zip(x, y):
            ax.annotate(f"{b:.3f}", (a, b), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        ax.set_title("validation: " + title, fontsize=10)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
    lr = [float(r["lr"]) for r in rows]
    axes[1][4].plot(it, lr, color="#c46a1c", lw=2)
    axes[1][4].set_title("generator learning rate", fontsize=10)
    axes[1][4].set_xlabel("iteration")
    axes[1][4].grid(alpha=0.3)
    axes[1][5].axis("off")
    fig.suptitle(f"{name}: training losses (top) and validation on 50 held-out tiles (bottom)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(run, "curves.png"), dpi=100)
    plt.close(fig)
    print("wrote", os.path.join(name, "curves.png"))
