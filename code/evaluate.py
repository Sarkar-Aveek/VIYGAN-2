"""
Scores every model on the held-out test places and draws everything.

Test data (collected by pipeline.py, never trained on): data/test/arcgis/{place}/images/{tile}.png and the 8-frame
L2A stacks in data/test/sentinel2_l2a (dates near the ArcGIS capture) and data/test/sentinel2_l2a_offseason
(dates ~6 months away: unseen conditions).

Models (each only if its weights exist):
    bicubic              median of the 8 frames, bicubic x4
    india20k             the L1C India ESRGAN on L2A input (the starting point; not trained on L2A)
    arcgis_A, arcgis_B   model 1 runs (ArcGIS colours)
    s2_maths             the chosen model 1 + colour_transfer onto the input: Sentinel-2 colours, computed
    s2colour             model 3 (Sentinel-2 colours, trained)

Metrics per tile: cPSNR, PSNR, SSIM, LPIPS (lower is better), edge-F1 and gradient correlation (structure,
colour-blind). Outputs in outputs/eval/:
    tiles.csv            every tile x model x test set
    summary.csv          means per model x test set x (all | kind | place)
    metrics.png          each metric per model, in-season vs off-season
    tradeoff.png         cPSNR vs LPIPS (the pixel / perceptual trade-off) per model
    by_kind.png          edge-F1 and LPIPS per land-cover kind, against india20k (the bias check)
    curves_<run>.png     training losses and validation metrics over iterations (from experiments/<run>/*.csv)
    sheets/<place>_<set>.png   each test place as a mosaic: target, input, every model

    python evaluate.py [--models ...] [--sets inseason offseason] [--no-sheets]
"""
import os
import sys
import csv
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import bootstrap  # noqa: F401

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import esrgan  # noqa: F401
from basicsr.metrics import calculate_psnr, calculate_ssim
from collectors import arcgis_collector as agc
from esrgan.dataset import pick_frames, split_frames
from esrgan.metrics import calculate_cpsnr, calculate_edge_f1, calculate_grad_corr, calculate_lpips
from esrgan.networks import SSR_RRDBNet, colour_transfer, frames_median

OUT = os.path.join("outputs", "eval")
TEST_ARCGIS = os.path.join("data", "test", "arcgis")
SETS = {"inseason": os.path.join("data", "test", "sentinel2_l2a"),
        "offseason": os.path.join("data", "test", "sentinel2_l2a_offseason")}
EXP = "experiments"
CHOICE = os.path.join("outputs", "model1_choice.txt")     # written by pipeline.py's pick step
METRICS = ["cpsnr", "psnr", "ssim", "lpips", "edge_f1", "grad_corr"]
LOWER_BETTER = {"lpips"}
BATCH = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def checkpoint(run):
    return os.path.join(EXP, run, "models", "net_g_latest.pth")


def load(path, lowpass=False):
    net = SSR_RRDBNet(24, 3, 4, input_lowpass=lowpass)
    ck = torch.load(path, map_location="cpu", weights_only=True)
    net.load_state_dict(ck["params_ema"] if "params_ema" in ck else ck["params"], strict=True)
    return net.to(device).eval()


def chosen_model1():
    if os.path.exists(CHOICE):
        return open(CHOICE).read().strip()
    return "arcgis_A" if os.path.exists(checkpoint("arcgis_A")) else None


def build_models(names):
    """name -> function(lr batch [b, 24, 32, 32] in 0-1) -> sr batch [b, 3, 128, 128]."""
    models = {}
    bicubic = lambda lr: F.interpolate(frames_median(lr), scale_factor=4, mode="bicubic",  # noqa: E731
                                       align_corners=False)
    candidates = {"bicubic": lambda: bicubic,
                  "india20k": lambda: load("weights/india_finetune_net_g_20000.pth")}
    for run in ("arcgis_A", "arcgis_B"):
        if os.path.exists(checkpoint(run)):
            candidates[run] = (lambda r: lambda: load(checkpoint(r)))(run)
    m1 = chosen_model1()
    if m1 and os.path.exists(checkpoint(m1)):
        def maths():
            net = load(checkpoint(m1))
            return lambda lr: colour_transfer(net(lr), frames_median(lr))
        candidates["s2_maths"] = maths
    if os.path.exists(checkpoint("s2colour")):
        candidates["s2colour"] = lambda: load(checkpoint("s2colour"), lowpass=True)
    for name in names or candidates:
        if name in candidates:
            models[name] = candidates[name]()
        else:
            print(f"skipping {name}: no weights yet")
    return models


def test_tiles(stack_root):
    """[(place, tile_id, x, y, hr uint8 HWC, lr tensor [24, 32, 32] uint8)] for every tile with both files."""
    tiles = []
    for place in agc.TEST:
        img_dir = os.path.join(TEST_ARCGIS, place, "images")
        s2_dir = os.path.join(stack_root, place)
        if not (os.path.isdir(img_dir) and os.path.isdir(s2_dir)):
            continue
        for fn in sorted(os.listdir(s2_dir)):
            if not fn.endswith(".png") or not os.path.exists(os.path.join(img_dir, fn)):
                continue
            hr = cv2.imread(os.path.join(img_dir, fn))[:, :, ::-1]
            s2 = cv2.imread(os.path.join(s2_dir, fn))
            if hr is None or s2 is None or s2.shape != (256, 32, 3) or (hr.sum(axis=-1) == 0).any():
                continue
            frames = split_frames(s2[:, :, ::-1].copy())
            lr = torch.cat([frames[i] for i in pick_frames(frames, 8, rand=False)])
            _, x, y = fn[:-4].split("_")
            tiles.append((place, fn[:-4], int(x), int(y), np.ascontiguousarray(hr), lr))
    return tiles


def cpu_metrics(pair):
    sr, gt = pair
    return {"cpsnr": calculate_cpsnr(sr, gt, crop_border=4),
            "psnr": calculate_psnr(sr, gt, crop_border=4, test_y_channel=False),
            "ssim": calculate_ssim(sr, gt, crop_border=4, test_y_channel=False),
            "edge_f1": calculate_edge_f1(sr, gt), "grad_corr": calculate_grad_corr(sr, gt)}


def run_set(set_name, tiles, models, pool):
    rows, outputs = [], defaultdict(dict)
    for name, fn in models.items():
        print(f"  {set_name}: {name}", flush=True)
        srs = []
        with torch.no_grad():
            for i in range(0, len(tiles), BATCH):
                lr = torch.stack([t[5] for t in tiles[i:i + BATCH]]).to(device).float() / 255
                sr = fn(lr).clamp(0, 1).mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
                srs.extend(sr)
        cpu = list(pool.map(cpu_metrics, [(sr, t[4]) for sr, t in zip(srs, tiles)], chunksize=16))
        for (place, tile_id, x, y, gt, _), sr, m in zip(tiles, srs, cpu):
            m["lpips"] = calculate_lpips(sr, gt)
            rows.append({"set": set_name, "model": name, "place": place, "kind": agc.TEST[place][2],
                         "tile_id": tile_id, **{k: round(float(m[k]), 5) for k in METRICS}})
            outputs[name][tile_id] = sr
    return rows, outputs


def summarise(rows):
    groups = defaultdict(list)
    for r in rows:
        for scope in ("all", f"kind:{r['kind']}", f"place:{r['place']}"):
            groups[(r["set"], r["model"], scope)].append(r)
    out = []
    for (s, m, scope), rs in sorted(groups.items()):
        out.append({"set": s, "model": m, "scope": scope, "tiles": len(rs),
                    **{k: round(float(np.mean([r[k] for r in rs])), 4) for k in METRICS}})
    return out


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_all(summary, model_order):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    allrows = {(r["set"], r["model"]): r for r in summary if r["scope"] == "all"}
    sets = [s for s in SETS if any((s, m) in allrows for m in model_order)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, metric in zip(axes.ravel(), METRICS):
        width = 0.8 / max(len(sets), 1)
        seen = []
        for j, s in enumerate(sets):
            vals = [allrows.get((s, m), {}).get(metric, np.nan) for m in model_order]
            seen += vals
            ax.bar(np.arange(len(model_order)) + j * width, vals, width, label=s)
        lo, hi = np.nanmin(seen), np.nanmax(seen)
        pad = (hi - lo) * 0.25 + 1e-3
        ax.set_ylim(lo - pad, hi + pad)       # zoomed to the data: the differences are what matter
        ax.set_xticks(np.arange(len(model_order)) + width * (len(sets) - 1) / 2)
        ax.set_xticklabels(model_order, rotation=30, ha="right", fontsize=8)
        ax.set_title(metric + (" (lower is better)" if metric in LOWER_BETTER else ""))
        ax.grid(axis="y", alpha=0.3)
    axes[0, 0].legend()
    fig.suptitle("Held-out test places: every metric per model")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "metrics.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for s, marker in zip(sets, "os"):
        for m in model_order:
            r = allrows.get((s, m))
            if r:
                ax.scatter(r["lpips"], r["cpsnr"], marker=marker, s=60)
                ax.annotate(f"{m} ({s})", (r["lpips"], r["cpsnr"]), fontsize=7, xytext=(4, 2),
                            textcoords="offset points")
    ax.set_xlabel("LPIPS (lower = more like the target to a person)")
    ax.set_ylabel("cPSNR dB (higher = closer pixel values)")
    ax.set_title("Pixel accuracy vs perceptual quality")
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "tradeoff.png"), dpi=120)
    plt.close(fig)

    kinds = sorted({r["scope"][5:] for r in summary if r["scope"].startswith("kind:")})
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, metric in zip(axes, ("edge_f1", "lpips")):
        width = 0.8 / max(len(model_order), 1)
        for j, m in enumerate(model_order):
            vals = [next((r[metric] for r in summary if r["set"] == "inseason" and r["model"] == m
                          and r["scope"] == f"kind:{k}"), np.nan) for k in kinds]
            ax.bar(np.arange(len(kinds)) + j * width, vals, width, label=m)
        ax.set_xticks(np.arange(len(kinds)) + width * (len(model_order) - 1) / 2)
        ax.set_xticklabels(kinds)
        ax.set_title(f"{metric} per land-cover kind (in season)" + (" - lower is better" if metric in LOWER_BETTER
                                                                     else ""))
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "by_kind.png"), dpi=120)
    plt.close(fig)


def plot_curves():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for run in sorted(os.listdir(EXP)) if os.path.isdir(EXP) else []:
        files = {k: os.path.join(EXP, run, f"{k}.csv") for k in ("losses", "val")}
        if not any(os.path.exists(p) for p in files.values()):
            continue
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        if os.path.exists(files["losses"]):
            rows = list(csv.DictReader(open(files["losses"])))
            it = [int(r["iter"]) for r in rows]
            for key in [k for k in rows[0] if k.startswith("l_")]:
                vals = np.array([float(r[key]) if r.get(key) else np.nan for r in rows])
                axes[0].plot(it, vals / np.nanmax(np.abs(vals)), label=key)
            axes[0].set_title(f"{run}: losses (each scaled to its max)")
            axes[0].set_xlabel("iteration")
            axes[0].legend(fontsize=7)
            axes[0].grid(alpha=0.3)
        if os.path.exists(files["val"]):
            rows = list(csv.DictReader(open(files["val"])))
            it = [int(r["iter"]) for r in rows]
            for key in [k for k in rows[0] if k not in ("iter", "epoch")]:
                vals = np.array([float(r[key]) for r in rows])
                axes[1].plot(it, (vals - vals[0]) * (-1 if key in LOWER_BETTER else 1), marker="o", ms=3,
                             label=f"{key} ({vals[0]:.3f} -> {vals[-1]:.3f})")
            axes[1].set_title(f"{run}: validation, change since the first check (up = better)")
            axes[1].set_xlabel("iteration")
            axes[1].legend(fontsize=7)
            axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"curves_{run}.png"), dpi=120)
        plt.close(fig)


def sheets(set_name, tiles, outputs, model_order):
    """Per test place: target, input, and each model as one mosaic (tiles at their grid positions)."""
    os.makedirs(os.path.join(OUT, "sheets"), exist_ok=True)
    by_place = defaultdict(list)
    for t in tiles:
        by_place[t[0]].append(t)
    for place, ts in by_place.items():
        x0, y0 = min(t[2] for t in ts), min(t[3] for t in ts)
        w, h = max(t[2] for t in ts) - x0 + 1, max(t[3] for t in ts) - y0 + 1
        panels = {"target": np.full((h * 128, w * 128, 3), 40, np.uint8),
                  "input": np.full((h * 128, w * 128, 3), 40, np.uint8)}
        panels.update({m: np.full((h * 128, w * 128, 3), 40, np.uint8) for m in model_order})
        for place_, tile_id, x, y, gt, lr in ts:
            r, c = (y - y0) * 128, (x - x0) * 128
            panels["target"][r:r + 128, c:c + 128] = gt
            med = np.median(lr.view(8, 3, 32, 32).numpy(), axis=0).transpose(1, 2, 0).astype(np.uint8)
            panels["input"][r:r + 128, c:c + 128] = cv2.resize(med, (128, 128), interpolation=cv2.INTER_NEAREST)
            for m in model_order:
                panels[m][r:r + 128, c:c + 128] = outputs[m][tile_id]
        names = list(panels)
        cols = 4
        scale = min(1.0, 1024 / (w * 128))
        size = (int(w * 128 * scale), int(h * 128 * scale))
        imgs = []
        for n in names:
            img = cv2.resize(panels[n], size, interpolation=cv2.INTER_AREA)
            img = np.ascontiguousarray(img)
            cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
            cv2.putText(img, n, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            imgs.append(img)
        while len(imgs) % cols:
            imgs.append(np.zeros_like(imgs[0]))
        grid = np.vstack([np.hstack(imgs[i:i + cols]) for i in range(0, len(imgs), cols)])
        cv2.imwrite(os.path.join(OUT, "sheets", f"{place}_{set_name}.png"), grid[:, :, ::-1])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", help="subset of models (default: every one with weights)")
    p.add_argument("--sets", nargs="+", choices=list(SETS), default=list(SETS))
    p.add_argument("--no-sheets", action="store_true")
    p.add_argument("--workers", type=int, default=4,
                   help="metric processes; each loads torch (~0.5 GB), so keep it small on 16 GB RAM (default 4)")
    args = p.parse_args()

    os.makedirs(OUT, exist_ok=True)
    plot_curves()
    models = build_models(args.models)
    order = list(models)
    rows = []
    with ProcessPoolExecutor(max(1, args.workers)) as pool:
        for set_name in args.sets:
            tiles = test_tiles(SETS[set_name])
            if not tiles:
                print(f"{set_name}: no test tiles in {SETS[set_name]}; skipped")
                continue
            print(f"{set_name}: {len(tiles)} tiles from {len({t[0] for t in tiles})} places")
            set_rows, outputs = run_set(set_name, tiles, models, pool)
            rows += set_rows
            if not args.no_sheets:
                sheets(set_name, tiles, outputs, order)
    if not rows:
        sys.exit("No test data: run pipeline.py --only test_data first.")
    write_csv(os.path.join(OUT, "tiles.csv"), rows)
    summary = summarise(rows)
    write_csv(os.path.join(OUT, "summary.csv"), summary)
    plot_all(summary, order)

    print(f"\n{'set':10} {'model':10} " + " ".join(f"{k:>9}" for k in METRICS))
    for r in summary:
        if r["scope"] == "all":
            print(f"{r['set']:10} {r['model']:10} " + " ".join(f"{r[k]:9.4f}" for k in METRICS))
    print(f"\nwrote {OUT}/ (tiles.csv, summary.csv, metrics.png, tradeoff.png, by_kind.png, curves_*.png, sheets/)")


if __name__ == "__main__":
    main()
