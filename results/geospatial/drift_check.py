"""Is opensr-test's 'spatial' drift a real shift or phase-correlation noise from added detail?

For every full 4x4 chip of the 7 in-season test places:
  A. signed shift (dy, dx) between opensr-test's own lr_RGB and sr_to_lr_RGB (same PCC, upsample 50)
  B. the same after a Gaussian blur of both (sigma 1.5 LR px): a real translation survives a blur, noise does not
  C. control with zero shift by construction: bicubic(LR) + the ArcGIS reference's high-pass detail
  D. at HR resolution: shift of SR vs the ArcGIS reference, compared with bicubic vs the reference
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "..", "code")      # needs code/data/test (the test set) to run
sys.path.insert(0, CODE)
os.chdir(CODE)
import bootstrap  # noqa
import numpy as np, torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation
from opensr_test.main import Metrics
import ps_proof as pp
import infer


def our_model(name):
    """Our two models from ../../weights; bicubic from evaluate.py."""
    if name == "bicubic_8date":
        return pp.ours(name)
    net = infer.load(name, os.path.join("..", "weights", f"{name}_generator.pth"), torch.device("cuda"))
    return net

dev = torch.device("cuda")
metric = Metrics(device="cuda")
tiles_by_place = pp.test_places("inseason")
models = {m: our_model(m) for m in ["bicubic_8date", "arcgis_B", "s2colour"]}


def pcc(a, b):
    s, _, _ = phase_cross_correlation(a, b, upsample_factor=50)
    return s  # (dy, dx)


def gray(x):
    return x.mean(0)


rows = []
for place, tiles in tiles_by_place.items():
    for cx, cy, members in pp.chips(tiles):
        if len(members) != 16:
            continue
        ids = list(members.values())

        def mosaic(get, px):
            m = np.zeros((4 * px, 4 * px, 3), np.float32)
            for (i, j), t in members.items():
                m[j * px:(j + 1) * px, i * px:(i + 1) * px] = get(t)
            return m.transpose(2, 0, 1) / 255

        lr = mosaic(lambda t: np.median(tiles[t]["lr8"].view(8, 3, 32, 32).numpy(), 0).transpose(1, 2, 0), 32)
        hr = mosaic(lambda t: tiles[t]["hr"], 128)
        # control C: bicubic of the LR + high-pass of the reference (detail with no shift of the low frequencies)
        up = F.interpolate(torch.from_numpy(lr)[None], scale_factor=4, mode="bicubic", align_corners=False)[0].numpy()
        hp = hr - np.stack([gaussian_filter(c, 4) for c in hr])
        srs = {"control_bicubic+ref_detail": np.clip(up + hp, 0, 1)}
        with torch.no_grad():
            x = torch.stack([tiles[t]["lr8"] for t in ids]).to(dev).float() / 255
            for m, fn in models.items():
                out = fn(x).clamp(0, 1).cpu().numpy()          # [16, 3, 128, 128]
                srs[m] = mosaic(lambda t: out[ids.index(t)].transpose(1, 2, 0) * 255, 128)
        for m, sr in srs.items():
            with torch.no_grad():
                r = metric.compute(lr=torch.from_numpy(lr).float(), sr=torch.from_numpy(sr).float(),
                                   hr=torch.from_numpy(hr).float())
            a = gray(metric.lr_RGB.cpu().numpy())
            b = gray(metric.sr_to_lr_RGB.cpu().numpy())
            s_raw = pcc(a, b)
            s_blur = pcc(gaussian_filter(a, 1.5), gaussian_filter(b, 1.5))
            s_hr = pcc(gray(hr), gray(sr))                     # in HR px
            rows.append((place, m, float(r["spatial"]), *s_raw, *s_blur, *(s_hr / 4)))
    print(place, len(rows), flush=True)

import collections
by = collections.defaultdict(list)
for r in rows:
    by[r[1]].append(r)
print(f"\n{'model':30s} n   opensr|  signed raw dy,dx (mean±sd)   |raw| | blurred mean dy,dx  |blur| | vs ArcGIS HR (LR px) mean dy,dx  |.|")
for m, rs in by.items():
    a = np.array([r[2:] for r in rs], float)
    sp, rdy, rdx, bdy, bdx, hdy, hdx = a.T
    print(f"{m:30s} {len(rs):3d} {np.nanmean(sp):.3f} | {rdy.mean():+.3f}±{rdy.std():.3f} {rdx.mean():+.3f}±{rdx.std():.3f}  "
          f"{np.hypot(rdy, rdx).mean():.3f} | {bdy.mean():+.3f} {bdx.mean():+.3f}  {np.hypot(bdy, bdx).mean():.3f} | "
          f"{hdy.mean():+.3f} {hdx.mean():+.3f}  {np.hypot(hdy, hdx).mean():.3f}")
import csv
with open(os.path.join(HERE, "drift_check.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["place", "model", "opensr_spatial", "raw_dy", "raw_dx", "blur_dy", "blur_dx", "vs_ref_dy", "vs_ref_dx"])
    for r in rows:
        w.writerow([r[0], r[1]] + [round(float(v), 4) for v in r[2:]])
