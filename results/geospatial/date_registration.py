"""How far does each single Sentinel-2 date sit from the 8-date consensus?

For every 4x4-tile chip (128 x 128 px at 10 m) of the 7 test places whose 16 tiles share the same 8 dates, each date's
mosaic is phase-correlated (skimage, 1/50 px) against the per-pixel median of the 8 dates. Both images are blurred
first (sigma 1 px) so that the estimate follows position, not cloud or crop texture.

    python date_registration.py <project data folder>     -> date_registration.csv + printed summary

A model fed one date inherits that date's offset. A model fed 8 dates sees their consensus (the median), which
averages the independent per-date geolocation errors.
"""
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation

HERE = os.path.dirname(os.path.abspath(__file__))
PX_M = 9.55   # metres per Sentinel-2 pixel on the zoom-17 Web-Mercator grid


def main(data):
    rows = []
    for place_dir in sorted(glob.glob(os.path.join(data, "test", "sentinel2_l2a", "*"))):
        place = os.path.basename(place_dir)
        tiles = {}
        for j in glob.glob(os.path.join(place_dir, "*.json")):
            d = json.load(open(j))
            x, y = map(int, d["tile_id"].split("_")[1:])
            tiles[(x, y)] = (d["dates"], j[:-5] + ".png")
        chips = defaultdict(dict)
        for (x, y), v in tiles.items():
            chips[(x // 4 * 4, y // 4 * 4)][(x % 4, y % 4)] = v
        for (cx, cy), members in sorted(chips.items()):
            if len(members) != 16:
                continue
            dates = {tuple(v[0][:8]) for v in members.values()}
            if len(dates) != 1 or len(next(iter(dates))) < 8:
                continue                       # dates not shared across the chip: frames would not be one image
            stack = np.zeros((8, 128, 128), np.float32)
            for (i, j), (_, png) in members.items():
                a = np.asarray(Image.open(png).convert("RGB"), np.float32).mean(2)      # [n*32, 32]
                for k in range(8):
                    stack[k, j * 32:(j + 1) * 32, i * 32:(i + 1) * 32] = a[k * 32:(k + 1) * 32]
            med = gaussian_filter(np.median(stack, 0), 1)
            for k, day in enumerate(next(iter(dates))):
                s, _, _ = phase_cross_correlation(med, gaussian_filter(stack[k], 1), upsample_factor=50)
                rows.append({"place": place, "chip": f"{cx}_{cy}", "date": day, "dy_px": round(float(s[0]), 3),
                             "dx_px": round(float(s[1]), 3), "offset_px": round(float(np.hypot(*s)), 3)})
    with open(os.path.join(HERE, "date_registration.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    off = np.array([r["offset_px"] for r in rows])
    print(f"{len(rows)} date x chip measurements, {len({(r['place'], r['chip']) for r in rows})} chips")
    print(f"single date vs 8-date consensus: mean {off.mean():.3f} px ({off.mean() * PX_M:.2f} m), "
          f"median {np.median(off):.3f} px, 90th pct {np.percentile(off, 90):.3f} px "
          f"({np.percentile(off, 90) * PX_M:.2f} m), max {off.max():.3f} px")
    for p in sorted({r["place"] for r in rows}):
        o = np.array([r["offset_px"] for r in rows if r["place"] == p])
        print(f"  {p:22s} n={len(o):4d} mean {o.mean():.3f} px  90th {np.percentile(o, 90):.3f} px")


if __name__ == "__main__":
    main(sys.argv[1])
