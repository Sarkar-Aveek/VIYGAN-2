"""Ablation: is model 3 (s2colour) better than model 1 (arcgis_B) with the same colour lock applied afterwards?

Same 8-date input, same test tiles as docs/05:
  s2colour        model 3 as released: trained from arcgis_B with the lock inside, colour-blind losses
  arcgis_B+lock   model 1, then esrgan.networks.lowpass_transfer(out, median of the 8 dates, 4, 3): the identical
                  3-step lock, but the network never trained with it
  arcgis_B        model 1, no lock (sanity check against results/benchmark/metrics/summary.csv)

Tile metrics as ps_proof.tile_metrics + colour-normalised LPIPS; opensr-test on full 4 x 4-tile chips (in season),
as ps_proof.score; lock error = |4 x 4 block mean of the output - 8-date median input| in reflectance. Paired
bootstrap (tiles, or chips for opensr-test) gives 95% intervals of s2colour - (arcgis_B+lock).

Needs code/data/test (python pipeline.py --only test_data).
    python lock_ablation.py   -> results.csv, examples.png in this folder

results.csv: mean of every metric per set and variant (tiles, or chips for opensr-test), then three in-season rows:
s2colour - (arcgis_B+lock) and its paired-bootstrap 95% interval (1,000 resamples over tiles or chips).
"""
import csv
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "..", "code")
sys.path.insert(0, CODE)
os.chdir(CODE)
import bootstrap  # noqa: F401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import infer  # noqa: E402
import ps_proof as pp  # noqa: E402
from esrgan.metrics import calculate_lpips  # noqa: E402
from esrgan.networks import frames_median, lowpass_transfer  # noqa: E402

VARIANTS = ["s2colour", "arcgis_B+lock", "arcgis_B"]
TILE_KEYS = ["psnr", "ssim", "cpsnr", "cn_psnr", "cn_ssim", "cn_lpips", "edge_f1", "grad_corr", "lock_err"]
CHIP_KEYS = ["im_metric", "om_metric", "ha_metric", "reflectance", "spectral"]
dev = torch.device("cuda")


def write_results(tile_rows, chip_rows):
    rows = []
    for set_name in ("inseason", "offseason"):
        for v in VARIANTS:
            t = [r for r in tile_rows if r["set"] == set_name and r["variant"] == v]
            c = [r for r in chip_rows if r["set"] == set_name and r["variant"] == v]
            row = {"set": set_name, "row": v, "tiles": len(t), "chips": len(c)}
            row.update({k: float(np.mean([r[k] for r in t])) for k in TILE_KEYS})
            row.update({k: (float(np.mean([r[k] for r in c])) if c else "") for k in CHIP_KEYS})
            rows.append(row)
    # paired bootstrap: s2colour - (arcgis_B+lock), in season
    rng = np.random.default_rng(0)
    diff = {"set": "inseason", "row": "s2colour - arcgis_B+lock"}
    low = {"set": "inseason", "row": "ci95 low"}
    high = {"set": "inseason", "row": "ci95 high"}
    for src, unit, keys in ((tile_rows, "tile", TILE_KEYS), (chip_rows, "chip", CHIP_KEYS)):
        by = defaultdict(dict)
        for r in src:
            if r["set"] == "inseason":
                by[r[unit]][r["variant"]] = r
        units = [u for u in by if "s2colour" in by[u] and "arcgis_B+lock" in by[u]]
        diff["tiles" if unit == "tile" else "chips"] = len(units)
        for k in keys:
            d = np.array([by[u]["s2colour"][k] - by[u]["arcgis_B+lock"][k] for u in units])
            means = d[rng.integers(0, len(d), (1000, len(d)))].mean(1)
            diff[k], low[k], high[k] = float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    rows += [diff, low, high]
    keys = ["set", "row", "tiles", "chips"] + TILE_KEYS + CHIP_KEYS
    with open(os.path.join(HERE, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows({k: (round(r[k], 5) if isinstance(r.get(k), float) else r.get(k, "")) for k in keys} for r in rows)


def main():
    from opensr_test.main import Metrics
    nets = {m: infer.load(m, os.path.join("..", "weights", f"{m}_generator.pth"), dev) for m in ("arcgis_B", "s2colour")}

    def run(variant, lr):
        if variant == "s2colour":
            return nets["s2colour"](lr)
        out = nets["arcgis_B"](lr)
        return lowpass_transfer(out, frames_median(lr), 4, 3) if variant == "arcgis_B+lock" else out

    metric = Metrics(device="cuda")
    tile_rows, chip_rows, keep = [], [], {}
    for set_name in ("inseason", "offseason"):
        places = pp.test_places(set_name)
        vis = {p: set(pp.pick_visual_tiles(t, 1)) for p, t in places.items()} if set_name == "inseason" else {}
        for place, tiles in places.items():
            print(set_name, place, len(tiles), flush=True)
            for cx, cy, members in pp.chips(tiles):
                ids = list(members.values())
                lr = torch.stack([tiles[t]["lr8"] for t in ids]).to(dev).float() / 255
                med = frames_median(lr)                                           # [b, 3, 32, 32]
                for v in VARIANTS:
                    with torch.no_grad():
                        y = run(v, lr).clamp(0, 1)
                        lock = (F.avg_pool2d(y, 4) - med).abs().mean((1, 2, 3)).mul(pp.TCI).cpu().numpy()
                    srs = y.mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
                    for t, sr, le in zip(ids, srs, lock):
                        gt = tiles[t]["hr"]
                        m = pp.tile_metrics((sr, gt))
                        m["cn_lpips"] = calculate_lpips(pp.colour_match(sr, gt), gt)
                        m["lock_err"] = float(le)
                        tile_rows.append({"set": set_name, "variant": v, "place": place, "tile": t, **m})
                        if t in vis.get(place, ()):
                            keep[(place, v)] = sr
                            keep[(place, "reference")] = gt
                    if set_name == "inseason" and len(members) == pp.CHIP * pp.CHIP:
                        def mosaic(get, px):
                            a = np.zeros((pp.CHIP * px, pp.CHIP * px, 3), np.float32)
                            for (i, j), t in members.items():
                                a[j * px:(j + 1) * px, i * px:(i + 1) * px] = get(t)
                            return torch.from_numpy(a.transpose(2, 0, 1) / 255).float()
                        sr_of = dict(zip(ids, srs))
                        low = mosaic(lambda t: np.median(tiles[t]["lr8"].view(8, 3, 32, 32).numpy(), 0)
                                     .transpose(1, 2, 0), 32)
                        with torch.no_grad():
                            r = metric.compute(lr=low, sr=mosaic(lambda t: sr_of[t], 128),
                                               hr=mosaic(lambda t: tiles[t]["hr"], 128))
                        chip_rows.append({"set": set_name, "variant": v, "place": place, "chip": f"{cx}_{cy}",
                                          **{k: float(r[k]) for k in CHIP_KEYS}})

    write_results(tile_rows, chip_rows)

    cols = ["reference", "s2colour", "arcgis_B+lock", "arcgis_B"]
    places = sorted({p for p, _ in keep})[:4]
    S = 256
    sheet = Image.new("RGB", (len(cols) * (S + 6), len(places) * (S + 6) + 22), "white")
    d = ImageDraw.Draw(sheet)
    for c, name in enumerate(cols):
        d.text((c * (S + 6) + 4, 4), name, fill="black")
    for r, p in enumerate(places):
        for c, v in enumerate(cols):
            sheet.paste(Image.fromarray(keep[(p, v)]).resize((S, S), Image.NEAREST), (c * (S + 6), 22 + r * (S + 6)))
    sheet.save(os.path.join(HERE, "examples.png"))
    print("done")


if __name__ == "__main__":
    main()
