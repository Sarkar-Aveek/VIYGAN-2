"""Independent check on Indian ground: our models against same-day VENµS 5 m surface reflectance (SEN2VENµS).

SEN2VENµS (Michel et al. 2022, https://zenodo.org/records/14603764) pairs Sentinel-2 L2A patches (10 m, 128 x 128 px)
with VENµS surface reflectance of the same place on the same day (5 m, 256 x 256 px). Its KUDALIAR site is in
Telangana, India. None of it was used for training or model selection.

Each Sentinel-2 patch (B04 B03 B02, scaled like ESA true colour) is given to the models as a single date repeated to
fill the 8 input slots (the 1-date setting of docs/05 5.6). The 2.5 m output is averaged 2 x 2 onto VENµS's 5 m grid
and compared with VENµS; bicubic from the same 10 m input is the baseline. VENµS is 5 m, so this checks the 10 m ->
5 m part of the super-resolution and the band values; detail finer than 5 m cannot be checked against it.

    python sen2venus_check.py   -> results.csv, examples.png, lock_steps_3_vs_10.png in this folder

The first run downloads only the patches it needs (about 175 MB of the 7.9 GB site archive, by HTTP range requests)
into code/data/sen2venus/. results.csv, one value per row (group, quantity, value):
  bicubic / s2colour / arcgis_B   band error vs VENµS (L1, per band), spectral angle, PSNR, SSIM, edge-F1, LPIPS
  sensors                         Sentinel-2 vs VENµS at 10 m: the two sensors' own disagreement
  venus_detail                    VENµS vs its own 10 m version: how much detail it holds beyond 10 m
  s2colour_lock                   block error after 3 steps; after 3 / 6 / 10 steps; 3 vs 10 steps image change
  confidence                      AUROC of the detail-added map for the worst 10% of errors against VENµS
  location                        nearest training and test place, patch extent
Licence of the data: Sentinel-2 patches Etalab 2.0, VENµS patches CC BY-NC 4.0 (code/data/sen2venus/LICENCE).
"""
import csv
import glob
import io
import os
import random
import sys
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "..", "code")
sys.path.insert(0, CODE)
os.chdir(CODE)
import bootstrap  # noqa: F401,E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import infer  # noqa: E402
from ps_proof import TCI, auroc, colour_match  # noqa: E402
from esrgan.metrics import calculate_edge_f1, calculate_grad_corr, calculate_lpips  # noqa: E402
from esrgan.networks import frames_median, lowpass_transfer  # noqa: E402
from basicsr.metrics import calculate_psnr, calculate_ssim  # noqa: E402

SITE = os.path.join("data", "sen2venus")
ZIP_URL = "https://zenodo.org/api/records/14603764/files/KUDALIAR.zip/content"
LICENCE_URL = "https://zenodo.org/api/records/6514159/files/LICENCE/content"
PAIRS = ["2019-08-27_44QKF", "2020-10-30_44QKF"]       # the two smallest dates of the tile, two seasons
N_MAX = 1000                 # patches scored at most (random, fixed seed)
N_LOCK = 60                  # patches for the lock-step comparison
SEED = 0
MODELS = ["bicubic", "s2colour", "arcgis_B"]
BORDER = 4
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- fetch only the needed patches

def _get(start, end):
    req = urllib.request.Request(ZIP_URL, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req) as r:
        return r.read()


class _HttpFile(io.RawIOBase):
    """Seekable read-only view of the remote zip, one range request per read."""
    def __init__(self):
        req = urllib.request.Request(ZIP_URL, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])
        self.pos = 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        d = _get(self.pos, self.pos + n - 1)
        b[:len(d)] = d
        self.pos += len(d)
        return len(d)


def fetch():
    """Download the inner zips named in PAIRS (B02 B03 B04 B08 at 10 m and 5 m) and the licence, if missing."""
    names = [f"KUDALIAR_{p}_b2b3b4b8_{r}.zip" for p in PAIRS for r in ("10m", "05m")]
    if all(os.path.exists(os.path.join(SITE, n)) for n in names):
        return
    os.makedirs(SITE, exist_ok=True)
    urllib.request.urlretrieve(LICENCE_URL, os.path.join(SITE, "LICENCE"))
    z = zipfile.ZipFile(io.BufferedReader(_HttpFile(), buffer_size=1 << 20))
    for name in names:
        dest = os.path.join(SITE, name)
        if os.path.exists(dest):
            continue
        info = z.getinfo(f"KUDALIAR/{name}")
        hdr = _get(info.header_offset, info.header_offset + 29)          # local header: name and extra lengths
        start = info.header_offset + 30 + int.from_bytes(hdr[26:28], "little") + int.from_bytes(hdr[28:30], "little")
        n, size = 8, info.compress_size                                  # 8 parallel ranges: Zenodo is slow per stream
        cuts = [start + size * i // n for i in range(n + 1)]
        with ThreadPoolExecutor(n) as ex:
            data = b"".join(ex.map(lambda i: _get(cuts[i], cuts[i + 1] - 1), range(n)))
        if info.compress_type == zipfile.ZIP_DEFLATED:
            data = zlib.decompressobj(-15).decompress(data)
        assert len(data) == info.file_size, (name, len(data), info.file_size)
        with open(dest, "wb") as f:
            f.write(data)
        print("fetched", name, round(len(data) / 1e6, 1), "MB")


# ---------------------------------------------------------------- data and metrics

def pairs():
    """[(s2 [4,128,128] int16, venus [4,256,256] int16, patch name, (lon, lat))]: one GeoTIFF per patch, bands
    B02 B03 B04 B08, reflectance x 10000."""
    from rasterio.io import MemoryFile
    from rasterio.warp import transform
    out = []
    for z10 in sorted(glob.glob(os.path.join(SITE, "*_b2b3b4b8_10m.zip"))):
        a, b = zipfile.ZipFile(z10), zipfile.ZipFile(z10.replace("_10m.zip", "_05m.zip"))
        for n10 in sorted(x for x in a.namelist() if x.endswith(".tif")):
            n05 = n10.replace("/10m/", "/05m/").replace("_10m.tif", "_05m.tif")
            with MemoryFile(a.read(n10)) as m, m.open() as d:
                s2 = torch.from_numpy(d.read())
                cx, cy = d.xy(d.height // 2, d.width // 2)
                lon, lat = transform(d.crs, "EPSG:4326", [cx], [cy])
            with MemoryFile(b.read(n05)) as m, m.open() as d:
                vn = torch.from_numpy(d.read())
            out.append((s2, vn, os.path.basename(n10).replace("_b2b3b4b8_10m.tif", ""), (lon[0], lat[0])))
    return out


def km(lon1, lat1, lon2, lat2):
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(lon2 - lon1)
    return float(6371 * np.arccos(np.clip(np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl), -1, 1)))


def to_tci(refl):
    """Reflectance -> ESA true-colour 8-bit scale, as the collector stores Sentinel-2 (1..255)."""
    return np.clip(np.round(refl * 255 / TCI), 1, 255)


def rgb(t):
    """[4, H, W] int16 (B02 B03 B04 B08, reflectance x 10000) -> [H, W, 3] float reflectance, R G B = B04 B03 B02."""
    return t[[2, 1, 0]].float().div(10000).permute(1, 2, 0).numpy()


def u8(a):
    return np.clip(np.round(a), 0, 255).astype(np.uint8)


def refl_metrics(out_tci, ref_refl):
    """Band-value agreement in reflectance, both sides clipped at the true-colour saturation."""
    a = out_tci / 255 * TCI
    b = np.clip(ref_refl, 0, TCI)
    l1 = np.abs(a - b).mean((0, 1))                                # per band: B04, B03, B02
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9)
    sam = np.degrees(np.arccos(np.clip(cos, -1, 1))).mean()
    return {"refl_l1": float(l1.mean()), "refl_b04": float(l1[0]), "refl_b03": float(l1[1]),
            "refl_b02": float(l1[2]), "sam_deg": float(sam)}


def image_metrics(out_u8, ref_u8):
    cn = colour_match(out_u8, ref_u8)
    return {"psnr": calculate_psnr(out_u8, ref_u8, BORDER, test_y_channel=False),
            "ssim": calculate_ssim(out_u8, ref_u8, BORDER, test_y_channel=False),
            "cn_psnr": calculate_psnr(cn, ref_u8, BORDER, test_y_channel=False),
            "edge_f1": calculate_edge_f1(out_u8, ref_u8), "grad_corr": calculate_grad_corr(out_u8, ref_u8),
            "lpips": calculate_lpips(out_u8, ref_u8), "cn_lpips": calculate_lpips(cn, ref_u8)}


def sheet(cols, rows, path):
    S = 256
    img = Image.new("RGB", (len(cols) * (S + 6), len(rows) * (S + 6) + 22), "white")
    d = ImageDraw.Draw(img)
    for c, name in enumerate(cols):
        d.text((c * (S + 6) + 4, 4), name, fill="black")
    for r, panels in enumerate(rows):
        for c, p in enumerate(panels):
            img.paste(Image.fromarray(p).resize((S, S), Image.NEAREST), (c * (S + 6), 22 + r * (S + 6)))
    img.save(path)


# ---------------------------------------------------------------- the check

def main():
    fetch()
    nets = {m: infer.load(m, os.path.join("..", "weights", f"{m}_generator.pth"), dev) for m in MODELS[1:]}
    raw = infer.load("s2colour", os.path.join("..", "weights", "s2colour_generator.pth"), dev)
    raw.input_lowpass = False                                     # s2colour's network before its lock
    usable = [p for p in pairs() if (p[0][:3] > 0).all() and (p[1][:3] > 0).all()]   # no no-data in either sensor
    good = list(usable)
    random.Random(SEED).shuffle(good)
    good = good[:N_MAX]
    print(f"{len(good)} patches scored ({dev})")
    out = []                                                      # (group, quantity, value)

    # where: distance to anything the models trained or were tested on
    from collectors import arcgis_collector as agc
    lons, lats = [p[3][0] for p in good], [p[3][1] for p in good]
    for set_name, places in (("train", agc.TRAIN), ("test", agc.TEST)):
        d, name = min((km(lo, la, p[0], p[1]), n) for n, p in places.items() for lo, la in zip(lons, lats))
        out += [("location", f"nearest {set_name} place", name), ("location", f"nearest {set_name} place km", round(d, 1))]
    out += [("location", "patch centres", f"lon {min(lons):.3f}-{max(lons):.3f}, lat {min(lats):.3f}-{max(lats):.3f}")]

    rows, extra, conf_rows, examples = [], [], [], []
    rng = np.random.default_rng(SEED)
    with torch.no_grad():
        for k, (s2, vn, pid, _) in enumerate(good):
            s2_refl, vn_refl = rgb(s2), rgb(vn)
            s2_tci = to_tci(s2_refl)                                               # [128,128,3]
            vn_tci = to_tci(vn_refl)
            vn_u8 = vn_tci.astype(np.uint8)                                        # [256,256,3]
            x = torch.from_numpy(s2_tci).permute(2, 0, 1)[None].float().to(dev) / 255
            outs5, outs25 = {}, {}
            outs5["bicubic"] = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False).clamp(0, 1)[0] \
                .permute(1, 2, 0).mul(255).cpu().numpy()
            for m, net in nets.items():
                y = net(x.repeat(1, 8, 1, 1)).clamp(0, 1)                         # [1,3,512,512], 2.5 m
                outs25[m] = y
                outs5[m] = F.avg_pool2d(y, 2)[0].permute(1, 2, 0).mul(255).cpu().numpy()
            for m in MODELS:
                rows.append({"model": m, **refl_metrics(outs5[m], vn_refl), **image_metrics(u8(outs5[m]), vn_u8)})
            # the sensors' own disagreement at 10 m; VENµS against its own 10 m version; s2colour's lock residual
            floor = refl_metrics(s2_tci, vn_refl.reshape(128, 2, 128, 2, 3).mean((1, 3)))
            vt = torch.from_numpy(vn_tci).permute(2, 0, 1)[None].float()
            v10 = u8(F.interpolate(F.avg_pool2d(vt, 2), scale_factor=2, mode="bicubic", align_corners=False)[0]
                     .permute(1, 2, 0).numpy())
            lock = F.avg_pool2d(outs25["s2colour"], 4)[0].permute(1, 2, 0).mul(255).cpu().numpy()
            extra.append({"sensors refl_l1": floor["refl_l1"], "sensors sam_deg": floor["sam_deg"],
                          "venus psnr": calculate_psnr(v10, vn_u8, BORDER, test_y_channel=False),
                          "venus ssim": calculate_ssim(v10, vn_u8, BORDER, test_y_channel=False),
                          "venus edge_f1": calculate_edge_f1(v10, vn_u8),
                          "lock refl": float(np.abs(lock - s2_tci).mean() / 255 * TCI)})
            # confidence map (detail added) vs the real error against VENµS, at 5 m
            bic25 = F.interpolate(x, scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1)
            conf = F.avg_pool2d((outs25["s2colour"] - bic25).abs().mean(1, keepdim=True), 2)[0, 0].cpu().numpy()
            err = np.abs(colour_match(u8(outs5["s2colour"]), vn_u8).astype(np.float32) - vn_u8).mean(-1)
            idx = rng.choice(conf.size, 300, replace=False)
            conf_rows.append((conf.ravel()[idx], err.ravel()[idx]))
            edges = (cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(vn_u8, cv2.COLOR_RGB2GRAY), (0, 0), 2), 50, 150) > 0).mean()
            examples.append((edges, s2_tci.astype(np.uint8), outs5, outs25, vn_u8))
            examples = sorted(examples, key=lambda e: -e[0])[:4]
            if k % 100 == 0:
                print(k, pid)

        # the lock's step count: residual and what more steps do to the image (first N_LOCK usable patches)
        steps_err, change, lock_rows = {3: [], 6: [], 10: []}, [], []
        for k, (s2, vn, pid, _) in enumerate(usable[:N_LOCK]):
            x = torch.from_numpy(to_tci(rgb(s2))).permute(2, 0, 1)[None].float().to(dev) / 255
            xx = x.repeat(1, 8, 1, 1)
            y0 = raw(xx)
            ys = {s: lowpass_transfer(y0, frames_median(xx), 4, s) for s in steps_err}
            for s, y in ys.items():
                steps_err[s].append(float((F.avg_pool2d(y, 4) - x).abs().mean()) * 255)
            a, b = ys[3].clamp(0, 1), ys[10].clamp(0, 1)
            d = (a - b).abs().mul(255)
            ua = a[0].permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
            ub = b[0].permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
            hp = lambda t: t - F.interpolate(F.avg_pool2d(t, 4), scale_factor=4, mode="bicubic", align_corners=False)  # noqa: E731
            change.append((float(d.mean()), float(d.max()), calculate_edge_f1(ua, ub),
                           float(torch.corrcoef(torch.stack([hp(a).flatten(), hp(b).flatten()]))[0, 1])))
            if k in (3, 17):
                diff = np.clip(d[0].permute(1, 2, 0).cpu().numpy() * 10, 0, 255).astype(np.uint8)
                lock_rows.append([ua[128:384, 128:384], ub[128:384, 128:384], diff[128:384, 128:384]])

    for m in MODELS:
        rs = [r for r in rows if r["model"] == m]
        out += [(m, "patches", len(rs))] + [(m, q, round(float(np.mean([r[q] for r in rs])), 5)) for q in rs[0] if q != "model"]
    for key in extra[0]:
        group, q = key.split(" ")
        out.append({"sensors": ("sensors", f"s2_vs_venus_10m {q}"), "venus": ("venus_detail", f"vs own 10m version {q}"),
                    "lock": ("s2colour_lock", "block error refl, 3 steps")}[group] + (round(float(np.mean([e[key] for e in extra])), 5),))
    out += [("s2colour_lock", f"block error of 255, {s} steps", round(float(np.mean(v)), 3)) for s, v in steps_err.items()]
    c = np.array(change)
    out += [("s2colour_lock", "3 vs 10 steps: mean pixel change of 255", round(float(c[:, 0].mean()), 2)),
            ("s2colour_lock", "3 vs 10 steps: max pixel change of 255", round(float(c[:, 1].max()), 1)),
            ("s2colour_lock", "3 vs 10 steps: edge-F1 between them", round(float(c[:, 2].mean()), 3)),
            ("s2colour_lock", "3 vs 10 steps: fine-detail correlation", round(float(c[:, 3].mean()), 4))]
    score = np.concatenate([s for s, _ in conf_rows])
    error = np.concatenate([e for _, e in conf_rows])
    out += [("confidence", "pixels", len(score)),
            ("confidence", "AUROC worst 10%, detail-added map", round(auroc(score, error >= np.quantile(error, 0.9)), 4))]
    with open(os.path.join(HERE, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "quantity", "value"])
        w.writerows(out)

    # figures: 4 patches with the most structure in VENµS (centre 640 m); the lock at 3 vs 10 steps
    crop = lambda a, n: a[n // 4: n // 4 + n // 2, n // 4: n // 4 + n // 2]  # noqa: E731
    t25 = lambda o25, m: o25[m][0].permute(1, 2, 0).mul(255).cpu().numpy()  # noqa: E731
    sheet(["Sentinel-2 10 m", "bicubic 5 m", "s2colour 5 m", "s2colour 2.5 m", "arcgis_B 2.5 m", "VENuS 5 m"],
          [[crop(s2u, 128), crop(u8(o5["bicubic"]), 256), crop(u8(o5["s2colour"]), 256),
            crop(u8(t25(o25, "s2colour")), 512), crop(u8(t25(o25, "arcgis_B")), 512), crop(vn, 256)]
           for _, s2u, o5, o25, vn in examples], os.path.join(HERE, "examples.png"))
    sheet(["3 steps (released)", "10 steps", "difference x 10"], lock_rows, os.path.join(HERE, "lock_steps_3_vs_10.png"))
    print("done")


if __name__ == "__main__":
    main()
