"""
Evidence for the SIH26142 problem statement, written to outputs/ps_proof/.

    python ps_proof.py audit        data + metadata audit, leak audit                    (CPU, ~5 min)
    python ps_proof.py score        every model on the 7 held-out places: tile metrics,
                                    opensr-test on 4x4-tile chips, visual crops          (GPU, ~1 h)
    python ps_proof.py uncertainty  per-pixel confidence (disjoint date subsets) + calibration against error
    python ps_proof.py frames       1 / 2 / 4 / 8 input dates: how much the multi-date input buys
    python ps_proof.py figures      plots and visual sheets from the CSVs above

Every table is a CSV in outputs/ps_proof/metrics/, every picture a PNG in outputs/ps_proof/visuals/.
"""
import csv
import glob
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import date

import bootstrap  # noqa: F401

import numpy as np
from scipy.spatial import cKDTree

from collectors import arcgis_collector as agc

OUT = os.path.join("outputs", "ps_proof")
MET = os.path.join(OUT, "metrics")
VIS = os.path.join(OUT, "visuals")


def write_csv(name, rows):
    os.makedirs(MET, exist_ok=True)
    path = os.path.join(MET, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


def haversine_km(a, b):
    (lon1, lat1), (lon2, lat2) = a, b
    p1, p2 = math.radians(lat1), math.radians(lat2)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def tile_lonlat(x, y, z=17):
    n = 2 ** z
    lon = (x + 0.5) / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 0.5) / n))))
    return lon, lat


# ============================================================
# AUDIT: data, metadata, leaks
# ============================================================
def read_json(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return None


def audit():
    report = list(csv.DictReader(open(os.path.join("data", "training", "report.csv"))))
    split = {r["tile_id"]: r for r in report}

    # --- per place: collected, kept, rejected by reason ----------------------------------------------------
    rows = []
    for place, (lon, lat, kind) in list(agc.TRAIN.items()) + list(agc.TEST.items()):
        test = place in agc.TEST
        root = agc.TEST_OUTPUT_DIR if test else agc.OUTPUT_DIR
        s2_root = os.path.join("data", "test", "sentinel2_l2a") if test else os.path.join("data", "sentinel2_l2a")
        hr = len(glob.glob(os.path.join(root, place, "images", "*.png")))
        s2 = len(glob.glob(os.path.join(s2_root, place, "*.png")))
        reasons = Counter(r["reason"] or r["split"] for r in report if r["city"] == place)
        rows.append({"place": place, "role": "test" if test else "train", "kind": kind, "lon": lon, "lat": lat,
                     "seen_by_india20k": place in agc.SEEN, "arcgis_tiles": hr, "s2_stacks": s2,
                     "train": reasons["train"], "val": reasons["val"],
                     **{k: reasons[k] for k in ("water", "mismatch", "hr_black", "bad_s2_shape", "duplicate_tile_id",
                                                "seen_cap")}})
    write_csv("data_places.csv", rows)

    # --- metadata of every ArcGIS tile, and the Sentinel-2 dates against it --------------------------------
    meta_rows, gaps, gap_rows = [], [], []
    for place in list(agc.TRAIN) + list(agc.TEST):
        test = place in agc.TEST
        root = agc.TEST_OUTPUT_DIR if test else agc.OUTPUT_DIR
        s2_root = os.path.join("data", "test", "sentinel2_l2a") if test else os.path.join("data", "sentinel2_l2a")
        res, src, dates, missing = [], Counter(), [], 0
        pg = []
        for mp in glob.glob(os.path.join(root, place, "metadata", "*.json")):
            tid = os.path.basename(mp)[:-5]
            if not test and split.get(tid, {}).get("split") not in ("train", "val"):
                continue                                   # audit what the model actually trained on
            m = read_json(mp)
            hr = (m or {}).get("hr", {})
            if not hr.get("acquisition_date") or hr.get("resolution_m") is None:
                missing += 1
                continue
            res.append(float(hr["resolution_m"]))
            src[f"{hr.get('provider', '?')}/{hr.get('platform', '?')}"] += 1
            dates.append(hr["acquisition_date"])
            s = read_json(os.path.join(s2_root, place, tid + ".json"))
            if s and s.get("dates"):
                a = date.fromisoformat(hr["acquisition_date"])
                d = [abs((date.fromisoformat(x) - a).days) for x in s["dates"]]
                pg.append(float(np.median(d)))
        if not res:
            continue
        gaps += pg
        meta_rows.append({"place": place, "role": "test" if test else "train", "tiles_with_metadata": len(res),
                          "missing_metadata": missing, "res_m_min": min(res), "res_m_median": float(np.median(res)),
                          "res_m_max": max(res), "share_le_1m": round(float(np.mean(np.array(res) <= 1.0)), 4),
                          "capture_first": min(dates), "capture_last": max(dates),
                          "main_source": src.most_common(1)[0][0], "sources": len(src),
                          "s2_gap_days_median": round(float(np.median(pg)), 1) if pg else "",
                          "s2_gap_days_p90": round(float(np.percentile(pg, 90)), 1) if pg else ""})
    write_csv("data_metadata.csv", meta_rows)
    np.save(os.path.join(MET, "s2_gap_days.npy"), np.array(gaps, np.float32))

    # --- leaks ---------------------------------------------------------------------------------------------
    train_ids = {t for t, r in split.items() if r["split"] == "train"}
    val_ids = {t for t, r in split.items() if r["split"] == "val"}
    test_ids = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(agc.TEST_OUTPUT_DIR, "*", "images", "*.png"))}
    trained_xy = [tuple(map(int, t.split("_")[1:])) for t in train_ids | val_ids]
    trained_ll = np.array([tile_lonlat(x, y) for x, y in trained_xy])

    def md5s(paths):
        return {hashlib.md5(open(p, "rb").read()).hexdigest() for p in paths}

    train_hr = [os.path.join("data", "training", "train", "arcgis", t, "rgb.png") for t in train_ids]
    test_hr = glob.glob(os.path.join(agc.TEST_OUTPUT_DIR, "*", "images", "*.png"))
    same_pixels = len(md5s(train_hr) & md5s(test_hr))

    leak = [{"check": "test tile ids also in train", "value": len(test_ids & train_ids), "pass": not (test_ids & train_ids)},
            {"check": "test tile ids also in val", "value": len(test_ids & val_ids), "pass": not (test_ids & val_ids)},
            {"check": "val tile ids also in train", "value": len(val_ids & train_ids), "pass": not (val_ids & train_ids)},
            {"check": "test ArcGIS images byte-identical to a training image", "value": same_pixels,
             "pass": same_pixels == 0},
            {"check": "test places that India 20k (the starting model) trained on",
             "value": len(set(agc.TEST) & agc.SEEN), "pass": not (set(agc.TEST) & agc.SEEN)}]
    dist_rows = []
    for place, (lon, lat, kind) in agc.TEST.items():
        near_place = min(agc.TRAIN, key=lambda p: haversine_km((lon, lat), agc.TRAIN[p][:2]))
        ll = np.array([tile_lonlat(*map(int, t.split("_")[1:])) for t in test_ids
                       if os.path.exists(os.path.join(agc.TEST_OUTPUT_DIR, place, "images", t + ".png"))])
        # nearest trained tile to any test tile of this place (equirectangular is exact enough at these distances)
        k = np.cos(np.radians(lat))
        d = cKDTree(trained_ll * [k, 1]).query(ll * [k, 1])[0].min() * 111.2
        dist_rows.append({"test_place": place, "kind": kind, "nearest_training_place": near_place,
                          "centre_distance_km": round(haversine_km((lon, lat), agc.TRAIN[near_place][:2]), 1),
                          "nearest_trained_tile_km": round(float(d), 1)})
    leak.append({"check": "min distance, test tile to any trained tile (km)",
                 "value": min(r["nearest_trained_tile_km"] for r in dist_rows),
                 "pass": min(r["nearest_trained_tile_km"] for r in dist_rows) > 20})
    write_csv("leak_checks.csv", leak)
    write_csv("leak_distances.csv", dist_rows)


# ============================================================
# SCORE: every model on the held-out places
# ============================================================
TCI = 0.3558                 # reflectance that maps to 255 (ESA true colour); every model is shown on this scale
CHIP = 4                     # opensr-test runs on 4x4-tile chips: 128 px in, 512 px out, the benchmark's own size
RIVAL_DIR = os.path.join("published_models", "loaders")   # SEN2SR Lite, LDSR-S2, Real-ESRGAN
OURS = ["bicubic_8date", "satlas_8s2", "arcgis_B", "s2colour"]     # our two models + the 8-date baselines
SINGLE = ["bicubic_1date", "sen2sr_lite", "ldsr_s2", "realesrgan_general"]
TILE_METRICS = ["psnr", "ssim", "cpsnr", "lpips", "cn_psnr", "cn_ssim", "cn_lpips", "edge_f1", "grad_corr"]
CHIP_METRICS = ["ha_metric", "om_metric", "im_metric", "reflectance", "spectral", "spatial", "synthesis"]


def ours(name):
    """name -> fn(lr [b, 24, 32, 32] 0-1) -> [b, 3, 128, 128] 0-1, as evaluate.py builds them."""
    import evaluate as ev
    if name == "bicubic_8date":
        return ev.build_models(["bicubic"])["bicubic"]
    if name == "satlas_8s2":
        return ev.load(os.path.join("weights", "esrgan_8S2.pth"))
    if name == "s2colour":                     # the final model 3 generator was renamed after training
        return ev.load(os.path.join(ev.EXP, "s2colour", "models", "net_g_latest_colour.pth"), lowpass=True)
    return ev.load(ev.checkpoint(name))


def single(name, device):
    """name -> fn(refl [b, 4, H, W] B02 B03 B04 B08) -> RGB reflectance [b, 3, 4H, 4W]."""
    import importlib
    import torch.nn.functional as F
    if name == "bicubic_1date":
        return lambda x: F.interpolate(x[:, [2, 1, 0]], scale_factor=4, mode="bicubic", align_corners=False)
    sys.path.insert(0, RIVAL_DIR)
    for mod in sorted(glob.glob(os.path.join(RIVAL_DIR, "[a-z]*.py"))):
        m = importlib.import_module(os.path.basename(mod)[:-3])
        if name in getattr(m, "NAMES", []):
            return m.load(name, device)
    raise KeyError(name)


def colour_match(sr, gt):
    """Per-channel mean/std of `sr` set to `gt`'s: removes the colour convention (ArcGIS vs Sentinel-2), keeps
    the structure. Used for the cn_* metrics so models of both colour families are compared on detail."""
    s, g = sr.astype(np.float32), gt.astype(np.float32)
    s = (s - s.mean((0, 1))) / (s.std((0, 1)) + 1e-6) * g.std((0, 1)) + g.mean((0, 1))
    return np.clip(s, 0, 255).round().astype(np.uint8)


def tile_metrics(pair):
    from basicsr.metrics import calculate_psnr, calculate_ssim
    from esrgan.metrics import calculate_cpsnr, calculate_edge_f1, calculate_grad_corr
    sr, gt = pair
    cn = colour_match(sr, gt)
    return {"psnr": calculate_psnr(sr, gt, 4, test_y_channel=False),
            "ssim": calculate_ssim(sr, gt, 4, test_y_channel=False),
            "cpsnr": calculate_cpsnr(sr, gt, 4),
            "cn_psnr": calculate_psnr(cn, gt, 4, test_y_channel=False),
            "cn_ssim": calculate_ssim(cn, gt, 4, test_y_channel=False),
            "edge_f1": calculate_edge_f1(sr, gt), "grad_corr": calculate_grad_corr(sr, gt)}


def band_tile(x, y, when):
    """[4, 32, 32] reflectance (B02 B03 B04 B08) of one zoom-17 tile on one date, from the cached block."""
    x0, y0 = x // 16 * 16, y // 16 * 16
    a = np.load(os.path.join("data", "cache", "bands", f"{x0}_{y0}_{when}_B02-B03-B04-B08.npy"))
    r, c = (y - y0) * 32, (x - x0) * 32
    return a[r:r + 32, c:c + 32].transpose(2, 0, 1).astype(np.float32) / 10000


def test_places(set_name):
    """{place: {tile_id: dict(x, y, hr, lr8 [24,32,32] uint8 tensor, date)}} for one test set."""
    import evaluate as ev
    out = defaultdict(dict)
    for place, tid, x, y, hr, lr in ev.test_tiles(ev.SETS[set_name]):
        d = read_json(os.path.join(ev.SETS[set_name], place, tid + ".json"))
        a = date.fromisoformat(d["arcgis_date"])
        near = min(d["dates"], key=lambda s: abs((date.fromisoformat(s) - a).days))
        out[place][tid] = {"x": x, "y": y, "hr": hr, "lr8": lr, "date": near}
    return out


def chips(tiles):
    """4x4-tile chips of one place: [(cx, cy, {(i, j): tile_id or None})]. A chip is scored by opensr-test only if
    all 16 tiles are present; its tiles are scored individually either way."""
    groups = defaultdict(dict)
    for tid, t in tiles.items():
        groups[(t["x"] // CHIP * CHIP, t["y"] // CHIP * CHIP)][(t["x"] % CHIP, t["y"] % CHIP)] = tid
    return [(cx, cy, g) for (cx, cy), g in sorted(groups.items())]


def chip_input(tiles, cx, cy, members):
    """Single-date reflectance [4, 128, 128] for a chip: each tile on its own nearest date; a gap tile takes the
    chip's most common date so the model still sees whole context."""
    common = Counter(tiles[t]["date"] for t in members.values()).most_common(1)[0][0]
    out = np.zeros((4, CHIP * 32, CHIP * 32), np.float32)
    for i in range(CHIP):
        for j in range(CHIP):
            tid = members.get((i, j))
            out[:, j * 32:(j + 1) * 32, i * 32:(i + 1) * 32] = band_tile(cx + i, cy + j,
                                                                         tiles[tid]["date"] if tid else common)
    return out


def pick_visual_tiles(tiles, n=3):
    """The n tiles with the most structure in the reference (edges of the blurred Canny used by edge-F1)."""
    import cv2
    from esrgan.metrics import _structure_gray
    score = {tid: (cv2.Canny(cv2.GaussianBlur(_structure_gray(t["hr"], 4), (0, 0), 2), 50, 150) > 0).mean()
             for tid, t in tiles.items()}
    return [t for t, _ in sorted(score.items(), key=lambda kv: -kv[1])[:n]]


def score(models=None):
    """All models, or (python ps_proof.py score NAME ...) only those, replacing their rows in the existing CSVs."""
    import torch
    from concurrent.futures import ProcessPoolExecutor
    from opensr_test.main import Metrics
    from esrgan.metrics import calculate_lpips
    device = torch.device("cuda")
    sets = {s: test_places(s) for s in ("inseason", "offseason")}
    visual = {s: {p: pick_visual_tiles(t) for p, t in sets[s].items()} for s in sets}
    keep = {}                                     # (set, place, tile_id, model) -> uint8 SR, for the figures
    tile_rows, chip_rows = [], []
    if models:
        tile_rows = [r for r in csv.DictReader(open(os.path.join(MET, "tiles.csv"))) if r["model"] not in models]
        chip_rows = [r for r in csv.DictReader(open(os.path.join(MET, "chips.csv"))) if r["model"] not in models]
        old = np.load(os.path.join(MET, "visual_tiles.npz"))
        keep = {tuple(k.split("|")): old[k] for k in old.files if k.split("|")[3] not in models}
    metric = Metrics(device="cuda")

    def run_chip(lr, sr, hr, model, set_name, place, cx, cy):
        with torch.no_grad():
            r = metric.compute(lr=torch.from_numpy(lr).float(), sr=torch.from_numpy(sr).float(),
                               hr=torch.from_numpy(hr).float())
        chip_rows.append({"set": set_name, "model": model, "place": place, "kind": agc.TEST[place][2],
                          "chip": f"{cx}_{cy}", **{k: round(float(r[k]), 5) for k in CHIP_METRICS}})

    with ProcessPoolExecutor(4) as pool:
        for model in models or OURS + SINGLE:
            fn = ours(model) if model in OURS else single(model, device)
            for set_name, places in sets.items():
                if model in SINGLE and set_name == "offseason":
                    continue                          # 4-band data was fetched for the in-season dates only
                print(f"{set_name}: {model}", flush=True)
                for place, tiles in places.items():
                    srs = {}
                    for cx, cy, members in chips(tiles):
                        with torch.no_grad():
                            if model in OURS:
                                ids = list(members.values())
                                lr = torch.stack([tiles[t]["lr8"] for t in ids]).to(device).float() / 255
                                out = fn(lr).clamp(0, 1).mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
                                srs.update(zip(ids, out))
                            else:
                                x = torch.from_numpy(chip_input(tiles, cx, cy, members))[None].to(device)
                                out = (fn(x)[0] / TCI).clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
                                for (i, j), t in members.items():
                                    srs[t] = np.ascontiguousarray(out[j * 128:(j + 1) * 128, i * 128:(i + 1) * 128])
                        if len(members) == CHIP * CHIP:   # opensr-test on the whole chip, in 0-1 display units
                            def mosaic(get, px):
                                m = np.zeros((CHIP * px, CHIP * px, 3), np.float32)
                                for (i, j), t in members.items():
                                    m[j * px:(j + 1) * px, i * px:(i + 1) * px] = get(t)
                                return m.transpose(2, 0, 1) / 255
                            if model in OURS:            # the model's own input: the median of its 8 frames
                                lr = mosaic(lambda t: np.median(tiles[t]["lr8"].view(8, 3, 32, 32).numpy(), 0)
                                            .transpose(1, 2, 0), 32)
                            else:                        # the single date it was given
                                lr = np.clip(chip_input(tiles, cx, cy, members)[[2, 1, 0]] / TCI, 0, 1)
                            run_chip(lr, mosaic(lambda t: srs[t], 128), mosaic(lambda t: tiles[t]["hr"], 128),
                                     model, set_name, place, cx, cy)
                    ids = list(srs)
                    cpu = list(pool.map(tile_metrics, [(srs[t], tiles[t]["hr"]) for t in ids], chunksize=16))
                    for t, m in zip(ids, cpu):
                        gt = tiles[t]["hr"]
                        m["lpips"] = calculate_lpips(srs[t], gt)
                        m["cn_lpips"] = calculate_lpips(colour_match(srs[t], gt), gt)
                        tile_rows.append({"set": set_name, "model": model, "place": place,
                                          "kind": agc.TEST[place][2], "tile_id": t,
                                          **{k: round(float(m[k]), 5) for k in TILE_METRICS}})
                        if t in visual[set_name][place]:
                            keep[(set_name, place, t, model)] = srs[t]
            del fn
            torch.cuda.empty_cache()
            write_csv("tiles.csv", tile_rows)          # after every model, so a crash keeps what was scored
            write_csv("chips.csv", chip_rows)

    for s, places in sets.items():                     # references and inputs of the visual tiles
        for p, ids in visual[s].items():
            for t in ids:
                tl = places[p][t]
                keep[(s, p, t, "reference")] = tl["hr"]
                keep[(s, p, t, "input_8date")] = np.median(tl["lr8"].view(8, 3, 32, 32).numpy(), 0) \
                    .transpose(1, 2, 0).astype(np.uint8)
                if s == "inseason":
                    keep[(s, p, t, "input_1date")] = np.clip(band_tile(tl["x"], tl["y"], tl["date"])[[2, 1, 0]]
                                                             / TCI * 255, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    np.savez_compressed(os.path.join(MET, "visual_tiles.npz"),
                        **{"|".join(k): v for k, v in keep.items()})
    summarise_scores(tile_rows, chip_rows)


def summarise_scores(tile_rows=None, chip_rows=None):
    """Means per set x model (all places) and per set x model x place; tiles and chips joined into one table."""
    tile_rows = tile_rows or list(csv.DictReader(open(os.path.join(MET, "tiles.csv"))))
    chip_rows = chip_rows or list(csv.DictReader(open(os.path.join(MET, "chips.csv"))))
    out = []
    for scope in ("all", "place"):
        g = defaultdict(lambda: ([], []))
        for rows, i, keys in ((tile_rows, 0, TILE_METRICS), (chip_rows, 1, CHIP_METRICS)):
            for r in rows:
                g[(r["set"], r["model"], "all" if scope == "all" else r["place"])][i].append(r)
        for (s, m, sc), (ts, cs) in sorted(g.items()):
            row = {"set": s, "model": m, "scope": sc, "tiles": len(ts), "chips": len(cs)}
            row.update({k: round(float(np.mean([float(r[k]) for r in ts])), 4) if ts else "" for k in TILE_METRICS})
            row.update({k: round(float(np.nanmean([float(r[k]) for r in cs])), 4) if cs else "" for k in CHIP_METRICS})
            out.append(row)
    write_csv("summary.csv", out)


# ============================================================
# UNCERTAINTY: does the per-pixel confidence predict the real error?
# ============================================================
UNC_MODELS = ["arcgis_B", "s2colour"]


def gray(img):
    return img.astype(np.float32).mean(axis=-1)


def auroc(score, label):
    """Area under the ROC curve of `score` for the binary `label` (Mann-Whitney U / (n1 n0))."""
    from scipy.stats import rankdata
    r = rankdata(score)
    n1 = float(label.sum())                # float: int32 on Windows overflows n1 * n0
    n0 = len(label) - n1
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def uncertainty():
    """sigma = per-pixel std between two runs on disjoint date subsets (gis_tools.dual_subsets / sigma, the
    dashboard's confidence layer). Real detail repeats across dates; invented detail does not. The test: pixels
    with high sigma should be the pixels that are actually wrong against the ArcGIS reference.

    Error = |colour-matched SR - reference| in grayscale (the colour convention removed, as in the cn_* metrics).
    Two naive predictors are scored the same way, to show sigma is more than 'edges are hard':
      gradient  Sobel magnitude of the SR image
      added     |SR - bicubic|: how much detail the model put in"""
    import cv2
    import torch
    import torch.nn.functional as F
    from scipy.stats import spearmanr
    import gis_tools as g
    from esrgan.networks import frames_median
    device = torch.device("cuda")
    rng = np.random.default_rng(0)
    places = test_places("inseason")
    rows, curve_rows, maps = [], [], {}
    for model in UNC_MODELS:
        fn = ours(model)
        pix = defaultdict(list)                    # predictor -> sampled values; plus "error", "grad_bin"
        tile_sigma = []
        for place, tiles in places.items():
            vis = set(pick_visual_tiles(tiles))
            for tid, t in tiles.items():
                frames = list(t["lr8"].view(8, 3, 32, 32))
                a, b = g.dual_subsets(frames)
                lr = torch.stack([t["lr8"], a, b]).to(device).float() / 255
                with torch.no_grad():
                    sr = fn(lr).clamp(0, 1)
                    bic = F.interpolate(frames_median(lr[:1]), scale_factor=4, mode="bicubic", align_corners=False)
                sig = g.sigma(sr[1], sr[2]).cpu().numpy() * 255
                full = sr[0].mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
                cm = colour_match(full, t["hr"])
                err = np.abs(gray(cm) - gray(t["hr"]))
                gs = gray(full)
                grad = np.hypot(cv2.Sobel(gs, cv2.CV_32F, 1, 0), cv2.Sobel(gs, cv2.CV_32F, 0, 1))
                added = (sr[0] - bic[0].clamp(0, 1)).abs().mean(0).cpu().numpy() * 255
                c = slice(4, -4)                   # the metrics' crop border
                idx = rng.choice(120 * 120, 300, replace=False)
                for k, v in (("sigma", sig), ("gradient", grad), ("added", added), ("error", err)):
                    pix[k].append(v[c, c].ravel()[idx])
                tile_sigma.append({"place": place, "tile_id": tid, "mean_sigma": float(sig[c, c].mean())})
                if tid in vis:
                    maps[f"{model}|{place}|{tid}|sigma"] = sig.astype(np.float16)
                    maps[f"{model}|{place}|{tid}|error"] = err.astype(np.float16)
        del fn
        torch.cuda.empty_cache()
        v = {k: np.concatenate(x) for k, x in pix.items()}
        bad = v["error"] > np.percentile(v["error"], 90)            # the worst 10% of pixels
        gbin = np.digitize(v["gradient"], np.percentile(v["gradient"], np.arange(10, 100, 10)))
        for pred in ("sigma", "gradient", "added"):
            within = [auroc(v[pred][gbin == b], bad[gbin == b]) for b in range(10)]
            rows.append({"model": model, "predictor": pred, "pixels": len(bad),
                         "spearman_vs_error": round(float(spearmanr(v[pred], v["error"])[0]), 4),
                         "auroc_worst10pct": round(auroc(v[pred], bad), 4),
                         "auroc_within_gradient_deciles": round(float(np.nanmean(within)), 4)})
            edges = np.percentile(v[pred], np.arange(0, 101, 10))
            for i in range(10):
                sel = (v[pred] >= edges[i]) & (v[pred] <= edges[i + 1])
                curve_rows.append({"model": model, "predictor": pred, "decile": i + 1,
                                   "predictor_mean": round(float(v[pred][sel].mean()), 4),
                                   "error_mean": round(float(v["error"][sel].mean()), 4),
                                   "share_worst10pct": round(float(bad[sel].mean()), 4)})
        # tile level: does a tile's mean sigma rank its perceptual / structural error?
        tr = {r["tile_id"]: r for r in csv.DictReader(open(os.path.join(MET, "tiles.csv")))
              if r["model"] == model and r["set"] == "inseason"}
        ms = [s["mean_sigma"] for s in tile_sigma if s["tile_id"] in tr]
        for metric in ("cn_lpips", "edge_f1"):
            y = [float(tr[s["tile_id"]][metric]) for s in tile_sigma if s["tile_id"] in tr]
            rows.append({"model": model, "predictor": f"tile mean sigma vs {metric}", "pixels": len(ms),
                         "spearman_vs_error": round(float(spearmanr(ms, y)[0]), 4),
                         "auroc_worst10pct": "", "auroc_within_gradient_deciles": ""})
    write_csv("uncertainty.csv", rows)
    write_csv("uncertainty_curve.csv", curve_rows)
    np.savez_compressed(os.path.join(MET, "uncertainty_maps.npz"), **maps)


# ============================================================
# FRAMES: what the 8 dates buy
# ============================================================
FRAME_MODELS = ["arcgis_B", "s2colour"]


def frames():
    """The same tiles with 1, 2, 4 and all 8 dates (nearest the ArcGIS date first, repeated to fill the 8 input
    slots the network takes). If the multi-date input is what makes the model good, quality should fall as dates
    are removed."""
    import cv2
    import torch
    from concurrent.futures import ProcessPoolExecutor
    from esrgan.dataset import split_frames
    from esrgan.metrics import calculate_lpips
    import evaluate as ev
    device = torch.device("cuda")
    root = ev.SETS["inseason"]
    stacks = []
    for place in agc.TEST:
        for p in sorted(glob.glob(os.path.join(root, place, "*.png"))):
            tid = os.path.basename(p)[:-4]
            hr_path = os.path.join(ev.TEST_ARCGIS, place, "images", tid + ".png")
            d = read_json(p[:-4] + ".json")
            s2, hr = cv2.imread(p), cv2.imread(hr_path)
            if hr is None or s2 is None or s2.shape != (256, 32, 3) or (hr.sum(-1) == 0).any() or not d:
                continue
            a = date.fromisoformat(d["arcgis_date"])
            order = np.argsort([abs((date.fromisoformat(x) - a).days) for x in d["dates"]])
            fr = split_frames(s2[:, :, ::-1].copy())
            stacks.append((place, tid, np.ascontiguousarray(hr[:, :, ::-1]), [fr[i] for i in order]))
    rows = []
    with ProcessPoolExecutor(4) as pool:
        for model in FRAME_MODELS:
            fn = ours(model)
            for n in (1, 2, 4, 8):
                srs = []
                for i in range(0, len(stacks), 64):
                    lr = torch.stack([torch.cat([f[k % n] for k in range(8)]) for *_, f in stacks[i:i + 64]])
                    with torch.no_grad():
                        out = fn(lr.to(device).float() / 255).clamp(0, 1)
                    srs.extend(out.mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy())
                ms = list(pool.map(tile_metrics, [(s, st[2]) for s, st in zip(srs, stacks)], chunksize=16))
                for s, st, m in zip(srs, stacks, ms):
                    m["cn_lpips"] = calculate_lpips(colour_match(s, st[2]), st[2])
                    rows.append({"model": model, "dates": n, "place": st[0], "tile_id": st[1],
                                 **{k: round(float(m[k]), 5) for k in ("cn_psnr", "cn_ssim", "cn_lpips", "edge_f1",
                                                                       "grad_corr")}})
                print(f"{model} {n} dates: edge-F1 {np.mean([r['edge_f1'] for r in rows[-len(stacks):]]):.4f}",
                      flush=True)
            del fn
            torch.cuda.empty_cache()
    summary = defaultdict(list)
    for r in rows:
        summary[(r["model"], r["dates"])].append(r)
    write_csv("frames.csv", [{"model": m, "dates": n, "tiles": len(rs),
                              **{k: round(float(np.mean([r[k] for r in rs])), 4)
                                 for k in ("cn_psnr", "cn_ssim", "cn_lpips", "edge_f1", "grad_corr")}}
                             for (m, n), rs in sorted(summary.items())])


# ============================================================
# FIGURES
# ============================================================
LABEL = {"bicubic_8date": "bicubic (8-date median)", "bicubic_1date": "bicubic (1 date)",
         "satlas_8s2": "Satlas 8S2 (US, original)",
         "arcgis_B": "OURS arcgis_B (ArcGIS colours)", "s2colour": "OURS s2colour (Sentinel-2 colours)",
         "sen2sr_lite": "SEN2SR Lite (ESA)", "ldsr_s2": "LDSR-S2 diffusion (ESA)",
         "realesrgan_general": "Real-ESRGAN (photos)"}
OUR_SET = {"arcgis_B", "s2colour"}


def _colour(m):
    return "#1f6f3f" if m in OUR_SET else "#9a9a9a" if m.startswith("bicubic") else "#c46a1c" if m in (
        "satlas_8s2", "sen2sr_lite", "ldsr_s2", "realesrgan_general") else "#3a64a8"


def figures():
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summ = list(csv.DictReader(open(os.path.join(MET, "summary.csv"))))
    allin = {r["model"]: r for r in summ if r["set"] == "inseason" and r["scope"] == "all"}
    order = [m for m in OURS + SINGLE if m in allin]

    # 1. every headline metric, in-season, all models
    panels = [("edge_f1", "edge-F1: structure edges in the right place", False),
              ("cn_lpips", "LPIPS after colour match (lower = looks more like the reference)", True),
              ("cn_ssim", "SSIM after colour match", False), ("grad_corr", "gradient correlation", False),
              ("im_metric", "opensr-test improvement (correct new detail)", False),
              ("ha_metric", "opensr-test hallucination (invented detail, lower better)", True)]
    fig, axes = plt.subplots(3, 2, figsize=(16, 17))
    for ax, (k, title, low) in zip(axes.ravel(), panels):
        ms = sorted(order, key=lambda m: float(allin[m][k]) * (1 if low else -1))
        ax.barh(range(len(ms)), [float(allin[m][k]) for m in ms], color=[_colour(m) for m in ms])
        ax.set_yticks(range(len(ms)))
        ax.set_yticklabels([LABEL[m] for m in ms], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", alpha=0.3)
        for i, m in enumerate(ms):
            ax.text(float(allin[m][k]), i, f" {float(allin[m][k]):.3f}", va="center", fontsize=8)
    fig.suptitle("7 held-out Indian places, in season: green = ours, orange = published models, "
                 "grey = bicubic", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(os.path.join(VIS, "01_metrics_all_models.png"), dpi=110)
    plt.close(fig)

    # 2. hallucination vs improvement (the opensr-test trade-off)
    fig, ax = plt.subplots(figsize=(9, 7))
    for m in order:
        x, y = float(allin[m]["ha_metric"]), float(allin[m]["im_metric"])
        ax.scatter(x, y, s=70, color=_colour(m))
        ax.annotate(LABEL[m], (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("hallucination: invented detail (lower better)")
    ax.set_ylabel("improvement: correct new detail (higher better)")
    ax.set_title("opensr-test on the 7 Indian places (4x4-tile chips, ArcGIS reference)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(VIS, "02_hallucination_vs_improvement.png"), dpi=120)
    plt.close(fig)

    # 3. per place heatmap of edge-F1
    places = list(agc.TEST)
    per = {(r["model"], r["scope"]): r for r in summ if r["set"] == "inseason" and r["scope"] != "all"}
    grid = np.array([[float(per[(m, p)]["edge_f1"]) if (m, p) in per else np.nan for p in places] for m in order])
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(places)))
    ax.set_xticklabels([f"{p}\n({agc.TEST[p][2]})" for p in places], fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABEL[m] for m in order], fontsize=8)
    for i in range(len(order)):
        for j in range(len(places)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="w" if grid[i, j] < np.nanmean(grid) else "k")
    fig.colorbar(im, ax=ax, label="edge-F1")
    ax.set_title("edge-F1 per held-out place (in season)")
    fig.tight_layout()
    fig.savefig(os.path.join(VIS, "03_edge_f1_per_place.png"), dpi=120)
    plt.close(fig)

    # 4. in season vs off season (ours; 4-band data exists for in-season only)
    off = {r["model"]: r for r in summ if r["set"] == "offseason" and r["scope"] == "all"}
    ms = [m for m in OURS if m in off]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, k in zip(axes, ("edge_f1", "cn_lpips", "im_metric")):
        w = 0.38
        ax.bar(np.arange(len(ms)) - w / 2, [float(allin[m][k]) for m in ms], w, label="in season")
        ax.bar(np.arange(len(ms)) + w / 2, [float(off[m][k]) for m in ms], w, label="~6 months away")
        ax.set_xticks(range(len(ms)))
        ax.set_xticklabels([LABEL[m] for m in ms], rotation=25, ha="right", fontsize=8)
        ax.set_title(k)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend()
    fig.suptitle("Dates far from the reference capture (crops, snow, water change): does quality hold?")
    fig.tight_layout()
    fig.savefig(os.path.join(VIS, "04_inseason_vs_offseason.png"), dpi=120)
    plt.close(fig)

    # 5. visual sheets: single tiles, 2x nearest so the 2.4 m detail is visible
    vt = np.load(os.path.join(MET, "visual_tiles.npz"))
    tiles = defaultdict(dict)
    for key in vt.files:
        s, p, t, m = key.split("|")
        tiles[(s, p, t)][m] = vt[key]

    def cell(img, label, size=256):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
        img = np.ascontiguousarray(img)
        cv2.rectangle(img, (0, 0), (size, 18), (0, 0, 0), -1)
        cv2.putText(img, label[:34], (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return img

    groups = {"05_visual_ours_vs_baselines": ["reference", "input_8date", "bicubic_8date", "satlas_8s2", "arcgis_B",
                                              "s2colour"],
              "06_visual_vs_published": ["reference", "input_1date", "arcgis_B", "s2colour", "sen2sr_lite",
                                         "ldsr_s2", "realesrgan_general"]}
    names = dict(LABEL, reference="REFERENCE ArcGIS 0.3-0.5 m", input_8date="INPUT 8-date median (10 m)",
                 input_1date="INPUT 1 date (10 m)")
    for gname, cols in groups.items():
        for place in agc.TEST:
            keys = [k for k in tiles if k[0] == "inseason" and k[1] == place]
            rows = [np.hstack([cell(tiles[k][c], names[c]) if c in tiles[k] else np.zeros((256, 256, 3), np.uint8)
                               for c in cols]) for k in keys]
            if rows:
                os.makedirs(os.path.join(VIS, gname), exist_ok=True)
                cv2.imwrite(os.path.join(VIS, gname, f"{place}.png"), np.vstack(rows)[:, :, ::-1])

    # 6. uncertainty: maps and calibration
    if os.path.exists(os.path.join(MET, "uncertainty_maps.npz")):
        um = np.load(os.path.join(MET, "uncertainty_maps.npz"))
        for model in UNC_MODELS:
            rows = []
            for key in sorted(k for k in um.files if k.startswith(model + "|") and k.endswith("|sigma")):
                _, place, t, _ = key.split("|")
                sig, err = um[key].astype(np.float32), um[f"{model}|{place}|{t}|error"].astype(np.float32)
                heat = lambda a: cv2.applyColorMap(  # noqa: E731  (each map scaled to its own 99th percentile)
                    np.clip(a / (np.percentile(a, 99) + 1e-6) * 255, 0, 255).astype(np.uint8),
                    cv2.COLORMAP_INFERNO)[:, :, ::-1]
                ref = tiles.get(("inseason", place, t), {})
                if model not in ref or "bicubic_8date" not in ref:
                    continue
                added = np.abs(ref[model].astype(np.float32) - ref["bicubic_8date"].astype(np.float32)).mean(-1)
                rows.append(np.hstack([cell(ref["reference"], f"reference {place}"), cell(ref[model], LABEL[model]),
                                       cell(heat(added), "CONFIDENCE: detail added (bright=unsure)"),
                                       cell(heat(sig), "alt: date-subset sigma"),
                                       cell(heat(err), "actual error vs reference")]))
            if rows:
                cv2.imwrite(os.path.join(VIS, f"07_uncertainty_maps_{model}.png"), np.vstack(rows[:9])[:, :, ::-1])
        cur = list(csv.DictReader(open(os.path.join(MET, "uncertainty_curve.csv"))))
        fig, axes = plt.subplots(1, len(UNC_MODELS), figsize=(13, 4.5))
        for ax, model in zip(np.atleast_1d(axes), UNC_MODELS):
            for pred, style in (("sigma", "-o"), ("gradient", "--s"), ("added", ":^")):
                rs = [r for r in cur if r["model"] == model and r["predictor"] == pred]
                ax.plot([int(r["decile"]) for r in rs], [float(r["share_worst10pct"]) for r in rs], style,
                        label=pred)
            ax.axhline(0.1, color="grey", lw=0.8, ls=":")
            ax.set_xlabel("predictor decile (1 = most confident)")
            ax.set_ylabel("share of the worst-10% error pixels")
            ax.set_title(LABEL[model])
            ax.grid(alpha=0.3)
            ax.legend()
        fig.suptitle("Calibration: do the pixels the confidence map flags hold the real errors? (grey = chance)")
        fig.tight_layout()
        fig.savefig(os.path.join(VIS, "08_uncertainty_calibration.png"), dpi=120)
        plt.close(fig)

    # 7. frames ablation
    if os.path.exists(os.path.join(MET, "frames.csv")):
        fr = list(csv.DictReader(open(os.path.join(MET, "frames.csv"))))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, k in zip(axes, ("edge_f1", "cn_lpips", "grad_corr")):
            for model in FRAME_MODELS:
                rs = [r for r in fr if r["model"] == model]
                ax.plot([int(r["dates"]) for r in rs], [float(r[k]) for r in rs], "-o", label=LABEL[model])
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 2, 4, 8])
            ax.set_xticklabels(["1", "2", "4", "8"])
            ax.set_xlabel("Sentinel-2 dates given to the model")
            ax.set_title(k + (" (lower better)" if k == "cn_lpips" else ""))
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.suptitle("What the multi-date input buys (same tiles, same weights)")
        fig.tight_layout()
        fig.savefig(os.path.join(VIS, "09_dates_ablation.png"), dpi=120)
        plt.close(fig)

    # 8. data audit
    meta = list(csv.DictReader(open(os.path.join(MET, "data_metadata.csv"))))
    gaps = np.load(os.path.join(MET, "s2_gap_days.npy"))
    places_csv = list(csv.DictReader(open(os.path.join(MET, "data_places.csv"))))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for r in places_csv:
        test = r["role"] == "test"
        axes[0].scatter(float(r["lon"]), float(r["lat"]), marker="*" if test else "o", s=120 if test else 30,
                        color="red" if test else "#1f6f3f")
        axes[0].annotate(r["place"], (float(r["lon"]), float(r["lat"])), fontsize=5.5, xytext=(2, 2),
                         textcoords="offset points")
    axes[0].set_title("55 training places (green) and 7 held-out test places (red stars)")
    axes[0].set_xlabel("longitude")
    axes[0].set_ylabel("latitude")
    axes[0].set_aspect("equal")
    axes[0].grid(alpha=0.3)
    axes[1].hist(np.clip(gaps, 0, 150), bins=50, color="#3a64a8")
    axes[1].set_title(f"days between the reference capture and the Sentinel-2 dates\n(per tile, median over its 8 "
                      f"dates; median {np.median(gaps):.0f} d, 90% within {np.percentile(gaps, 90):.0f} d)")
    axes[1].set_xlabel("days")
    rs = sorted(meta, key=lambda r: float(r["res_m_median"]))
    axes[2].barh(range(len(rs)), [float(r["res_m_median"]) for r in rs],
                 color=["red" if r["role"] == "test" else "#1f6f3f" for r in rs])
    axes[2].set_yticks(range(len(rs)))
    axes[2].set_yticklabels([r["place"] for r in rs], fontsize=5)
    axes[2].set_title("reference resolution per place (median, metres)")
    axes[2].axvline(1.0, color="k", ls=":", lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(VIS, "10_data_audit.png"), dpi=120)
    plt.close(fig)
    print(f"  wrote {VIS}/")


if __name__ == "__main__":
    cmds = {"audit": audit, "score": score, "summary": summarise_scores, "uncertainty": uncertainty,
            "frames": frames, "figures": figures}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    os.makedirs(MET, exist_ok=True)
    os.makedirs(VIS, exist_ok=True)
    cmds[sys.argv[1]](*([sys.argv[2:]] if len(sys.argv) > 2 else []))
