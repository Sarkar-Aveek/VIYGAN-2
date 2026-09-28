"""
Step 4: builds the train/val layout from the collected data, filters out bad pairs, keeps every tile that passes
(or, with --target N, trims to N with URBAN_SHARE of it from cities), and makes a validation split.

Output (hard links; read by the configs in configs/):
    data/training/{train,val}/sentinel2/{tile}/tci.png     8 L2A frames, [8*32, 32, 3]
    data/training/{train,val}/sentinel2/{tile}/dates.json  the date of each frame
    data/training/{train,val}/arcgis/{tile}/rgb.png        128x128 ArcGIS target
    data/training/report.csv                               every tile: place, kind, split or rejection reason

Filters:
    bad_s2_shape   Sentinel-2 PNG is not [NUM_IMAGES * 32, 32, 3].
    hr_black       HR image contains pure black pixels.
    water          HR image is mostly textureless (open water).
    mismatch       HR and the Sentinel-2 median disagree (land change between dates, snow, HR cloud):
                   correlation between the HR image at 32x32 and the median S2 frame, both grayscale.
    seen_cap       a place India 20k trained on (agc.SEEN) beyond its SEEN_CAP tiles nearest the centre.
    over_target    more tiles passed than needed: cities fill URBAN_SHARE of TARGET_TILES and the other places
                   the rest, each place an equal share within its group, nearest its centre first.

Validation split: VAL_TILES (50) tiles, taken in turn from the places India 20k never saw, in a fixed pseudo-random
order; everything else trains. Val only watches training: the held-out test places (collectors/arcgis_collector.py TEST)
are the real check, collected separately and never built here.

The data/training folder is deleted and rebuilt on every run; the collected data is never touched.
    python collectors/build_training_data.py [--dry-run] [--target N]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bootstrap  # noqa: E402,F401

import csv  # noqa: E402
import shutil  # noqa: E402
import hashlib  # noqa: E402
import argparse  # noqa: E402
from collections import defaultdict  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from collectors import arcgis_collector as agc  # noqa: E402

ARCGIS_DIR = agc.OUTPUT_DIR
SENTINEL_DIR = os.path.join("data", "sentinel2_l2a")
OUTPUT_DIR = os.path.join("data", "training")
TARGET_TILES = None         # keep every tile that passes; --target N trims to N (URBAN_SHARE from cities)
URBAN_SHARE = 0.42          # of the kept tiles; spread over all the cities in agc.TRAIN

NUM_IMAGES = 8
S2_SIZE = 32

# Water: fraction of 8x8 HR patches whose std is below FLAT_PATCH_STD.
FLAT_PATCH_STD = 3.0
WATER_FLAT_FRAC = 0.8

# Mismatch: correlation between the HR image (32x32 grayscale) and the median S2 frame.
MIN_S2_HR_CORR = 0.2
MIN_HR_STRUCTURE = 4.0      # skip the mismatch test when the HR image has almost no structure

VAL_TILES = 50              # only watches training; the held-out test places are the real check

COLUMNS = ["train", "val", "water", "mismatch", "hr_black", "bad_s2_shape", "duplicate_tile_id", "seen_cap",
           "over_target"]


def kind(place):
    return agc.TRAIN[place][2] if place in agc.TRAIN else "unknown"


def correlation(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def check_tile(job):
    """Returns (place, tile_id, x, y, reason or None, stats)."""
    place, tile_id = job
    _, x, y = tile_id.split("_")
    x, y = int(x), int(y)
    s2 = np.array(Image.open(os.path.join(SENTINEL_DIR, place, f"{tile_id}.png")))
    if s2.shape != (NUM_IMAGES * S2_SIZE, S2_SIZE, 3):
        return place, tile_id, x, y, "bad_s2_shape", {}
    hr_img = Image.open(os.path.join(ARCGIS_DIR, place, "images", f"{tile_id}.png")).convert("RGB")
    hr = np.array(hr_img).astype(np.float32)
    if (hr.sum(axis=-1) == 0).any():
        return place, tile_id, x, y, "hr_black", {}

    gray = hr.mean(axis=-1)
    h, w = gray.shape
    patches = gray[:h - h % 8, :w - w % 8].reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)
    flat = float((patches.reshape(h // 8, w // 8, 64).std(axis=-1) < FLAT_PATCH_STD).mean())

    hr_small = np.array(hr_img.convert("L").resize((S2_SIZE, S2_SIZE), Image.BOX), np.float32)
    s2_median = np.median(s2.reshape(-1, S2_SIZE, S2_SIZE, 3).astype(np.float32).mean(axis=-1), axis=0)
    corr = correlation(hr_small, s2_median)
    stats = {"flat": round(flat, 3), "corr": round(corr, 3)}

    if flat > WATER_FLAT_FRAC:
        return place, tile_id, x, y, "water", stats
    if hr_small.std() >= MIN_HR_STRUCTURE and corr < MIN_S2_HR_CORR:
        return place, tile_id, x, y, "mismatch", stats
    return place, tile_id, x, y, None, stats


def round_robin(kept, target):
    """kept: list of (place, tile_id, x, y). Equal shares per place (a place with fewer tiles gives its unused
    share to the others), nearest the place centre first. Returns the set of tile ids to keep."""
    by_place = defaultdict(list)
    for place, tile_id, x, y in kept:
        by_place[place].append((agc.distance_rank(place, x, y), tile_id))
    queues = {p: [t for _, t in sorted(v)] for p, v in by_place.items()}
    chosen, i = set(), 0
    while len(chosen) < min(target, len(kept)):
        for place, q in queues.items():
            if i < len(q) and len(chosen) < target:
                chosen.add(q[i])
        i += 1
    return chosen


def cap_seen(kept, cap=agc.SEEN_CAP):
    """Tile ids to drop: all but the `cap` tiles nearest the centre of each place in agc.SEEN."""
    by_place = defaultdict(list)
    for place, tile_id, x, y in kept:
        if place in agc.SEEN:
            by_place[place].append((agc.distance_rank(place, x, y), tile_id))
    return {t for v in by_place.values() for _, t in sorted(v)[cap:]}


def trim_to_target(kept, target, urban_share=URBAN_SHARE):
    """Cities get urban_share of the target, the other places the rest; if one group runs short, the other
    takes up the difference."""
    urban = [k for k in kept if kind(k[0]) == "urban"]
    other = [k for k in kept if kind(k[0]) != "urban"]
    want_urban = min(round(target * urban_share), len(urban))
    want_other = min(target - want_urban, len(other))
    want_urban = min(target - want_other, len(urban))
    return round_robin(urban, want_urban) | round_robin(other, want_other)


def stable_order(key):
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def choose_val_tiles(kept, n=VAL_TILES):
    """kept: list of (place, tile_id, x, y). Returns n tile ids for val: taken in turn from each place not in
    agc.SEEN (India 20k never saw those, so val doesn't favour the fine-tuned start), in a fixed pseudo-random
    order within each place."""
    queues = defaultdict(list)
    for place, tile_id, _, _ in kept:
        if place not in agc.SEEN:
            queues[place].append(tile_id)
    queues = [sorted(q, key=stable_order) for _, q in sorted(queues.items())]
    chosen, i = set(), 0
    while len(chosen) < min(n, sum(map(len, queues))):
        for q in queues:
            if i < len(q) and len(chosen) < n:
                chosen.add(q[i])
        i += 1
    return chosen


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)  # e.g. a filesystem without hard links


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="only print what would be kept/rejected")
    parser.add_argument("--target", type=int, default=TARGET_TILES, help=f"tiles to keep (default {TARGET_TILES})")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    jobs = []
    missing_hr = 0
    for place in sorted(os.listdir(SENTINEL_DIR)):
        place_dir = os.path.join(SENTINEL_DIR, place)
        if place.startswith("_") or not os.path.isdir(place_dir) or place not in agc.TRAIN:
            continue
        for fn in sorted(os.listdir(place_dir)):
            if not fn.endswith(".png"):
                continue
            if not os.path.exists(os.path.join(ARCGIS_DIR, place, "images", fn)):
                missing_hr += 1
                continue
            jobs.append((place, fn[:-4]))

    print(f"Checking {len(jobs)} tile pairs with {args.workers} workers...")
    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(check_tile, jobs, chunksize=64))

    # The same tile id in two places would overwrite each other in the flat layout.
    seen = set()
    for i, (place, tile_id, x, y, reason, stats) in enumerate(results):
        if tile_id in seen:
            results[i] = (place, tile_id, x, y, "duplicate_tile_id", stats)
        seen.add(tile_id)

    passed = [(p, t, x, y) for p, t, x, y, reason, _ in results if reason is None]
    capped = cap_seen(passed)
    uncapped = [k for k in passed if k[1] not in capped]
    keep = {k[1] for k in uncapped} if args.target is None else trim_to_target(uncapped, args.target)
    results = [(p, t, x, y, reason or ("seen_cap" if t in capped else None if t in keep else "over_target"), s)
               for p, t, x, y, reason, s in results]
    kept = [(p, t, x, y) for p, t, x, y, reason, _ in results if reason is None]
    val_tiles = choose_val_tiles(kept)

    rows = []
    counts = defaultdict(lambda: defaultdict(int))
    for place, tile_id, x, y, reason, stats in results:
        if reason is None:
            split = "val" if tile_id in val_tiles else "train"
        else:
            split = "rejected"
        counts[place][reason or split] += 1
        rows.append({"city": place, "kind": kind(place), "tile_id": tile_id, "split": split, "reason": reason or "",
                     "flat": stats.get("flat", ""), "corr": stats.get("corr", "")})

    print(f"\n{'place':22} {'kind':9} " + " ".join(f"{c[:11]:>11}" for c in COLUMNS))
    totals, by_kind = defaultdict(int), defaultdict(int)
    for place in sorted(counts):
        print(f"{place:22} {kind(place):9} " + " ".join(f"{counts[place][c]:11d}" for c in COLUMNS))
        for c in COLUMNS:
            totals[c] += counts[place][c]
        by_kind[kind(place)] += counts[place]["train"] + counts[place]["val"]
    print(f"{'TOTAL':32} " + " ".join(f"{totals[c]:11d}" for c in COLUMNS))
    n_kept = totals["train"] + totals["val"]
    print("\nkept by kind: " + ", ".join(f"{k} {v} ({v / max(n_kept, 1):.0%})" for k, v in sorted(by_kind.items())))
    if missing_hr:
        print(f"(skipped {missing_hr} Sentinel-2 files with no matching HR image)")
    if args.target is not None and n_kept < args.target:
        print(f"NOTE: {n_kept} tiles passed, fewer than the target {args.target}; raise the tile counts in "
              "collectors/arcgis_collector.py and collect again for the full size.")

    if args.dry_run:
        print("\nDry run: nothing written.")
        return

    if os.path.isdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    for row in rows:
        if row["split"] == "rejected":
            continue
        root = os.path.join(OUTPUT_DIR, row["split"])
        place, tile_id = row["city"], row["tile_id"]
        link(os.path.join(SENTINEL_DIR, place, f"{tile_id}.png"), os.path.join(root, "sentinel2", tile_id, "tci.png"))
        dates = os.path.join(SENTINEL_DIR, place, f"{tile_id}.json")
        if os.path.exists(dates):
            link(dates, os.path.join(root, "sentinel2", tile_id, "dates.json"))
        link(os.path.join(ARCGIS_DIR, place, "images", f"{tile_id}.png"), os.path.join(root, "arcgis", tile_id, "rgb.png"))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "report.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["city", "kind", "tile_id", "split", "reason", "flat", "corr"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {totals['train']} train and {totals['val']} val tiles to {OUTPUT_DIR}/ "
          f"(details in {OUTPUT_DIR}/report.csv).")


if __name__ == "__main__":
    sys.exit(main())
