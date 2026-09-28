"""Condense the project's data folder (images left out) into the CSVs in this folder.

    python build_metadata.py <path to the project's data folder>

Writes, next to this script:
  reference_tiles.csv      one row per ArcGIS World Imagery tile: place, split, tile position, capture date, sensor,
                           provider, source resolution and positional accuracy (from the World Imagery citation
                           layer), and whether the collector dropped it as mostly water
  sentinel2_stacks.csv     one row per 8-date Sentinel-2 L2A stack: the reference date it was matched to, the 8 dates,
                           how many are block-shared, and the gap in days between each date and the reference
  build_report.csv         the training-set build: every tile, its split (train / val) or the filter that removed it
  folder_structure.md      the data folder tree with file counts
"""
import csv
import glob
import json
import math
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))


def lonlat(x, y, z=17):
    n = 2 ** z
    lon = (x + 0.5) / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 0.5) / n))))
    return round(lon, 6), round(lat, 6)


def read_ref(args):
    path, split = args
    d = json.load(open(path, encoding="utf-8"))
    hr, t = d.get("hr", {}), d["tile"]
    lon, lat = lonlat(t["x"], t["y"], t["zoom"])
    return {"split": split, "place": d.get("city"), "tile_id": d["tile_id"], "x": t["x"], "y": t["y"],
            "lon": lon, "lat": lat, "capture_date": hr.get("acquisition_date"), "platform": hr.get("platform"),
            "provider": hr.get("provider"), "source": hr.get("source"), "resolution_m": hr.get("resolution_m"),
            "accuracy_m": hr.get("accuracy_m")}


def read_s2(args):
    path, split = args
    d = json.load(open(path, encoding="utf-8"))
    ref = d.get("arcgis_date") or d.get("hr_date")      # the first 15 places call it hr_date
    gaps = [abs((date.fromisoformat(s) - date.fromisoformat(ref)).days) for s in d["dates"]] if ref else []
    return {"split": split, "place": os.path.basename(os.path.dirname(path)), "tile_id": d["tile_id"],
            "collection": d.get("collection"), "reference_date": ref, "n_dates": len(d["dates"]),
            "dates": " ".join(d["dates"]), "n_block_shared": sum(d.get("shared", [])) if "shared" in d else "",
            "gap_days_min": min(gaps) if gaps else "", "gap_days_median": sorted(gaps)[len(gaps) // 2] if gaps else "",
            "gap_days_max": max(gaps) if gaps else ""}


def write(name, rows):
    with open(os.path.join(HERE, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def tree(root):
    """Directory tree with file counts; per-tile folders are summarised, not listed."""
    lines = ["# Data folder structure", "", "Image files are not included in this repository; this is the layout the "
             "pipeline creates (`code/pipeline.py`). Counts are from the final build.", "", "```text"]

    def walk(d, depth, max_list):
        if os.path.basename(d) == "_state":
            lines.append(f"{'  ' * depth}_state/   (collector cache: catalogue searches and downloaded frames)")
            return
        subs = sorted(e for e in os.listdir(d) if os.path.isdir(os.path.join(d, e)))
        files = [e for e in os.listdir(d) if os.path.isfile(os.path.join(d, e))]
        kinds = Counter(os.path.splitext(f)[1] or f for f in files)
        name = os.path.basename(d) or d
        desc = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items()))
        lines.append(f"{'  ' * depth}{name}/" + (f"   ({desc})" if desc else ""))
        if len(subs) > max_list:
            for s in subs[:2]:
                walk(os.path.join(d, s), depth + 1, 3)
            lines.append(f"{'  ' * (depth + 1)}... {len(subs) - 2} more folders like these")
        else:
            for s in subs:
                walk(os.path.join(d, s), depth + 1, max_list)

    walk(root, 0, 12)
    lines.append("```")
    open(os.path.join(HERE, "folder_structure.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("folder_structure.md")


def main(data):
    jobs = [(p, "train_collection") for p in glob.glob(os.path.join(data, "arcgis", "*", "metadata", "*.json"))]
    jobs += [(p, "test") for p in glob.glob(os.path.join(data, "test", "arcgis", "*", "metadata", "*.json"))]
    with ProcessPoolExecutor(8) as pool:
        refs = list(pool.map(read_ref, jobs, chunksize=500))
    water = set()
    for rej in glob.glob(os.path.join(data, "arcgis", "*", "rejected.txt")) + \
            glob.glob(os.path.join(data, "test", "arcgis", "*", "rejected.txt")):
        water |= {line.split()[0] for line in open(rej) if line.strip()}
    for r in refs:
        r["dropped_as_water"] = r["tile_id"] in water
    refs.sort(key=lambda r: (r["split"], r["place"] or "", r["tile_id"]))
    write("reference_tiles.csv", refs)

    jobs = [(p, "train_collection") for p in glob.glob(os.path.join(data, "sentinel2_l2a", "*", "*.json"))]
    jobs += [(p, "test_inseason") for p in glob.glob(os.path.join(data, "test", "sentinel2_l2a", "*", "*.json"))]
    jobs += [(p, "test_offseason") for p in
             glob.glob(os.path.join(data, "test", "sentinel2_l2a_offseason", "*", "*.json"))]
    jobs = [j for j in jobs if f"{os.sep}_state{os.sep}" not in j[0]]      # collector caches, not tiles
    with ProcessPoolExecutor(8) as pool:
        s2 = [r for r in pool.map(read_s2, jobs, chunksize=500)]
    s2.sort(key=lambda r: (r["split"], r["place"], r["tile_id"]))
    write("sentinel2_stacks.csv", s2)

    report = os.path.join(data, "training", "report.csv")
    if os.path.exists(report):
        write("build_report.csv", list(csv.DictReader(open(report, encoding="utf-8"))))
    tree(data)


if __name__ == "__main__":
    main(sys.argv[1])
