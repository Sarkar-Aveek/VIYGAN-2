"""
Steps 1-2: the places, their ArcGIS World Imagery targets, and the targets' capture metadata.

Places: TRAIN (55 places; 23 cities hold ~42% of the kept tiles, the rest covers farmland, forest, hills and tea,
coast and backwaters, a river island, rocky and arid land and Himalayan snow) and TEST (7 places > 50 km from every
training place; the evaluation runs on these). One list, so the collectors, the build step and the evaluation agree.

--arcgis    zoom-17 tiles (128x128 px, ~2.4 m) nearest each place's centre, skipping mostly-water tiles, until the
            place's count is saved (tiles_for: cities 2,300, other places 2,450, test places 400; headroom for what
            later steps drop). -> {root}/{place}/images/{tile}.png, metadata/{tile}.json, rejected.txt (water)
--metadata  capture date, provider and resolution per tile from the World Imagery citation layer (the Sentinel-2
            collector needs hr.acquisition_date and skips sources coarser than 1 m)
No step flag runs both. Root: data/arcgis, or data/test/arcgis with --test. Every step resumes: saved tiles and
tiles that already have metadata are skipped, so after a failure run the same command again.

    python collectors/arcgis_collector.py [--test] [--places NAME ...] [--arcgis] [--metadata] [--dry-run]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bootstrap  # noqa: E402,F401

import json  # noqa: E402
import math  # noqa: E402
import argparse  # noqa: E402
from io import BytesIO  # noqa: E402
from threading import Lock  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402
import requests  # noqa: E402
from PIL import Image  # noqa: E402

# ============================================================
# PLACES   name: (lon, lat, kind)
# ============================================================
URBAN_TILES = 2300          # ArcGIS tiles collected per city (cities lose ~15% to the filters; ~7% headroom)
OTHER_TILES = 2450          # per non-urban place (~10% lost)
TEST_TILES = 400            # per test place: a ~20 x 20-tile square (~5.6 km) around the centre

TRAIN = {
    # --- cities (India ESRGAN's 7, the 5 comparison cities, 11 new) ---
    "howrah": (88.3246, 22.5958, "urban"),
    "kota": (75.8390, 25.1815, "urban"),
    "kanpur": (80.3319, 26.4499, "urban"),
    "ahmedabad": (72.5714, 23.0225, "urban"),
    "pune_core": (73.8567, 18.5204, "urban"),
    "chennai_dense": (80.2707, 13.0827, "urban"),
    "surat": (72.8311, 21.1702, "urban"),
    "bhopal": (77.4126, 23.2599, "urban"),
    "jaipur": (75.7873, 26.9124, "urban"),
    "bengaluru": (77.5946, 12.9716, "urban"),
    "nagpur": (79.0882, 21.1458, "urban"),
    "guwahati": (91.7362, 26.1445, "urban"),
    "delhi_old": (77.2300, 28.6560, "urban"),          # Chandni Chowk: very dense old core
    "mumbai": (72.8550, 19.0430, "urban"),             # Dharavi / Sion: informal + dense
    "lucknow": (80.9462, 26.8467, "urban"),
    "patna": (85.1376, 25.5941, "urban"),
    "indore": (75.8577, 22.7196, "urban"),
    "chandigarh": (76.7794, 30.7333, "urban"),         # planned grid
    "visakhapatnam": (83.2185, 17.6868, "urban"),      # port city between hills and sea
    "coimbatore": (76.9558, 11.0168, "urban"),
    "madurai": (78.1198, 9.9252, "urban"),
    "raipur": (81.6296, 21.2514, "urban"),
    "srinagar": (74.7973, 34.0837, "urban"),           # Himalayan valley city
    # --- farmland ---
    "punjab_agri_1": (75.8573, 30.9010, "farmland"),   # Ludhiana outskirts
    "punjab_agri_2": (74.8723, 31.6340, "farmland"),   # Amritsar outskirts
    "up_agri": (78.0081, 27.8974, "farmland"),         # Aligarh
    "haryana_agri": (76.7821, 30.3165, "farmland"),    # Ambala
    "hisar_agri": (75.7217, 29.1492, "farmland"),      # Haryana wheat
    "bardhaman_agri": (87.8615, 23.2324, "farmland"),  # West Bengal paddy
    "guntur_agri": (80.4365, 16.3067, "farmland"),     # Andhra chilli / cotton
    "latur_agri": (76.5604, 18.4088, "farmland"),      # Deccan soybean / sugarcane
    "malwa_agri": (75.7800, 23.1800, "farmland"),      # Ujjain plateau
    "cuttack_agri": (85.9000, 20.4000, "farmland"),    # Mahanadi delta rice
    "kolhapur_agri": (74.2400, 16.7050, "farmland"),   # sugarcane
    # --- dry, desert, rocky ---
    "jaisalmer_barren": (70.9161, 26.9157, "desert"),
    "bikaner_barren": (73.3119, 28.0229, "desert"),
    "kutch_barren": (69.8597, 23.7337, "desert"),      # salt flats
    "jodhpur_arid": (72.9000, 26.3000, "desert"),      # Thar edge
    "hampi_rocky": (76.4600, 15.3350, "rocky"),        # granite boulders
    "anantapur_arid": (77.6000, 14.6800, "rocky"),     # semi-arid scrub
    "ladakh_barren": (77.5771, 34.1526, "mountain"),   # cold desert
    # --- forest, hills, tea ---
    "pachmarhi_forest": (78.4336, 22.4675, "forest"),  # Satpura hills
    "similipal_forest": (86.3500, 21.8000, "forest"),
    "agumbe_ghats": (75.0880, 13.5030, "forest"),      # Western Ghats rainforest
    "kanha_forest": (80.6100, 22.3350, "forest"),
    "bastar_forest": (81.9500, 19.0700, "forest"),
    "darjeeling_tea": (88.2627, 27.0360, "hills"),     # steep tea slopes
    "munnar_tea": (77.0595, 10.0889, "hills"),
    # --- water's edge ---
    "alleppey_backwaters": (76.3388, 9.4981, "coast"),
    "goa_coast": (73.8000, 15.5000, "coast"),
    "majuli_river": (94.1600, 26.9500, "river"),       # Brahmaputra river island
    # --- snow: Himalayan places whose ArcGIS image is from winter (checked 2026-09-18, 0.31-0.46 m) ---
    "drass_snow": (75.7550, 34.4300, "snow"),          # Kargil district, ArcGIS 2023-12-05
    "manali_solang": (77.1560, 32.3160, "snow"),       # Himachal, 2025-01-27
    "auli_joshimath": (79.5650, 30.5300, "snow"),      # Uttarakhand, 2025-01-25
    "tawang_snow": (91.8590, 27.5860, "snow"),         # Arunachal, 2023-01 / 2023-11
}

# The 15 places India 20k (the starting model) was trained on, every tile for ~8 passes. The build keeps only the
# SEEN_CAP tiles nearest each centre, so targets the model has already fitted stay a small share (~15%) of the set.
SEEN = {"howrah", "kota", "kanpur", "ahmedabad", "pune_core", "chennai_dense", "surat", "punjab_agri_1",
        "punjab_agri_2", "up_agri", "haryana_agri", "jaisalmer_barren", "bikaner_barren", "kutch_barren",
        "ladakh_barren"}
SEEN_CAP = 1000

TEST = {
    # the four sites of the earlier comparison, plus three new kinds
    "hyderabad_charminar": (78.4747, 17.3616, "urban"),
    "varanasi_ghats": (83.0104, 25.3109, "urban"),
    "thanjavur_paddy": (79.1378, 10.7870, "farmland"),
    "barmer_desert": (71.3967, 25.7521, "desert"),
    "wayanad_forest": (76.1320, 11.6854, "forest"),
    "shillong_hills": (91.8933, 25.5788, "hills"),
    "lachung_sikkim": (88.7450, 27.6890, "snow"),      # North Sikkim, ArcGIS 2025-03 / 2025-11
}


def tiles_for(name):
    if name in TEST:
        return TEST_TILES
    return URBAN_TILES if TRAIN[name][2] == "urban" else OTHER_TILES


# ============================================================
# CONFIGURATION
# ============================================================
OUTPUT_DIR = os.path.join("data", "arcgis")
TEST_OUTPUT_DIR = os.path.join("data", "test", "arcgis")
ARCGIS_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
CITATION_URL = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/4/query"

ZOOM = 17
HR_WIDTH = HR_HEIGHT = 128
MAX_WORKERS = 32
# name: (lon, lat); set to TEST by --test
PLACES = {name: (lon, lat) for name, (lon, lat, _kind) in TRAIN.items()}

WEB_MERCATOR_HALF_WORLD = 20037508.342789244
WATER_STD = 12.0            # 8-bit RGB std below this = open water


# ============================================================
# TILE GEOMETRY
# ============================================================
def lonlat_to_tile(lon, lat, zoom):
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2 ** zoom
    return int((lon + 180.0) / 360.0 * n), int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)


def tile_xy_to_bbox(x, y, zoom):
    tile_m = 2 * WEB_MERCATOR_HALF_WORLD / (2 ** zoom)
    min_x = -WEB_MERCATOR_HALF_WORLD + x * tile_m
    max_y = WEB_MERCATOR_HALF_WORLD - y * tile_m
    return (min_x, max_y - tile_m, min_x + tile_m, max_y)


def webmercator_to_lonlat(x, y):
    lon = x / WEB_MERCATOR_HALF_WORLD * 180.0
    lat = math.degrees(2 * math.atan(math.exp(y / WEB_MERCATOR_HALF_WORLD * math.pi)) - math.pi / 2)
    return lon, lat


def tiles_by_distance(lon, lat, count, zoom=ZOOM):
    """At least `count` tiles around (lon, lat), nearest first (a disc, grown in rings)."""
    cx, cy = lonlat_to_tile(lon, lat, zoom)
    r = math.ceil(math.sqrt(count / math.pi)) + 2
    cells = [(cx + dx, cy + dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)]
    cells.sort(key=lambda t: ((t[0] - cx) ** 2 + (t[1] - cy) ** 2, t))
    return cells


def distance_rank(place, x, y):
    lon, lat = PLACES[place]
    cx, cy = lonlat_to_tile(lon, lat, ZOOM)
    return (x - cx) ** 2 + (y - cy) ** 2


def build_metadata(place, x, y, zoom):
    min_x, min_y, max_x, max_y = tile_xy_to_bbox(x, y, zoom)
    min_lon, min_lat = webmercator_to_lonlat(min_x, min_y)
    max_lon, max_lat = webmercator_to_lonlat(max_x, max_y)
    return {
        "tile_id": f"{zoom}_{x}_{y}",
        "city": place,
        "country": "India",
        "tile": {"x": x, "y": y, "zoom": zoom},
        "crs": "EPSG:3857",
        "bbox_3857": [min_x, min_y, max_x, max_y],
        "bbox_wgs84": [min_lon, min_lat, max_lon, max_lat],
        "hr": {"source": "ArcGIS World Imagery", "width": HR_WIDTH, "height": HR_HEIGHT, "channels": 3,
               "format": "RGB PNG"},
    }


# ============================================================
# STEP --arcgis: TILES
# ============================================================
def is_mostly_water(image):
    return float(np.std(np.array(image))) < WATER_STD


def download_tile(city, x, y, zoom):
    """Saves {OUTPUT_DIR}/{city}/images/{tile}.png and metadata. Returns (tile_id, status)."""
    tile_id = f"{zoom}_{x}_{y}"
    image_path = os.path.join(OUTPUT_DIR, city, "images", f"{tile_id}.png")
    metadata_path = os.path.join(OUTPUT_DIR, city, "metadata", f"{tile_id}.json")
    if os.path.exists(image_path):
        return tile_id, "exists"

    min_x, min_y, max_x, max_y = tile_xy_to_bbox(x, y, zoom)
    params = {"bbox": f"{min_x},{min_y},{max_x},{max_y}", "bboxSR": 3857, "imageSR": 3857,
              "size": f"{HR_WIDTH},{HR_HEIGHT}", "format": "png", "pixelType": "U8",
              "interpolation": "RSP_BilinearInterpolation", "compressionQuality": 100, "transparent": "false",
              "f": "image"}
    response = requests.get(ARCGIS_URL, params=params, timeout=60)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")
    if image.size != (HR_WIDTH, HR_HEIGHT):
        raise RuntimeError(f"Wrong image size: {image.size}")
    if is_mostly_water(image):
        return tile_id, "water_rejected"

    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    image.save(image_path, "PNG")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(build_metadata(city, x, y, zoom), f, indent=2)
    return tile_id, "downloaded"


def collect_place(place, lon, lat, target):
    rejected_path = os.path.join(OUTPUT_DIR, place, "rejected.txt")
    rejected = set()
    if os.path.exists(rejected_path):
        with open(rejected_path) as f:
            rejected = {line.strip() for line in f if line.strip()}
    lock = Lock()
    counts = {"downloaded": 0, "exists": 0, "water_rejected": 0, "failed": 0}

    def one(xy):
        try:
            tile_id, status = download_tile(place, xy[0], xy[1], ZOOM)
        except Exception as e:
            print(f"  {place} {ZOOM}_{xy[0]}_{xy[1]} FAILED: {e}")
            status, tile_id = "failed", None
        with lock:
            counts[status] += 1
            if status == "water_rejected":
                rejected.add(tile_id)
        return status in ("downloaded", "exists")

    candidates = [c for c in tiles_by_distance(lon, lat, target * 3)
                  if f"{ZOOM}_{c[0]}_{c[1]}" not in rejected]
    saved, pos = 0, 0
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        while saved < target and pos < len(candidates):
            batch = candidates[pos:pos + target - saved]
            pos += len(batch)
            saved += sum(ex.map(one, batch))
            print(f"  {place}: {saved}/{target} tiles")

    os.makedirs(os.path.dirname(rejected_path), exist_ok=True)
    with open(rejected_path, "w") as f:
        f.write("\n".join(sorted(rejected)))
    print(f"{place}: {saved} tiles | new {counts['downloaded']} | existing {counts['exists']} | "
          f"water {counts['water_rejected']} (plus {len(rejected) - counts['water_rejected']} known) | "
          f"failed {counts['failed']}")
    return saved


def collect_tiles(names):
    """Returns the places that ended short of their tile count."""
    return [n for n in names if collect_place(n, *PLACES[n], tiles_for(n)) < tiles_for(n)]


# ============================================================
# STEP --metadata: CAPTURE DATE / RESOLUTION
# ============================================================
def choose_best_citation(citations):
    """The finest-resolution citation (SRC_RES 99999 means unknown)."""
    valid = []
    for c in citations:
        try:
            res = float(c.get("SRC_RES"))
        except (TypeError, ValueError):
            continue
        if res < 99999:
            valid.append((res, c))
    if not valid:
        return citations[0] if citations else None
    return min(valid, key=lambda v: v[0])[1]


def get_citation(metadata_path):
    """Returns True when the tile's metadata has citations (fetched now or before)."""
    try:
        with open(metadata_path) as f:
            data = json.load(f)
        if "arcgis_citations" in data:
            return True

        xmin, ymin, xmax, ymax = data["bbox_3857"]
        x, y = (xmin + xmax) / 2, (ymin + ymax) / 2
        params = {"where": "1=1", "geometry": f"{x},{y}", "geometryType": "esriGeometryPoint", "inSR": "3857",
                  "spatialRel": "esriSpatialRelIntersects", "outFields": "*", "returnGeometry": "false", "f": "json"}
        response = requests.get(CITATION_URL, params=params, timeout=30)
        response.raise_for_status()
        result = response.json()
        if "error" in result:
            print("ERROR:", metadata_path, result["error"])
            return False
        citations = [feat.get("attributes", {}) for feat in result.get("features", [])]
        if not citations:
            print("NO METADATA:", metadata_path)   # stored as checked; the tile has no date and is skipped later

        data["arcgis_citations"] = citations
        data["arcgis_metadata_query"] = {"layer": "World Imagery Citations", "layer_url": CITATION_URL.rsplit("/", 1)[0],
                                         "query_x_3857": x, "query_y_3857": y}
        best = choose_best_citation(citations)
        if best:
            hr = data["hr"]
            hr["provider"] = best.get("NICE_NAME")
            hr["source"] = best.get("NICE_DESC")
            hr["platform"] = best.get("SRC_DESC")
            hr["resolution_m"] = best.get("SRC_RES")
            hr["accuracy_m"] = best.get("SRC_ACC")
            hr["acquisition_date_raw"] = best.get("SRC_DATE")
            date = str(best.get("SRC_DATE") or "")
            if len(date) == 8:   # YYYYMMDD
                hr["acquisition_date"] = f"{date[:4]}-{date[4:6]}-{date[6:]}"

        with open(metadata_path, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print("FAILED:", metadata_path, "|", e)
        return False


def collect_metadata(names):
    """Returns the number of tiles still without metadata."""
    files = []
    for place in names:
        meta_dir = os.path.join(OUTPUT_DIR, place, "metadata")
        if os.path.isdir(meta_dir):
            files += [os.path.join(meta_dir, fn) for fn in sorted(os.listdir(meta_dir)) if fn.endswith(".json")]
    print(f"Metadata: {len(files)} tiles; using {MAX_WORKERS} workers")

    ok = failed = 0
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futures = [ex.submit(get_citation, path) for path in files]
        for i, fut in enumerate(as_completed(futures), 1):
            if fut.result():
                ok += 1
            else:
                failed += 1
            if i % 1000 == 0:
                print(f"Progress: {i}/{len(files)} | ok {ok} | failed {failed}")
    print(f"Metadata done: {ok} with metadata, {failed} failed")
    return failed


def main():
    global OUTPUT_DIR, PLACES
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test", action="store_true", help="the held-out test places, into data/test/arcgis")
    p.add_argument("--places", nargs="+", help="only these places")
    p.add_argument("--arcgis", action="store_true", help="only the tile download step")
    p.add_argument("--metadata", action="store_true", help="only the metadata step")
    p.add_argument("--dry-run", action="store_true", help="print the plan only")
    args = p.parse_args()
    if args.test:
        OUTPUT_DIR = TEST_OUTPUT_DIR
        PLACES = {name: (lon, lat) for name, (lon, lat, _kind) in TEST.items()}
    both = not (args.arcgis or args.metadata)

    names = args.places or list(PLACES)
    unknown = [n for n in names if n not in PLACES]
    if unknown:
        sys.exit(f"Unknown place(s): {', '.join(unknown)}")
    print(f"{len(names)} places, {sum(tiles_for(n) for n in names)} tiles in all -> {OUTPUT_DIR}; zoom {ZOOM}")
    if args.dry_run:
        for name in names:
            print(f"  {name:22} lon {PLACES[name][0]:9.4f} lat {PLACES[name][1]:8.4f}  {tiles_for(name)} tiles")
        return

    if args.arcgis or both:
        short = collect_tiles(names)
        if short:
            sys.exit(f"Not enough tiles for: {', '.join(short)} (failed downloads? run again with --arcgis)")
    if args.metadata or both:
        if collect_metadata(names):
            sys.exit("Some tiles have no metadata yet; run again with --metadata (finished tiles are skipped).")


if __name__ == "__main__":
    main()
