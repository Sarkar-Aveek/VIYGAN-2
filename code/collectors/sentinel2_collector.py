"""
Step 3: Sentinel-2 L2A collector for the ArcGIS tiles (Copernicus Data Space / Sentinel Hub).

8 frames per tile of surface reflectance (atmospheric correction applied), scaled like the ESA true-colour
product (reflectance / 0.3558 -> 0-255), so every model in this repo sees L2A.

Shared dates per block: the 8 frames of every tile in a 16x16-tile block come from the same dates wherever those
are clean for the tile (the dates clean for the most tiles win); a tile only falls back to its own dates where it
is cloudy on a shared one. Neighbouring tiles then show the same day's light, so a mosaic in Sentinel-2 colours has
no tile patchwork. The dates of each stack are written next to it.

Quota-friendly design:
  * One Catalog search per (city, HR acquisition date) group instead of one per tile.
  * One Process request per (16x16-tile block, date) with a single-day time range, so each
    request is billed exactly one data sample over a <=512x512 px area.
  * Dates closest to the HR acquisition are fetched first; a block stops early once every
    tile in it has NUM_IMAGES clean frames.
  * Catalog results and downloaded block frames are cached under data/sentinel2/_state, so an
    interrupted run never pays for the same data twice. Tiles with no clean frames are
    recorded in failed.json and skipped on later runs.
  * A shared rate limiter stays under the per-minute limits, and the whole run stops on
    the first sign that the monthly quota is exhausted.

Output: data/sentinel2_l2a/{place}/{tile_id}.png of shape [NUM_IMAGES * 32, 32, 3], and {tile_id}.json with the
date of each frame. --test reads data/test/arcgis and writes data/test/sentinel2_l2a; --test --offseason DAYS
searches around (ArcGIS date + DAYS) instead and writes data/test/sentinel2_l2a_offseason (the "unseen dates" test).

Credentials are read from the CDSE_CLIENT_ID and CDSE_CLIENT_SECRET environment variables (or .env).
    python collectors/sentinel2_collector.py --dry-run [--catalog]
    python collectors/sentinel2_collector.py [--city PLACE] [--max-blocks N] [--retry-failed] [--preview N]
    python collectors/sentinel2_collector.py --test [--offseason 180]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bootstrap  # noqa: E402,F401

import json  # noqa: E402
import time
import argparse
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timedelta
from io import BytesIO
from threading import Lock

import numpy as np
import requests
from PIL import Image, ImageDraw
from requests.adapters import HTTPAdapter

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_ROOT = os.path.join("data", "arcgis")
OUTPUT_DIR = os.path.join("data", "sentinel2_l2a")
OFFSEASON_DAYS = 0             # --offseason: centre the date search this many days after the ArcGIS date


def set_paths(dataset_root, output_dir):
    global DATASET_ROOT, OUTPUT_DIR, STATE_DIR, CATALOG_CACHE_DIR, FRAME_CACHE_DIR, PREVIEW_DIR, FAILED_PATH
    DATASET_ROOT, OUTPUT_DIR = dataset_root, output_dir
    STATE_DIR = os.path.join(OUTPUT_DIR, "_state")
    CATALOG_CACHE_DIR = os.path.join(STATE_DIR, "catalog")
    FRAME_CACHE_DIR = os.path.join(STATE_DIR, "frames")
    PREVIEW_DIR = os.path.join(STATE_DIR, "preview")
    FAILED_PATH = os.path.join(STATE_DIR, "failed.json")


set_paths(DATASET_ROOT, OUTPUT_DIR)

COLLECTION = "sentinel-2-l2a"  # surface reflectance; the evalscript scales it like the true-colour product
TCI_SATURATION = 0.3558        # ESA TCI: reflectance that maps to 255

NUM_IMAGES = 8
IMG_SIZE = 32                  # Sentinel-2 pixels per zoom-17 tile (9.55 m/px)
BLOCK_TILES = 16               # blocks of 16x16 tiles -> at most 512x512 px per request

WINDOW_DAYS = 120              # search +-120 days around the HR acquisition date
MIN_DAYS_GAP = 4
MAX_DATES = 20
MIN_DATES_BEFORE_STOP = 10     # fetch at least this many dates before stopping a block early
CATALOG_MAX_CLOUD = 30         # scene-level cloud cover filter (%)

MAX_HR_RES_M = 1.0             # skip tiles whose ArcGIS source is coarser than this (e.g. 15 m TerraColor)

# Cloud/haze/shadow test: each frame is compared with the tile's per-pixel median over all dates.
#   shift  = mean |frame colour offset| vs the median (thick cloud, shadow, strong haze)
#   struct = mean |difference| after removing that offset (partial clouds, thin haze, shadows)
# Persistent bright ground (snow, salt, sand) is part of the median, so it is not flagged.
CLOUD_SHIFT_MAX = 20           # reject if the uniform colour offset is larger than this (0-255 scale)
CLOUD_STRUCT_MIN = 5.0         # never reject for structure below this
CLOUD_STRUCT_RATIO = 2.5       # reject if struct > ratio * the tile's median struct (and > CLOUD_STRUCT_MIN)

MAX_WORKERS = 6
REQUESTS_PER_MIN = 240         # CDSE limit is 300/min
PU_PER_MIN = 240               # CDSE limit is 300/min

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CATALOG_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

WEB_MERCATOR_HALF_WORLD = 20037508.342789244

EVALSCRIPT = f"""
//VERSION=3
function setup() {{
    return {{
        input: ["B04", "B03", "B02", "dataMask"],
        output: {{ bands: 3, sampleType: "UINT8" }}
    }};
}}
function scale(v) {{
    return Math.max(1, Math.min(255, Math.round(v * 255 / {TCI_SATURATION})));
}}
function evaluatePixel(s) {{
    if (s.dataMask === 0) return [0, 0, 0];
    return [scale(s.B04), scale(s.B03), scale(s.B02)];
}}
"""


# ============================================================
# SHARED STATE
# ============================================================
class RunState:
    def __init__(self):
        self.stop = threading.Event()
        self.stop_reason = None
        self.lock = Lock()
        self.requests = 0
        self.pu = 0.0
        self.cached = 0
        self.consecutive_5xx = 0

    def request_stop(self, reason):
        with self.lock:
            if self.stop_reason is None:
                self.stop_reason = reason
        self.stop.set()

    def count_request(self, pu):
        with self.lock:
            self.requests += 1
            self.pu += pu


class TokenManager:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self._token = None
        self._expires_at = 0
        self._lock = Lock()

    def get_token(self):
        with self._lock:
            if self._token is None or time.time() >= self._expires_at - 120:
                payload = {
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret
                }
                resp = requests.post(TOKEN_URL, data=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                self._token = data["access_token"]
                self._expires_at = time.time() + data.get("expires_in", 3600)
            return self._token

    def invalidate(self):
        with self._lock:
            self._token = None
            self._expires_at = 0


class RateLimiter:
    """Sliding 60 s window over both request count and processing units."""

    def __init__(self, max_requests, max_pu):
        self.max_requests = max_requests
        self.max_pu = max_pu
        self.events = deque()  # (timestamp, pu)
        self.lock = Lock()

    def acquire(self, pu, stop_event):
        while not stop_event.is_set():
            with self.lock:
                now = time.time()
                while self.events and now - self.events[0][0] >= 60:
                    self.events.popleft()
                used_pu = sum(p for _, p in self.events)
                if len(self.events) < self.max_requests and used_pu + pu <= self.max_pu:
                    self.events.append((now, pu))
                    return True
                wait_s = 60 - (now - self.events[0][0]) + 0.05
            time.sleep(min(max(wait_s, 0.05), 1.0))
        return False


def create_session():
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=MAX_WORKERS * 2, pool_maxsize=MAX_WORKERS * 2)
    session.mount("https://", adapter)
    return session


def post_with_rate_limit(session, url, json_payload, token_manager, limiter, state, pu, max_attempts=5):
    """POST with rate limiting and quota protection. Returns a successful response, an
    unsuccessful non-retryable response, or None if the run was stopped."""
    for attempt in range(max_attempts):
        if state.stop.is_set() or not limiter.acquire(pu, state.stop):
            return None

        headers = {
            "Authorization": f"Bearer {token_manager.get_token()}",
            "Content-Type": "application/json"
        }
        try:
            resp = session.post(url, json=json_payload, headers=headers, timeout=60)
        except requests.RequestException as e:
            print(f"  network error ({e}); retrying")
            time.sleep(2 * (attempt + 1))
            continue
        state.count_request(pu)

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait_s = float(retry_after) / 1000.0 if retry_after else 2.0 * (attempt + 1)
            if wait_s > 120:
                state.request_stop(f"429 with Retry-After {wait_s:.0f}s (quota exhausted?): {resp.text[:300]}")
                return None
            time.sleep(wait_s)
            continue

        if resp.status_code == 401:
            token_manager.invalidate()
            continue

        if resp.status_code == 403:
            state.request_stop(f"403 Forbidden (quota exhausted or no access): {resp.text[:300]}")
            return None

        if resp.status_code >= 500:
            with state.lock:
                state.consecutive_5xx += 1
                too_many = state.consecutive_5xx >= 3
            if too_many:
                state.request_stop(f"3 consecutive server errors, last {resp.status_code}: {resp.text[:300]}")
                return None
            time.sleep(3 * (attempt + 1))
            continue

        with state.lock:
            state.consecutive_5xx = 0
        return resp

    state.request_stop(f"request to {url} still failing after {max_attempts} attempts")
    return None


# ============================================================
# FILE HELPERS
# ============================================================
def atomic_write_bytes(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def atomic_write_json(path, obj):
    atomic_write_bytes(path, json.dumps(obj, indent=2).encode("utf-8"))


def save_png(path, array):
    buf = BytesIO()
    Image.fromarray(array).save(buf, "PNG")
    atomic_write_bytes(path, buf.getvalue())


class FailedStore:
    def __init__(self, path):
        self.path = path
        self.lock = Lock()
        self.data = {}
        if os.path.exists(path):
            with open(path, "r") as f:
                self.data = json.load(f)

    def add(self, tile_id, city, reason):
        with self.lock:
            self.data[tile_id] = {"city": city, "reason": reason}
            atomic_write_json(self.path, self.data)

    def remove(self, tile_id):
        with self.lock:
            if self.data.pop(tile_id, None) is not None:
                atomic_write_json(self.path, self.data)


# ============================================================
# PLANNING: TILES -> GROUPS -> BLOCKS
# ============================================================
def load_tiles(cities, failed, retry_failed):
    """Returns (eligible, skipped_lowres, skipped_failed, existing). Every eligible tile is returned so
    block geometry stays stable across runs; tiles with t["todo"] False are only used for geometry."""
    eligible, skipped_lowres, skipped_failed, existing = [], defaultdict(int), 0, 0
    for city in sorted(os.listdir(DATASET_ROOT)):
        if cities and city not in cities:
            continue
        meta_dir = os.path.join(DATASET_ROOT, city, "metadata")
        if not os.path.isdir(meta_dir):
            continue
        for fn in sorted(os.listdir(meta_dir)):
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(meta_dir, fn), "r") as f:
                meta = json.load(f)
            tile_id = meta["tile_id"]
            hr = meta.get("hr", {})
            res = hr.get("resolution_m")
            if not hr.get("acquisition_date") or (res is not None and float(res) > MAX_HR_RES_M):
                skipped_lowres[city] += 1
                continue
            todo = True
            if os.path.exists(os.path.join(OUTPUT_DIR, city, f"{tile_id}.png")):
                existing += 1
                todo = False
            elif tile_id in failed.data and not retry_failed:
                skipped_failed += 1
                todo = False
            search = hr["acquisition_date"]
            if OFFSEASON_DAYS:
                search = (datetime.strptime(search, "%Y-%m-%d") + timedelta(days=OFFSEASON_DAYS)).strftime("%Y-%m-%d")
            eligible.append({
                "city": city,
                "tile_id": tile_id,
                "x": meta["tile"]["x"],
                "y": meta["tile"]["y"],
                "zoom": meta["tile"]["zoom"],
                "bbox_wgs84": meta["bbox_wgs84"],
                "hr_date": search,                        # centre of the date search (groups, caches)
                "arcgis_date": hr["acquisition_date"],
                "todo": todo,
            })
    return eligible, skipped_lowres, skipped_failed, existing


def group_tiles(tiles):
    groups = defaultdict(list)
    for t in tiles:
        groups[(t["city"], t["hr_date"])].append(t)
    return groups


def catalog_cache_path(city, hr_date):
    return os.path.join(CATALOG_CACHE_DIR, f"{city}_{hr_date}.json")


def fetch_catalog(city, hr_date, tiles, session, token_manager, limiter, state):
    """Returns {date: min scene cloud cover} for the group, from cache or the Catalog API."""
    path = catalog_cache_path(city, hr_date)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)["date_cloud"]

    bbox = [
        min(t["bbox_wgs84"][0] for t in tiles), min(t["bbox_wgs84"][1] for t in tiles),
        max(t["bbox_wgs84"][2] for t in tiles), max(t["bbox_wgs84"][3] for t in tiles),
    ]
    hr = datetime.strptime(hr_date, "%Y-%m-%d")
    start = (hr - timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%dT00:00:00Z")
    end = (hr + timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%dT23:59:59Z")

    date_cloud = {}
    payload = {"collections": [COLLECTION], "bbox": bbox, "datetime": f"{start}/{end}", "limit": 100}
    while True:
        resp = post_with_rate_limit(session, CATALOG_URL, payload, token_manager, limiter, state, pu=0)
        if resp is None:
            return None
        if not resp.ok:
            print(f"CATALOG ERROR {city} {hr_date}: {resp.status_code} {resp.text[:300]}")
            return None
        body = resp.json()
        for feat in body.get("features", []):
            props = feat.get("properties", {})
            date = props["datetime"].split("T")[0]
            cloud = props.get("eo:cloud_cover", 100)
            date_cloud[date] = min(cloud, date_cloud.get(date, 100))
        next_token = body.get("context", {}).get("next")
        if next_token is None:
            break
        payload["next"] = next_token

    atomic_write_json(path, {"city": city, "hr_date": hr_date, "bbox": bbox, "date_cloud": date_cloud})
    return date_cloud


def select_dates(date_cloud, hr_date):
    """Clear-enough dates, closest to the HR acquisition first, at least MIN_DAYS_GAP apart."""
    hr = datetime.strptime(hr_date, "%Y-%m-%d")
    candidates = [d for d, c in date_cloud.items() if c < CATALOG_MAX_CLOUD]
    candidates.sort(key=lambda d: (abs((datetime.strptime(d, "%Y-%m-%d") - hr).days), d))
    chosen = []
    for d in candidates:
        cur = datetime.strptime(d, "%Y-%m-%d")
        if all(abs((cur - datetime.strptime(p, "%Y-%m-%d")).days) >= MIN_DAYS_GAP for p in chosen):
            chosen.append(d)
        if len(chosen) == MAX_DATES:
            break
    return chosen


def build_blocks(city, hr_date, tiles, dates):
    by_block = defaultdict(list)
    for t in tiles:
        by_block[(t["x"] // BLOCK_TILES, t["y"] // BLOCK_TILES)].append(t)

    blocks = []
    for (bx, by), all_tiles in sorted(by_block.items()):
        block_tiles = [t for t in all_tiles if t["todo"]]
        if not block_tiles:
            continue
        # Geometry comes from all eligible tiles so it (and the frame cache) is stable across runs.
        zoom = all_tiles[0]["zoom"]
        xmin, xmax = min(t["x"] for t in all_tiles), max(t["x"] for t in all_tiles)
        ymin, ymax = min(t["y"] for t in all_tiles), max(t["y"] for t in all_tiles)
        tile_m = 2 * WEB_MERCATOR_HALF_WORLD / (2 ** zoom)
        bbox_3857 = [
            -WEB_MERCATOR_HALF_WORLD + xmin * tile_m,
            WEB_MERCATOR_HALF_WORLD - (ymax + 1) * tile_m,
            -WEB_MERCATOR_HALF_WORLD + (xmax + 1) * tile_m,
            WEB_MERCATOR_HALF_WORLD - ymin * tile_m,
        ]
        width = (xmax - xmin + 1) * IMG_SIZE
        height = (ymax - ymin + 1) * IMG_SIZE
        blocks.append({
            "city": city, "hr_date": hr_date, "bx": bx, "by": by,
            "xmin": xmin, "ymin": ymin, "bbox_3857": bbox_3857,
            "width": width, "height": height,
            "pu": max(width * height / (512 * 512), 0.01),
            "tiles": block_tiles, "dates": dates,
        })
    return blocks


# ============================================================
# CLOUD / VALIDITY TEST
# ============================================================
def classify_frames(frames):
    """Returns a list of labels ('kept', 'cloud', 'nodata') for a tile's frames of shape (32, 32, 3)."""
    labels = []
    valid = []
    for f in frames:
        if np.all(f == 0, axis=-1).any():
            labels.append("nodata")
        else:
            labels.append(None)
            valid.append(f)

    if len(valid) >= 3:
        stack = np.stack(valid).astype(np.float32)                        # (n, 32, 32, 3)

        def scores(reference):
            offset = stack.mean(axis=(1, 2), keepdims=True) - reference.mean(axis=(0, 1), keepdims=True)
            shift = np.abs(offset).mean(axis=(1, 2, 3))                    # (n,)
            struct = np.abs(stack - offset - reference).mean(axis=(1, 2, 3))
            return shift, struct

        # Pass 1: median of all frames. Pass 2: median of the more typical half only, so that many
        # cloudy or snow-covered dates can't drag the reference away from the clear ground.
        shift, _ = scores(np.median(stack, axis=0))
        typical = stack[shift <= np.median(shift)]
        if len(typical) >= 3:
            shift, struct = scores(np.median(typical, axis=0))
        else:
            shift, struct = scores(np.median(stack, axis=0))
        struct_limit = max(CLOUD_STRUCT_MIN, CLOUD_STRUCT_RATIO * float(np.median(struct)))
        cloudy = (shift > CLOUD_SHIFT_MAX) | (struct > struct_limit)
    else:
        # Too few dates for a median: fall back to an absolute white/black check.
        cloudy = []
        for f in valid:
            white = np.all(f > 245, axis=-1).mean()
            black = np.all(f < 10, axis=-1).mean()
            cloudy.append(white + black > 0.10)

    it = iter(cloudy)
    return [lab if lab is not None else ("cloud" if next(it) else "kept") for lab in labels]


# ============================================================
# BLOCK PROCESSING
# ============================================================
def frame_cache_path(block, date):
    name = (f"{block['hr_date']}_{block['bx']}_{block['by']}_"
            f"{block['xmin']}_{block['ymin']}_{block['width']}x{block['height']}")
    return os.path.join(FRAME_CACHE_DIR, block["city"], name, f"{date}.png")


def fetch_block_frame(block, date, session, token_manager, limiter, state):
    """Returns (array HxWx3, 'cached'|'fetched') or (None, reason)."""
    path = frame_cache_path(block, date)
    if os.path.exists(path):
        array = np.array(Image.open(path).convert("RGB"))
        if array.shape == (block["height"], block["width"], 3):
            with state.lock:
                state.cached += 1
            return array, "cached"

    payload = {
        "input": {
            "bounds": {
                "bbox": block["bbox_3857"],
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/3857"}
            },
            "data": [{
                "type": COLLECTION,
                "dataFilter": {"timeRange": {"from": f"{date}T00:00:00Z", "to": f"{date}T23:59:59Z"}},
                "processing": {"upsampling": "BILINEAR", "downsampling": "BILINEAR"}
            }]
        },
        "output": {
            "width": block["width"],
            "height": block["height"],
            "responses": [{"identifier": "default", "format": {"type": "image/png"}}]
        },
        "evalscript": EVALSCRIPT
    }
    resp = post_with_rate_limit(session, PROCESS_URL, payload, token_manager, limiter, state, pu=block["pu"])
    if resp is None:
        return None, "stopped"
    if not resp.ok:
        return None, f"process error {resp.status_code}: {resp.text[:200]}"

    array = np.array(Image.open(BytesIO(resp.content)).convert("RGB"))
    if array.shape != (block["height"], block["width"], 3):
        return None, f"unexpected shape {array.shape}"
    atomic_write_bytes(path, resp.content)
    return array, "fetched"


def tile_slice(block, tile, array):
    r = (tile["y"] - block["ymin"]) * IMG_SIZE
    c = (tile["x"] - block["xmin"]) * IMG_SIZE
    return array[r:r + IMG_SIZE, c:c + IMG_SIZE]


def save_preview(tile, frames, dates, labels):
    cell = IMG_SIZE * 4
    pad = 4
    label_h = 14
    colors = {"kept": (0, 200, 0), "cloud": (220, 0, 0), "nodata": (128, 128, 128), "hr": (0, 120, 255)}
    n = len(frames) + 1
    sheet = Image.new("RGB", (n * (cell + 2 * pad), cell + 2 * pad + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    hr_path = os.path.join(DATASET_ROOT, tile["city"], "images", f"{tile['tile_id']}.png")
    items = [(np.array(Image.open(hr_path).convert("RGB")), "HR " + tile["hr_date"], "hr")]
    items += [(f, d, lab) for f, d, lab in zip(frames, dates, labels)]
    for i, (img, text, lab) in enumerate(items):
        x0 = i * (cell + 2 * pad)
        draw.rectangle([x0, 0, x0 + cell + 2 * pad - 1, cell + 2 * pad - 1], fill=colors[lab])
        sheet.paste(Image.fromarray(img).resize((cell, cell), Image.NEAREST), (x0 + pad, pad))
        draw.text((x0 + pad, cell + 2 * pad), f"{text} {lab if lab != 'hr' else ''}", fill=(0, 0, 0))

    os.makedirs(PREVIEW_DIR, exist_ok=True)
    sheet.save(os.path.join(PREVIEW_DIR, f"{tile['tile_id']}.png"))


def process_block(block, session, token_manager, limiter, state, failed, preview_budget):
    tag = f"[{block['city']} {block['hr_date']} b{block['bx']}_{block['by']} ({len(block['tiles'])} tiles)]"
    tiles = block["tiles"]
    frames = {t["tile_id"]: [] for t in tiles}
    used_dates = []

    for date in block["dates"]:
        if state.stop.is_set():
            return f"{tag} STOPPED (cached dates are kept)"
        array, status = fetch_block_frame(block, date, session, token_manager, limiter, state)
        if array is None:
            if status == "stopped":
                return f"{tag} STOPPED (cached dates are kept)"
            print(f"{tag} {date} skipped: {status}")
            continue
        print(f"{tag} {date} {status}")
        used_dates.append(date)
        for t in tiles:
            frames[t["tile_id"]].append(tile_slice(block, t, array))

        if len(used_dates) >= MIN_DATES_BEFORE_STOP:
            if all(classify_frames(frames[t["tile_id"]]).count("kept") >= NUM_IMAGES for t in tiles):
                break

    labels = {t["tile_id"]: classify_frames(frames[t["tile_id"]]) for t in tiles}
    picks, shared = choose_frames(labels, used_dates, block["hr_date"])

    written = padded = n_failed = on_shared = 0
    for t in tiles:
        tile_frames = frames[t["tile_id"]]
        if preview_budget.take():
            save_preview(t, tile_frames, used_dates, labels[t["tile_id"]])
        idx = picks[t["tile_id"]]
        if not idx:
            failed.add(t["tile_id"], t["city"], f"0 clean frames out of {len(tile_frames)} dates")
            n_failed += 1
            continue
        if len(idx) < NUM_IMAGES:
            idx = idx + [idx[i] for i in np.random.choice(len(idx), NUM_IMAGES - len(idx), replace=True)]
            padded += 1
        on_shared += all(used_dates[i] in shared for i in idx)

        base = os.path.join(OUTPUT_DIR, t["city"], t["tile_id"])
        save_png(base + ".png", np.vstack([tile_frames[i] for i in idx]))
        atomic_write_json(base + ".json", {"tile_id": t["tile_id"], "collection": COLLECTION,
                                           "arcgis_date": t["arcgis_date"], "search_centre": t["hr_date"],
                                           "dates": [used_dates[i] for i in idx],
                                           "shared": [used_dates[i] in shared for i in idx]})
        failed.remove(t["tile_id"])
        written += 1

    return (f"{tag} DONE dates={len(used_dates)} written={written} all-shared={on_shared} padded={padded} "
            f"failed={n_failed}")


def choose_frames(labels, dates, centre):
    """Picks NUM_IMAGES frame indices per tile, preferring dates shared across the block.

    labels: {tile_id: ['kept' | 'cloud' | 'nodata', ...]} per date in `dates` (same order).
    The block's shared dates are the NUM_IMAGES dates clean for the most tiles (ties: closest to `centre`).
    Each tile takes the shared dates it is clean on, then tops up with its own clean dates in the same ranking.
    Returns ({tile_id: [date indices, chronological]}, set of shared dates).
    """
    if not labels or not dates:
        return {t: [] for t in labels}, set()
    c = datetime.strptime(centre, "%Y-%m-%d")
    n_clean = [sum(lab[i] == "kept" for lab in labels.values()) for i in range(len(dates))]
    order = sorted(range(len(dates)),
                   key=lambda i: (-n_clean[i], abs((datetime.strptime(dates[i], "%Y-%m-%d") - c).days), dates[i]))
    shared = {dates[i] for i in order[:NUM_IMAGES] if n_clean[i] > 0}
    picks = {}
    for tile, lab in labels.items():
        clean = [i for i in order if lab[i] == "kept"]
        first = [i for i in clean if dates[i] in shared]
        rest = [i for i in clean if dates[i] not in shared]
        picks[tile] = sorted((first + rest)[:NUM_IMAGES], key=lambda i: dates[i])
    return picks, shared


class Budget:
    def __init__(self, n):
        self.n = n
        self.lock = Lock()

    def take(self):
        with self.lock:
            if self.n > 0:
                self.n -= 1
                return True
            return False


# ============================================================
# MAIN
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="plan only and print request/PU estimates")
    p.add_argument("--catalog", action="store_true", help="with --dry-run: also run (and cache) catalog searches")
    p.add_argument("--city", action="append", default=[], help="limit to a city (repeatable)")
    p.add_argument("--max-blocks", type=int, default=None, help="process at most N blocks")
    p.add_argument("--retry-failed", action="store_true", help="retry tiles listed in failed.json")
    p.add_argument("--preview", type=int, default=0, help="save preview sheets for N tiles")
    p.add_argument("--test", action="store_true", help="the held-out test places (data/test/...)")
    p.add_argument("--offseason", type=int, default=0,
                   help="with --test: search around ArcGIS date + DAYS (the unseen-dates test set)")
    return p.parse_args()


def main():
    global OFFSEASON_DAYS
    args = parse_args()
    if args.offseason and not args.test:
        sys.exit("--offseason is for the test set; add --test")
    if args.test:
        OFFSEASON_DAYS = args.offseason
        out = "sentinel2_l2a_offseason" if args.offseason else "sentinel2_l2a"
        set_paths(os.path.join("data", "test", "arcgis"), os.path.join("data", "test", out))
    os.makedirs(STATE_DIR, exist_ok=True)

    failed = FailedStore(FAILED_PATH)
    tiles, skipped_lowres, skipped_failed, existing = load_tiles(set(args.city), failed, args.retry_failed)
    groups = {k: g for k, g in group_tiles(tiles).items() if any(t["todo"] for t in g)}

    print(f"Tiles to collect: {sum(t['todo'] for t in tiles)} | already done: {existing} | "
          f"skipped failed: {skipped_failed} | SKIPPED_LOWRES: {sum(skipped_lowres.values())} {dict(skipped_lowres)}")
    print(f"Groups (city, HR date): {len(groups)}")

    need_api = not args.dry_run or args.catalog
    state = RunState()
    limiter = RateLimiter(REQUESTS_PER_MIN, PU_PER_MIN)
    session = token_manager = None
    if need_api:
        client_id = os.environ.get("CDSE_CLIENT_ID")
        client_secret = os.environ.get("CDSE_CLIENT_SECRET")
        if not client_id or not client_secret:
            sys.exit("Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET in the environment or in .env.")
        token_manager = TokenManager(client_id, client_secret)
        session = create_session()
        token_manager.get_token()
        print("Authentication successful.")

    # Catalog phase (one search per group, cached).
    blocks = []
    uncached_catalogs = 0
    for (city, hr_date), group in sorted(groups.items()):
        cached = os.path.exists(catalog_cache_path(city, hr_date))
        if need_api:
            date_cloud = fetch_catalog(city, hr_date, group, session, token_manager, limiter, state)
            if date_cloud is None:
                if state.stop.is_set():
                    break
                continue
            dates = select_dates(date_cloud, hr_date)
        elif cached:
            with open(catalog_cache_path(city, hr_date), "r") as f:
                dates = select_dates(json.load(f)["date_cloud"], hr_date)
        else:
            uncached_catalogs += 1
            dates = None
        blocks.extend(build_blocks(city, hr_date, group, dates))

    if args.dry_run:
        exp_req = worst_req = exp_pu = worst_pu = 0.0
        per_city = defaultdict(lambda: [0, 0, 0.0, 0.0])
        for b in blocks:
            n_worst = MAX_DATES if b["dates"] is None else len(b["dates"])
            n_exp = min(MIN_DATES_BEFORE_STOP, n_worst)
            exp_req += n_exp
            worst_req += n_worst
            exp_pu += n_exp * b["pu"]
            worst_pu += n_worst * b["pu"]
            c = per_city[b["city"]]
            c[0] += len(b["tiles"])
            c[1] += 1
            c[2] += n_exp * b["pu"]
            c[3] += n_worst * b["pu"]
        print(f"\n{'city':18} {'tiles':>6} {'blocks':>6} {'PU exp':>8} {'PU worst':>9}")
        for city, (nt, nb, pe, pw) in sorted(per_city.items()):
            print(f"{city:18} {nt:6d} {nb:6d} {pe:8.0f} {pw:9.0f}")
        print(f"\nBlocks: {len(blocks)}")
        print(f"Catalog requests still needed: ~{uncached_catalogs} (0 PU)")
        print(f"Process requests: expected ~{exp_req:.0f}, worst {worst_req:.0f}")
        print(f"Processing units: expected ~{exp_pu:.0f}, worst {worst_pu:.0f}  (check the account's monthly quota)")
        if uncached_catalogs:
            print("Blocks without cached catalog dates assume MAX_DATES for the worst case; "
                  "run with --dry-run --catalog for exact date counts.")
        if need_api:
            print(f"Requests used by this dry run: {state.requests}")
        return

    if state.stop.is_set():
        sys.exit(f"\nSTOPPED during catalog phase: {state.stop_reason}")

    if args.max_blocks is not None:
        blocks = blocks[:args.max_blocks]
    print(f"Processing {len(blocks)} blocks with {MAX_WORKERS} workers...\n")

    preview_budget = Budget(args.preview)
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    pending = {executor.submit(process_block, b, session, token_manager, limiter, state, failed, preview_budget)
               for b in blocks}
    done_count = 0
    try:
        while pending:
            finished, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for fut in finished:
                done_count += 1
                try:
                    print(f"({done_count}/{len(blocks)}) {fut.result()}")
                except Exception as e:
                    print(f"({done_count}/{len(blocks)}) BLOCK ERROR: {e!r}")
    except KeyboardInterrupt:
        state.request_stop("interrupted by user (Ctrl+C)")
        print("\nStopping... waiting for workers to finish their current request.")
    executor.shutdown(wait=True, cancel_futures=True)

    remaining = [t for t in load_tiles(set(args.city), FailedStore(FAILED_PATH), False)[0] if t["todo"]]
    print("\n==============================")
    print(f"Process/catalog requests sent: {state.requests} | estimated PU: {state.pu:.1f} | "
          f"cached frames reused: {state.cached}")
    print(f"Failed tiles recorded: {len(FailedStore(FAILED_PATH).data)} (retry with --retry-failed)")
    print(f"Tiles still to collect{' (selected cities)' if args.city else ''}: {len(remaining)}")
    if state.stop_reason:
        print(f"STOPPED: {state.stop_reason}")
        print("Run the same command again later; cached data will not be downloaded twice.")
        sys.exit(2)


if __name__ == "__main__":
    main()
