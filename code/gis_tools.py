"""Georeferencing, Sentinel-2 band download and the per-pixel confidence map, on top of the super-resolution output.

    block_bands     4-band (or any-band) Sentinel-2 L2A reflectance for a 16x16-tile block, cached (used to feed the
                    single-image published models in ps_proof.py)
    dual_subsets,   the date-subset confidence check: two runs on disjoint halves of the dates and their per-pixel
    sigma           standard deviation (docs/07)
    world_file,     write outputs so QGIS and ArcGIS Pro open them in place (EPSG:3857)
    geotiff

    python gis_tools.py            # self-check, offline, no processing units
"""
import json
import os

import numpy as np
import torch

import bootstrap  # noqa: F401  (repo root on sys.path, basicsr shim)
from collectors import sentinel2_collector as sc   # Copernicus Data Space client (token, rate limit, quota)

HALF_WORLD = 20037508.342789244
ZOOM = 17
BLOCK_TILES = sc.BLOCK_TILES                  # 16 x 16 zoom-17 tiles per block
TILE_M = 2 * HALF_WORLD / 2 ** ZOOM           # 305.748 m at zoom 17
BLOCK_M = TILE_M * BLOCK_TILES                # 4891.97 m
SR_PX = TILE_M / 128                          # 2.388657 m
LR_PX = TILE_M / 32                           # 9.554629 m
BAND_CACHE = os.path.join("data", "cache", "bands")

PRJ_3857 = ('PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,'
            '298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
            'PROJECTION["Mercator_1SP"],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],'
            'PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1],'
            'AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","3857"]]')


def block_bbox_3857(x0, y0):
    """[xmin, ymin, xmax, ymax] in EPSG:3857 of the 16x16-tile block whose top-left tile is (x0, y0)."""
    return [-HALF_WORLD + x0 * TILE_M, HALF_WORLD - (y0 + BLOCK_TILES) * TILE_M,
            -HALF_WORLD + (x0 + BLOCK_TILES) * TILE_M, HALF_WORLD - y0 * TILE_M]


def _post_process(payload, pu):
    """One Sentinel Hub Process API request through the collector's client (credentials from the environment)."""
    state = sc.RunState()
    return sc.post_with_rate_limit(sc.create_session(), sc.PROCESS_URL, payload,
                                   sc.TokenManager(os.environ["CDSE_CLIENT_ID"], os.environ["CDSE_CLIENT_SECRET"]),
                                   sc.RateLimiter(sc.REQUESTS_PER_MIN, sc.PU_PER_MIN), state, pu=pu)


# ============================================================
# BANDS
# ============================================================
def _evalscript(bands):
    return ('//VERSION=3\nfunction setup() {{ return {{input: [{{bands: {names}, units: "REFLECTANCE"}}],\n'
            '  output: {{bands: {n}, sampleType: "UINT16"}}}}; }}\n'
            'function evaluatePixel(s) {{ return [{vals}]; }}\n').format(
                names=json.dumps(bands), n=len(bands), vals=", ".join(f"s.{b} * 10000" for b in bands))


def block_bands(x0, y0, date, bands, size=512):
    """{band: float32 [size, size] reflectance} for one 16x16 block on one date. Cached; costs len(bands)/3 PU.

    Every band comes back on the `size` grid, so 20 m bands are resampled to 10 m server-side and all the arrays
    line up.
    """
    path = os.path.join(BAND_CACHE, f"{x0}_{y0}_{date}_{'-'.join(bands)}.npy")
    if os.path.exists(path):
        arr = np.load(path)
    else:
        import tifffile
        payload = {
            "input": {"bounds": {"bbox": block_bbox_3857(x0, y0),
                                 "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/3857"}},
                      "data": [{"type": sc.COLLECTION,
                                "dataFilter": {"timeRange": {"from": f"{date}T00:00:00Z", "to": f"{date}T23:59:59Z"}},
                                "processing": {"upsampling": "BILINEAR", "downsampling": "BILINEAR"}}]},
            "output": {"width": size, "height": size,
                       "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
            "evalscript": _evalscript(bands),
        }
        resp = _post_process(payload, pu=len(bands) / 3)
        if resp is None or not resp.ok:
            raise RuntimeError(f"band fetch failed for {x0},{y0} {date}: "
                               f"{resp.status_code if resp is not None else 'stopped'}")
        os.makedirs(BAND_CACHE, exist_ok=True)
        sc.atomic_write_bytes(path + ".tif", resp.content)
        arr = np.atleast_3d(tifffile.imread(path + ".tif")).astype(np.uint16)
        np.save(path, arr)
        os.remove(path + ".tif")
    if arr.max() == 0:                 # the date has no scene over this block; do not keep an empty cache entry
        os.remove(path)
        raise RuntimeError(f"no Sentinel-2 data for {date} over this block")
    return {b: arr[:, :, i].astype(np.float32) / 10000.0 for i, b in enumerate(bands)}


# ============================================================
# CONFIDENCE
# ============================================================
def dual_subsets(frames):
    """[n, 3, h, w] frames (n >= 4) -> two 24-channel stacks from disjoint dates, for a repeatability check."""
    if len(frames) < 4:
        raise ValueError(f"confidence needs at least 4 dates, got {len(frames)}")

    def stack(sub):
        return torch.cat([sub[i % len(sub)] for i in range(8)])
    return stack(frames[0::2]), stack(frames[1::2])


def sigma(sr_a, sr_b):
    """Per-pixel standard deviation between two independent-date outputs [3, H, W] -> [H, W].

    Real detail repeats across dates, invented detail does not, so this is high where the model guessed.
    """
    return torch.stack([sr_a, sr_b]).std(dim=0).mean(dim=0)


# ============================================================
# EXPORT
# ============================================================
def world_file(raster_path, x0, y0, px):
    """Write the .wld and .prj beside a PNG so QGIS and ArcGIS Pro open it in place (EPSG:3857)."""
    xmin, _, _, ymax = block_bbox_3857(x0, y0)
    stem = os.path.splitext(raster_path)[0]
    with open(stem + ".wld", "w") as f:
        f.write(f"{px!r}\n0.0\n0.0\n{-px!r}\n{xmin + px / 2!r}\n{ymax - px / 2!r}\n")
    with open(stem + ".prj", "w") as f:
        f.write(PRJ_3857 + "\n")
    return stem + ".wld"


def geotiff(path, array, x0, y0, px, nodata=None):
    """Write a georeferenced GeoTIFF: uint8 [H, W, 3] imagery, or float32 [H, W] for a confidence map."""
    import rasterio
    from rasterio.transform import from_origin
    xmin, _, _, ymax = block_bbox_3857(x0, y0)
    bands = array.transpose(2, 0, 1) if array.ndim == 3 else array[None]
    with rasterio.open(path, "w", driver="GTiff", height=bands.shape[1], width=bands.shape[2],
                       count=bands.shape[0], dtype=bands.dtype, crs="EPSG:3857",
                       transform=from_origin(xmin, ymax, px, px), nodata=nodata, compress="deflate") as dst:
        dst.write(bands)
    return path


# ============================================================
# SELF-CHECK
# ============================================================
def demo():
    """Offline, no processing units: every function above with an answer known in advance."""
    x0, y0 = 97791 // 16 * 16, 55173 // 16 * 16
    xmin, ymin, xmax, ymax = block_bbox_3857(x0, y0)
    assert abs((xmax - xmin) - BLOCK_M) < 1e-6 and abs((ymax - ymin) - BLOCK_M) < 1e-6

    # confidence: identical outputs mean zero sigma, and 4 dates are the minimum
    frames = torch.rand(6, 3, 8, 8)
    a, c = dual_subsets(frames)
    assert a.shape == (24, 8, 8) and not torch.equal(a, c)
    assert float(sigma(torch.ones(3, 4, 4), torch.ones(3, 4, 4)).max()) == 0.0
    try:
        dual_subsets(torch.rand(3, 3, 8, 8))
        raise AssertionError("3 dates should have been refused")
    except ValueError:
        pass

    # world file: pixel size, negative y step, and the top-left pixel centre half a pixel inside the block
    tmp = os.path.join(os.environ.get("TEMP", "."), "gis_tools_check.png")
    lines = open(world_file(tmp, x0, y0, SR_PX)).read().split()
    assert abs(float(lines[0]) - SR_PX) < 1e-9 and abs(float(lines[3]) + SR_PX) < 1e-9, lines
    assert abs(float(lines[4]) - (xmin + SR_PX / 2)) < 1e-6, lines
    assert abs(float(lines[5]) - (ymax - SR_PX / 2)) < 1e-6, lines
    for ext in (".wld", ".prj"):
        os.remove(os.path.splitext(tmp)[0] + ext)

    # GeoTIFF: CRS, origin and pixel size read back as written
    import rasterio
    tif = os.path.join(os.environ.get("TEMP", "."), "gis_tools_check.tif")
    geotiff(tif, np.zeros((128, 128, 3), np.uint8), x0, y0, SR_PX)
    with rasterio.open(tif) as src:
        assert src.crs.to_epsg() == 3857 and src.count == 3, src.crs
        assert abs(src.transform.c - xmin) < 1e-6 and abs(src.transform.f - ymax) < 1e-6
        assert abs(src.transform.a - SR_PX) < 1e-9 and abs(src.transform.e + SR_PX) < 1e-9
    os.remove(tif)

    print(f"gis_tools self-check passed  (SR {SR_PX:.6f} m/px, Sentinel {LR_PX:.6f} m/px, block {BLOCK_M:.2f} m)")


if __name__ == "__main__":
    demo()
