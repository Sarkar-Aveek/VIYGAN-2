# 2. Data

[← 1. Approach](01_approach_and_decisions.md) · next: [3. Models](03_models.md)

Everything here is produced by `code/pipeline.py` (steps `arcgis`, `metadata`, `l2a`, `build`, `test_data`). Images
are not in this repository. Every tile's metadata is, in [`data_metadata/`](../data_metadata/):

| file | rows | what |
|---|---|---|
| `reference_tiles.csv` | 135,679 | every ArcGIS reference tile: place, split, tile x/y, lon/lat, **capture date, satellite, provider, source resolution, positional accuracy**, dropped-as-water flag |
| `sentinel2_stacks.csv` | 137,744 | every 8-date Sentinel-2 stack: the reference date it was matched to, the 8 dates, how many are block-shared, the gap in days to the reference |
| `build_report.csv` | 132,146 | the training-set build: every tile's split (train / val) or the filter that removed it |
| `folder_structure.md` | | the data folder tree with file counts |
| `samples/` | | one raw metadata file of each kind |

`build_metadata.py` regenerates the CSVs from a data folder.

## 2.1 Sources

| what | source | resolution | licence / cost |
|---|---|---|---|
| **Input** | Copernicus Sentinel-2 **L2A** (surface reflectance, atmosphere removed), bands B04/B03/B02, through the Copernicus Data Space Ecosystem Sentinel Hub Process API | 10 m (9.55 m/px on the zoom-17 grid) | free; quota in processing units |
| **Reference** | Esri **ArcGIS World Imagery** (WorldView-2/3, GeoEye-1, Legion), zoom 17 | 0.31–0.5 m source, delivered at 2.39 m/px | free tile export |
| **Reference metadata** | ArcGIS *World Imagery Citations* layer: capture date, sensor, provider, resolution, positional accuracy for every tile | — | free |

The input is scaled like ESA's true-colour product (`reflectance / 0.3558 → 0–255`), so every model in this
repository is shown and scored on the same scale.

## 2.2 Places

- **55 training places** (listed in `code/collectors/arcgis_collector.py`): 23 cities of different kinds (old dense
  cores such as Chandni Chowk and Dharavi, planned Chandigarh, hill and valley cities), 11 farmland regions (Punjab
  wheat, Bengal paddy, Andhra chilli, Deccan soybean, sugarcane), desert and salt flats, granite and scrub, cold
  desert, 5 forests, 2 tea-hill regions, backwaters and coast, a Brahmaputra river island, and 4 Himalayan snow
  places (with winter reference images, checked by hand).
- **Kept tiles by land kind:** urban 43,419 (41.8%), farmland 20,263, forest 10,856, snow 8,590, desert 5,445,
  rocky 4,788, hills 4,678, coast 2,897, river 2,050, mountain 1,000.
- **7 test places**, collected separately into `data/test/` and never used in training or model selection:
  Hyderabad old city (urban), Varanasi ghats (urban), Thanjavur (paddy), Barmer (desert), Wayanad (forest),
  Shillong (hills), Lachung in North Sikkim (snow). 400 tiles each, **2,800 tiles**.
- Map and per-place counts: [`results/benchmark/figures/10_data_audit.png`](../results/benchmark/figures/10_data_audit.png),
  [`results/benchmark/metrics/data_places.csv`](../results/benchmark/metrics/data_places.csv).

## 2.3 How a training pair is made (pre-processing)

1. **Reference.** The zoom-17 ArcGIS tiles nearest each place's centre (128 × 128 px, a 305.7 m square). Tiles that
   are mostly water are skipped (colour standard deviation < 12, listed in `rejected.txt`).
2. **Metadata.** For every tile, the citation layer gives capture date, sensor and resolution. Tiles whose imagery is
   coarser than **1 m** are skipped. This excluded, for example, 84 Chennai tiles served from 15 m TerraColor.
3. **Input: 8 Sentinel-2 L2A dates within ±120 days of the reference capture, nearest first.**
   - Scenes over 30% cloud are rejected at catalogue level.
   - Each frame is then checked against the tile's own median of all candidate dates. A colour shift over 20/255, or
     structure over max(5, 2.5 × the tile's median), marks it as cloud or haze.
   - Within a 16 × 16-tile block, all tiles share the same dates wherever those dates are clean for them, so
     mosaics have no patchwork and a block is fetched in one request per date (quota).
   - Stored as one PNG per tile: 8 frames of 32 × 32 px stacked vertically, plus a JSON with the dates.
4. **Filters** (`code/collectors/build_training_data.py`):

| filter | rule | tiles removed |
|---|---|---|
| `water` | > 80% of the 8 × 8 patches of the reference are textureless | 3,075 |
| `mismatch` | the reference (reduced to 32 px) and the Sentinel-2 median correlate below 0.2: land changed between the dates, snow in one image only, or cloud in the reference | 6,411 |
| `hr_black`, `bad_s2_shape`, `duplicate_tile_id` | no-data pixels, a broken stack, the same tile in two places | 0 |
| `seen_cap` | the 15 places India 20k had already trained on are capped at 1,000 tiles each, so memorised places cannot dominate | 18,674 |

From 132,879 reference tiles (132,146 with an 8-date stack), the result is **103,936 training + 50 validation
tiles**. Validation tiles come from places the starting model never saw and are only used to watch training; the 7
test places are the real check.

## 2.4 Every tile checked through its metadata

- **103,986 / 103,986** training and validation tiles have a capture date, sensor, provider and resolution. None
  are missing.
- **100%** come from imagery at **≤ 1 m**: 0.31 m (WorldView-3, 32,787 tiles), 0.34 m (Legion, 32,902), 0.46 m
  (GeoEye-1, 8,967), 0.5 m (WorldView-2, 29,330). Providers: Vantor Vivid Advanced 55,541, Vivid 48,445.
- **Date agreement:** the median distance between a tile's Sentinel-2 dates and its reference capture is
  **20 days**. 90% are within 54 days; the maximum is 106 days, inside the ±120-day window by construction.
- Capture dates run from 2016 to 2026 (most 2023–2026). An old reference is paired with Sentinel-2 dates near
  **its own** capture date, not with today's imagery.

## 2.5 No data leak

[`results/benchmark/metrics/leak_checks.csv`](../results/benchmark/metrics/leak_checks.csv):

| check | result |
|---|---|
| test tile IDs also in training | **0** |
| test tile IDs also in validation | **0** |
| validation tile IDs also in training | **0** |
| test reference images byte-identical to a training image | **0** |
| test places the starting model (India 20k) trained on | **0** |
| distance from any test tile to the nearest trained tile | **≥ 54.3 km** (Shillong; the others 76–223 km) |

Leaks were prevented by design, not only checked afterwards:
- **Geographic hold-out.** The test places were chosen as whole places ≥ 50 km from every training place and
  collected into a separate folder that the build step never reads.
- **The pretrained start cannot know them.** Satlas was trained on US imagery only; India 20k on 15 places, none a
  test place.
- **Model selection never used test data.** arcgis_B was chosen over run A, and checkpoints were chosen, on the 50
  validation tiles.
- **Time.** Test Sentinel-2 dates are matched to the test reference capture exactly as in training. An off-season
  set (dates about 6 months away) is a further check that the model is not recalling one day's appearance.

**Caveat, stated plainly:** the test places were looked at during development (visual sheets), so a human could in
principle have been influenced by them. No weight, threshold or checkpoint was chosen on them.

## 2.6 Folder layout

See [`data_metadata/folder_structure.md`](../data_metadata/folder_structure.md). In short:

```text
data/arcgis/<place>/images/<tile>.png            reference, 128 x 128, 2.39 m/px
data/arcgis/<place>/metadata/<tile>.json         capture date, sensor, resolution, citations (samples/)
data/sentinel2_l2a/<place>/<tile>.png            8 dates x 32 x 32 px, stacked vertically
data/sentinel2_l2a/<place>/<tile>.json           the 8 dates and the reference date they were matched to
data/training/{train,val}/{sentinel2,arcgis}/<tile>/   hard links used by the training loop
data/test/{arcgis,sentinel2_l2a,sentinel2_l2a_offseason}/<place>/...   the 7 held-out places
```

Tile IDs are `17_<x>_<y>`: zoom-17 Web-Mercator (EPSG:3857) tile coordinates, so every tile, and every output
pixel, has an exact map position.
