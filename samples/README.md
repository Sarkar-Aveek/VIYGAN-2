# Samples

One 4 × 4-tile chip (1.2 km × 1.2 km) of **Hyderabad's old city**, a test place the models never saw.

| folder | contents |
|---|---|
| `input_sentinel2/hyderabad_charminar/` | 16 Sentinel-2 L2A stacks: `<tile>.png` is 8 dates × 32 × 32 px (10 m) stacked vertically, `<tile>.json` lists the dates and the reference capture they were matched to |
| `output_arcgis_B/`, `output_s2colour/` | for each tile, `<tile>_sr.png` (128 × 128, 2.39 m) and `<tile>_lr.png` (the first input date, 32 × 32) |

Regenerate the outputs:

```powershell
cd code
python infer.py --model arcgis_B --input ..\samples\input_sentinel2 --output ..\samples\output_arcgis_B
python infer.py --model s2colour --input ..\samples\input_sentinel2 --output ..\samples\output_s2colour
```

Tile IDs are `17_<x>_<y>` (zoom-17 Web-Mercator, EPSG:3857); the 16 tiles form the square x 94104–94107,
y 59120–59123. Contains modified Copernicus Sentinel data.
