# Data metadata

The metadata of every tile the project collected, without the images. Explained in [docs/02](../docs/02_data.md).

| file | rows | one row per |
|---|---|---|
| `reference_tiles.csv` | 135,679 | ArcGIS reference tile: `split` (train_collection / test), place, tile x/y, lon/lat, capture date, platform (WV02, WV03, GE01, LG01–06), provider, source, resolution (m), positional accuracy (m), dropped as water |
| `sentinel2_stacks.csv` | 137,744 | 8-date Sentinel-2 L2A stack: `split` (train_collection / test_inseason / test_offseason), the reference date it was matched to, the 8 dates, how many are shared with the whole 16 × 16-tile block, min / median / max gap to the reference in days |
| `build_report.csv` | 132,146 | tile in the training-set build: land kind, `split` (train / val / rejected), rejection reason, flat fraction, reference–input correlation |
| `folder_structure.md` | | the data folder tree with file counts |
| `samples/` | | one raw file of each kind: ArcGIS tile metadata (with the full citation record), Sentinel-2 dates, a training pair's dates, `rejected.txt` |

`python build_metadata.py <data folder>` regenerates everything here from a data folder built by
`code/pipeline.py`. The 15 places copied from the first collection carry per-tile dates rather than block-shared
ones, so their stacks have an empty `n_block_shared`.
