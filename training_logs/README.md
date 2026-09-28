# Training logs

Explained in [docs/04](../docs/04_training.md).

| folder | run | role |
|---|---|---|
| `arcgis_B/` | 24,000 iterations from India 20k | **released model** (ArcGIS colours) |
| `s2colour/stage1_s2colour_24k_v1/` | 24,000 iterations from arcgis_B, colour lock | s2colour stage 1 (its 18k checkpoint continues below) |
| `s2colour/` | +17,322 iterations from stage 1 @ 18k | **released model** (Sentinel-2 colours) |
| `arcgis_A_comparison_run/` | 24,000 iterations from India 20k, Satlas recipe unchanged | comparison run that decided the arcgis_B recipe |

Each folder: the exact config used (`*.yml`), the full BasicSR log (`train_*.log`), `losses.csv` (every loss every
200 iterations), `val.csv` (validation on 50 held-out tiles) and `curves.png`. `python plot_curves.py` redraws the
curves from the CSVs. Stage 1's files were extracted from the pipeline log (`code/pipeline.py` records every run);
its losses start at iteration 600, after 500 discriminator-only warm-up iterations.
