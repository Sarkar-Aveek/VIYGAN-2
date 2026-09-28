# Results

## `benchmark/`: our two models against published models on 7 unseen Indian places ([docs/05](../docs/05_evaluation.md), [docs/07](../docs/07_uncertainty.md))

| file | contents |
|---|---|
| `metrics/summary.csv` | mean of every metric per test set (in season / off season) × model, over all places and per place |
| `metrics/tiles.csv` | every tile × model × set: psnr, ssim, cpsnr, lpips, cn_psnr, cn_ssim, cn_lpips, edge_f1, grad_corr |
| `metrics/chips.csv` | every 4 × 4-tile chip × model × set: opensr-test improvement / omission / hallucination, reflectance, spectral, spatial, synthesis |
| `metrics/frames.csv` | 1 / 2 / 4 / 8 input dates |
| `metrics/uncertainty.csv`, `uncertainty_curve.csv` | does the confidence map predict the real error? |
| `metrics/data_places.csv`, `data_metadata.csv` | per place: tiles collected / kept / rejected by reason; reference resolution, sensors, dates |
| `metrics/leak_checks.csv`, `leak_distances.csv` | the six leak checks |
| `metrics/*.npz`, `*.npy` | raw arrays behind the figures |
| `figures/01`–`10` | every metric per model; hallucination vs improvement; per place; in vs off season; visual sheets; uncertainty maps and calibration; dates ablation; data audit |
| `opensr_test_spain_urban.json` | ESA opensr-test on 10 Spanish city scenes, per scene |

Models: `arcgis_B`, `s2colour` (ours); `satlas_8s2` (our starting point); `ldsr_s2`, `sen2sr_lite` (ESA);
`realesrgan_general`; `bicubic_1date`, `bicubic_8date`.

## `architecture_comparison/`: how the architecture was chosen ([docs/05 §5.10](../docs/05_evaluation.md#510-how-the-architecture-was-chosen-phase-2))

7 published models as released, our India fine-tune, and 11 one-epoch fine-tunes of them, on 4 Indian sites: `results.csv`,
`summary_by_model.csv`, `finetune_runs.csv`, and a sheet per site with every output. See its README.

## `geospatial/`: why 8 dates keep the geometry ([docs/06](../docs/06_geospatial_consistency.md))

| file | contents |
|---|---|
| `date_registration.py`, `.csv` | offset of each single Sentinel-2 date from the 8-date consensus (39 chips, 312 measurements) |
| `drift_check.py`, `.csv` | opensr-test's spatial score vs signed and blurred shifts, with a zero-shift control (117 chips) |
| `alignment_figure.py`, `alignment_s2colour.png`, `alignment_s2colour_flicker.gif` | input vs s2colour output edges, and a flicker comparison |

The scripts need the test set in `code/data/test/` (`python pipeline.py --only test_data`).
