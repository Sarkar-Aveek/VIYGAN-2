# Code

Everything that produced the results in this repository. Run every script from this folder (`bootstrap.py` makes
each script use it as its working directory). Python 3.11, PyTorch 2.11 (CUDA 12.8), BasicSR 1.4.2; see
`requirements.txt`. The evaluation against ESA's benchmark and the published models also needs
`requirements-evaluation.txt`.

| file | what it does |
|---|---|
| `pipeline.py` | all steps in order: `arcgis`, `metadata`, `l2a`, `build`, `test_data`, `train_A`, `train_B`, `pick`, `train_s2`, `evaluate`. Every step resumes; `--dry-run` lists them. Before downloading, a free catalogue dry run estimates the Copernicus processing units and stops above `--pu-budget`. |
| `collectors/arcgis_collector.py` | the 55 training and 7 test places; ArcGIS World Imagery tiles and their citation metadata |
| `collectors/sentinel2_collector.py` | 8 clean Sentinel-2 L2A dates per tile (Copernicus Data Space Sentinel Hub): date selection, cloud/haze test, block-shared dates, quota control |
| `collectors/build_training_data.py` | filters (water, mismatch, no-data, duplicates, cap on previously seen places), train/val split, `report.csv` |
| `esrgan/networks.py` | generator `SSR_RRDBNet` (with the colour lock `input_lowpass`), discriminator `SSR_UNetDiscriminatorSN` |
| `esrgan/s2colour.py` | **s2colour's own code, PolyForm Noncommercial ([`LICENSE-s2colour.md`](esrgan/LICENSE-s2colour.md))**: the colour lock `lowpass_transfer`, `frames_median`, `colour_transfer`, the colour-blind loss shift `match_mean`, the high-pass discriminator input `highpass` |
| `esrgan/model.py` | the training step: L1 + perceptual on the sharpened target, GAN, Sobel edge loss, colour-blind losses, EMA |
| `esrgan/dataset.py` | 8 frames per tile (random during training), augmentation |
| `esrgan/metrics.py` | cPSNR, LPIPS, edge-F1, gradient correlation |
| `train.py` | BasicSR training loop; writes `losses.csv` and `val.csv` |
| `infer.py` | run `--model arcgis_B` or `--model s2colour` on any folder of 8-date stacks |
| `evaluate.py` | test-place scoring, plots and mosaics used during development |
| `ps_proof.py` | the evaluation behind `docs/05` and `docs/07`: audit, score, uncertainty, frames, figures |
| `gis_tools.py` | GeoTIFF / world-file export (EPSG:3857), Sentinel-2 band download for the published models, the date-subset confidence map; `python gis_tools.py` runs an offline self-check |
| `published_models/loaders/` | how ESA SEN2SR Lite, ESA LDSR-S2 and Real-ESRGAN were run, with notes quoting their authors' code; weights download on first use |
| `configs/` | `arcgis_A.yml` (comparison run), `arcgis_B.yml`, `s2colour_stage1.yml` → `s2colour.yml` |

Credentials for data collection: copy `.env.example` to `.env` (never committed) or set `CDSE_CLIENT_ID` and
`CDSE_CLIENT_SECRET`. The pipeline writes `data/`, `experiments/` and `outputs/` inside this folder; they are not in
the repository.

Training starts from the India 20k checkpoint (`weights/india_finetune_net_g_20000.pth` and `_net_d_`), which is
not included; it is itself Satlas `esrgan_8S2.pth`
([allenai/satlas-super-resolution](https://github.com/allenai/satlas-super-resolution)) fine-tuned for 20,000
iterations on 15 Indian places ([docs/04](../docs/04_training.md)).
