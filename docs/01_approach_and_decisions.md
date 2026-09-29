# 1. Approach and design decisions

[← back to README](../README.md) · next: [2. Data](02_data.md)

## 1.1 The problem in one paragraph

Sentinel-2 gives free 10 m imagery of all of India every 5 days, but at 10 m a narrow road, a small building or a
field boundary is a single mixed pixel. SIH26142, *Deep Learning Based Super Resolution Mapping (SRM) from Medium Resolution Satellite Imageries*, asks
for a framework that turns this into a sharper product
(< 4 m) **without breaking what makes Sentinel-2 trustworthy**: its geolocation and its measured colours
(reflectance). It also has to say honestly which details were observed and which were inferred.

## 1.2 Our answer

A **multi-date generative super-resolution network**:

```text
 8 Sentinel-2 L2A dates of the same place (10 m, 32 x 32 px each, stacked: 24 channels)
        │   pre-processing: date selection around the reference date, per-tile cloud/haze test,
        │   same dates for a whole 16 x 16-tile block, reflectance scaled like ESA's true-colour product
        ▼
 generator: RRDB-ESRGAN (23 residual-in-residual dense blocks, 16.7 M parameters), x4
        ▼
 128 x 128 px RGB at 2.39 m/px on the zoom-17 Web-Mercator grid (EPSG:3857), two colour versions:
   • arcgis_B  : looks like the ArcGIS basemap (for mapping and visual interpretation)
   • s2colour  : keeps Sentinel-2's measured colours per 10 m cell (for anything quantitative)
        +
 a per-pixel confidence layer (which details were added by the model)
```

It is the Allen AI **Satlas** super-resolution generator (trained on 1.2 million US Sentinel-2 → NAIP pairs),
fine-tuned twice on Indian data: 103,936 real pairs from 55 Indian places, with ArcGIS World Imagery
as the reference (zoom-17 tiles at 2.39 m/px, resampled from 0.31–0.5 m satellite captures). It is tested on 7 whole places it never saw.

## 1.3 How the project went

Every decision was **measured before it was kept**. Ideas that lost were dropped and are listed below.

| # | dates (2026) | phase | what happened | outcome |
|---|---|---|---|---|
| 1 | Sep 13–14 | **First Indian fine-tune** | Collectors written for the Copernicus quota (block batching, cache, cloud test). Satlas fine-tuned on 15 Indian places (L1C input, ~30k tiles, 20k iterations, 2 h 42 min) → *India 20k*. Learning-rate test 2e-5 / 5e-5 / 1e-4 over 3k iterations. | Validation PSNR 16.94 (Satlas as published) → **21.79**. Unseen Hyderabad 15.18 → 17.35 dB, Dharavi 14.56 → 17.56 dB. |
| 2 | Sep 16–17 | **Choose the architecture by measurement** | 7 published super-resolution models (GANs, Swin transformers, ESA's CNN and diffusion model) and our India fine-tune run on 4 unseen Indian sites, plus 11 one-epoch fine-tunes of them on the same Indian tiles (on a separate A6000 machine). | India 20k first on LPIPS at **every** site (0.171; the best other 0.211). At equal budget the ESRGAN family still led (0.211–0.218 vs 0.253+). Swin2-MoSE led cPSNR by 0.4 dB only by blurring (LPIPS 0.484). Decision: keep the multi-date ESRGAN. [5.10](05_evaluation.md#510-how-the-architecture-was-chosen-phase-2) |
| 3 | Sep 18 | **Diagnose the remaining error** | 400 validation tiles: how much error is brightness, how much misalignment? L1C vs L2A input. | Brightness correction +1.07 dB, best shift only +0.09 dB → the targets are aligned; the remaining error is ArcGIS's per-tile colour. **L2A** cuts the input-to-target contrast gap from 20.5 to 12.5 → switch to L2A. |
| 4 | Sep 18–21 | **Scale the data, train the final models** | 55 training places, 132,879 raw tiles → 103,986 clean; 7 test places collected separately. Two ArcGIS-colour runs (A: the Satlas recipe; B: less GAN + an edge loss), then the Sentinel-colour model. | Our two models: **arcgis_B** and **s2colour** ([4. Training](04_training.md)). |
| 5 | Sep 21–28 | **Independent checks** | ESA's opensr-test benchmark (India and Spanish cities), published models run on the test places, geospatial-consistency analysis, this repository. | [5. Evaluation](05_evaluation.md), [6. Geospatial consistency](06_geospatial_consistency.md). |

## 1.4 Why we chose what we chose

| decision | alternatives considered | why this one (evidence) |
|---|---|---|
| **Generative model (GAN), not a pixel-loss CNN** | EDSR / RCAN / SwinIR trained on L1/L2 | A network trained on pixel losses outputs the *average* of all plausible high-resolution images, which is blurry. The literature is explicit about this ([references](../references/README.md): ESRGAN, SRGAN, Blau & Michaeli). Our own phase 2 confirmed it: the best-cPSNR model (Swin2-MoSE) had LPIPS 0.484 against our 0.171 ([5.10](05_evaluation.md#510-how-the-architecture-was-chosen-phase-2)). |
| **Multi-date input (8 dates)** | single image (what almost every published model does) | Eight looks at the same ground are shifted by fractions of a pixel. That is real information about sub-10 m detail, not noise (HighRes-net, PROBA-V, WorldStrat in [references](../references/README.md)). WorldStrat reports "a clear improvement when going from 4 revisits to 8". The dates also let haze and a partial cloud on one day be outvoted. Measured here: [5.6](05_evaluation.md#56-what-the-8-dates-buy). And the dates are co-registered to 0.03 px, so stacking them does **not** blur geometry: [6. Geospatial consistency](06_geospatial_consistency.md). |
| **Start from Satlas, fine-tune** | train from scratch | Satlas learned 2.5 m texture from 1.2 M pairs. Fine-tuning keeps that and replaces US textures with Indian ones in ~3 h per run on a laptop GPU (RTX 5060, 8 GB). |
| **Sentinel-2 L2A input** | L1C (top of atmosphere) | Phase 3: L2A (atmosphere removed) cuts the contrast gap to the reference from 20.5 to 12.5 and helps most on hazy dates. It is also the product users would feed in. |
| **RGB bands only (B04/B03/B02), in and out** | adding near-infrared, red-edge or short-wave infrared bands to the input | The reference imagery has only three bands (R, G, B), so no other band can be supervised or output. As inputs they would cost more quota, discard the pretrained first layer, and add 20–60 m detail to a 10 m signal. Satlas's default released model (our start) is RGB too. [1.6](#16-why-the-input-and-output-are-rgb-only) |
| **Real pairs, ArcGIS World Imagery as the reference** | NAIP (USA only), SPOT/Pléiades (paid), synthetic downsampling of Sentinel-2 | Free, sharper than the 2.39 m target grid (0.31–0.5 m captures, exported at 2.39 m), covers India, and publishes capture date, sensor and resolution for **every** tile, which makes a full metadata audit possible ([2.4](02_data.md#24-every-tile-checked-through-its-metadata)). Synthetic downsampling teaches a network to undo its own blur, not to recover real sub-10 m detail. |
| **2.39 m output (x4 on the zoom-17 grid)** | x2 (5 m), x3 (3.3 m) | Meets < 4 m with margin, and is the grid Satlas was trained on, so its weights transfer. The output grid is an exact Web-Mercator tile grid, so every pixel has a known map position. |
| **Two models, two colour conventions** | one model | Users need two different things. **arcgis_B** looks like the familiar basemap. **s2colour** keeps the measured Sentinel-2 colour of every 10 m cell (locked inside the network), for change detection and anything quantitative: the PS's "spectral consistency". |
| **arcgis_B over run A** | run A (the Satlas recipe unchanged) | Decided on the 50 validation tiles, never on test places. B was better on 5 of 6 validation metrics (cPSNR 22.79 vs 22.62, SSIM 0.518 vs 0.507, edge-F1 0.688 vs 0.683, gradient correlation 0.391 vs 0.379; LPIPS 0.141 vs 0.139). Its losses are designed to invent less ([3.4](03_models.md#34-losses)). |
| **Held-out places, not held-out tiles** | a random 10% of tiles | Neighbouring tiles share buildings, fields and dates, so a random split leaks. Whole places ≥ 54 km from any training tile cannot. |
| **Snow, desert, forest, coast, farmland in training** | cities only | The PS names crop monitoring, urban analysis and disaster assessment. Floods, landslides and avalanches happen outside cities. |
| **Headline metrics: edge-F1, LPIPS, opensr-test** | PSNR / SSIM | PSNR rewards blur ([5.1](05_evaluation.md#51-why-psnr-and-ssim-are-not-the-headline)). We still report PSNR, SSIM and cPSNR for every model. |

**Tried and dropped (measured, lost):** a colour-constrained loss (v2); attention-based fusion of the dates;
per-date sun-angle conditioning; per-place expert models; training the Sentinel-colour model against a recoloured
target without the colour lock (no better than recolouring arcgis_B after the fact: LPIPS 0.3005 vs 0.3008).


## 1.5 What the Satlas authors tested, and what we took from it

Our generator is the one from Wolters, Bastani & Kembhavi, *Zooming Out on Zooming In: Advancing Super-Resolution
for Remote Sensing* (Allen AI, 2023; [references](../references/README.md#5-satlas-super-resolution-zooming-out-on-zooming-in)).
They built S2-NAIP (1.2 million pairs of a Sentinel-2 time series and a US NAIP aerial image, on the same zoom-17
Web-Mercator tiles we use) and ran these studies:

| their study | what they found (their words) | what it meant for us |
|---|---|---|
| **Human judgement of metrics.** People ranked SR outputs; every metric was checked against their choices | "Finding 1. PSNR and SSIM, the most widely used metrics, are insufficient and poorly correlate with human judgments." CLIP-based similarity agreed with people best (84.6% for CLIPA-v2) | PSNR / SSIM are reported but not used to rank ([5.1](05_evaluation.md#51-why-psnr-and-ssim-are-not-the-headline)); LPIPS and structure metrics lead |
| **Four method families** on S2-NAIP, PROBA-V, WorldStrat and OLI2MSI: SRCNN and HighRes-net (pixel loss), ESRGAN (GAN), SR3 (diffusion) | "Finding 4. GANs are capable super-resolution methods." "Compared to training only on L2 loss, like SRCNN and HighResNet, the outputs are sharper and more similar to the target images" | a GAN, not a pixel-loss network; confirmed on Indian data in our phase 2 ([5.10](05_evaluation.md#510-how-the-architecture-was-chosen-phase-2)) |
| **Building-count study** (an expert counted buildings in 256 held-out outputs) | "ESRGAN accurately generates the buildings from the high-resolution target image 94.70% of the time, while SR3 only 81.77% of the time." "Finding 5. Diffusion models generate realistic but inaccurate images." Also: "the ESRGAN is 200x faster" | no diffusion model; ESA's LDSR-S2 diffusion model is in our benchmark as a check ([5.3](05_evaluation.md#53-results-7-unseen-indian-places-in-season)) |
| **Data and model scale** (5 data splits × 3 ESRGAN sizes: 17 M, 87 M, 347 M parameters) | "Finding 3. Performance scales with dataset and model size." "Between the smallest and largest data splits, there is a ten point improvement in CLIPSCORE for the smallest model." | collect as much Indian data as the quota allowed (103,936 pairs from 55 places), rather than a few cities |
| **Number of input dates** (released models for 1, 2, 4, 8 and 16 dates; results in their supplement) | the default released model uses 8 | 8 dates; our own ablation shows most of the gain arrives by 4 ([5.6](05_evaluation.md#56-what-the-8-dates-buy)) |
| **Input bands** (released models with the 10 m, 20 m and 60 m band sets; results in their supplement) | "Most experiments utilize just TCI" (ESA's true-colour RGB) | RGB only ([1.6](#16-why-the-input-and-output-are-rgb-only)) |
| **Super-resolved images for machine learning** (5 downstream tasks) | "Finding 7. Super-resolution outputs are ineffective for machine consumption." ... "using SUPER-RES outputs does not outperform inputting the original low-resolution images" | we present the output as imagery for people to interpret, with a confidence map, and do not claim it improves automatic classifiers ([8](08_limitations.md)) |

Their supplementary material (which holds the date and band ablations) is not part of the arXiv paper, so we
quote only the main text and the released model list.

## 1.6 Why the input and output are RGB only

Sentinel-2 has 13 bands; we use three (B04 red, B03 green, B02 blue, all 10 m). The reasons:

1. **The reference has only three bands.** ArcGIS World Imagery is an 8-bit **RGB** product: there is no
   near-infrared or short-wave infrared truth at < 4 m to train against. A network can only learn to super-resolve
   what its loss can see, so a super-resolved NIR, red-edge or SWIR band could not be trained or validated here. The
   same holds for Satlas: its NAIP targets were stored as RGB ("Each image is 128x128px with RGB channels").
2. **As extra inputs, more bands cost more than they can give here.**
   - **Quota.** Sentinel Hub counts every three bands as one processing unit, so adding NIR and the SWIR bands would
     roughly double or triple the processing units of every download (each download step runs under a
     15,000-unit budget).
   - **Pretrained weights.** The Satlas and India 20k generators take 24 channels (8 dates × RGB). Extra bands mean a
     new first layer trained from scratch, giving up part of the 1.2-million-pair start.
   - **Coarser detail.** The red-edge and SWIR bands are 20 m and the atmospheric bands 60 m, so they carry less
     spatial detail than the 10 m RGB we already use.
   - **Satlas kept RGB.** They released band variants but used TCI (RGB) for most experiments, and their default
     released model (`esrgan_8S2`, our starting point) is RGB.
3. **What we did not test.** Whether NIR as an extra *input* would sharpen vegetation edges was not measured here;
   the evidence above is about cost, supervision and precedent, not an ablation. Indices that need NIR or SWIR (NDVI,
   NDWI, NBR) stay at Sentinel-2's native resolution ([8](08_limitations.md)).
