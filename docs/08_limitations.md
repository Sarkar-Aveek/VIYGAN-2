# 8. Limitations

[← 7. Uncertainty](07_uncertainty.md) · next: [9. Why it's real, and the maths](09_why_its_real_and_the_maths.md)

Stated plainly, so that no result in this repository is read as more than it is.

1. **Pixel spacing is not resolution.** The output grid is 2.39 m. The finest detail actually resolved is coarser:
   the 8 inputs are 10 m, and anything smaller than a few metres is inferred, not observed. The confidence map
   ([7.2](07_uncertainty.md#72-a-per-pixel-confidence-map-tested-against-the-real-error)) shows where.
2. **arcgis_B keeps a small geometric offset (≈ 0.08 input px, ≈ 0.75 m, mostly north–south)** inherited from the
   ArcGIS targets it was trained to imitate. s2colour has none measurable (0.010 px). The larger 0.13–0.18 px
   reported by opensr-test's spatial score is mostly a measurement artefact of added detail
   ([6.4](06_geospatial_consistency.md#64-why-opensr-tests-spatial-score-reports-drift-that-is-not-there)).
3. **Pixel-accuracy metrics favour blur.** On cPSNR and colour-normalised PSNR, ESA's models and even bicubic score
   slightly above ours. This is the perception–distortion trade-off ([5.1](05_evaluation.md#51-why-psnr-and-ssim-are-not-the-headline)),
   which is why structure and perceptual metrics are the headline. It also means the output should not be used as
   if every 2.39 m pixel were a measurement.
4. **opensr-test's three correctness scores are relative weights** that sum to 1 per model. Bicubic, which adds
   nothing, still gets improvement ≈ 0.38–0.42. Read every model against the bicubic row.
5. **The reference is a different camera on a different day.** ArcGIS imagery is a median of 20 days from the
   Sentinel-2 dates (90% within 54 days), so crops, water and shadows can genuinely differ. This affects every
   model equally, and the mismatch filter removes the worst cases from training only.
6. **The test is in-distribution in kind.** Test places are unseen, but their kinds of land (city, paddy, desert,
   forest, hills, snow) are represented in training. A landscape unlike any of the 55 places (for example open
   glaciers or large reservoirs) is not covered by this evidence. Outside India (Spanish cities) the models still
   work but were not tuned for it.
7. **RGB only.** The models super-resolve B04/B03/B02. Near-infrared and short-wave infrared bands (needed
   for NDVI, NDWI, NBR) stay at Sentinel-2's native 10–20 m; the reference imagery has no such bands to train
   against ([1.6](01_approach_and_decisions.md#16-why-the-input-and-output-are-rgb-only)).
8. **Snow and dense forest are the weakest cases** (edge-F1 0.39 at Lachung, 0.55 at Wayanad), because the reference
   and the Sentinel-2 dates rarely agree on snow cover, and closed canopy has few edges to recover.
9. **The architecture comparison is small** (4 scenes of 1.2 km, mostly L1C input, 1-epoch fine-tunes;
   [5.10](05_evaluation.md#510-how-the-architecture-was-chosen-phase-2)). It chose the architecture; the final
   claims rest on the 2,800-tile evaluation. A larger training run planned on the same Linux machine did not
   finish; all released models were trained on the laptop. The India 20k checkpoint (the start of arcgis_B) is not
   included; arcgis_B and s2colour are.
10. **Training ran on a laptop** (RTX 5060, 8 GB). Batch 12 and ~2.7 h per run were set by that hardware, not tuned
    for the best possible result.
