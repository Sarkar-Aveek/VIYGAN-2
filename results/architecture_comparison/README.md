# Architecture comparison (phase 2, 2026-09-16/17)

How the architecture was chosen before the main training ([docs/05 §5.10](../../docs/05_evaluation.md#510-how-the-architecture-was-chosen-phase-2)).
Run on a separate Linux machine (A6000 GPU).

| file | contents |
|---|---|
| `results.csv` | every model × site: PSNR, SSIM, cPSNR, LPIPS, inference seconds, and the input it was given |
| `summary_by_model.csv` | mean over the 4 sites, and each model's LPIPS rank at each site (Hyderabad, Varanasi, Thanjavur, Barmer) |
| `finetune_runs.csv` | the 11 quick fine-tunes: epochs, tiles, batch, iterations, minutes, validation PSNR / SSIM |
| `<site>.jpg` | every model's output on that site's scene, with its scores, next to the reference and the inputs |

- **Sites:** Hyderabad old city (dense urban), Varanasi ghats (urban), Thanjavur (paddy), Barmer (desert): one
  512 × 512 px scene (1.2 km × 1.2 km) each, never trained on, scored against ArcGIS World Imagery.
- **Models, as published (7), plus ours:** Real-ESRGAN x4plus, SwinIR-L real-world x4, Satlas ESRGAN 1-frame and 2-frame (US),
  ESA SEN2SR Lite, ESA LDSR-S2 (latent diffusion, with its uncertainty map), Swin2-MoSE (×2 + bicubic ×2), and our
  India fine-tune of the Satlas 8-frame ESRGAN (`india_finetune_8`, the model arcgis_B was later trained from).
  Plus bicubic.
- **Quick fine-tunes (11):** Real-ESRGAN, SwinIR-L, Satlas 1/2/8-frame, SEN2SR Lite and Swin2-MoSE, each fine-tuned for
  **one epoch on the same 18,794 Indian tiles** (1–19 minutes), with recipe A (L1 + perceptual + GAN, the
  discriminator sees the input) and, where it applies, recipe B (the model's own loss, or a discriminator that sees
  the image only).
- **Input.** At the time the Indian data was Sentinel-2 L1C; the L2A switch came a day later. Models built for
  L2A four-band input (SEN2SR, LDSR-S2, Swin2-MoSE) were given one L2A date with B02/B03/B04/B08.
