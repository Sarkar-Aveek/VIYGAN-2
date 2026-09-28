# LDSR-S2, ESA OpenSR latent diffusion (`ldsr_s2`)

Loader: `ldsr.py`.

- **Source code:** https://github.com/ESAOpenSR/opensr-model (package `opensr_model` 1.0.8 installed, MIT; the
  latent-diffusion part is adapted from CompVis LDM, MIT). Paper: Donike et al., "Trustworthy Super-Resolution of
  Multispectral Sentinel-2 Imagery With Latent Diffusion", IEEE JSTARS 18, 6940-6952, 2025.
- **Config:** `opensr_model/configs/config_10m.yaml` at tag v1.0.8 (identical to `main`, which is what the README's
  "Minimal Example" fetches) -> `code/published_models/weights_public/ldsr_s2/config_10m_v1.0.8.yaml`.
- **Weights:** `ckpt_version: "opensr-ldsrs2_v1_0_0.ckpt"` from
  https://huggingface.co/simon-donike/RS-SR-LTDF/resolve/main/opensr-ldsrs2_v1_0_0.ckpt (1,130,715,795 bytes) ->
  `code/published_models/weights_public/ldsr_s2/`. HF card licence: MIT. Loaded with the package's `load_pretrained`
  (strict=True, loss weights removed); the file was downloaded beforehand so no download happens at load.
- **Architecture:** KL autoencoder (4 ch in/out, 4-ch latent, ch 128, ch_mult [1,2,4] = 4x down; 55.3 M params) +
  conditional denoising UNet in latent space (160 ch, mult [1,2,2,4], attention at 16/8; 113.6 M params); the LR
  image is bilinearly upsampled x4, encoded by the same AE and concatenated as conditioning (8 input ch). DDIM
  sampling, eps prediction, 1000 training timesteps. Total ~169 M params used at inference (the checkpoint also
  carries a 113.6 M EMA copy of the UNet which the official inference path does not use; neither does the loader).
- **Training data:** HF card: "LDSR-S2 was developed as part of the ESA OpenSR project in conjunction with the
  SEN2NAIP Sentinel-2 super-resolution dataset" (card tags: tacofoundation/SEN2NAIPv2, isp-uv-es/SEN2NEON). USA
  NAIP aerial targets.

## Preprocessing (quoted)

HF card: "The expected channel order is therefore **R, G, B, NIR (`B04`, `B03`, `B02`, `B08`)**" ...
"Expected reflectance range: approximately `[0, 1]`". demo.py: `lr = (lr/10_000).to(torch.float32).to(device)`.
README: `sr = model.forward(torch.rand(1,4,128,128), sampling_steps=100) # run SR`.
Config: `apply_normalization: False`, `encode_conditioning: True`, `sampling_eta: 0.95`,
`sampling_temperature: 1.0`, `sampling_steps: 100`, `cond_stage_config.image_size: 128`.
Inside `forward`: NaN -> 0, inputs < 128 px reflect-padded to 128, zero pixels become a no-data mask applied to the
output, and each output band is histogram-matched to the LR band (`histogram_matching=True` default).
opensr-utils (the official large-scene pipeline): `window_size=(128, 128)`, `overlap=12`,
`eliminate_border_px=2`, "a sigmoid weight curve" for blending.

## What `fn` does

1. Reorders (B02, B03, B04, B08) -> (B04, B03, B02, B08); values stay reflectance.
2. `torch.manual_seed(0)` at every call (sampling is stochastic; this makes outputs reproducible; verified
   identical on repeat).
3. Tiles into 128x128 LR windows with 12 px overlap, discards 2 px at internal patch borders, sigmoid blending
   (opensr-utils defaults, reimplemented in `_tiling.py`). Inputs < 128 are reflect-padded to 128 and cropped back.
   One patch per sampler call (batch 4 failed with `CUDNN_STATUS_EXECUTION_FAILED` on the 8 GB GPU).
4. `model.forward(patch)` with config defaults (100 DDIM steps, eta 0.95, temperature 1.0, histogram matching on).
5. Returns channels 0..2 = (R, G, B) reflectance.

## Check (Hyderabad block 94096_59104, 2025-11-13)

- 128 crop -> [1,3,512,512], range 0.032..0.450, p1/p50/p99 0.047/0.113/0.222; channel means within 6e-5 of
  bicubic (histogram matching). 256x256 LR -> [1,3,1024,1024] in 45.9 s (9 patches), no visible seams, column
  gradient near the seam 0.0063 vs 0.0059 overall.
- `check_ldsr_s2.png`: left bicubic, right LDSR-S2. Colours correct; clearly sharper, plausible building blocks and
  street grid, with invented detail (generative model).
- **Runtime:** 4.7 s per 128 px LR patch (100 DDIM steps, fp32, RTX 5060 Laptop, batch 1); ~5.1 s per patch when
  tiling. A 512x512 LR block = 25 patches, ~2 min. A lone 32x32 tile costs a full 128 patch (~4.7 s). Peak VRAM
  4.2 GB. Load 5.5 s.

## Fairness notes

- ~300x slower than SEN2SR / our GAN; benchmark on fewer tiles or feed 128+ px blocks rather than 32 px tiles
  (which are padded with mirrored content and waste 15/16 of the compute).
- Stochastic: results depend on the seed (fixed to 0). The package's `uncertainty_map` is not used.
- Histogram matching to the LR input makes its per-band distribution equal to S2 (like our model 2/3), so it
  cannot match ArcGIS colours; compare it against S2-colour models.
- Single image (one date) vs our 8 frames; trained on USA NAIP.
