# Real-ESRGAN realesr-general-x4v3 (`realesrgan_general`)

Loader: `realesrgan.py`.

- **Source:** https://github.com/xinntao/Real-ESRGAN (BSD-3-Clause).
- **Weights:** https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth
  -> `code/published_models/weights_public/realesrgan/realesr-general-x4v3.pth` (4,885,111 bytes,
  sha256 8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292). Single key `params`.
- **Architecture:** `SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4,
  act_type='prelu')` from basicsr 1.4.2: a plain conv stack at LR resolution + PixelShuffle, plus a nearest
  upsampled input residual. Loaded with `strict=True`.
- **Parameters:** 1,213,296.
- **Training data:** generic natural photographs with the Real-ESRGAN synthetic degradation pipeline (blur, noise,
  resize, JPEG, second-order); no satellite imagery. It is a "general scene" blind SR model.

## Preprocessing

Official `RealESRGANer.pre_process/enhance` (realesrgan/utils.py): image read with cv2 (BGR), converted to RGB,
`img = img / max_range` (255) -> 0..1 float, model output `output_img.data.squeeze().float().cpu().clamp_(0, 1)`.

Our adaptation (as specified by the caller): input `clip(refl / 0.3558, 0, 1)` in RGB order = (B04, B03, B02)
(the ESA TCI scale our data uses); output `clamp(0, 1) * 0.3558` -> reflectance. NIR is not used.

Not replicated: the official `inference_realesrgan.py` CLI has `--denoise_strength` default **0.5**, which for this
model interpolates weights with `realesr-general-wdn-x4v3` (DNI). The loader uses the pure x4v3 weights
(= denoise_strength 1). No tiling needed up to 512x512 LR (whole image in one pass); larger inputs are tiled 256 px
with 16 px overlap.

## Check (Hyderabad block 94096_59104, 2025-11-13)

- 128 crop -> [1,3,512,512], range 0.029..0.3558 (saturates at the TCI max by construction), p1/p50/p99
  0.048/0.108/0.249. Full 512x512 -> [1,3,2048,2048].
- Channel means 0.116/0.112/0.103 vs bicubic 0.123/0.116/0.104: the model darkens and adds contrast.
- `check_realesrgan_general.png`: left bicubic, right Real-ESRGAN. Colours correct (not swapped); crisp but
  painterly, invented edges and blob texture in the dense urban area, as expected of a photo model.
- **Runtime:** 0.009 s per 128 px LR patch; 0.11 s for 512x512 LR (fp32). Peak VRAM 0.29 GB.

## Fairness notes

- Out of domain (photos, 8-bit display space); it is a baseline for "generic SR", not a remote-sensing model.
- Anything above reflectance 0.3558 (bright roofs, sand, snow) is clipped on input and output, which will cost it
  on bright scenes. Single image, no NIR, no multi-frame.
- Output is not reflectance-constrained: brightness shifts (darker here) will hurt reflectance/spectral metrics.
