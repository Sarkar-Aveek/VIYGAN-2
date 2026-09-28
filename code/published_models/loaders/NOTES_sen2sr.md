# SEN2SR Lite, NonReference_RGBN_x4 (`sen2sr_lite`)

Loader: `sen2sr_lite.py` (named so because a file called `sen2sr.py` shadows the `sen2sr` package when the loaders
folder is on `sys.path`).

- **Source code:** https://github.com/ESAOpenSR/SEN2SR (package `sen2sr` 0.6.1, MIT, per the README badge).
- **Weights:** https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/NonReference_RGBN_x4/mlm.json
  fetched with `mlstac.download` into `code/published_models/weights_public/SEN2SRLite_RGBN/` (model.safetensor 2.3 MB,
  hard_constraint.safetensor 1.0 MB, load.py, mlm.json, example_data). HF model licence tag: **CC0-1.0**.
- **Architecture:** `CNNSR(4, 4, 24, 4, True, False, 6)` from `sen2sr/models/opensr_baseline/cnn.py` = SPAN (Swift
  Parameter-free Attention Network), 24 feature channels, 6 blocks, x4, followed by `HardConstraint`: the SR output
  is clamped >= 0, then its FFT low frequencies (a fixed 512x512 mask, ~radius 50 of the 256 HR Nyquist, i.e.
  ~0.78 of the LR Nyquist) are replaced by those of `interpolate(lr, mode="bicubic", antialias=True)`.
- **Parameters:** 572,336 (counted from the safetensors; mlm.json says `mlm:total_parameters: 472496`).
- **Training data:** mlm.json: "A Swift Parameter-free Attention Network (SPAN) trained on the SEN2NAIPv2 dataset
  to enhance RGBN Sentinel-2 bands, improving spatial resolution from 10 meters to 2.5 meters." (Targets: NAIP,
  USA aerial imagery, harmonised to S2 reflectance.)

## Preprocessing (quoted)

README, "From 10m Sentinel-2 bands to 2.5m":
```python
bands=["B04", "B03", "B02", "B08"],
...
edge_size=128,
...
original_s2_numpy = (da[11].compute().to_numpy() / 10_000).astype("float32")
X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
model = mlstac.load("model/SEN2SRLite_RGBN").compiled_model(device=device)
superX = model(X[None]).squeeze(0)
```
mlm.json input: `"bands": ["B04","B03","B02","B08"]`, shape `[-1, 4, 128, 128]`. README on large images:
"Although the model is trained to operate on fixed-size 128x128 patches, the `sen2sr.predict_large` utility
automatically segments larger inputs into these tiles ... An overlap margin (e.g., 32 pixels)".

## What `fn` does

1. Reorders (B02, B03, B04, B08) -> (B04, B03, B02, B08). Values stay reflectance (DN/10000), as in the README.
2. Tiles into 128x128 LR windows, 32 px overlap, hard cut at half the overlap (same scheme as `predict_large`,
   reimplemented because `predict_large` only handles square single images and returns CPU tensors).
   Inputs smaller than 128 (e.g. a 32x32 tile) are reflect-padded to 128 and the output cropped back.
3. Returns channels 0..2 = (R, G, B) reflectance. NIR output dropped.

## Check (Hyderabad block 94096_59104, 2025-11-13)

- 128x128 crop -> [1,3,512,512], range 0.03..0.45, p1/p50/p99 0.047/0.113/0.225. Channel means equal bicubic to
  1e-6 (the HardConstraint at work). Full 512x512 -> [1,3,2048,2048]; crop region of the full run vs the
  standalone crop run: mean |diff| 1.7e-4 (tiling is seamless).
- `check_sen2sr_lite.png`: left bicubic x4, right SEN2SR. Colours correct, mild sharpening, no hallucinated texture.
- **Runtime:** 0.015 s per 128 px LR patch (fp32, RTX 5060 Laptop); 0.12 s for 512x512 LR. Peak VRAM 1.65 GB.

## Fairness notes

- Single image (one date) vs our 8 frames: the caller has to pick which frame(s) to feed.
- A 32x32 tile is padded to 128 with mirrored content; SEN2SR sees fake context. Feeding 128+ px blocks and
  cropping is fairer.
- The HardConstraint forces low frequencies to the bicubic input, so SEN2SR is spectrally faithful to S2 by
  construction and cannot match ArcGIS colours; compare it to model 2/3 (S2 colours), not to model 1's colours.
- Trained on USA (NAIP) targets, not Indian scenes.
