# Weights

| file | model | size | SHA-256 |
|---|---|---|---|
| `arcgis_B_generator.pth` | arcgis_B (ArcGIS colours), final checkpoint (24,000 iterations) | 67 MB | `e9cd9a50156cb2454ce47dbd57c9f95df0664580ee69216df7e287a222c796a1` |
| `s2colour_generator.pth` | s2colour (Sentinel-2 colours), final checkpoint (stage 2, 17,322 iterations) | 67 MB | `1e22a2d62a38b8de4224811d82e90e080a9423e90628f54e593fc048d9559ea6` |

- Each file holds one key, `params_ema`: the exponential moving average of the generator weights (decay 0.999),
  which is what validation and every result in this repository used. The training checkpoints also held the raw
  weights (`params`); they were dropped to halve the size. The outputs are identical (maximum difference 0.0).
- Architecture: `code/esrgan/networks.py` `SSR_RRDBNet(24, 3, 4)`, 16.7 M parameters. **s2colour must be built with
  `input_lowpass=True`** (its colour lock); `code/infer.py --model s2colour` does this.
- Stored with Git LFS: run `git lfs pull` after cloning.

```python
from esrgan.networks import SSR_RRDBNet
import torch
net = SSR_RRDBNet(24, 3, 4, input_lowpass=True)          # False for arcgis_B
net.load_state_dict(torch.load("../weights/s2colour_generator.pth", weights_only=True)["params_ema"], strict=True)
```
