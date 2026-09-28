"""Stock Real-ESRGAN realesr-general-x4v3 (SRVGGNetCompact, generic photo SR, BSD-3), pure x4v3 weights
(denoise_strength 1; the official CLI default 0.5 would blend in realesr-general-wdn-x4v3).

Input: the Sentinel-2 true-colour image clip(refl / 0.3558, 0, 1) in RGB order (B04, B03, B02); output is
clamped to 0..1 like the official RealESRGANer and converted back with * 0.3558, so reflectance > 0.3558 saturates.
NIR is ignored. Fully convolutional: runs whole inputs up to 512x512 LR, larger ones in 256 px tiles (overlap 16).
"""
import os
import pathlib
import sys
import urllib.request

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tiling import tiled  # noqa: E402

NAMES = ["realesrgan_general"]
WEIGHTS = pathlib.Path(__file__).resolve().parents[1] / "weights_public" / "realesrgan" / "realesr-general-x4v3.pth"
URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
TCI = 0.3558


def load(name, device):
    assert name in NAMES, name
    from basicsr.archs.srvgg_arch import SRVGGNetCompact
    if not WEIGHTS.exists():
        WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, WEIGHTS)
    net = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
    sd = torch.load(WEIGHTS, map_location='cpu', weights_only=True)
    net.load_state_dict(sd['params'], strict=True)
    net = net.eval().to(device)

    def run(p):
        return net(p).clamp(0, 1)

    @torch.no_grad()
    def fn(x):
        rgb = (x.float()[:, [2, 1, 0]] / TCI).clamp(0, 1)  # (B02,B03,B04) -> (R,G,B) display space
        if rgb.shape[-2] * rgb.shape[-1] <= 512 * 512:
            out = torch.cat([run(rgb[i:i + 1]) for i in range(rgb.shape[0])])
        else:
            out = tiled(run, rgb, win=256, overlap=16, scale=4, mode='crop', batch=1)
        return out * TCI

    return fn
