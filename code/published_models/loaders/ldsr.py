"""ESA OpenSR LDSR-S2 (latent diffusion, RGBN x4, MIT). Config = opensr-model v1.0.8 `configs/config_10m.yaml`
(identical to main), checkpoint `opensr-ldsrs2_v1_0_0.ckpt` from huggingface.co/simon-donike/RS-SR-LTDF.

Model input: [N, 4, 128, 128] reflectance 0..1 in (R, G, B, NIR) = (B04, B03, B02, B08) order (opensr-utils:
"Sentinel-2 10m bands (R-G-B-NIR)"; demo.py: `lr = (lr/10_000)`). Sampling uses the config defaults
(DDIM 100 steps, eta 0.95, temperature 1.0) and the package's histogram matching of each SR band to the LR band.
Large inputs are tiled like opensr-utils defaults: 128 px LR windows, 12 px overlap, 2 px border discarded,
sigmoid blending. Inputs < 128 px are reflect-padded to 128 (the package does the same) and cropped back.
Sampling is stochastic: fn seeds torch with SEED at every call so outputs are reproducible.
"""
import os
import pathlib
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tiling import tiled  # noqa: E402

NAMES = ["ldsr_s2"]
DIR = pathlib.Path(__file__).resolve().parents[1] / "weights_public" / "ldsr_s2"
CONFIG = DIR / "config_10m_v1.0.8.yaml"
CKPT = DIR / "opensr-ldsrs2_v1_0_0.ckpt"
SEED = 0
BATCH = 1  # 128x128 patches per sampler call (batch 4 fails with a cuDNN execution error on the 8 GB 5060)


def load(name, device):
    assert name in NAMES, name
    from omegaconf import OmegaConf
    import opensr_model
    config = OmegaConf.load(CONFIG)
    model = opensr_model.SRLatentDiffusion(config, device=device)
    # load_pretrained downloads from HF when the path does not exist; the file is already in weights_public
    model.load_pretrained(str(CKPT))
    model.eval()

    def run(p):  # [N, 4, 128, 128] (R,G,B,N) -> [N, 4, 512, 512]
        return model.forward(p)  # README defaults: sampling_steps etc. from config (100)

    @torch.no_grad()
    def fn(x):
        torch.manual_seed(SEED)
        rgbn = x.float()[:, [2, 1, 0, 3]].to(device)  # (B02,B03,B04,B08) -> (B04,B03,B02,B08)
        sr = tiled(run, rgbn, win=128, overlap=12, scale=4, mode='sigmoid', border=2, batch=BATCH)
        return sr[:, :3].contiguous()  # (R, G, B)

    return fn
