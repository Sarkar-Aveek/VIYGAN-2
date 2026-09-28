"""SEN2SR Lite (NonReference_RGBN_x4) loader. Official weights via mlstac, kept with its HardConstraint.

Model input: [N, 4, 128, 128] reflectance (0..1) in band order (B04, B03, B02, B08) = R, G, B, NIR
(mlm.json "bands": ["B04","B03","B02","B08"]). The HardConstraint's low-pass mask is a fixed 512x512 tensor, so
the model only accepts 128x128 LR patches: larger inputs are tiled like `sen2sr.predict_large` (128 px windows,
32 px overlap, hard cut at half the overlap); smaller ones are reflect-padded to 128 and cropped back.
"""
import os
import pathlib
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tiling import tiled  # noqa: E402

NAMES = ["sen2sr_lite"]
ROOT = pathlib.Path(__file__).resolve().parents[1] / "weights_public"
MODEL_DIR = ROOT / "SEN2SRLite_RGBN"
MLM_URL = "https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/NonReference_RGBN_x4/mlm.json"


def load(name, device):
    assert name in NAMES, name
    import mlstac
    if not (MODEL_DIR / "model.safetensor").exists():
        mlstac.download(file=MLM_URL, output_dir=str(MODEL_DIR))
    model = mlstac.load(str(MODEL_DIR)).compiled_model(device=device)
    model.eval()

    def run(p):  # p: [N, 4, 128, 128] in (R, G, B, N)
        return model(p)

    @torch.no_grad()
    def fn(x):
        x = x.float()
        rgbn = x[:, [2, 1, 0, 3]]  # (B02,B03,B04,B08) -> (B04,B03,B02,B08)
        sr = tiled(run, rgbn, win=128, overlap=32, scale=4, mode='crop', batch=16)
        return sr[:, :3].contiguous()  # already (R, G, B) = (B04, B03, B02)

    return fn
