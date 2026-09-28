"""
Super-resolve Sentinel-2 stacks with one of our two models (no ground truth needed).

Every png under --input must be a stack of 32x32 RGB frames, shape [n*32, 32, 3] (the format in data/sentinel2_l2a and
the training folders; n >= 8). For each input it saves <name>_sr.png (128x128, 2.39 m) and <name>_lr.png (the first
frame used, 32x32, 9.55 m) in --output, keeping the input's sub-folders.

--model chooses the network:
  arcgis_B  output in ArcGIS World Imagery colours (looks like the familiar basemap)
  s2colour  output in Sentinel-2's measured colours: every 10 m cell keeps the input's colour (colour lock inside
            the network, `input_lowpass`), so the image stays quantitatively comparable with Sentinel-2

    python infer.py --model s2colour --input <folder of stacks> --output results/
"""
import os
import glob
import argparse

import bootstrap  # noqa: F401  (code folder as working directory, basicsr compatibility)
import numpy as np
import torch
from PIL import Image

from esrgan.dataset import split_frames, pick_frames
from esrgan.networks import SSR_RRDBNet

WEIGHTS = {"arcgis_B": os.path.join("..", "weights", "arcgis_B_generator.pth"),
           "s2colour": os.path.join("..", "weights", "s2colour_generator.pth")}


def load(model, weights, device):
    net = SSR_RRDBNet(24, 3, 4, input_lowpass=(model == "s2colour"))
    ckpt = torch.load(weights, map_location='cpu', weights_only=True)
    net.load_state_dict(ckpt['params_ema'] if 'params_ema' in ckpt else ckpt['params'], strict=True)
    return net.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=sorted(WEIGHTS), required=True)
    parser.add_argument('--input', required=True, help='folder searched recursively for Sentinel-2 stack pngs')
    parser.add_argument('--output', required=True)
    parser.add_argument('--weights', default=None, help='default: ../weights/<model>_generator.pth')
    parser.add_argument('--limit', type=int, default=None, help='only process the first N pngs')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = load(args.model, args.weights or WEIGHTS[args.model], device)

    pngs = sorted(glob.glob(os.path.join(args.input, '**', '*.png'), recursive=True))[:args.limit]
    print(f"Running {args.model} on {len(pngs)} images ({device}).")

    with torch.no_grad():
        for png in pngs:
            frames = split_frames(np.array(Image.open(png).convert('RGB')))
            if len(frames) < 8:
                print(f"Skipping {png}: {len(frames)} frames, need 8")
                continue
            chosen = pick_frames(frames, 8, rand=False)
            lr = torch.cat([frames[i] for i in chosen]).unsqueeze(0).to(device).float() / 255
            sr = net(lr).clamp(0, 1).mul(255).round().byte()[0].permute(1, 2, 0).cpu().numpy()

            rel = os.path.splitext(os.path.relpath(png, args.input))[0]
            out_base = os.path.join(args.output, rel)
            os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
            Image.fromarray(sr).save(f'{out_base}_sr.png')
            Image.fromarray(frames[chosen[0]].permute(1, 2, 0).numpy()).save(f'{out_base}_lr.png')

    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
