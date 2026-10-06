# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright (c) 2026 Aveek Sarkar (https://github.com/Sarkar-Aveek)
# Non-commercial use only; see LICENSE-s2colour.md in this folder. The rest of code/ is Apache 2.0.
"""
s2colour: the parts that make the Sentinel-colour model (model 3).

  frames_median      the per-pixel median of the input frames: the colour each 10 m cell must keep
  lowpass_transfer   the colour lock: push every scale x scale block mean of the output onto the input's colour
                     with smooth corrections, keeping the network's detail (used inside SSR_RRDBNet with
                     `input_lowpass`, and after any model to put its output into Sentinel-2 colours)
  colour_transfer    lowpass_transfer with the detail capped, for dark scenes (the `gt_input_colour` option)
  match_mean         the colour-blind pixel and perceptual losses (`colour_invariant_loss`)
  highpass           what the high-pass discriminator sees (`disc_highpass`)
"""
import torch.nn.functional as F


def frames_median(x):
    """[b, n*3, h, w] frame stack -> [b, 3, h, w] per-pixel median over the frames."""
    return x.view(x.shape[0], -1, 3, x.shape[2], x.shape[3]).median(dim=1).values


def lowpass_transfer(out, ref, scale=4, steps=3):
    """Give `out` [b, 3, H, W] the colours of `ref` [b, 3, H/scale, W/scale] while keeping out's detail: its
    scale x scale block means are pushed onto ref with smooth corrections. Used inside the generator
    (`input_lowpass`) and after it, to put any model's output into Sentinel-2 colours."""
    for _ in range(steps):
        diff = ref - F.avg_pool2d(out, scale)
        out = out + F.interpolate(diff, scale_factor=scale, mode='bicubic', align_corners=False)
    return out


# Detail strength (mean |detail| / mean brightness, detail = finer than one input pixel) that a tile may keep.
# Measured on 800 held-out test tiles: the ArcGIS targets sit at 0.145 (p90 0.202) and lowpass_transfer's outputs
# at 0.152, so 0.22 leaves ordinary tiles untouched; a dark winter Shimla block sits at 0.299 and is pulled back.
MAX_DETAIL_RATIO = 0.22


def colour_transfer(out, ref, scale=4, steps=3):
    """`lowpass_transfer` without the grain it causes on dark scenes: the detail is capped at MAX_DETAIL_RATIO.

    The transfer only moves block means, so a scene it darkens (a winter hillside: Sentinel mean 0.10 where the
    model's output was 0.25) keeps its detail at the old strength. Detail that was 24% of the brightness becomes
    37%, and that over-contrast reads as grain. Here the detail is scaled back afterwards so its strength relative
    to brightness is the one the model's own output had. Block means, i.e. the colours, are unchanged: the detail
    has zero mean per block.
    """
    transferred = lowpass_transfer(out, ref, scale, steps)
    low = F.interpolate(F.avg_pool2d(transferred, scale), scale_factor=scale, mode='bicubic', align_corners=False)
    det = transferred - low
    ratio = det.abs().mean(dim=(1, 2, 3), keepdim=True) / low.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-3)
    return (low + det * (MAX_DETAIL_RATIO / ratio.clamp_min(1e-6)).clamp(max=1.0)).clamp(0, 1)


def match_mean(ref, img):
    """Shift `img` to `ref`'s per-channel mean, without passing gradients through the shift."""
    return img + (ref.mean(dim=(2, 3), keepdim=True) - img.mean(dim=(2, 3), keepdim=True)).detach()


def highpass(img, scale):
    """Detail finer than one input pixel: the image minus its smooth block-mean version."""
    return img - F.interpolate(F.avg_pool2d(img, scale), scale_factor=scale, mode='bicubic', align_corners=False)
