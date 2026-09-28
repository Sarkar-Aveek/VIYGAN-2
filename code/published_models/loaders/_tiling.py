"""Shared helpers for the published-model loaders: fixed-size patch tiling with blending, and padding for small inputs."""
import math

import torch
import torch.nn.functional as F


def reflect_pad_to(x, size):
    """Reflect-pad [B,C,H,W] (centred) up to at least size x size. Repeats reflection when the pad exceeds
    the input size (torch's reflect needs pad < dim), which is what a 32x32 tile needs to reach 128."""
    pads = [0, 0, 0, 0]  # left, right, top, bottom
    while x.shape[-2] < size or x.shape[-1] < size:
        ph = max(0, size - x.shape[-2])
        pw = max(0, size - x.shape[-1])
        ph = min(ph, 2 * (x.shape[-2] - 1))
        pw = min(pw, 2 * (x.shape[-1] - 1))
        p = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
        x = F.pad(x, p, mode='reflect')
        pads = [a + b for a, b in zip(pads, p)]
    return x, pads


def _starts(n, win, step):
    if n <= win:
        return [0]
    s = list(range(0, n - win, step))
    s.append(n - win)
    return sorted(set(s))


def _ramp(win_hr, lo_nb, hi_nb, overlap_hr, border_hr, mode):
    """1-D weight along one axis of an HR patch. lo_nb/hi_nb: whether a neighbouring patch exists on that side
    (no ramp at the image edge, so every pixel keeps a non-zero weight)."""
    d = torch.arange(win_hr, dtype=torch.float32) + 0.5
    w = torch.ones(win_hr)
    for has_nb, dist in ((lo_nb, d), (hi_nb, win_hr - d)):
        if not has_nb:
            continue
        if mode == 'crop':  # hard cut at half the overlap (sen2sr.predict_large)
            w = w * (dist >= overlap_hr / 2).float()
        elif mode == 'sigmoid':  # opensr-utils style: discard border px, sigmoid ramp across the overlap
            t = (dist - border_hr) / max(overlap_hr - border_hr, 1)
            s = torch.sigmoid((t - 0.5) * 12)
            s[dist < border_hr] = 0
            w = w * s.clamp_min(0)
        else:
            raise ValueError(mode)
    return w


@torch.no_grad()
def tiled(run, x, win, overlap, scale=4, mode='sigmoid', border=0, batch=8, out_ch=None):
    """Run `run` ([N,C,win,win] -> [N,Co,win*s,win*s]) over x [B,C,H,W] in win x win LR patches with `overlap`
    LR px, blend the HR outputs, return [B,Co,H*s,W*s] on x.device. Inputs smaller than win are reflect-padded
    to win first and cropped back afterwards."""
    B, C, H0, W0 = x.shape
    x, pads = reflect_pad_to(x, win)
    _, _, H, W = x.shape
    ys, xs = _starts(H, win, win - overlap), _starts(W, win, win - overlap)
    whr = win * scale
    out = None
    acc = torch.zeros(1, 1, H * scale, W * scale, device=x.device)
    coords = [(y, xx) for y in ys for xx in xs]
    for b in range(B):
        for i in range(0, len(coords), batch):
            chunk = coords[i:i + batch]
            patches = torch.stack([x[b, :, y:y + win, xx:xx + win] for y, xx in chunk])
            sr = run(patches).float()
            if out is None:
                out = torch.zeros(B, sr.shape[1], H * scale, W * scale, device=x.device)
            for (y, xx), p in zip(chunk, sr):
                wy = _ramp(whr, y > 0, y + win < H, overlap * scale, border * scale, mode)
                wx = _ramp(whr, xx > 0, xx + win < W, overlap * scale, border * scale, mode)
                w = (wy[:, None] * wx[None, :]).to(x.device)
                out[b, :, y * scale:y * scale + whr, xx * scale:xx * scale + whr] += p * w
                if b == 0:
                    acc[0, 0, y * scale:y * scale + whr, xx * scale:xx * scale + whr] += w
    out = out / acc.clamp_min(1e-8)
    l, r, t, bt = [p * scale for p in pads]
    return out[:, :, t:t + H0 * scale, l:l + W0 * scale]
