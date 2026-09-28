"""
Extra metrics, registered with BasicSR so configs can use them next to calculate_psnr / calculate_ssim:
  calculate_cpsnr      PSNR maximised over small shifts and per-channel brightness bias (PROBA-V)
  calculate_lpips      perceptual distance, lower is better (lpips_model: alexnet | vgg)
  calculate_edge_f1    are the output's structure edges where the target's are (blurred Canny, 1 px tolerance), 0-1
  calculate_grad_corr  correlation of Sobel gradient magnitudes of output and target, -1..1
The two structure metrics work on contrast-stretched grayscale, so they ignore colour: an ArcGIS-colour and a
Sentinel-colour output with the same edges score the same.
All take uint8 HWC images `img` (SR) and `img2` (GT). Models are loaded once and cached.
"""
import cv2
import numpy as np
import torch

from basicsr.utils.registry import METRIC_REGISTRY

_models = {}


def _to_tensor(img, device):
    return torch.as_tensor(img).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255


def _structure_gray(img, crop_border):
    """Grayscale, cropped, stretched so its 2nd-98th percentiles span 0-255 (colour and contrast removed)."""
    g = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2GRAY).astype(np.float32)
    if crop_border:
        g = g[crop_border:-crop_border, crop_border:-crop_border]
    lo, hi = np.percentile(g, (2, 98))
    return np.clip((g - lo) / max(hi - lo, 1e-3) * 255, 0, 255).astype(np.uint8)


@METRIC_REGISTRY.register()
def calculate_edge_f1(img, img2, crop_border=4, tolerance=1, sigma=2.0, **kwargs):
    """Structure edges only: Gaussian blur (sigma px) before Canny, so texture doesn't count as edges.

    Settings chosen on 100 val tiles (2026-09-18): without the blur ~33% of pixels are 'edges' and any output
    scores near 1; with sigma 2, Canny 50-150, 1 px tolerance the target's edge density is 12% and India 20k scores
    0.76, the US Satlas model 0.54, bicubic 0.42, and another tile's output (chance level) 0.38.
    """
    a, b = _structure_gray(img, crop_border), _structure_gray(img2, crop_border)
    if sigma:
        a, b = cv2.GaussianBlur(a, (0, 0), sigma), cv2.GaussianBlur(b, (0, 0), sigma)
    ea, eb = cv2.Canny(a, 50, 150) > 0, cv2.Canny(b, 50, 150) > 0
    if not ea.any() and not eb.any():
        return 1.0                      # both flat: nothing to disagree on
    if not ea.any() or not eb.any():
        return 0.0
    kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    near_a = cv2.dilate(ea.astype(np.uint8), kernel) > 0
    near_b = cv2.dilate(eb.astype(np.uint8), kernel) > 0
    precision = (ea & near_b).sum() / ea.sum()
    recall = (eb & near_a).sum() / eb.sum()
    return 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))


@METRIC_REGISTRY.register()
def calculate_grad_corr(img, img2, crop_border=4, **kwargs):
    def magnitude(img_):
        g = _structure_gray(img_, crop_border).astype(np.float32)
        return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)).ravel()
    ma, mb = magnitude(img), magnitude(img2)
    if ma.std() < 1e-6 or mb.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(ma, mb)[0, 1])


@METRIC_REGISTRY.register()
def calculate_cpsnr(img, img2, crop_border, max_offset=8, **kwargs):
    img1 = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    assert img1.shape == img2.shape, f'Image shapes are different: {img1.shape}, {img2.shape}.'
    if crop_border:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border]

    # Crop img1 with top-left at (row, col) and img2 at (max_offset - row, max_offset - col); keep the best MSE.
    crop_h, crop_w = img1.shape[0] - max_offset, img1.shape[1] - max_offset
    best_mse = None
    for row in range(max_offset + 1):
        for col in range(max_offset + 1):
            a = img1[row:row + crop_h, col:col + crop_w]
            b = img2[max_offset - row:max_offset - row + crop_h, max_offset - col:max_offset - col + crop_w]
            b = b + (a - b).mean(axis=(0, 1))  # remove the per-channel brightness bias
            mse = np.mean((a - b) ** 2)
            if best_mse is None or mse < best_mse:
                best_mse = mse
    return float('inf') if best_mse == 0 else 10. * np.log10(255. ** 2 / best_mse)


@METRIC_REGISTRY.register()
def calculate_lpips(img, img2, lpips_model='alexnet', **kwargs):
    import lpips
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    key = ('lpips', lpips_model)
    if key not in _models:
        _models[key] = lpips.LPIPS(net={'alexnet': 'alex', 'vgg': 'vgg'}[lpips_model], verbose=False).to(device)
    with torch.no_grad():
        return _models[key](_to_tensor(img, device), _to_tensor(img2, device)).item()
