"""
Paired Sentinel-2 / ArcGIS dataset, in the layout written by collectors/build_training_data.py:

    {sentinel2_path}/{tile}/tci.png   8 stacked 32x32 L2A true-colour frames, shape [8*32, 32, 3]
    {hr_path}/{tile}/rgb.png          128x128 RGB ArcGIS image
"""
import os
import glob
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torchvision
from torch.utils import data

from basicsr.utils.registry import DATASET_REGISTRY

S2_SIZE = 32


def has_black_pixels(img):
    """True if any pixel of a [C, H, W] tensor is 0 in every channel (no data)."""
    return bool((img.sum(dim=0) == 0).any())


def split_frames(stack):
    """[T*32, 32, C] image (HWC numpy or CHW tensor) -> list of CHW tensors, one per frame."""
    if isinstance(stack, np.ndarray):
        stack = torch.from_numpy(np.ascontiguousarray(stack)).permute(2, 0, 1)
    return list(stack.reshape(stack.shape[0], -1, S2_SIZE, S2_SIZE).permute(1, 0, 2, 3))


def pick_frames(frames, n, rand=True):
    """Pick n frame indices, preferring frames without no-data pixels. rand=False takes the first valid ones."""
    goods = [i for i, f in enumerate(frames) if not has_black_pixels(f)]
    bads = [i for i in range(len(frames)) if i not in goods]
    if len(goods) >= n:
        return random.sample(goods, n) if rand else goods[:n]
    need = n - len(goods)
    return goods + (random.sample(bads, need) if rand else bads[:need])


def augment(lr, hr):
    """The same random flip / 90-degree rotation for the input stack [C, 32, 32] and the target [3, 128, 128]."""
    k = random.randrange(4)
    if k:
        lr, hr = torch.rot90(lr, k, (1, 2)), torch.rot90(hr, k, (1, 2))
    if random.random() < 0.5:
        lr, hr = lr.flip(2), hr.flip(2)
    return lr.contiguous(), hr.contiguous()


@DATASET_REGISTRY.register()
class S2ArcGISDataset(data.Dataset):
    """
    Returns {'hr': [3, 128, 128] uint8, 'lr': [n_s2_images*3, 32, 32] uint8 (the frames stacked), 'Index', 'Phase',
    'Chip'}. The model divides by 255.

    Config keys: sentinel2_path, hr_path, n_s2_images, phase (set by the options parser), and optionally
      max_tiles      keep only this many tiles, evenly spaced over the sorted tile list (e.g. a quick val set),
      random_frames  pick frames at random (default true); false takes the first clean ones, for repeatable val.
      augment        random flips / 90-degree rotations, the same for input and target (default false).
      preload        decode every tile once into two uint8 arrays in RAM (~74 KB per tile,
                     7.4 GB for 100k). Workers share them copy-on-write when they are forked (Linux); on Windows
                     each worker would get its own copy, so use num_worker_per_gpu: 0 there.
      preload_workers  threads for the initial decode (default 16).
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.split = opt['phase']
        self.n_s2_images = int(opt['n_s2_images'])
        self.random_frames = opt.get('random_frames', True)
        self.augment = bool(opt.get('augment', False))
        self.s2_path = opt['sentinel2_path']
        self.hr_path = opt['hr_path']
        if not (os.path.exists(self.s2_path) and os.path.exists(self.hr_path)):
            raise FileNotFoundError(f"Data folders not found: {self.s2_path}, {self.hr_path} "
                                    "(build them with collectors/build_training_data.py)")

        self.datapoints = []
        for hr in sorted(glob.glob(os.path.join(self.hr_path, '**', '*.png'), recursive=True)):
            tile = os.path.basename(os.path.dirname(hr))
            self.datapoints.append((hr, os.path.join(self.s2_path, tile, 'tci.png'), tile))
        max_tiles = opt.get('max_tiles')
        if max_tiles and len(self.datapoints) > max_tiles:
            step = (len(self.datapoints) - 1) / max(max_tiles - 1, 1)
            self.datapoints = [self.datapoints[round(i * step)] for i in range(max_tiles)]

        self.preloaded = False
        if opt.get('preload'):
            self._preload(int(opt.get('preload_workers', 16)))
        print(f"Number of datapoints for split {self.split}: {len(self.datapoints)}"
              + (" (preloaded into RAM)" if self.preloaded else ""))

    def _preload(self, workers):
        """Decode every pair once. Pairs that the loader would skip (HR no-data, short stacks) are dropped here."""
        def load(dp):
            hr = torchvision.io.read_image(dp[0])
            if has_black_pixels(hr):
                return None
            s2 = torchvision.io.read_image(dp[1])
            frames = split_frames(s2)
            if len(frames) < self.n_s2_images:
                return None
            return hr.numpy(), torch.stack(frames).numpy()

        with ThreadPoolExecutor(workers) as ex:
            loaded = list(ex.map(load, self.datapoints, chunksize=256))
        keep = [i for i, x in enumerate(loaded) if x is not None]
        n_frames = max(loaded[i][1].shape[0] for i in keep)
        self.hr_all = np.zeros((len(keep), 3, 128, 128), np.uint8)
        self.lr_all = np.zeros((len(keep), n_frames, 3, S2_SIZE, S2_SIZE), np.uint8)
        for j, i in enumerate(keep):
            self.hr_all[j] = loaded[i][0]
            self.lr_all[j, :loaded[i][1].shape[0]] = loaded[i][1]
        self.datapoints = [self.datapoints[i] for i in keep]
        self.preloaded = True

    def __len__(self):
        return len(self.datapoints)

    def _frames_item(self, index):
        """(hr [3,128,128] uint8, list of [3,32,32] frames) or None if the pair should be skipped."""
        if self.preloaded:
            return torch.from_numpy(self.hr_all[index]), list(torch.from_numpy(self.lr_all[index]))
        hr_path, s2_path, tile = self.datapoints[index]
        img_hr = torchvision.io.read_image(hr_path)
        if not os.path.exists(s2_path):
            raise FileNotFoundError(f"Missing Sentinel-2 file for tile {tile}: {s2_path}")
        if has_black_pixels(img_hr):
            return None
        frames = split_frames(torchvision.io.read_image(s2_path))
        return None if len(frames) < self.n_s2_images else (img_hr, frames)

    def __getitem__(self, index):
        # Skip the rare pairs whose HR image has no-data pixels or whose S2 stack has too few frames.
        for _ in range(len(self.datapoints)):
            tile = self.datapoints[index][2]
            item = self._frames_item(index)
            if item is None:
                index = (index + 1) % len(self.datapoints)
                continue
            img_hr, frames = item
            img_lr = torch.cat([frames[i] for i in pick_frames(frames, self.n_s2_images, rand=self.random_frames)])
            if self.augment:
                img_lr, img_hr = augment(img_lr, img_hr)
            return {'hr': img_hr, 'lr': img_lr, 'Index': index, 'Phase': self.split, 'Chip': tile}
        raise RuntimeError("No valid datapoints in the dataset.")
