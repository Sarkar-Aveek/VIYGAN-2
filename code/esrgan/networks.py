"""
Generator and discriminator.

Generator: ESRGAN's RRDBNet, taking a stack of Sentinel-2 frames as input (8 frames x 3 channels = 24 channels).
    Adapted from https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/rrdbnet_arch.py
Discriminator: Real-ESRGAN's U-Net discriminator with spectral norm.
    Taken from https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/archs/discriminator_arch.py

Class names and parameter names are unchanged from the Satlas code, so esrgan_8S2.pth and the India checkpoints
load with strict=True.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.utils import spectral_norm

from basicsr.utils.registry import ARCH_REGISTRY

# The colour lock and the other s2colour parts live in s2colour.py (PolyForm Noncommercial, LICENSE-s2colour.md);
# re-exported here so existing imports keep working.
from esrgan.s2colour import MAX_DETAIL_RATIO, colour_transfer, frames_median, lowpass_transfer  # noqa: F401


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x  # residual scaling from the ESRGAN paper


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""

    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


@ARCH_REGISTRY.register()
class SSR_RRDBNet(nn.Module):
    """ESRGAN generator, 4x only. Input [b, n_frames*3, 32, 32] in 0-1, output [b, 3, 128, 128].

    With `input_lowpass`, the output keeps the input's colours: its 4x4 block means are pushed onto the median of
    the input frames and only the network's detail is its own (the Sentinel-colour model). The weights are
    unchanged either way, so esrgan_8S2.pth and the India checkpoints load with strict=True.
    """

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32,
                 input_lowpass=False):
        super().__init__()
        if scale != 4:
            raise ValueError('only scale 4 is implemented')
        self.scale = scale
        self.input_lowpass = input_lowpass
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))

        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        if self.input_lowpass:
            out = self.add_input_lowpass(out, x)
        return out

    def add_input_lowpass(self, out, x, steps=3):
        """Push the output's block means onto the median of the input frames, keeping its detail.

        Each step adds a smooth correction, so the high frequencies stay the network's. Bicubic upsampling is
        only an approximate inverse of the averaging, so the correction is repeated; measured on 24 val tiles,
        the mean block-mean error falls 12.7 -> 1.65 -> 0.60 -> 0.25 of 255 over three steps, with the detail
        still 0.98 correlated with the unconstrained output. Nearest upsampling would be exact in one step but
        leaves visible steps at the 4 px block boundaries.
        """
        return lowpass_transfer(out, frames_median(x), self.scale, steps)


@ARCH_REGISTRY.register()
class SSR_UNetDiscriminatorSN(nn.Module):
    """U-Net discriminator with spectral norm. Input is the SR/GT image, optionally with the upsampled LR stack."""

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super().__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x6 = x6 + x0

        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        return self.conv9(out)


def load_generator(path, device, n_frames=8):
    """Build the generator and load weights from a checkpoint (prefers the EMA weights, as validation used them)."""
    net = SSR_RRDBNet(num_in_ch=n_frames * 3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32)
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    net.load_state_dict(ckpt['params_ema'] if 'params_ema' in ckpt else ckpt['params'], strict=True)
    return net.to(device).eval()
