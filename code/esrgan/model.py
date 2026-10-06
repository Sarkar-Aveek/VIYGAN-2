"""
GAN training logic: generator + discriminator updates, EMA, validation.
Adapted from https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/models/esrgan_model.py
"""
import os
from collections import OrderedDict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import USMSharp, get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY

from esrgan import s2colour  # PolyForm Noncommercial, see LICENSE-s2colour.md
from esrgan.s2colour import colour_transfer, frames_median


@MODEL_REGISTRY.register()
class SSRESRGANModel(SRGANModel):
    """
    The generator gets a stack of Sentinel-2 frames and produces a 4x image. The discriminator sees the SR or GT
    image, optionally concatenated with the same LR frames upsampled to 128 px (`feed_disc_lr`).

    Optional:
      train:grad_opt: {loss_weight: w}   L1 between the Sobel gradients of the output and the (sharpened) target:
                                         rewards edges in the right place, ignores colour and flat areas.
    For the Sentinel-colour model (network_g input_lowpass), trained against ArcGIS targets whose colour the input
    can't predict:
      colour_invariant_loss: true        the targets are shifted to the output's per-channel mean before the pixel
                                         and perceptual losses (score structure, not colour)
      disc_highpass: true                the discriminator sees only the high frequencies of the SR/GT image
    Or, instead of those three, the target itself is put into the input's colours before every loss and before the
    discriminator, so the model learns to output Sentinel colours (and their contrast) on its own:
      gt_input_colour: true              target -> esrgan.networks.colour_transfer onto the median input frame
    """

    def __init__(self, opt):
        super().__init__(opt)
        self.usm_sharpener = USMSharp().to(self.device)

    def init_training_settings(self):
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            get_root_logger().info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # net_g_ema is used for validation and saving
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weights
            self.net_g_ema.eval()

        self.net_d = self.model_to_device(build_network(self.opt['network_d']))
        self.print_network(self.net_d)
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)
        self.net_d.train()
        self.net_g.train()

        self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device) if train_opt.get('pixel_opt') else None
        self.cri_perceptual = (build_loss(train_opt['perceptual_opt']).to(self.device)
                               if train_opt.get('perceptual_opt') else None)
        self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
        self.grad_weight = float(train_opt.get('grad_opt', {}).get('loss_weight', 0))
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        self.sobel = torch.stack([sobel_x, sobel_x.t()]).unsqueeze(1).to(self.device)   # [2, 1, 3, 3]

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        self.setup_optimizers()
        self.setup_schedulers()

    @torch.no_grad()
    def feed_data(self, data):
        self.lr = data['lr'].to(self.device).float() / 255
        if 'hr' in data:
            self.gt = data['hr'].to(self.device).float() / 255
            if self.opt.get('gt_input_colour'):
                self.gt = colour_transfer(self.gt, frames_median(self.lr))
            self.gt_usm = self.usm_sharpener(self.gt)  # sharpened ground truth, as in Real-ESRGAN

    @staticmethod
    def match_mean(ref, img):
        return s2colour.match_mean(ref, img)

    def highpass(self, img):
        return s2colour.highpass(img, self.opt['scale'])

    def disc_input(self, img, lr_up):
        # `disc_highpass`: judge texture, not colour (the GT's colour is not predictable from the input).
        if self.opt.get('disc_highpass'):
            img = self.highpass(img)
        return torch.cat((img, lr_up), dim=1) if self.opt.get('feed_disc_lr') else img

    def grad_loss(self, output, target):
        """L1 between Sobel gradients (x and y, every channel) of the output and the target."""
        k = self.sobel.to(output.dtype)
        c = output.shape[1]
        grad = lambda img: F.conv2d(F.pad(img, (1, 1, 1, 1), mode='replicate'), k.repeat(c, 1, 1, 1),  # noqa: E731
                                    groups=c)
        return F.l1_loss(grad(output), grad(target))

    def optimize_parameters(self, current_iter):
        l1_gt = self.gt_usm if self.opt['l1_gt_usm'] else self.gt
        percep_gt = self.gt_usm if self.opt['percep_gt_usm'] else self.gt
        gan_gt = self.gt_usm if self.opt['gan_gt_usm'] else self.gt

        lr_up = F.interpolate(self.lr, scale_factor=4)  # so the LR frames can be stacked onto the 128 px image

        # generator
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lr)

        loss_dict = OrderedDict()
        if current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters:
            l_g_total = 0
            # `colour_invariant_loss`: score structure only. Each ArcGIS tile carries a colour offset the model
            # cannot predict (mean |bias| 7/255 on val, 83% of it tile-by-tile rather than per acquisition), so
            # the targets are shifted to the output's per-channel mean before the pixel and perceptual losses.
            if self.opt.get('colour_invariant_loss'):
                l1_gt = self.match_mean(self.output, l1_gt)
                percep_gt = self.match_mean(self.output, percep_gt)
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, l1_gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            if self.grad_weight:
                l_g_grad = self.grad_weight * self.grad_loss(self.output, l1_gt)
                l_g_total += l_g_grad
                loss_dict['l_g_grad'] = l_g_grad
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style

            fake_g_pred = self.net_d(self.disc_input(self.output, lr_up))
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # discriminator
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        real_d_pred = self.net_d(self.disc_input(gan_gt, lr_up))
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()

        fake_d_pred = self.net_d(self.disc_input(self.output, lr_up).detach().clone())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        net = self.net_g_ema if hasattr(self, 'net_g_ema') else self.net_g
        was_training = net.training
        net.eval()
        with torch.no_grad():
            self.output = net(self.lr)
        if was_training:
            net.train()

    def get_current_visuals(self):
        out = OrderedDict(lr=self.lr.detach().cpu(), result=self.output.detach().cpu())
        if hasattr(self, 'gt'):
            out['gt'] = self.gt.detach().cpu()
        return out

    def _initialize_best_metric_results(self, dataset_name, metrics2run):
        if not hasattr(self, 'best_metric_results'):
            self.best_metric_results = dict()
        if dataset_name in self.best_metric_results:
            return
        record = dict()
        for metric, content in metrics2run.items():
            better = content.get('better', 'higher')
            record[metric] = dict(better=better, val=float('-inf') if better == 'higher' else float('inf'), iter=-1)
        self.best_metric_results[dataset_name] = record

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        section = self.opt['val']
        metrics2run = section.get('metrics')
        if metrics2run:
            self.metric_results = {metric: 0 for metric in metrics2run}
            self._initialize_best_metric_results(dataset_name, metrics2run)

        pbar = tqdm(total=len(dataloader), unit='image') if section.get('pbar', False) else None
        vis_dir = self.opt['path']['visualization']

        for idx, val_data in enumerate(dataloader):
            img_name = str(idx)  # batch size 1
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            metric_data = {'img': tensor2img([visuals['result']])}
            save_base = os.path.join(vis_dir, img_name, f'{img_name}_{current_iter}')

            if 'gt' in visuals:
                metric_data['img2'] = tensor2img([visuals['gt']])
                if save_img:
                    imwrite(metric_data['img2'], f'{save_base}_gt.png')
                del self.gt
            if save_img:
                imwrite(metric_data['img'], f'{save_base}.png')

            del self.lr
            del self.output
            torch.cuda.empty_cache()

            if metrics2run:
                for name, opt_ in metrics2run.items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        if metrics2run:
            for metric in self.metric_results:
                self.metric_results[metric] /= (idx + 1)
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
