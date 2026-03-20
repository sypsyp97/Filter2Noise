import os
import sys
import time
import math
from datetime import datetime

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader

TRAIN_FRACTION = 0.10


def to_np(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().float().numpy()
    return np.asarray(t)


class BoundedSoftplus(nn.Module):
    def __init__(self, beta=1.0, threshold=6.0, eps=1e-6):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.eps = eps

    def forward(self, x):
        return torch.clamp(F.softplus(x, beta=self.beta), max=self.threshold) + self.eps


class DeformableAttnBlock(nn.Module):
    def __init__(self, channels, num_points=25, spacing=1, max_offset=4.0):
        super().__init__()
        self.num_points = num_points
        self.max_offset = max_offset

        self.param_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=0, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, 3 * num_points, 1),
        )
        nn.init.zeros_(self.param_conv[-1].weight)
        nn.init.zeros_(self.param_conv[-1].bias)

        k = int(math.sqrt(num_points))
        half_k = k // 2
        oy, ox = torch.meshgrid(
            torch.arange(-half_k, half_k + 1, dtype=torch.float32) * spacing,
            torch.arange(-half_k, half_k + 1, dtype=torch.float32) * spacing,
            indexing='ij',
        )
        self.register_buffer('base_offsets', torch.stack([ox.reshape(-1), oy.reshape(-1)], dim=-1))

    def forward(self, x):
        B, C, H, W = x.shape
        K = self.num_points

        x_padded = F.pad(x, [1, 1, 1, 1], mode='reflect')
        params = self.param_conv(x_padded)

        offsets = torch.tanh(params[:, :2 * K]) * self.max_offset
        attn_logits = params[:, 2 * K:]

        offsets = offsets.view(B, K, 2, H, W)
        base = self.base_offsets.view(1, K, 2, 1, 1)
        total = base + offsets

        dx = total[:, :, 0]
        dy = total[:, :, 1]

        base_x = torch.arange(W, device=x.device, dtype=x.dtype).view(1, 1, 1, W)
        base_y = torch.arange(H, device=x.device, dtype=x.dtype).view(1, 1, H, 1)

        grid_x = (base_x + dx) / (W - 1) * 2 - 1
        grid_y = (base_y + dy) / (H - 1) * 2 - 1

        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(B, K * H, W, 2)

        sampled = F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=True)
        sampled = sampled.view(B, C, K, H, W)

        attn = F.softmax(attn_logits, dim=1).unsqueeze(1)
        return (sampled * attn).sum(dim=2)


class RMSNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        rms = x.norm(2, dim=1, keepdim=True) / math.sqrt(x.shape[1])
        return x / (rms + self.eps) * self.weight


class DLKASigmaPredictor(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=16, aux_drop_prob=0.5):
        super().__init__()
        self.aux_drop_prob = aux_drop_prob
        self.proj_main = nn.Conv2d(1, hidden_dim, 1)
        self.proj_aux = nn.Conv2d(in_channels - 1, hidden_dim, 1) if in_channels > 1 else None
        self.norm1 = RMSNorm2d(hidden_dim)
        self.attn_fine = DeformableAttnBlock(hidden_dim, num_points=25, spacing=1, max_offset=4.0)
        self.attn_coarse = DeformableAttnBlock(hidden_dim, num_points=25, spacing=3, max_offset=6.0)
        self.norm2 = RMSNorm2d(hidden_dim)
        self.gate_proj = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.proj_out = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.sigma_head = nn.Conv2d(hidden_dim, 3, 1)
        self.activation = BoundedSoftplus()

    def _aux_feat(self, x):
        if self.proj_aux is None or x.shape[1] <= 1:
            return 0.0
        aux = self.proj_aux(x[:, 1:])
        if self.training:
            keep = (torch.rand(1, device=x.device) >= self.aux_drop_prob).float()
            return aux * keep
        return aux

    def forward(self, x):
        shortcut = self.proj_main(x[:, 0:1]) + self._aux_feat(x)
        feat = F.gelu(shortcut)
        attn = self.attn_fine(self.norm1(feat))
        attn = self.attn_coarse(attn)
        attn = self.gate_proj(self.norm2(attn))
        gated = feat * attn
        out = self.proj_out(gated) + shortcut
        sigmas = self.activation(self.sigma_head(out))
        return sigmas.permute(0, 2, 3, 1), out


class DAGBF(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=16,
                 num_sample_points=25, max_offset=5.0, aux_drop_prob=0.5):
        super().__init__()
        self.num_sample_points = num_sample_points
        self.max_offset = max_offset

        self.sigma_predictor = DLKASigmaPredictor(
            in_channels=in_channels, hidden_dim=hidden_dim,
            aux_drop_prob=aux_drop_prob,
        )

        self.offset_head = nn.Conv2d(hidden_dim, 2 * num_sample_points, 1)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)

        k = int(math.sqrt(num_sample_points))
        half_k = k // 2
        oy, ox = torch.meshgrid(
            torch.arange(-half_k, half_k + 1, dtype=torch.float32),
            torch.arange(-half_k, half_k + 1, dtype=torch.float32),
            indexing='ij',
        )
        self.register_buffer(
            'regular_offsets',
            torch.stack([ox.reshape(-1), oy.reshape(-1)], dim=-1),
        )

    def forward(self, x, return_sigmas=False):
        x = x.float()
        B, C, H, W = x.shape
        K = self.num_sample_points

        s, feat = self.sigma_predictor(x)
        sx, sy, sr = s[..., 0], s[..., 1], s[..., 2]

        offsets = torch.tanh(self.offset_head(feat)) * self.max_offset
        offsets = offsets.view(B, K, 2, H, W)

        total = self.regular_offsets.view(1, K, 2, 1, 1) + offsets
        dx, dy = total[:, :, 0], total[:, :, 1]

        base_x = torch.arange(W, device=x.device, dtype=x.dtype).view(1, 1, 1, W)
        base_y = torch.arange(H, device=x.device, dtype=x.dtype).view(1, 1, H, 1)
        grid_x = (base_x + dx) / (W - 1) * 2 - 1
        grid_y = (base_y + dy) / (H - 1) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(B * K, H, W, 2)

        x_ch0 = x[:, 0:1]
        x_expanded = x_ch0.unsqueeze(1).expand(B, K, 1, H, W).reshape(B * K, 1, H, W)
        sampled = F.grid_sample(
            x_expanded, grid, mode='bilinear',
            padding_mode='reflection', align_corners=True,
        )
        patches = sampled.view(B, K, 1, H, W).permute(0, 2, 3, 4, 1)

        Cf = feat.shape[1]
        feat_expanded = feat.unsqueeze(1).expand(B, K, Cf, H, W).reshape(B * K, Cf, H, W)
        feat_sampled = F.grid_sample(
            feat_expanded, grid, mode='bilinear',
            padding_mode='reflection', align_corners=True,
        )
        feat_patches = feat_sampled.view(B, K, Cf, H, W).permute(0, 2, 3, 4, 1)

        deformed_dx = dx.permute(0, 2, 3, 1)
        deformed_dy = dy.permute(0, 2, 3, 1)
        spatial_kernel = torch.exp(
            -(deformed_dx ** 2 / (2 * sx.unsqueeze(-1) ** 2)
              + deformed_dy ** 2 / (2 * sy.unsqueeze(-1) ** 2))
        ).unsqueeze(1)

        feat_center = feat.unsqueeze(-1)
        feat_dist = ((feat_center - feat_patches) ** 2).sum(dim=1, keepdim=True)
        range_kernel = torch.exp(
            -feat_dist / (2 * sr.unsqueeze(1).unsqueeze(-1) ** 2)
        )

        combined = spatial_kernel * range_kernel
        normalized = combined / (combined.sum(dim=-1, keepdim=True) + 1e-8)
        output = (patches * normalized).sum(dim=-1)

        if return_sigmas:
            return output, s, offsets
        return output


class DenoisingPipeline(nn.Module):
    def __init__(self, num_stages=1, in_channels=4,
                 hidden_dim=16, num_sample_points=25, max_offset=5.0):
        super().__init__()
        self.stages = nn.ModuleList([
            DAGBF(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                num_sample_points=num_sample_points,
                max_offset=max_offset,
            )
            for _ in range(num_stages)
        ])

    def forward(self, x, return_sigmas=False, output_channel=None):
        sigmas_list, offsets_list = [], []
        current_x = x
        for stage in self.stages:
            if return_sigmas:
                current_x, s, off = stage(current_x, return_sigmas=True)
                sigmas_list.append(s)
                offsets_list.append(off)
            else:
                current_x = stage(current_x)
        if output_channel is not None:
            current_x = current_x[:, output_channel:output_channel + 1]
        if return_sigmas:
            return current_x, sigmas_list, offsets_list
        return current_x


def _gaussian_kernel_1d(sigma, device):
    radius = int(3 * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def gaussian_smooth_2d(image, sigma):
    if sigma <= 0:
        return image
    k1d = _gaussian_kernel_1d(sigma, image.device)
    r = k1d.shape[0] // 2
    c = image.shape[1]
    kh = k1d.view(1, 1, 1, -1).expand(c, -1, -1, -1)
    kv = k1d.view(1, 1, -1, 1).expand(c, -1, -1, -1)
    out = F.conv2d(F.pad(image, [r, r, 0, 0], mode='reflect'), kh, groups=c)
    out = F.conv2d(F.pad(out, [0, 0, r, r], mode='reflect'), kv, groups=c)
    return out


def bilateral_smooth_2d(image, sigma_spatial, sigma_range):
    if sigma_spatial <= 0:
        return image
    b, c, h, w = image.shape
    r = int(3 * sigma_spatial + 0.5)
    k = 2 * r + 1
    padded = F.pad(image, [r] * 4, mode='reflect')
    patches = F.unfold(padded, kernel_size=k, stride=1).view(b, c, k * k, h, w)
    center = image.view(b, c, 1, h, w)
    range_w = torch.exp(
        -((patches - center) ** 2).sum(dim=1, keepdim=True) / (2 * sigma_range ** 2 + 1e-8)
    )
    gy, gx = torch.meshgrid(
        torch.arange(-r, r + 1, dtype=torch.float32, device=image.device),
        torch.arange(-r, r + 1, dtype=torch.float32, device=image.device),
        indexing='ij',
    )
    spatial_w = torch.exp(-(gx ** 2 + gy ** 2) / (2 * sigma_spatial ** 2)).view(1, 1, k * k, 1, 1)
    combined = spatial_w * range_w
    normalized = combined / (combined.sum(dim=2, keepdim=True) + 1e-8)
    return (patches * normalized).sum(dim=2)


def warp_image(image, displacement):
    b, c, h, w = image.shape
    gy, gx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=image.device),
        torch.arange(w, dtype=torch.float32, device=image.device),
        indexing='ij',
    )
    new_y = gy.unsqueeze(0).expand(b, -1, -1) - displacement[:, 0]
    new_x = gx.unsqueeze(0).expand(b, -1, -1) - displacement[:, 1]
    grid = torch.stack([2.0 * new_x / (w - 1) - 1.0, 2.0 * new_y / (h - 1) - 1.0], dim=-1)
    return F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True)


class DemonsRegistration:
    def __init__(self, num_scales=3, num_iters=50, sigma_fluid=2.0,
                 sigma_diffusion=1.5, sigma_presmooth=1.5, sigma_range=0.1):
        self.num_scales = num_scales
        self.num_iters = num_iters
        self.sigma_fluid = sigma_fluid
        self.sigma_diffusion = sigma_diffusion
        self.sigma_presmooth = sigma_presmooth
        self.sigma_range = sigma_range

    def _build_pyramid(self, img):
        pyramid = [img]
        current = img
        for _ in range(self.num_scales - 1):
            current = F.avg_pool2d(current, kernel_size=2, stride=2)
            pyramid.append(current)
        return pyramid

    @torch.no_grad()
    def register(self, fixed, moving):
        eps = 1e-8
        fixed_s = bilateral_smooth_2d(fixed, self.sigma_presmooth, self.sigma_range)
        moving_s = bilateral_smooth_2d(moving, self.sigma_presmooth, self.sigma_range)
        fixed_pyr = self._build_pyramid(fixed_s)
        moving_pyr = self._build_pyramid(moving_s)

        b = fixed.shape[0]
        coarsest = fixed_pyr[-1]
        disp = torch.zeros(b, 2, coarsest.shape[2], coarsest.shape[3],
                           device=fixed.device, dtype=fixed.dtype)

        for level in range(self.num_scales - 1, -1, -1):
            fl = fixed_pyr[level]
            ml = moving_pyr[level]
            hl, wl = fl.shape[2], fl.shape[3]

            if disp.shape[2] != hl or disp.shape[3] != wl:
                sy = hl / disp.shape[2]
                sx = wl / disp.shape[3]
                disp = F.interpolate(disp, size=(hl, wl), mode='bilinear', align_corners=True)
                disp[:, 0] *= sy
                disp[:, 1] *= sx

            for _ in range(self.num_iters):
                warped = warp_image(ml, disp)
                diff = fl - warped
                grad_y = torch.zeros_like(warped)
                grad_x = torch.zeros_like(warped)
                grad_y[:, :, 1:, :] = warped[:, :, 1:, :] - warped[:, :, :-1, :]
                grad_x[:, :, :, 1:] = warped[:, :, :, 1:] - warped[:, :, :, :-1]
                denom = grad_y ** 2 + grad_x ** 2 + diff ** 2 + eps
                update = torch.cat([
                    (diff * grad_y / denom).mean(dim=1, keepdim=True),
                    (diff * grad_x / denom).mean(dim=1, keepdim=True),
                ], dim=1)
                update = gaussian_smooth_2d(update, self.sigma_fluid)
                disp = gaussian_smooth_2d(disp - update, self.sigma_diffusion)

        return disp

    @torch.no_grad()
    def register_batched(self, fixed_list, moving_list, device, batch_size=30):
        disps = []
        for start in range(0, len(fixed_list), batch_size):
            end = min(start + batch_size, len(fixed_list))
            fb = torch.stack(fixed_list[start:end]).to(device)
            mb = torch.stack(moving_list[start:end]).to(device)
            d = self.register(fb, mb)
            disps.append(d.cpu())
            del fb, mb, d
        return torch.cat(disps, dim=0)


class MotionFieldComputer:
    def __init__(self, num_scales=3, num_iters=50, sigma_fluid=2.0,
                 sigma_diffusion=1.5, sigma_presmooth=1.5, sigma_range=0.1,
                 batch_size=30):
        self.registration = DemonsRegistration(
            num_scales=num_scales, num_iters=num_iters,
            sigma_fluid=sigma_fluid, sigma_diffusion=sigma_diffusion,
            sigma_presmooth=sigma_presmooth, sigma_range=sigma_range,
        )
        self.batch_size = batch_size

    def compute_for_indices(self, volume, device, zt_indices):
        x_dim, y_dim, z_dim, t_dim = volume.shape
        warped_prev, warped_next, motion_mag = {}, {}, {}

        unique_zt = sorted(set(zt_indices))
        logger.info(f"Computing motion fields for {len(unique_zt)} (z,t) pairs")

        fixed_prev, moving_prev, keys_prev = [], [], []
        fixed_next, moving_next, keys_next = [], [], []

        for z, t in unique_zt:
            ft = torch.from_numpy(volume[:, :, z, t]).float().unsqueeze(0)
            if t > 0:
                pt = torch.from_numpy(volume[:, :, z, t - 1]).float().unsqueeze(0)
                fixed_prev.append(ft)
                moving_prev.append(pt)
                keys_prev.append((z, t))
            if t < t_dim - 1:
                nt = torch.from_numpy(volume[:, :, z, t + 1]).float().unsqueeze(0)
                fixed_next.append(ft)
                moving_next.append(nt)
                keys_next.append((z, t))

        prev_disp_map = {}
        if fixed_prev:
            logger.info(f"Registering {len(fixed_prev)} prev-frame pairs...")
            disp_prev = self.registration.register_batched(
                fixed_prev, moving_prev, device, self.batch_size,
            )
            for i, (z, t) in enumerate(keys_prev):
                d = disp_prev[i].unsqueeze(0)
                wp = warp_image(moving_prev[i].unsqueeze(0), d)
                warped_prev[(z, t)] = wp.squeeze().numpy()
                prev_disp_map[(z, t)] = disp_prev[i]

        next_disp_map = {}
        if fixed_next:
            logger.info(f"Registering {len(fixed_next)} next-frame pairs...")
            disp_next = self.registration.register_batched(
                fixed_next, moving_next, device, self.batch_size,
            )
            for i, (z, t) in enumerate(keys_next):
                d = disp_next[i].unsqueeze(0)
                wn = warp_image(moving_next[i].unsqueeze(0), d)
                warped_next[(z, t)] = wn.squeeze().numpy()
                next_disp_map[(z, t)] = disp_next[i]

        for z, t in unique_zt:
            if (z, t) not in warped_prev:
                warped_prev[(z, t)] = volume[:, :, z, t].copy()
            if (z, t) not in warped_next:
                warped_next[(z, t)] = volume[:, :, z, t].copy()
            accum = []
            if (z, t) in prev_disp_map:
                accum.append(prev_disp_map[(z, t)])
            if (z, t) in next_disp_map:
                accum.append(next_disp_map[(z, t)])
            if accum:
                avg_d = torch.stack(accum, dim=0).mean(dim=0)
                motion_mag[(z, t)] = torch.sqrt(avg_d[0] ** 2 + avg_d[1] ** 2).numpy()
            else:
                motion_mag[(z, t)] = np.zeros((x_dim, y_dim), dtype=np.float32)

        return {
            'warped_prev': warped_prev,
            'warped_next': warped_next,
            'motion_magnitude': motion_mag,
        }


def setup_environment(seed=77):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_logger():
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
               "<level>{level: <8}</level> | <cyan>{name}</cyan>:"
               "<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )


def load_4d_volume(filepath):
    with h5py.File(filepath, 'r') as f:
        volume = np.transpose(f['volume'][:], (1, 2, 0, 3))
    logger.info(f"Loaded 4D volume: {volume.shape}")
    return volume


def normalize_volume(volume):
    vmin, vmax = volume.min(), volume.max()
    logger.info(f"Normalizing [{vmin:.3f}, {vmax:.3f}] -> [0, 1]")
    return (volume - vmin) / (vmax - vmin)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def compute_spatial_warps(volume, zt_indices, registration, device, batch_size=30):
    z_dim = volume.shape[2]
    fixed_minus, moving_minus, keys_minus = [], [], []
    fixed_plus, moving_plus, keys_plus = [], [], []

    for z, t in zt_indices:
        fixed = torch.from_numpy(volume[:, :, z, t]).float().unsqueeze(0)
        if z > 0:
            mv = torch.from_numpy(volume[:, :, z - 1, t]).float().unsqueeze(0)
            fixed_minus.append(fixed)
            moving_minus.append(mv)
            keys_minus.append((z, t))
        if z < z_dim - 1:
            mv = torch.from_numpy(volume[:, :, z + 1, t]).float().unsqueeze(0)
            fixed_plus.append(fixed)
            moving_plus.append(mv)
            keys_plus.append((z, t))

    warped_z_minus, warped_z_plus = {}, {}

    if fixed_minus:
        logger.info(f"Registering {len(fixed_minus)} spatial z-1->z pairs...")
        disp = registration.register_batched(fixed_minus, moving_minus, device, batch_size)
        for i, (z, t) in enumerate(keys_minus):
            d = disp[i].unsqueeze(0)
            w = warp_image(moving_minus[i].unsqueeze(0), d)
            warped_z_minus[(z, t)] = w.squeeze().numpy()

    if fixed_plus:
        logger.info(f"Registering {len(fixed_plus)} spatial z+1->z pairs...")
        disp = registration.register_batched(fixed_plus, moving_plus, device, batch_size)
        for i, (z, t) in enumerate(keys_plus):
            d = disp[i].unsqueeze(0)
            w = warp_image(moving_plus[i].unsqueeze(0), d)
            warped_z_plus[(z, t)] = w.squeeze().numpy()

    return warped_z_minus, warped_z_plus


class RegularizedN2NLoss:
    def __init__(self, gamma_max=2.0, n_epochs=100):
        self.gamma_max = gamma_max
        self.n_epochs = n_epochs

    def __call__(self, C_batch, D_batch, model, epoch):
        denoised_D = model(D_batch, output_channel=0)
        target = C_batch[:, 0:1]

        with torch.no_grad():
            denoised_C_sg = model(C_batch, output_channel=0)

        diff = denoised_D - target
        exp_diff = denoised_D.detach() - denoised_C_sg

        gamma = epoch / self.n_epochs * self.gamma_max

        loss_rec = F.l1_loss(denoised_D, target)
        loss_reg = gamma * torch.mean((diff - exp_diff) ** 2)

        return loss_rec + loss_reg


class TrainingPairDataset(Dataset):
    def __init__(self, C_tensor, D_tensor):
        self.C = C_tensor
        self.D = D_tensor

    def __len__(self):
        return self.C.shape[0]

    def __getitem__(self, idx):
        return self.C[idx], self.D[idx]


def create_training_pairs(volume, motion_fields, zt_indices):
    C_list, D_list = [], []
    wp = motion_fields['warped_prev']
    wn = motion_fields['warped_next']
    mm = motion_fields['motion_magnitude']

    for z, t in zt_indices:
        D_ch0 = volume[:, :, z, t]
        warped_p, warped_n, motion_m = wp[(z, t)], wn[(z, t)], mm[(z, t)]
        C_ch0 = 0.5 * (warped_p + warped_n)

        C_list.append(np.stack([C_ch0, warped_p, warped_n, motion_m], axis=0))
        D_list.append(np.stack([D_ch0, warped_p, warped_n, motion_m], axis=0))

    C_tensor = torch.from_numpy(np.stack(C_list)).float()
    D_tensor = torch.from_numpy(np.stack(D_list)).float()
    logger.info(f"Created {len(C_list)} training pairs ({TRAIN_FRACTION*100:.0f}% subset), "
                f"tensor shape: {C_tensor.shape}")
    return TrainingPairDataset(C_tensor, D_tensor)


def train(model, dataset, loss_fn, optimizer, scheduler, device, epochs, batch_size):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, persistent_workers=True,
        drop_last=False,
    )
    model.train()

    for epoch in range(epochs):
        total_loss, n_batches = 0.0, 0

        for C_batch, D_batch in loader:
            C_batch = C_batch.to(device, non_blocking=True)
            D_batch = D_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = loss_fn(C_batch, D_batch, model, epoch)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / n_batches:.6f}")


def denoise_volume(model, volume, device):
    model.eval()
    _, _, z_dim, t_dim = volume.shape
    result = np.zeros_like(volume)

    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for z in range(z_dim):
            for t in range(t_dim):
                inp = torch.from_numpy(volume[:, :, z, t]).float().unsqueeze(0).unsqueeze(0).to(device)
                result[:, :, z, t] = to_np(model(inp).squeeze())

    return result


def compute_volume_metrics(clean, test, name="test"):
    z_dim, t_dim = clean.shape[2], clean.shape[3]
    psnr_vals, ssim_vals, rmse_vals = [], [], []

    for z in range(z_dim):
        for t in range(t_dim):
            cs, ts = clean[:, :, z, t], test[:, :, z, t]
            psnr_vals.append(psnr(cs, ts, data_range=1.0))
            ssim_vals.append(ssim(cs, ts, data_range=1.0))
            rmse_vals.append(np.sqrt(np.mean((cs - ts) ** 2)))

    metrics = {'PSNR': np.mean(psnr_vals), 'SSIM': np.mean(ssim_vals), 'RMSE': np.mean(rmse_vals)}
    logger.info(f"{name.upper()} - PSNR: {metrics['PSNR']:.4f} dB, "
                f"SSIM: {metrics['SSIM']:.6f}, RMSE: {metrics['RMSE']:.6f}")
    return metrics


def compute_per_slice_metrics(clean, test, name="test"):
    z_dim, t_dim = clean.shape[2], clean.shape[3]
    psnr_mat = np.zeros((z_dim, t_dim))
    ssim_mat = np.zeros((z_dim, t_dim))
    rmse_mat = np.zeros((z_dim, t_dim))

    for z in range(z_dim):
        for t in range(t_dim):
            cs, ts = clean[:, :, z, t], test[:, :, z, t]
            psnr_mat[z, t] = psnr(cs, ts, data_range=1.0)
            ssim_mat[z, t] = ssim(cs, ts, data_range=1.0)
            rmse_mat[z, t] = np.sqrt(np.mean((cs - ts) ** 2))

    stats = {}
    for metric_name, mat in [('PSNR', psnr_mat), ('SSIM', ssim_mat), ('RMSE', rmse_mat)]:
        stats[metric_name] = {
            'mean': np.mean(mat), 'std': np.std(mat),
            'min': np.min(mat), 'max': np.max(mat), 'median': np.median(mat),
        }

    logger.info(f"{name.upper()} per-slice: "
                f"PSNR={stats['PSNR']['mean']:.4f}+-{stats['PSNR']['std']:.4f}, "
                f"SSIM={stats['SSIM']['mean']:.6f}+-{stats['SSIM']['std']:.6f}")

    return {'psnr_matrix': psnr_mat, 'ssim_matrix': ssim_mat, 'rmse_matrix': rmse_mat, 'statistics': stats}


def save_metrics(metrics, output_dir, prefix):
    ensure_dir(output_dir)
    for name, key in [('psnr', 'psnr_matrix'), ('ssim', 'ssim_matrix'), ('rmse', 'rmse_matrix')]:
        np.save(os.path.join(output_dir, f'{prefix}_{name}_matrix.npy'), metrics[key])
        np.savetxt(os.path.join(output_dir, f'{prefix}_{name}_matrix.txt'),
                   metrics[key], fmt='%.6f', delimiter='\t')

    with open(os.path.join(output_dir, f'{prefix}_statistics.txt'), 'w') as f:
        for metric_name, s in metrics['statistics'].items():
            f.write(f"{metric_name}: mean={s['mean']:.6f} std={s['std']:.6f} "
                    f"min={s['min']:.6f} max={s['max']:.6f} median={s['median']:.6f}\n")


def plot_metrics_heatmaps(metrics, output_dir, prefix, title):
    ensure_dir(output_dir)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{title} - Metrics', fontsize=16)

    for ax, (key, label, cmap) in zip(axes, [
        ('psnr_matrix', 'PSNR (dB)', 'viridis'),
        ('ssim_matrix', 'SSIM', 'viridis'),
        ('rmse_matrix', 'RMSE', 'plasma'),
    ]):
        im = ax.imshow(metrics[key], aspect='auto', cmap=cmap, interpolation='nearest')
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('Z')
        ax.set_title(label)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{prefix}_metrics_heatmaps.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def visualize_comparisons(clean, noisy, denoised, output_dir, num_slices=8):
    ensure_dir(output_dir)
    z_dim = clean.shape[2]
    t_mid = clean.shape[3] // 2

    for z in np.linspace(0, z_dim - 1, num_slices, dtype=int):
        cs = np.rot90(clean[:, :, z, t_mid], k=-1)
        ns = np.rot90(noisy[:, :, z, t_mid], k=-1)
        ds = np.rot90(denoised[:, :, z, t_mid], k=-1)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'z={z}, t={t_mid}', fontsize=16)
        for ax, data, title in zip(axes, [cs, ns, ds], ["Clean", "Noisy", "Denoised"]):
            ax.imshow(data, cmap='gray', vmin=0, vmax=1)
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'comparison_z{z}_t{t_mid}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)


def visualize_sigma_and_offsets(model, clean_vol, noisy_vol, device, output_dir):
    sigma_dir = os.path.join(output_dir, 'sigma_maps')
    ensure_dir(sigma_dir)
    model.eval()
    z_dim, t_dim = noisy_vol.shape[2], noisy_vol.shape[3]
    t_mid = t_dim // 2

    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for z in np.linspace(0, z_dim - 1, 6, dtype=int):
            noisy_slice = noisy_vol[:, :, z, t_mid]
            inp = torch.from_numpy(noisy_slice).float().unsqueeze(0).unsqueeze(0).to(device)

            denoised_out, sigmas_list, offsets_list = model(inp, return_sigmas=True)
            denoised_slice = to_np(denoised_out.squeeze())
            clean_slice = clean_vol[:, :, z, t_mid]

            s = sigmas_list[0]
            sx = to_np(s[..., 0].squeeze())
            sy = to_np(s[..., 1].squeeze())
            sr = to_np(s[..., 2].squeeze())
            learned_offsets = to_np(offsets_list[0].squeeze())

            np.savez(os.path.join(sigma_dir, f'sigma_z{z}_t{t_mid}.npz'),
                     sigma_x=sx, sigma_y=sy, sigma_r=sr, offsets=learned_offsets)

            psnr_n = psnr(clean_slice, noisy_slice, data_range=1.0)
            ssim_n = ssim(clean_slice, noisy_slice, data_range=1.0)
            psnr_d = psnr(clean_slice, denoised_slice, data_range=1.0)
            ssim_d = ssim(clean_slice, denoised_slice, data_range=1.0)

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'DA-JBF  z={z}, t={t_mid}', fontsize=16)
            axes[0, 0].imshow(np.rot90(clean_slice, k=-1), cmap='gray', vmin=0, vmax=1)
            axes[0, 0].set_title('Clean'); axes[0, 0].axis('off')
            axes[0, 1].imshow(np.rot90(noisy_slice, k=-1), cmap='gray', vmin=0, vmax=1)
            axes[0, 1].set_title(f'Noisy PSNR={psnr_n:.2f} SSIM={ssim_n:.4f}'); axes[0, 1].axis('off')
            axes[0, 2].imshow(np.rot90(denoised_slice, k=-1), cmap='gray', vmin=0, vmax=1)
            axes[0, 2].set_title(f'Denoised PSNR={psnr_d:.2f} SSIM={ssim_d:.4f}'); axes[0, 2].axis('off')

            im3 = axes[1, 0].imshow(np.rot90(np.abs(clean_slice - noisy_slice), k=-1),
                                     cmap='hot', vmin=0, vmax=0.15)
            axes[1, 0].set_title('|Clean - Noisy|'); axes[1, 0].axis('off')
            plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
            im4 = axes[1, 1].imshow(np.rot90(np.abs(clean_slice - denoised_slice), k=-1),
                                     cmap='hot', vmin=0, vmax=0.15)
            axes[1, 1].set_title('|Clean - Denoised|'); axes[1, 1].axis('off')
            plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

            axes[1, 2].axis('off')
            axes[1, 2].text(0.5, 0.5,
                f'PSNR: {psnr_n:.2f} -> {psnr_d:.2f} dB\nSSIM: {ssim_n:.4f} -> {ssim_d:.4f}',
                transform=axes[1, 2].transAxes, fontsize=14, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            plt.tight_layout()
            plt.savefig(os.path.join(sigma_dir, f'comparison_z{z}_t{t_mid}.png'), dpi=200, bbox_inches='tight')
            plt.close()

            fig, axes = plt.subplots(1, 4, figsize=(24, 5))
            fig.suptitle(f'DA-JBF Sigma Maps  z={z}, t={t_mid}', fontsize=14)
            axes[0].imshow(np.rot90(noisy_slice, k=-1), cmap='gray', vmin=0, vmax=1)
            axes[0].set_title('Input (Noisy)'); axes[0].axis('off')
            for ax, data, label in zip(axes[1:], [sx, sy, sr],
                                       ['Sigma_x', 'Sigma_y', 'Sigma_r']):
                im = ax.imshow(np.rot90(data, k=-1), cmap='rainbow')
                ax.set_title(label); ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(os.path.join(sigma_dir, f'sigma_z{z}_t{t_mid}.png'), dpi=200, bbox_inches='tight')
            plt.close()

            K = learned_offsets.shape[0]
            H, W = noisy_slice.shape
            off_dx = learned_offsets[:, 0]
            off_dy = learned_offsets[:, 1]
            offset_mag = np.sqrt(off_dx ** 2 + off_dy ** 2).mean(axis=0)

            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            fig.suptitle(f'DA-JBF Offsets  z={z}, t={t_mid}', fontsize=14)

            im0 = axes[0].imshow(np.rot90(offset_mag, k=-1), cmap='magma')
            axes[0].set_title('Mean offset magnitude (px)'); axes[0].axis('off')
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            axes[1].imshow(np.rot90(noisy_slice, k=-1), cmap='gray', vmin=0, vmax=1, alpha=0.5)
            step = max(H, W) // 16
            gy, gx = np.mgrid[step//2:H:step, step//2:W:step]
            mean_dx = off_dx.mean(axis=0)
            mean_dy = off_dy.mean(axis=0)
            axes[1].quiver(
                np.rot90(gx, k=-1), np.rot90(gy, k=-1),
                np.rot90(mean_dx[gy, gx], k=-1), -np.rot90(mean_dy[gy, gx], k=-1),
                color='cyan', scale=50, width=0.003,
            )
            axes[1].set_title('Mean offset direction'); axes[1].axis('off')

            reg_off = model.stages[0].regular_offsets.cpu().numpy()
            reg_dx, reg_dy = reg_off[:, 0], reg_off[:, 1]
            ax2 = axes[2]
            ax2.imshow(np.rot90(noisy_slice, k=-1), cmap='gray', vmin=0, vmax=1, alpha=0.4)
            cy, cx = H // 2, W // 2
            pts_x = cx + reg_dx + off_dx[:, cy, cx]
            pts_y = cy + reg_dy + off_dy[:, cy, cx]
            rot_cx, rot_cy = W - 1 - cy, cx
            rot_pts_x = W - 1 - pts_y
            rot_pts_y = pts_x
            ax2.plot(rot_cx, rot_cy, 'r+', markersize=12, markeredgewidth=2)
            ax2.scatter(rot_pts_x, rot_pts_y, c='lime', s=15, zorder=5)
            for i in range(K):
                ax2.plot([rot_cx, rot_pts_x[i]], [rot_cy, rot_pts_y[i]], 'lime', alpha=0.3, linewidth=0.5)
            ax2.set_xlim(rot_cx - 15, rot_cx + 15)
            ax2.set_ylim(rot_cy + 15, rot_cy - 15)
            ax2.set_title(f'Sampling at ({cx},{cy})'); ax2.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(sigma_dir, f'offsets_z{z}_t{t_mid}.png'), dpi=200, bbox_inches='tight')
            plt.close()

    logger.info(f"Saved sigma maps and offsets to {sigma_dir}/")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Motion-aware DA-JBF for 4D CT denoising")
    parser.add_argument('--clean', type=str, required=True, help='Path to clean 4D volume (HDF5)')
    parser.add_argument('--noisy', type=str, required=True, help='Path to noisy 4D volume (HDF5)')
    parser.add_argument('--output', type=str, default='denoised_4d.h5', help='Output denoised volume path')
    parser.add_argument('--output-dir', type=str, default='output_4d', help='Output directory for figures and metrics')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--gamma-max', type=float, default=2.0)
    parser.add_argument('--train-fraction', type=float, default=TRAIN_FRACTION)
    parser.add_argument('--seed', type=int, default=77)
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logger()
    device = setup_environment(seed=args.seed)

    clean_path = args.clean
    noisy_path = args.noisy
    output_path = args.output
    output_figures = args.output_dir

    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay

    reg_config = dict(num_scales=3, num_iters=50, sigma_fluid=2.0,
                      sigma_diffusion=1.5, sigma_presmooth=1.5, sigma_range=0.1)

    gamma_max = args.gamma_max
    train_fraction = args.train_fraction

    logger.info("DA-JBF (Deformable Attention-Guided Joint Bilateral Filter) for 4D CT")
    logger.info(f"Training on {train_fraction*100:.0f}% of data")

    if not os.path.exists(clean_path) or not os.path.exists(noisy_path):
        logger.error("Missing input files.")
        return

    clean_volume = normalize_volume(load_4d_volume(clean_path))
    noisy_volume = normalize_volume(load_4d_volume(noisy_path))
    _, _, z_dim, t_dim = noisy_volume.shape

    mfc = MotionFieldComputer(**reg_config)

    all_valid_zt = [(z, t) for z in range(z_dim) for t in range(1, t_dim - 1)]
    n_total = len(all_valid_zt)
    n_train = max(1, int(n_total * train_fraction))
    rng = np.random.default_rng(42)
    train_zt = [all_valid_zt[i] for i in rng.choice(n_total, size=n_train, replace=False)]
    logger.info(f"Selected {n_train}/{n_total} (z,t) pairs for training")

    motion_start = time.time()
    train_motion = mfc.compute_for_indices(noisy_volume, device, train_zt)
    motion_time = time.time() - motion_start
    logger.info(f"Motion field computation: {motion_time:.2f}s")

    dataset = create_training_pairs(noisy_volume, train_motion, train_zt)
    del train_motion

    model = DenoisingPipeline(
        num_stages=1, in_channels=4, hidden_dim=16,
        num_sample_points=25, max_offset=5.0,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(dataset) / batch_size)
    scheduler = OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch)
    loss_fn = RegularizedN2NLoss(gamma_max=gamma_max, n_epochs=epochs)

    train_start = time.time()
    train(model, dataset, loss_fn, optimizer, scheduler, device, epochs, batch_size)
    train_time = time.time() - train_start
    logger.info(f"Training: {train_time:.2f}s")
    del dataset

    logger.info("Denoising full 4D volume (1-channel inference)...")
    denoised_volume = denoise_volume(model, noisy_volume, device)

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('volume', data=denoised_volume, compression='gzip', compression_opts=9)
        f.attrs['original_shape'] = clean_volume.shape

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_figures, f"run_{timestamp}")
    ensure_dir(run_dir)

    logger.info("Computing metrics...")
    noisy_4d = compute_volume_metrics(clean_volume, noisy_volume, "noisy")
    denoised_4d = compute_volume_metrics(clean_volume, denoised_volume, "denoised")
    noisy_per_slice = compute_per_slice_metrics(clean_volume, noisy_volume, "noisy")
    denoised_per_slice = compute_per_slice_metrics(clean_volume, denoised_volume, "denoised")

    metrics_dir = os.path.join(run_dir, 'metrics')
    ensure_dir(metrics_dir)

    with open(os.path.join(metrics_dir, 'overall_metrics.txt'), 'w') as f:
        f.write(f"DA-JBF gamma={gamma_max} - {train_fraction*100:.0f}% training\n{'='*60}\n\n")
        for label, m in [("NOISY", noisy_4d), ("DENOISED", denoised_4d)]:
            f.write(f"{label}: PSNR={m['PSNR']:.4f} dB, SSIM={m['SSIM']:.6f}, RMSE={m['RMSE']:.6f}\n")
        f.write(f"\nMotion time: {motion_time:.2f}s\nTraining time: {train_time:.2f}s\n")

    np.save(os.path.join(metrics_dir, 'noisy_4d_metrics.npy'), noisy_4d)
    np.save(os.path.join(metrics_dir, 'denoised_4d_metrics.npy'), denoised_4d)
    save_metrics(noisy_per_slice, metrics_dir, 'noisy')
    save_metrics(denoised_per_slice, metrics_dir, 'denoised')
    plot_metrics_heatmaps(noisy_per_slice, metrics_dir, 'noisy', 'Noisy Volume')
    plot_metrics_heatmaps(denoised_per_slice, metrics_dir, 'denoised', 'Denoised Volume')

    visualize_comparisons(clean_volume, noisy_volume, denoised_volume, run_dir, num_slices=8)
    visualize_sigma_and_offsets(model, clean_volume, noisy_volume, device, run_dir)

    logger.info("Done!")


if __name__ == '__main__':
    main()
