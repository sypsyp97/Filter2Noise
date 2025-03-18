import os
import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt

from loguru import logger
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR


###############################################################################
#                                Utilities                                    #
###############################################################################

def setup_environment(seed=77):
    """
    Configure the random seeds and CUDA environment for reproducibility.

    This function:
      - Sets NumPy and torch manual seeds to ensure reproducible results.
      - Checks if CUDA is available. If so, sets seeds for all CUDA devices,
        and enables non-deterministic benchmark mode for potential performance gains.

    Args:
        seed (int, optional): The random seed value. Default is 77.

    Returns:
        torch.device: The device object ('cuda' if CUDA is available, otherwise 'cpu').
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Enable benchmark mode to speed up convolution operations.
        torch.backends.cudnn.benchmark = True
        # Turn off deterministic mode for better performance, but slightly less reproducible results.
        torch.backends.cudnn.deterministic = False
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_logger():
    """
    Configure logging settings for the loguru logger.

    Removes any default handlers and adds a new handler that logs to stderr
    with a specific format including timestamp and code location.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
               "<level>{level: <8}</level> | <cyan>{name}</cyan>:"
               "<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )


def prepare_image(image, device='cpu'):
    """
    Convert a numpy array or torch.Tensor to a 4D float Tensor on the specified device.

    If the input is a 2D array, it is reshaped to (1, 1, H, W).
    If it's a 3D array (H, W, C), it is permuted to (1, C, H, W).
    The result is moved to the given device.

    Args:
        image (np.ndarray or torch.Tensor): The input image.
        device (str, optional): The device to move the tensor to. Default is 'cpu'.

    Returns:
        torch.Tensor: The image as a 4D float Tensor (N, C, H, W).

    Raises:
        TypeError: If the image type is not supported.
        ValueError: If the image has an unsupported shape.
    """
    if isinstance(image, np.ndarray):
        tensor = torch.from_numpy(image).float()
    elif isinstance(image, torch.Tensor):
        tensor = image.float()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    if tensor.dim() == 2:
        # (H, W) -> (1, 1, H, W)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:
        # (H, W, C) -> (1, C, H, W)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported image shape: {tensor.shape}")

    return tensor.to(device)


def restore_image(tensor):
    """
    Convert a 4D Tensor (N, C, H, W) back to a numpy array in HWC or HW format.

    Args:
        tensor (torch.Tensor): The 4D input tensor to be converted.

    Returns:
        np.ndarray: The corresponding numpy array with shape (H, W) or (H, W, C).
    """
    if tensor.shape[1] == 1:
        # Single-channel
        return tensor.squeeze().cpu().numpy()
    # Multi-channel case
    return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()


def Downsampler(img):
    """
    Downsample an image into two outputs using fixed 2x2 stride-2 convolutions.

    Two specific 2x2 filters are applied to create two separate downsampled images,
    helping to retain different local structure information.

    Args:
        img (torch.Tensor): Input tensor of shape (B, C, H, W).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Two downsampled images with shape (B, C, H/2, W/2).
    """
    c = img.shape[1]
    device = img.device

    filter1 = torch.FloatTensor([[[[0, 0.5],
                                   [0.5, 0]]]]).to(device).repeat(c, 1, 1, 1)
    filter2 = torch.FloatTensor([[[[0.5, 0],
                                   [0, 0.5]]]]).to(device).repeat(c, 1, 1, 1)

    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)
    return output1, output2


def ELS(img):
    """
    Perform Euclidean Local Shuffle (ELS) on a 4D tensor by swapping pixels in each 2x2 block.

    For each 2x2 block, find the pair of pixels with the smallest Euclidean distance and swap them.
    This aims to challenge the model to learn invariant local structure.

    Args:
        img (torch.Tensor): The input image of shape (B, C, H, W).
            H and W must be even.

    Returns:
        torch.Tensor: An image with swapped pixels in each 2x2 block, same shape as input.

    Raises:
        AssertionError: If H or W are not even.
    """
    B, C, H, W = img.shape
    assert (H % 2 == 0) and (W % 2 == 0), "Height and Width must be even for 2x2 blocks."

    # Unfold the image into blocks of size 2x2 and reshape
    blocks = img.unfold(2, 2, 2).unfold(3, 2, 2).permute(0, 2, 3, 1, 4, 5)
    M = B * (H // 2) * (W // 2)
    flat_blocks = blocks.reshape(M, C, 2, 2).permute(0, 2, 3, 1).reshape(M, 4, C)

    # Compute pairwise squared distances
    diff = flat_blocks.unsqueeze(2) - flat_blocks.unsqueeze(1)
    dists = (diff ** 2).sum(dim=-1)
    eye_mask = torch.eye(4, device=img.device, dtype=bool).unsqueeze(0)
    dists_min = dists.masked_fill(eye_mask, float("inf"))

    # Identify indices of min distances and swap
    idx_min = torch.argmin(dists_min.view(M, -1), dim=1)
    p_min = idx_min // 4
    q_min = idx_min % 4

    swapped_blocks = flat_blocks.clone()
    batch_indices = torch.arange(M, device=img.device)
    temp = swapped_blocks[batch_indices, p_min, :].clone()
    swapped_blocks[batch_indices, p_min, :] = swapped_blocks[batch_indices, q_min, :]
    swapped_blocks[batch_indices, q_min, :] = temp

    # Reshape back to original
    x = swapped_blocks.view(M, 2, 2, C).permute(0, 3, 1, 2)
    x = x.view(B, H // 2, W // 2, C, 2, 2).permute(0, 3, 1, 4, 2, 5)
    x = x.contiguous().view(B, C, H, W)
    return x


def prepare_image_for_display(img):
    """
    Convert an image in [0,1] with shape (H, W) or (H, W, C) to a displayable 4D Tensor in [-1,1].

    Args:
        img (np.ndarray): The input image with shape (H, W) or (H, W, C) in [0, 1].

    Returns:
        torch.Tensor: A 4D tensor (N, C, H, W) scaled to [-1, 1].
    """
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img_tensor.float() * 2 - 1


def prepare_for_display(img):
    """
    Prepare an image (tensor or numpy array) for display by squeezing out
    any singleton dimensions if appropriate.

    Args:
        img (np.ndarray or torch.Tensor): The input image.

    Returns:
        np.ndarray or torch.Tensor: The squeezed image if it's a numpy array.
            If it's a torch.Tensor, it returns the tensor with squeezed 
            batch dimension if batch size is 1.
    """
    if isinstance(img, np.ndarray):
        # For numpy arrays
        if img.ndim == 3 and (img.shape[0] == 1 or img.shape[2] == 1):
            return np.squeeze(img)
        return img
    # If it's a torch.Tensor
    return img.squeeze(0) if img.shape[0] == 1 else img


###############################################################################
#                             Plotting Functions                              #
###############################################################################

def plot_results(original, noisy, denoised, metrics, save_path=None):
    """
    Plot original, noisy, and denoised images side by side with PSNR and SSIM annotations.

    Args:
        original (np.ndarray): The clean/original image in [0,1].
        noisy (np.ndarray): The noisy image in [0,1].
        denoised (np.ndarray): The denoised image in [0,1].
        metrics (dict): A dictionary with 'psnr_noisy', 'psnr_denoised',
            'ssim_noisy', and 'ssim_denoised'.
        save_path (str, optional): If provided, the plot is saved in this directory
            under a predefined filename.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    original_disp = prepare_for_display(original)
    noisy_disp = prepare_for_display(noisy)
    denoised_disp = prepare_for_display(denoised)
    cmap = 'gray' if original_disp.ndim == 2 else None

    axes[0].imshow(original_disp, cmap=cmap)
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(noisy_disp, cmap=cmap)
    axes[1].set_title("Noisy\nPSNR: {:.2f} dB\nSSIM: {:.4f}".format(
        metrics['psnr_noisy'], metrics['ssim_noisy']))
    axes[1].axis('off')

    axes[2].imshow(denoised_disp, cmap=cmap)
    axes[2].set_title("Denoised\nPSNR: {:.2f} dB\nSSIM: {:.4f}".format(
        metrics['psnr_denoised'], metrics['ssim_denoised']))
    axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(
            os.path.join(save_path, 'results_with_metrics_pixel.png'),
            bbox_inches='tight',
            pad_inches=0,
            dpi=300
        )
    plt.show()


def plot_sigmas(sigmas_list, save_path=None):
    """
    Plot sigma maps (sigma_x, sigma_y, sigma_r) for each stage in a multi-stage pipeline.

    Args:
        sigmas_list (list of torch.Tensor): Each tensor is a predicted sigma map of shape [B, H, W, 3].
        save_path (str, optional): Directory path to save the composite figure.
    """
    num_stages = len(sigmas_list)
    fig, axes = plt.subplots(num_stages, 3, figsize=(15, 5 * num_stages))

    if num_stages == 1:
        axes = [axes]

    for stage_idx, sigmas in enumerate(sigmas_list):
        sigma_sx = sigmas[..., 0].cpu().numpy()
        sigma_sy = sigmas[..., 1].cpu().numpy()
        sigma_r = sigmas[..., 2].cpu().numpy()

        im0 = axes[stage_idx][0].imshow(sigma_sx.squeeze(), cmap='rainbow')
        axes[stage_idx][0].set_title(f"Stage {stage_idx + 1}\nSigma_x")
        plt.colorbar(im0, ax=axes[stage_idx][0], fraction=0.046, pad=0.04)

        im1 = axes[stage_idx][1].imshow(sigma_sy.squeeze(), cmap='rainbow')
        axes[stage_idx][1].set_title(f"Stage {stage_idx + 1}\nSigma_y")
        plt.colorbar(im1, ax=axes[stage_idx][1], fraction=0.046, pad=0.04)

        im2 = axes[stage_idx][2].imshow(sigma_r.squeeze(), cmap='rainbow')
        axes[stage_idx][2].set_title(f"Stage {stage_idx + 1}\nSigma_r")
        plt.colorbar(im2, ax=axes[stage_idx][2], fraction=0.046, pad=0.04)

        for ax in axes[stage_idx]:
            ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(
            os.path.join(save_path, 'sigmas_all_stages.png'),
            bbox_inches='tight',
            pad_inches=0,
            dpi=300
        )
    plt.show()


def save_sigmas_maps_separately(sigmas_list, save_path=None):
    """
    Save sigma maps (sigma_x, sigma_y, sigma_r) from each stage in separate image files.

    Each file is saved with a color bar, DPI=300, and no axis. Logs the saved filename.

    Args:
        sigmas_list (list of torch.Tensor): Each element corresponds to a stage's predicted
            sigma map of shape [B, H, W, 3].
        save_path (str, optional): Directory path to save the sigma map files. If None,
            the files are not saved.
    """
    for stage_idx, sigmas in enumerate(sigmas_list):
        sigma_x = sigmas[..., 0].cpu().numpy()
        sigma_y = sigmas[..., 1].cpu().numpy()
        sigma_r = sigmas[..., 2].cpu().numpy()

        channels = {"x": sigma_x, "y": sigma_y, "r": sigma_r}

        for channel, sigma in channels.items():
            plt.imshow(sigma.squeeze(), cmap='rainbow')
            plt.axis('off')
            plt.colorbar()
            plt.tight_layout()

            if save_path:
                filename = os.path.join(
                    save_path, f"sigmas_{channel}_stage_{stage_idx + 1}.png"
                )
                plt.savefig(
                    filename,
                    bbox_inches='tight',
                    pad_inches=0,
                    dpi=300
                )
                logger.info(f"Saved {filename}")

            plt.close()


###############################################################################
#                             Model Components                                #
###############################################################################

class BoundedSoftplus(nn.Module):
    """
    BoundedSoftplus activation that prevents extremely large values by clamping.

    Attributes:
        beta (float): Parameter for the softplus function.
        threshold (float): Maximum clamp value.
        eps (float): A small epsilon value added to the final output.
    """

    def __init__(self, beta=1.0, threshold=6.0, eps=1e-6):
        """
        Args:
            beta (float, optional): Beta for F.softplus. Defaults to 1.0.
            threshold (float, optional): Max clamp value. Defaults to 6.0.
            eps (float, optional): Small additive constant. Defaults to 1e-6.
        """
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.eps = eps

    def forward(self, x):
        """
        Forward pass for the BoundedSoftplus layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after clamped softplus.
        """
        softplus_val = F.softplus(x, beta=self.beta)
        clamped_val = torch.clamp(softplus_val, max=self.threshold)
        return clamped_val + self.eps


class SigmaPredictor(nn.Module):
    """
    Predicts sigma maps (sigma_x, sigma_y, sigma_r) for bilateral filtering
    using a patch-based attention mechanism.

    Attributes:
        patch_size (int): The size of the non-overlapping image patches.
        hidden_dim (int): Dimension for latent representation in attention.
        query, key, value (nn.Linear): Attention transformation for the main features.
        sigma_query, sigma_key, sigma_value (nn.Linear): Attention for sigma features.
        norm (nn.LayerNorm): LayerNorm to normalize the sigma features.
        sigma_proj (nn.Linear): Final linear layer to project to 3 sigma channels.
        activation (BoundedSoftplus): Activation to ensure the sigmas remain positive.
        attention_scale (float): Scale factor for scaled dot-product attention.
    """

    def __init__(self, patch_size=8):
        """
        Args:
            patch_size (int, optional): Patch size for non-overlapping patches. Defaults to 8.
            in_channels (int, optional): Number of channels of the input image. Defaults to 1.
        """
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = 8

        self.query = nn.LazyLinear(self.hidden_dim)
        self.key = nn.LazyLinear(self.hidden_dim)
        self.value = nn.LazyLinear(self.hidden_dim)

        self.sigma_query = nn.LazyLinear(self.hidden_dim)
        self.sigma_key = nn.LazyLinear(self.hidden_dim)
        self.sigma_value = nn.LazyLinear(self.hidden_dim)

        self.norm = nn.LayerNorm(self.hidden_dim)
        self.sigma_proj = nn.LazyLinear(3)
        self.activation = BoundedSoftplus(threshold=6)

        self.attention_scale = self.hidden_dim ** -0.5

    def _attention(self, q, k, v):
        """
        Compute scaled dot-product attention.

        Args:
            q (torch.Tensor): Query of shape [B, N, D].
            k (torch.Tensor): Key of shape [B, N, D].
            v (torch.Tensor): Value of shape [B, N, D].

        Returns:
            torch.Tensor: Attention output of shape [B, N, D].
        """
        scores = torch.bmm(q, k.transpose(1, 2)) * self.attention_scale
        attn_weights = F.softmax(scores, dim=-1)
        return torch.bmm(attn_weights, v)

    def forward(self, x):
        """
        Predict sigmas from the input image.

        Args:
            x (torch.Tensor): Input image of shape (B, C, H, W).

        Returns:
            torch.Tensor: Predicted sigma map of shape (B, H, W, 3).
        """
        b, c, h, w = x.shape
        ps = self.patch_size
        assert h % ps == 0 and w % ps == 0, "Image dimensions must be divisible by patch_size"

        # Extract non-overlapping patches
        patches = x.view(b, c, h // ps, ps, w // ps, ps)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.view(b, (h // ps) * (w // ps), -1)

        # Feature attention
        q_main = self.query(patches)
        k_main = self.key(patches)
        v_main = self.value(patches)
        feats = self._attention(q_main, k_main, v_main)

        # Sigma attention
        q_sigma = self.sigma_query(feats)
        k_sigma = self.sigma_key(feats)
        v_sigma = self.sigma_value(feats)
        sigmas_patch = self._attention(q_sigma, k_sigma, v_sigma)
        sigmas_patch_norm = self.norm(sigmas_patch)
        sigmas_patch_proj = self.sigma_proj(sigmas_patch_norm)
        sigmas = self.activation(sigmas_patch_proj)  # [B, N, 3]

        # Reshape -> [B, 3, H//ps, W//ps]
        sigmas = sigmas.view(b, h // ps, w // ps, 3).permute(0, 3, 1, 2)

        # Upsample to original resolution via nearest neighbor
        sigmas_resized = F.interpolate(sigmas, size=(h, w), mode='nearest').permute(0, 2, 3, 1)
        return sigmas_resized


class AGBF(nn.Module):
    """
    Attention-Guided Bilateral Filtering (AGBF) module that uses predicted sigmas
    from SigmaPredictor. Combines both spatial and range kernels in a single pass.

    This class also caches grid coordinates for performance.

    Attributes:
        _cached_grids (dict): Class-level cache for grid coordinates.
        sigma_predictor (SigmaPredictor): Predicts the sigma maps.
    """
    _cached_grids = {}

    def __init__(self, patch_size=16):
        """
        Args:
            patch_size (int, optional): Patch size for the SigmaPredictor. Defaults to 16.
            in_channels (int, optional): Number of input channels. Defaults to 1.
        """
        super().__init__()
        self.sigma_predictor = SigmaPredictor(patch_size=patch_size)

    def compute_spatial_kernel(self, sx, sy, k, device):
        """
        Compute the 2D spatial Gaussian kernel.

        Args:
            sx (torch.Tensor): Sigma_x values of shape [B, H, W].
            sy (torch.Tensor): Sigma_y values of shape [B, H, W].
            k (int): Kernel size.
            device (torch.device): Device to use.

        Returns:
            torch.Tensor: Spatial kernel of shape [B, 1, H, W, k*k].
        """
        if k not in AGBF._cached_grids:
            half_k = k // 2
            y_coords, x_coords = torch.meshgrid(
                torch.arange(-half_k, half_k + 1, device=device),
                torch.arange(-half_k, half_k + 1, device=device),
                indexing='ij'
            )
            x_coords = x_coords.float().view(1, 1, 1, k, k)
            y_coords = y_coords.float().view(1, 1, 1, k, k)
            AGBF._cached_grids[k] = (x_coords, y_coords)
        else:
            x_coords, y_coords = AGBF._cached_grids[k]
            if x_coords.device != device:
                x_coords = x_coords.to(device)
                y_coords = y_coords.to(device)

        b, h, w = sx.shape
        sx_expanded = sx.view(b, h, w, 1, 1)
        sy_expanded = sy.view(b, h, w, 1, 1)

        spatial_kernel = torch.exp(
            -((x_coords**2) / (2 * sx_expanded**2) + (y_coords**2) / (2 * sy_expanded**2))
        )
        return spatial_kernel

    def compute_range_kernel(self, center_values, neighbor_values, sigma_r):
        """
        Compute the range kernel for bilateral filtering.

        Args:
            center_values (torch.Tensor): Center pixel values, shape [B, C, H, W, 1].
            neighbor_values (torch.Tensor): Neighboring pixel values, shape [B, C, H, W, k*k].
            sigma_r (torch.Tensor): Range sigma values, shape [B, H, W].

        Returns:
            torch.Tensor: Range kernel, shape [B, C, H, W, k*k].
        """
        diff = center_values - neighbor_values
        sq_diff = diff**2
        if center_values.shape[1] > 1:
            sq_diff = sq_diff.sum(dim=1, keepdim=True)  # Sum across channels if multi-channel

        sigma_r_expanded = sigma_r.unsqueeze(1).unsqueeze(-1)
        range_kernel = torch.exp(-sq_diff / (2 * sigma_r_expanded**2))
        return range_kernel

    def forward(self, x, return_sigmas=False):
        """
        Forward pass for the AGBF module. Predicts sigmas, computes bilateral filtering.

        Args:
            x (torch.Tensor): Input of shape [B, C, H, W].
            return_sigmas (bool, optional): If True, also return the predicted sigmas.

        Returns:
            torch.Tensor: Filtered output of shape [B, C, H, W].
            (Optional) torch.Tensor: Sigma map of shape [B, H, W, 3] if return_sigmas=True.
        """
        x = x.float()
        b, c, h, w = x.shape

        # Predict sigma maps
        s = self.sigma_predictor(x)
        sx = s[..., 0]
        sy = s[..., 1]
        sr = s[..., 2]

        # Determine kernel size dynamically
        m = torch.max(sx.max(), sy.max()).item()
        k = int(2 * math.ceil(m) + 1)
        if k % 2 == 0:
            k += 1
        p = k // 2

        # Pad the input
        xp = F.pad(x, (p, p, p, p), mode='reflect')
        patches = F.unfold(xp, kernel_size=k, stride=1).view(b, c, k*k, h, w).permute(0, 1, 3, 4, 2)

        # Compute spatial kernel
        spatial_kernel = self.compute_spatial_kernel(sx, sy, k, x.device)
        spatial_kernel = spatial_kernel.view(b, 1, h, w, k * k)

        # Compute range kernel
        center_values = x.view(b, c, h, w, 1)
        range_kernel = self.compute_range_kernel(center_values, patches, sr)

        # Combine kernels
        combined_kernel = spatial_kernel * range_kernel
        norm_factor = combined_kernel.sum(dim=-1, keepdim=True)
        normalized_kernel = combined_kernel / (norm_factor + 1e-8)

        # Apply filter
        patches.mul_(normalized_kernel)
        output = patches.sum(dim=-1)

        if return_sigmas:
            return output, s
        return output


class DenoisingPipeline(nn.Module):
    """
    Denoising pipeline with multiple stages of AGBF in sequence.

    Attributes:
        stages (nn.ModuleList): A list of AGBF stages.
    """

    def __init__(self, num_stages=2, patch_size=8):
        """
        Args:
            num_stages (int, optional): Number of AGBF stages in the pipeline. Defaults to 2.
            patch_size (int, optional): Patch size for the SigmaPredictor. Defaults to 8.
            in_channels (int, optional): Number of input channels. Defaults to 1.
        """
        super().__init__()
        self.stages = nn.ModuleList(
            [AGBF(patch_size=patch_size) for _ in range(num_stages)]
        )

    def forward(self, x, return_sigmas=False):
        """
        Forward pass through all stages.

        Args:
            x (torch.Tensor): Input of shape [B, C, H, W].
            return_sigmas (bool, optional): If True, return sigma maps from each stage.

        Returns:
            torch.Tensor: Denoised output.
            (Optional) list: A list of sigma maps if return_sigmas=True.
        """
        sigmas_list = []
        current_x = x
        for stage in self.stages:
            if return_sigmas:
                current_x, s = stage(current_x, return_sigmas=True)
                sigmas_list.append(s)
            else:
                current_x = stage(current_x)
        if return_sigmas:
            return current_x, sigmas_list
        return current_x


###############################################################################
#                           Loss and Training                                 #
###############################################################################

class LossFunction:
    """
    A self-supervised loss function for image denoising that combines:
      - Multi-scale consistency
      - Edge preservation using a Difference of Gaussians (DoG) filter

    Attributes:
        kernel_size (int): Size of the DoG kernel.
        sigma_narrow (float): Narrow Gaussian sigma.
        sigma_wide (float): Wide Gaussian sigma.
        device (torch.device): Target device.
        edge_loss_weight (float): Weight for the edge loss term.
        base_dog_kernel (torch.Tensor): The DoG kernel used for edge preservation.
    """

    def __init__(self, device, lambda_=350):
        """
        Args:
            device (torch.device): Device for kernel computations.
            lambda_ (float, optional): Weight for the edge loss term. Defaults to 350.
        """
        self.kernel_size = 7
        self.sigma_narrow = 9
        self.sigma_wide = 10
        self.device = device
        self.edge_loss_weight = lambda_
        self._initialize_dog_kernel()

    def _initialize_dog_kernel(self):
        """
        Create a Difference of Gaussians (DoG) kernel used for edge detection.
        """
        grid_coords = torch.arange(self.kernel_size, dtype=torch.float32,
                                   device=self.device) - self.kernel_size // 2
        grid_y, grid_x = torch.meshgrid(grid_coords, grid_coords, indexing='ij')

        gaussian_narrow = torch.exp(-(grid_x**2 + grid_y**2) / (2 * self.sigma_narrow**2))
        gaussian_narrow = gaussian_narrow / gaussian_narrow.sum()

        gaussian_wide = torch.exp(-(grid_x**2 + grid_y**2) / (2 * self.sigma_wide**2))
        gaussian_wide = gaussian_wide / gaussian_wide.sum()

        dog_kernel = (gaussian_narrow - gaussian_wide).unsqueeze(0).unsqueeze(0)
        self.base_dog_kernel = dog_kernel

    def __call__(self, noisy_input, model, alpha=None):
        """
        Calculate the self-supervised loss on the given noisy image and model outputs.

        Args:
            noisy_input (torch.Tensor): A noisy image of shape (B, C, H, W).
            model (nn.Module): The denoising model.
            alpha (float, optional): If given, overrides the edge_loss_weight.

        Returns:
            torch.Tensor: The total loss value.
        """
        if alpha is not None:
            self.edge_loss_weight = alpha

        dog_kernel = self.base_dog_kernel.repeat(noisy_input.shape[1], 1, 1, 1)

        # Generate multi-scale pairs
        noisy_scale1, noisy_scale2 = Downsampler(noisy_input)

        # Shuffle pixels to avoid trivial identity mapping
        shuffled_noisy_scale1 = ELS(noisy_scale1)
        shuffled_noisy_scale2 = ELS(noisy_scale2)

        # Denoise each shuffled scale
        denoised_scale1 = model(shuffled_noisy_scale1)
        denoised_scale2 = model(shuffled_noisy_scale2)

        # Loss term: difference between the two scales
        loss_resolution = (1.0 / 3.0) * F.l1_loss(denoised_scale1, denoised_scale2)

        # Full-resolution denoised
        denoised_full = model(noisy_input)

        # Downsample full-resolution denoised
        downsampled_denoised1, downsampled_denoised2 = Downsampler(denoised_full)

        # Cross-scale consistency
        loss_cross_scale = (1.0 / 3.0) * (
            F.l1_loss(denoised_scale1, downsampled_denoised1) +
            F.l1_loss(denoised_scale2, downsampled_denoised2)
        )

        # Additional denoising consistency
        loss_denoise = (1.0 / 3.0) * F.l1_loss(downsampled_denoised1, downsampled_denoised2)

        # Edge preservation using DoG
        edges_noisy = F.conv2d(noisy_input, dog_kernel, padding=self.kernel_size // 2,
                               groups=noisy_input.shape[1])
        edges_denoised = F.conv2d(denoised_full, dog_kernel, padding=self.kernel_size // 2,
                                  groups=noisy_input.shape[1])
        loss_edge = self.edge_loss_weight * F.l1_loss(
            torch.abs(edges_noisy), torch.abs(edges_denoised)
        )

        return loss_resolution + loss_cross_scale + loss_denoise + loss_edge


def train_model(model, noisy, loss_function, optimizer, scheduler, epochs=500):
    """
    Generic training loop for the denoising model using a self-supervised loss.

    Args:
        model (nn.Module): The denoising pipeline.
        noisy (torch.Tensor): The noisy input image of shape (B, C, H, W).
        loss_function (LossFunction): The self-supervised loss to optimize.
        optimizer (torch.optim.Optimizer): The optimizer (e.g., AdamW).
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler (e.g., OneCycleLR).
        epochs (int, optional): Number of epochs to train. Defaults to 500.
    """
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(noisy, model)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.6f}")


###############################################################################
#                         Metrics and Data Loading                            #
###############################################################################

def compute_metrics(image, noisy_image, denoised_image, in_channels):
    """
    Compute PSNR and SSIM metrics for noisy and denoised images relative to the reference.

    Args:
        image (np.ndarray): The clean reference image in [0,1].
        noisy_image (np.ndarray): The noisy image in [0,1].
        denoised_image (np.ndarray): The denoised image in [0,1].
        in_channels (int): The number of channels (e.g., 1 for grayscale, 3 for RGB).

    Returns:
        dict: Contains 'psnr_noisy', 'psnr_denoised', 'ssim_noisy', and 'ssim_denoised'.
    """
    metrics = {}
    metrics['psnr_noisy'] = compare_psnr(image, noisy_image, data_range=1)
    metrics['psnr_denoised'] = compare_psnr(image, denoised_image, data_range=1)

    if in_channels == 1:
        ssim_noisy, _ = compare_ssim(image, noisy_image, data_range=1, full=True)
        ssim_denoised, _ = compare_ssim(image, denoised_image, data_range=1, full=True)
    else:
        ssim_noisy, _ = compare_ssim(image, noisy_image, data_range=1, channel_axis=2, full=True)
        ssim_denoised, _ = compare_ssim(
            image, denoised_image, data_range=1, channel_axis=2, full=True
        )
    metrics['ssim_noisy'] = ssim_noisy
    metrics['ssim_denoised'] = ssim_denoised

    return metrics


def load_data():
    """
    Load sample data for demonstration. Replace this with your own data loading logic.

    Returns:
        tuple: (clean_image, noisy_image, in_channels)
    """
    image = np.load('clean.npy')
    noisy_image = np.load('noisy.npy')
    in_channels = 1  # e.g., 1 for grayscale
    return image, noisy_image, in_channels


def save_image(image_np, filename, in_channels):
    """
    Save a numpy array as an image file (PNG) without borders or axis.

    Args:
        image_np (np.ndarray): The image array in [0,1] with shape (H, W) or (H, W, C).
        filename (str): The path/filename to save the image.
        in_channels (int): Number of channels (1 for grayscale, 3 for RGB).
    """
    disp_image = prepare_for_display(image_np)
    plt.figure(frameon=False)
    plt.imshow(disp_image, cmap='gray' if in_channels == 1 else None)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    logger.info(f"Saved {filename}")


###############################################################################
#                                   Main                                      #
###############################################################################

def main():
    """
    The main entry point for the denoising pipeline.

    Steps:
      1. Set up environment and logger.
      2. Load data (clean image, noisy image, in_channels).
      3. Prepare the noisy image as a torch.Tensor.
      4. Define the denoising model and training components.
      5. Train the model using self-supervised loss.
      6. Perform inference and measure training time.
      7. Evaluate the results using PSNR/SSIM metrics.
      8. Save and visualize the denoised image, along with sigma maps.
    """
    device = setup_environment()
    setup_logger()
    logger.info("Starting denoising pipeline...")

    # Load Data
    image, noisy_image, in_channels = load_data()
    noisy = prepare_image(noisy_image, device)

    # Initialize model and training utilities
    model = DenoisingPipeline(num_stages=2, patch_size=8).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=0.001, epochs=500, steps_per_epoch=1)
    loss_function = LossFunction(device, lambda_=400)

    # Train model
    start_time = time.time()
    train_model(model, noisy, loss_function, optimizer, scheduler, epochs=500)
    inference_time = time.time() - start_time

    model.eval()
    logger.info(
        "Number of trainable parameters: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )

    # Inference
    with torch.no_grad():
        denoised, sigmas_list = model(noisy, return_sigmas=True)
        denoised_image = restore_image(denoised)

    logger.info(f"Inference time: {inference_time:.4f} seconds")

    # Compute and print metrics
    metrics = compute_metrics(image, noisy_image, denoised_image, in_channels)
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.4f}")

    # Example of saving a final denoised image (uncomment if needed):
    # save_image(denoised_image, "denoised_image_result.png", in_channels)

    plot_results(image, noisy_image, denoised_image, metrics, save_path='.')
    plot_sigmas(sigmas_list, save_path='.')

    # Optionally, save sigma maps for each stage:
    # save_sigmas_maps_separately(sigmas_list, save_path='sigmas_maps')


if __name__ == '__main__':
    main()