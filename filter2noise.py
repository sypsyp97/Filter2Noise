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
        and enables non-deterministic, benchmark modes for potential performance gains.

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
    logger.remove()  # Remove default logger handlers.
    logger.add(sys.stderr,
               format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                      "<level>{level: <8}</level> | <cyan>{name}</cyan>:"
                      "<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")


def prepare_image(image, device='cpu'):
    """
    Convert a numpy array or torch.Tensor to a 4D float Tensor on the specified device.

    This function handles two scenarios: a numpy array or a torch.Tensor as input.
    - If the input is a 2D array, it is reshaped to (1, 1, H, W).
    - If it's a 3D array (H, W, C), it is permuted to (1, C, H, W).
    The result is moved to the given device.

    Args:
        image (numpy.ndarray or torch.Tensor): The image to convert.
        device (str, optional): The device to move the tensor to. Default is 'cpu'.

    Returns:
        torch.Tensor: The image as a 4D float Tensor (N, C, H, W).
    """
    if isinstance(image, np.ndarray):
        tensor = torch.from_numpy(image).float()
    elif isinstance(image, torch.Tensor):
        tensor = image.float()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Handle shapes:
    #  - If 2D, add batch and channel dimensions.
    #  - If 3D, permute for PyTorch's (C, H, W) format, then add batch dimension.
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    elif tensor.dim() == 3:
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
    else:
        raise ValueError(f"Unsupported image shape: {tensor.shape}")

    return tensor.to(device)


def restore_image(tensor):
    """
    Convert a 4D Tensor (N, C, H, W) back to a numpy array in HWC or HW format.

    This function supports:
    - Single-channel: returns a 2D array if possible (HW).
    - Multi-channel: returns a 3D array (H, W, C).

    Args:
        tensor (torch.Tensor): The 4D input tensor to be converted.

    Returns:
        numpy.ndarray: The corresponding numpy array with shape (H, W) or (H, W, C).
    """
    if tensor.shape[1] == 1:
        # Single-channel (e.g. grayscale)
        return tensor.squeeze().cpu().numpy()
    # Multi-channel (e.g. RGB)
    return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()


def Downsampler(img):
    """
    Downsample an image into two outputs using fixed 2x2 stride-2 convolutions.

    Two specific 2x2 filters are applied to create two separate downsampled images.
    This helps to retain different local structure information.

    Args:
        img (torch.Tensor): Input tensor of shape (B, C, H, W).

    Returns:
        (torch.Tensor, torch.Tensor): Two downsampled images with shapes (B, C, H/2, W/2).
    """
    c = img.shape[1]
    device = img.device

    # Define two filters for stride=2 convolution:
    filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(device).repeat(c, 1, 1, 1)
    filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(device).repeat(c, 1, 1, 1)

    # Convolve and downsample
    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)
    return output1, output2


def ELS(img):
    """
    Euclidean Local Shuffle (ELS) operation on a 4D tensor by swapping pixels in each 2x2 block.

    For each 2x2 block in the input tensor, find the pair of pixels with the smallest Euclidean
    distance and swap them. This aims to challenge the model to learn invariant local structure.

    Args:
        img (torch.Tensor): The input image of shape (B, C, H, W).
                            H and W must be even.

    Returns:
        torch.Tensor: An image with swapped pixels in each 2x2 block, same shape as input.
    """
    B, C, H, W = img.shape
    assert (H % 2 == 0) and (W % 2 == 0), "Height and Width must be even for 2x2 blocks."

    # Unfold the image into blocks of size 2x2
    blocks = img.unfold(2, 2, 2).unfold(3, 2, 2).permute(0, 2, 3, 1, 4, 5)
    M = B * (H // 2) * (W // 2)

    # Flatten each 2x2 block to shape [M, 4, C]
    flat_blocks = blocks.reshape(M, C, 2, 2).permute(0, 2, 3, 1).reshape(M, 4, C)

    # Compute pairwise squared distances between pixels in each 2x2
    diff = flat_blocks.unsqueeze(2) - flat_blocks.unsqueeze(1)
    dists = (diff ** 2).sum(dim=-1)
    eye_mask = torch.eye(4, device=img.device, dtype=bool).unsqueeze(0)
    # Mask out diagonal distances (distance to itself)
    dists_min = dists.masked_fill(eye_mask, float("inf"))

    # Identify indices of min distances
    idx_min = torch.argmin(dists_min.view(M, -1), dim=1)
    p_min = idx_min // 4
    q_min = idx_min % 4

    # Swap the pair with smallest distance
    swapped_blocks = flat_blocks.clone()
    batch_indices = torch.arange(M, device=img.device)
    temp = swapped_blocks[batch_indices, p_min, :].clone()
    swapped_blocks[batch_indices, p_min, :] = swapped_blocks[batch_indices, q_min, :]
    swapped_blocks[batch_indices, q_min, :] = temp

    # Reshape back to original (B, C, H, W)
    x = swapped_blocks.view(M, 2, 2, C).permute(0, 3, 1, 2)
    x = x.view(B, H // 2, W // 2, C, 2, 2).permute(0, 3, 1, 4, 2, 5)
    x = x.contiguous().view(B, C, H, W)
    return x


def prepare_image_for_display(img):
    """
    Convert an image (HW or HWC) in [0,1] to a displayable 4D Tensor in the range [-1,1],
    suitable for certain visualization frameworks.

    Args:
        img (numpy.ndarray): The input image with shape (H, W) or (H, W, C) in [0, 1].

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
        img (numpy.ndarray or torch.Tensor): The image to be displayed.

    Returns:
        numpy.ndarray or torch.Tensor: The squeezed image if it's a numpy array.
                                       If it's a torch.Tensor, it returns the squeezed
                                       tensor along the batch dimension if batch size is 1.
    """
    if isinstance(img, np.ndarray):
        # For numpy arrays, if there's a singleton dimension in channel or batch, squeeze it.
        if img.ndim == 3 and (img.shape[0] == 1 or img.shape[2] == 1):
            return np.squeeze(img)
        return img
    # If it's a torch.Tensor, just squeeze the batch dimension if it's 1.
    return img.squeeze(0) if img.shape[0] == 1 else img


###############################################################################
#                             Plotting Functions                              #
###############################################################################

def plot_results(original, noisy, denoised, metrics, save_path=None):
    """
    Plot original, noisy, and denoised images side by side with PSNR and SSIM annotations.

    Args:
        original (numpy.ndarray): The clean/original image in [0,1].
        noisy (numpy.ndarray): The noisy image in [0,1].
        denoised (numpy.ndarray): The denoised image in [0,1].
        metrics (dict): A dictionary with 'psnr_noisy', 'psnr_denoised',
                        'ssim_noisy', and 'ssim_denoised'.
        save_path (str, optional): If provided, the plot is saved in this directory
                                   with a predefined filename.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    original_disp = prepare_for_display(original)
    noisy_disp = prepare_for_display(noisy)
    denoised_disp = prepare_for_display(denoised)
    cmap = 'gray' if original_disp.ndim == 2 else None

    # Plot original
    axes[0].imshow(original_disp, cmap=cmap)
    axes[0].set_title("Original")
    axes[0].axis('off')

    # Plot noisy with metrics
    axes[1].imshow(noisy_disp, cmap=cmap)
    axes[1].set_title("Noisy\nPSNR: {:.2f} dB\nSSIM: {:.4f}".format(metrics['psnr_noisy'],
                                                                    metrics['ssim_noisy']))
    axes[1].axis('off')

    # Plot denoised with metrics
    axes[2].imshow(denoised_disp, cmap=cmap)
    axes[2].set_title("Denoised\nPSNR: {:.2f} dB\nSSIM: {:.4f}".format(metrics['psnr_denoised'],
                                                                      metrics['ssim_denoised']))
    axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'results_with_metrics_pixel.png'),
                    bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()


def plot_sigmas(sigmas_list, save_path=None):
    """
    Plot sigma maps (sigma_x, sigma_y, sigma_r) for each stage in a multi-stage pipeline.

    Args:
        sigmas_list (list of torch.Tensor): Each tensor is the predicted sigma values
                                            of shape [B, H, W, 3], for each stage.
        save_path (str, optional): If provided, save the composite figure into this path.
    """
    num_stages = len(sigmas_list)
    fig, axes = plt.subplots(num_stages, 3, figsize=(15, 5 * num_stages))

    # If there's only one stage, wrap axes
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
        plt.savefig(os.path.join(save_path, 'sigmas_all_stages.png'),
                    bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()


def save_sigmas_maps_separately(sigmas_list, save_path=None):
    """
    Save sigma maps (sigma_x, sigma_y, sigma_r) from each stage in separate image files.

    Each file is saved with a color bar, dpi=300, and no axis.
    A log statement prints the filename once saved.

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
        
        # Save each channel separately
        for channel, sigma in channels.items():
            plt.imshow(sigma.squeeze(), cmap='rainbow')
            plt.axis('off')
            plt.colorbar()
            plt.tight_layout()

            if save_path:
                filename = os.path.join(save_path,
                                        f"sigmas_{channel}_stage_{stage_idx + 1}.png")
                plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=300)
                logger.info(f"Saved {filename}")

            plt.close()


###############################################################################
#                             Model Components                                #
###############################################################################

class BoundedSoftplus(nn.Module):
    """
    A Softplus variant that is clipped at a threshold and adds a small epsilon.

    This activation:
      - Applies a standard Softplus function: log(1 + exp(beta * x)) / beta.
      - Clamps the result to a maximum threshold, preventing extremely large values.
      - Adds a small epsilon to avoid zero outputs.

    Args:
        beta (float, optional): Controls steepness of the Softplus. Default is 1.
        threshold (float, optional): Maximum value to clamp the Softplus output. Default is 6.
        eps (float, optional): Small constant added to the output. Default is 1e-6.
    """
    def __init__(self, beta=1, threshold=6, eps=1e-6):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.eps = eps

    def forward(self, x):
        """
        Forward pass of the BoundedSoftplus activation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after Softplus, clamping, and epsilon addition.
        """
        return torch.clamp_(F.softplus(x, beta=self.beta), max=self.threshold) + self.eps


class SigmaPredictor(nn.Module):
    """
    A self-attention based predictor for sigma maps (σx, σy, σr).

    This class:
      - Unfolds the image into patches.
      - Uses an attention mechanism to create a feature embedding.
      - Predicts three sigma values (σx, σy, σr) at a coarse patch level.
      - Upscales and reshapes them to match the input image resolution via nearest neighbor.

    Args:
        patch_size (int, optional): The patch size for unfolding. Default is 8.
        in_channels (int, optional): Number of input channels. Default is 1.
    """
    def __init__(self, patch_size=8, in_channels=1):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = 8

        # Query/Key/Value for the main attention
        self.query = nn.LazyLinear(self.hidden_dim)
        self.key = nn.LazyLinear(self.hidden_dim)
        self.value = nn.LazyLinear(self.hidden_dim)

        # Separate linear layers for sigma-related attention
        self.sigma_query = nn.LazyLinear(self.hidden_dim)
        self.sigma_key = nn.LazyLinear(self.hidden_dim)
        self.sigma_value = nn.LazyLinear(self.hidden_dim)

        # Post-attention layer normalization
        self.norm = nn.LayerNorm(self.hidden_dim)

        # Final linear projection to 3 sigma channels
        self.sigma_proj = nn.LazyLinear(3)

        self.activation = BoundedSoftplus(threshold=6)
        self.attention_scale = self.hidden_dim ** -0.5

    def _attention(self, q, k, v):
        """
        Compute scaled dot-product attention for q, k, v sequences.

        Args:
            q, k, v (torch.Tensor): Query, Key, Value of shape [B, Length, Dim].

        Returns:
            torch.Tensor: Attention output of shape [B, Length, Dim].
        """
        attn = torch.bmm(q, k.transpose(1, 2)) * self.attention_scale
        attn = F.softmax(attn, dim=-1)
        return torch.bmm(attn, v)

    def _compute_attention(self, x):
        """
        Compute the main attention pass over feature embeddings.

        Args:
            x (torch.Tensor): Flattened patch features of shape [B, SeqLen, FeatDim].

        Returns:
            torch.Tensor: Attention output features.
        """
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        return self._attention(q, k, v)

    def _predict_sigmas(self, x):
        """
        Compute sigma predictions using a second attention pass.

        Args:
            x (torch.Tensor): Flattened patch features of shape [B, SeqLen, FeatDim].

        Returns:
            torch.Tensor: Predicted sigma maps of shape [B, SeqLen, 3].
        """
        q = self.sigma_query(x)
        k = self.sigma_key(x)
        v = self.sigma_value(x)
        out = self._attention(q, k, v)
        out = self.norm(out)
        return self.activation(self.sigma_proj(out))

    def forward(self, x):
        """
        Forward pass to predict sigma maps (σx, σy, σr) for the input image.

        Args:
            x (torch.Tensor): Image of shape [B, C, H, W].

        Returns:
            torch.Tensor: Sigma maps of shape [B, H, W, 3].
        """
        b, c, h, w = x.shape

        # Unfold image into patches: [B, C, H//ps, W//ps, ps, ps]
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        B, C, H_p, W_p, _, _ = patches.size()

        # Reshape: [B, (H_p*W_p), (C * patch_size^2)]
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, H_p * W_p, -1)

        # Attention-based feature encoding
        feats = self._compute_attention(patches)

        # Predict sigma channels
        sigmas = self._predict_sigmas(feats)  # [B, (H_p*W_p), 3]

        # Reshape to [B, 3, H_p, W_p]
        sigmas = sigmas.view(B, H_p, W_p, 3).permute(0, 3, 1, 2)

        # Upsample to match original image size via nearest neighbor
        sigmas = F.interpolate(sigmas, size=(h, w), mode='nearest').permute(0, 2, 3, 1)

        return sigmas


class AGBF(nn.Module):
    """
    Attention-Guided Bilateral Filter (AGBF) stage that uses predicted sigma maps.

    AGBF performs:
      - Spatial filtering based on σx, σy, which define anisotropic kernels.
      - Range filtering based on σr, which controls how intensities are weighted.

    Args:
        patch_size (int, optional): Patch size for the SigmaPredictor. Default is 16.
        in_channels (int, optional): Number of input channels. Default is 1.
    """
    def __init__(self, patch_size=16, in_channels=1):
        super().__init__()
        self.sigma_predictor = SigmaPredictor(patch_size=patch_size, in_channels=in_channels)
        self.patch_size = patch_size

    def compute_spatial_kernel(self, sx, sy, k, device):
        """
        Compute a spatial Gaussian kernel for each pixel, shaped by sigma_x and sigma_y.

        Args:
            sx (torch.Tensor): Sigma_x values (flattened).
            sy (torch.Tensor): Sigma_y values (flattened).
            k (int): Size of the kernel (always odd).
            device (torch.device): The device to perform computations on.

        Returns:
            torch.Tensor: A spatial kernel of shape [B, 1, 1, k, k].
        """
        half = k // 2
        yy, xx = torch.meshgrid(
            torch.arange(-half, half + 1, device=device),
            torch.arange(-half, half + 1, device=device),
            indexing='ij'
        )
        xx = xx.float().view(1, 1, 1, *xx.shape)
        yy = yy.float().view(1, 1, 1, *yy.shape)
        sx = sx.unsqueeze(-1).unsqueeze(-1)
        sy = sy.unsqueeze(-1).unsqueeze(-1)

        # Gaussian function: exp(-((x^2)/(2sx^2) + (y^2)/(2sy^2)))
        return torch.exp(- (xx**2)/(2*sx**2) - (yy**2)/(2*sy**2))

    def compute_range_kernel(self, center_values, neighbor_values, sigma_r):
        """
        Compute the range kernel based on intensity differences.

        Args:
            center_values (torch.Tensor): Center pixel values. Shape [B, C, H, W, 1].
            neighbor_values (torch.Tensor): Neighbor patches. Shape [B, C, H, W, k*k].
            sigma_r (torch.Tensor): Range sigma. Shape [B, H, W].

        Returns:
            torch.Tensor: Range kernel, same shape as neighbor_values (B, C, H, W, k*k).
        """
        diff = center_values - neighbor_values
        # For multi-channel, sum squared differences across channels
        if center_values.shape[1] > 1:
            sq = (diff**2).sum(dim=1, keepdim=True)
        else:
            sq = diff**2
        sigma_r = sigma_r.unsqueeze(1).unsqueeze(-1)
        return torch.exp(-sq / (2*sigma_r**2))

    def forward(self, x, return_sigmas=False):
        """
        Forward pass of the AGBF stage.

        1. Predict sigma maps (σx, σy, σr) using SigmaPredictor.
        2. Determine kernel size k based on max(sigma_x, sigma_y).
        3. Pad the input, unfold patches, apply spatial & range kernels, normalize, and
           sum to get the filtered output.

        Args:
            x (torch.Tensor): Input image of shape [B, C, H, W].
            return_sigmas (bool, optional): If True, returns the sigma maps in addition
                                            to the output image.

        Returns:
            torch.Tensor: Filtered image of the same shape as the input.
            (Optional) torch.Tensor: The predicted sigma maps with shape [B, H, W, 3].
        """
        x = x.float()
        b, c, h, w = x.shape

        # Predict sigma maps
        s = self.sigma_predictor(x)
        sx = s[..., 0]
        sy = s[..., 1]
        sr = s[..., 2]

        # Determine the kernel size k from max(sigma_x, sigma_y)
        m = max(sx.max().item(), sy.max().item())
        k = int(2 * math.ceil(m + 1))
        if k % 2 == 0:
            k += 1  # ensure odd kernel size
        p = k // 2

        # Pad the input
        xp = F.pad(x, [p]*4, mode='constant', value=0)

        # Unfold patches to shape [B, C, H, W, k*k]
        patches = xp.unfold(2, k, 1).unfold(3, k, 1).contiguous().view(b, c, h, w, -1)

        # Spatial kernel: shape [B, H, W, k*k], each pixel has a distinct kernel
        sxr = sx.view(-1)
        syr = sy.view(-1)
        sp = self.compute_spatial_kernel(sxr, syr, k, x.device).view(-1, k*k)
        sp = sp.view(b, h, w, k*k)

        # Range kernel: shape [B, C, H, W, k*k]
        cv = x.view(b, c, h, w, 1)  # center_values
        rg = self.compute_range_kernel(cv, patches, sr)

        # Combine spatial and range kernels
        # sp: (B, 1, H, W, k*k), rg: (B, C, H, W, k*k)
        comb = sp.view(b, 1, h, w, k*k) * rg
        s_ = comb.sum(dim=-1, keepdim=True)  # Normalization factor

        # Normalized output
        norm = comb / (s_ + 1e-8)
        out = (patches * norm).sum(dim=-1)  # Weighted sum along k*k dimension

        if return_sigmas:
            return out, s
        return out


class DenoisingPipeline(nn.Module):
    """
    A multi-stage denoising pipeline that sequentially applies several AGBF stages.

    Args:
        num_stages (int, optional): How many AGBF stages to apply. Default = 2.
        patch_size (int, optional): Patch size for the SigmaPredictor. Default = 8.
        in_channels (int, optional): Number of input channels. Default = 1.
    """
    def __init__(self, num_stages=2, patch_size=8, in_channels=1):
        super().__init__()
        self.stages = nn.ModuleList([
            AGBF(patch_size=patch_size, in_channels=in_channels) for _ in range(num_stages)
        ])

    def forward(self, x, return_sigmas=False):
        """
        Forward pass through multiple AGBF stages.

        Args:
            x (torch.Tensor): Input image of shape [B, C, H, W].
            return_sigmas (bool, optional): If True, returns sigma maps from each stage
                                            in a list.

        Returns:
            torch.Tensor: The resulting image after all stages.
            (Optional) list of torch.Tensor: Sigma maps for each stage if return_sigmas=True.
        """
        sigmas_list = []
        for stage in self.stages:
            if return_sigmas:
                x, s = stage(x, return_sigmas=True)
                sigmas_list.append(s)
            else:
                x = stage(x)
        if return_sigmas:
            return x, sigmas_list
        return x


###############################################################################
#                           Loss and Training                                 #
###############################################################################

class LossFunction:
    """
    A self-supervised loss function for image denoising that combines:
      - Multi-scale consistency
      - Edge preservation using a Difference of Gaussians (DoG) filter

    Args:
        device (torch.device): The device for kernel computations.
        lambda_ (float, optional): Weight for the edge loss term. Default = 350.
    """
    def __init__(self, device, lambda_=350):
        self.kernel_size = 7
        self.sigma_narrow = 9
        self.sigma_wide = 10
        self.device = device
        self.edge_loss_weight = lambda_
        self._initialize_dog_kernel()

    def _initialize_dog_kernel(self):
        """
        Create a Difference of Gaussians (DoG) kernel, used for edge detection.
        """
        grid_coords = torch.arange(self.kernel_size, dtype=torch.float32,
                                   device=self.device) - self.kernel_size // 2
        grid_y, grid_x = torch.meshgrid(grid_coords, grid_coords, indexing='ij')

        gaussian_narrow = torch.exp(-(grid_x**2 + grid_y**2) / (2 * self.sigma_narrow**2))
        gaussian_narrow = gaussian_narrow / gaussian_narrow.sum()

        gaussian_wide = torch.exp(-(grid_x**2 + grid_y**2) / (2 * self.sigma_wide**2))
        gaussian_wide = gaussian_wide / gaussian_wide.sum()

        # DoG: narrower Gaussian - wider Gaussian
        self.base_dog_kernel = (gaussian_narrow - gaussian_wide).unsqueeze(0).unsqueeze(0)

    def __call__(self, noisy_input, model, alpha=None):
        """
        Calculate the self-supervised loss on the given noisy image and model.

        Components of the loss:
          1) Multi-scale consistency: by comparing downsampled pairs (scale1 vs scale2).
          2) Cross-scale consistency: comparing denoised versions of the same image
             at different scales.
          3) Edge preservation: L1 loss on DoG-filtered images.

        Args:
            noisy_input (torch.Tensor): A noisy image of shape (B, C, H, W).
            model (nn.Module): The denoising model to use (DenoisingPipeline).
            alpha (float, optional): If given, overrides the edge_loss_weight.

        Returns:
            torch.Tensor: The total loss value.
        """
        if alpha is not None:
            self.edge_loss_weight = alpha

        dog_kernel = self.base_dog_kernel.repeat(noisy_input.shape[1], 1, 1, 1)

        # Generate multi-scale pairs of the noisy input
        noisy_scale1, noisy_scale2 = Downsampler(noisy_input)

        # Shuffle pixels within 2x2 blocks to prevent trivial identity solutions
        shuffled_noisy_scale1 = ELS(noisy_scale1)
        shuffled_noisy_scale2 = ELS(noisy_scale2)

        # Denoise each shuffled scale
        denoised_scale1 = model(shuffled_noisy_scale1)
        denoised_scale2 = model(shuffled_noisy_scale2)

        # Loss term: difference between the two scales' denoised outputs
        loss_resolution = (1.0 / 3.0) * F.l1_loss(denoised_scale1, denoised_scale2)

        # Full-resolution denoised
        denoised_full = model(noisy_input)

        # Downsample the full-resolution denoised image
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
        # L1 loss between magnitudes of edges
        loss_edge = self.edge_loss_weight * F.l1_loss(torch.abs(edges_noisy), torch.abs(edges_denoised))

        return loss_resolution + loss_cross_scale + loss_denoise + loss_edge


def train_model(model, noisy, loss_function, optimizer, scheduler, epochs=500):
    """
    Generic training loop for the denoising model using a self-supervised loss.

    Args:
        model (nn.Module): The denoising pipeline to be trained.
        noisy (torch.Tensor): The noisy input image of shape (B, C, H, W).
        loss_function (LossFunction): The self-supervised loss to optimize.
        optimizer (torch.optim.Optimizer): The optimizer (e.g., AdamW).
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler (e.g., OneCycleLR).
        epochs (int, optional): Number of epochs to train for. Default is 500.
    """
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)

        # Compute loss
        loss = loss_function(noisy, model)

        # Backprop
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.6f}")


###############################################################################
#                         Metrics and Data Loading                            #
###############################################################################

def compute_metrics(image, noisy_image, denoised_image, in_channels, device):
    """
    Compute PSNR and SSIM metrics for noisy and denoised images relative to the reference.

    Args:
        image (numpy.ndarray): The clean reference image in [0,1].
        noisy_image (numpy.ndarray): The noisy image in [0,1].
        denoised_image (numpy.ndarray): The denoised image in [0,1].
        in_channels (int): The number of channels. 1 for grayscale, 3 for RGB, etc.
        device (torch.device): Device used for computations (currently not used).

    Returns:
        dict: A dictionary containing:
              'psnr_noisy', 'psnr_denoised', 'ssim_noisy', 'ssim_denoised'.
    """
    metrics = {}

    metrics['psnr_noisy'] = compare_psnr(image, noisy_image, data_range=1)
    metrics['psnr_denoised'] = compare_psnr(image, denoised_image, data_range=1)

    if in_channels == 1:
        # 2D, grayscale
        metrics['ssim_noisy'], _ = compare_ssim(image, noisy_image, data_range=1, full=True)
        metrics['ssim_denoised'], _ = compare_ssim(image, denoised_image, data_range=1, full=True)
    else:
        # Multi-channel assumption (often channel_axis=2 for HWC)
        metrics['ssim_noisy'], _ = compare_ssim(image, noisy_image, data_range=1,
                                                channel_axis=2, full=True)
        metrics['ssim_denoised'], _ = compare_ssim(image, denoised_image, data_range=1,
                                                   channel_axis=2, full=True)
    return metrics


def load_data():
    """
    Load sample data for demonstration.
    Replace the file paths or logic as needed for your own data.

    Returns:
        tuple: (clean_image, noisy_image, in_channels)
    """
    # Example data loaded via NumPy arrays
    image = np.load('clean.npy')
    noisy_image = np.load('noisy.npy')

    # For grayscale tasks, in_channels = 1
    in_channels = 1
    return image, noisy_image, in_channels


def save_image(image_np, filename, in_channels):
    """
    Save a numpy array as an image file (PNG) without borders or axis.

    Args:
        image_np (numpy.ndarray): The image array in [0,1] with shape (H, W) or (H, W, C).
        filename (str): The path/filename to save the image.
        in_channels (int): Number of channels. If 1, applies a gray colormap.
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
      2. Load data (clean image, noisy image, channel count).
      3. Prepare the noisy image as a torch.Tensor.
      4. Define the denoising model and training components.
      5. Train the model using self-supervised loss.
      6. Perform inference and measure training time.
      7. Evaluate the results using PSNR/SSIM metrics.
      8. Save and visualize the denoised image, along with sigma maps.
    """
    # Environment setup
    device = setup_environment()
    setup_logger()
    logger.info("Starting denoising pipeline...")

    # Load Data
    image, noisy_image, in_channels = load_data()
    noisy = prepare_image(noisy_image, device)

    # Initialize model and training utilities
    model = DenoisingPipeline(num_stages=2, patch_size=8, in_channels=in_channels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=0.001, epochs=500, steps_per_epoch=1)
    loss_function = LossFunction(device, lambda_=400)

    # Train model
    start_time = time.time()
    train_model(model, noisy, loss_function, optimizer, scheduler, epochs=500)
    inference_time = time.time() - start_time

    # Model evaluation
    model.eval()
    logger.info("Number of trainable parameters: {}".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad))
    )

    # Inference (denoising)
    with torch.no_grad():
        denoised, sigmas_list = model(noisy, return_sigmas=True)
        denoised_image = restore_image(denoised)

    logger.info(f"Inference time: {inference_time:.4f} seconds")

    # Compute and log metrics
    metrics = compute_metrics(image, noisy_image, denoised_image, in_channels, device)
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.4f}")

    # Save denoised result
    # save_image(denoised_image, "denoised_image_result.png", in_channels)

    # Plot composite results
    plot_results(image, noisy_image, denoised_image, metrics, save_path='.')
    # Plot sigma maps from each stage
    plot_sigmas(sigmas_list, save_path='.')

    # (Optional) Save sigma maps for each stage separately
    # save_sigmas_maps_separately(sigmas_list, save_path='sigmas_maps')


if __name__ == '__main__':
    main()
