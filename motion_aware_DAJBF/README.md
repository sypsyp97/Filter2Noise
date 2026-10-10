# Motion-Aware Deformable Attention-Guided Joint Bilateral Filter (DA-JBF) for 4D CT Denoising

This folder contains the code for our motion-aware extension of the Filter2Noise framework, designed for 4D (3D + time) low-dose CT denoising. The method combines deformable attention-guided joint bilateral filtering with a self-supervised training strategy that exploits temporal motion information.

## Method

The pipeline consists of three main components:

1. **Demons-based Motion Field Estimation**: Multi-scale Demons registration computes dense displacement fields between consecutive temporal frames. Gaussian smoothing of velocity updates uses `sigma_fluid=2.0` pixels. Warped neighboring frames and displacement magnitude maps provide auxiliary guidance during both training and inference.

2. **Deformable Attention-Guided Joint Bilateral Filter (DA-JBF)**: Extends the original Filter2Noise bilateral filter with:
   - A deformable attention-based sigma predictor (`DLKASigmaPredictor`) that uses multi-scale deformable attention blocks to predict spatially varying filter parameters
   - Learnable sampling offsets that adapt the filter kernel shape to local image content and motion patterns
   - Joint bilateral filtering with anisotropic spatial kernels and a learned 16-dimensional feature-space range kernel
   - A single-stage model with **4,443 trainable parameters**, two attention scales (spacing 1 and 3), and 25 sampling points; learned filter offsets are bounded to ±5 pixels and sigma maps to 6.0
   - Strict value isolation: only the target noisy slice supplies filtering values; auxiliary channels predict weights and offsets. Auxiliary features are dropped with probability 0.5 during training.

3. **Regularized Self-Supervised Loss**: The pseudo-target is the average of the warped previous and next frames. The target and input share the same auxiliary guidance. An L1 reconstruction loss is combined with the stopped-gradient consistency term, whose weight increases from 0 to 2.0 across training. The optional Sobel gradient L1 loss is disabled in the primary objective.

## Input Format

The code expects calibrated **Hounsfield units (HU)** in an HDF5 `volume` dataset stored as `(D, H, W, T)`. The internal layout is `(H, W, D, T)`. Model inputs are `(B, 4, H, W)`.

Clean and noisy volumes use the same fixed mapping: clip to `[-1024, 3071]` HU, then scale to `[0, 1]`. There is no independent volume min–max normalization. Denoised HDF5 outputs preserve the input file's `(D, H, W, T)` layout and HU units; metrics use the fixed normalized range. Comparison images use the separate lung display window (WL = −600 HU, WW = 1500 HU).

Temporal morphing expects independently acquired noise across frames before registration. Any temporal resampling should precede acquisition-noise generation; interpolating an already noisy sequence introduces noise correlations. Registration and warping can still introduce residual dependencies in the pseudo-target.

## Usage

```bash
python filter2noise_4d_dajbf.py \
    --noisy path/to/noisy_volume.h5 \
    --output denoised_4d.h5
```

Add `--clean path/to/clean_volume.h5` to compute metrics and diagnostic plots. The clean reference is used only for evaluation. Training samples interior slice-phase pairs (`t=1,...,T-2`); inference covers every frame, using the target slice as guidance when a temporal neighbor is absent.

| Argument | Default | Description |
|----------|---------|-------------|
| `--clean` | *(optional)* | Clean HU reference (HDF5, evaluation only) |
| `--noisy` | *(required)* | Path to noisy 4D volume (HDF5) |
| `--output` | `denoised_4d.h5` | Output denoised volume path |
| `--output-dir` | `output_4d` | Output directory for figures and metrics |
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 16 | Training batch size |
| `--lr` | 0.003 | AdamW learning rate / OneCycleLR maximum |
| `--weight-decay` | 0.0001 | AdamW weight decay |
| `--gamma-max` | 2.0 | Maximum regularization weight for the self-supervised loss |
| `--edge-weight` | 0.0 | Optional Sobel edge-loss weight; positive values enable edge regularization |
| `--no-rmsnorm` | false | Disable RMSNorm; enabled by default |
| `--train-fraction` | 0.10 | Fraction of (z, t) pairs used for training |
| `--seed` | 42 | Model initialization and training-pair sampling seed |

OneCycleLR uses 30% warmup over the requested epochs. CUDA training and model inference use bfloat16 autocast; CPU execution uses float32. Demons registration operates in float32. Use `--train-fraction 1.0` for full-data training; the default is 10%. Different `--seed` values also select different subsets.

## Evaluation Outputs

With `--clean`, the script writes metrics and diagnostic plots under `--output-dir`. `overall_metrics.txt` reports **T00 (frame 0)** and **FULL_4D (all frames)** separately:

- PSNR uses a normalized peak range of 1.0.
- SSIM uses an 11×11 Gaussian window with σ=1.5, K1=0.01, K2=0.03 and population covariance.
- Absolute ΔPSNR is measured against the noisy input.
- PSNR-implied MSE reduction is `100 * (1 - 10 ** (-delta_psnr / 10))`, corresponding to geometric-mean squared error across slices.
- Direct voxel MSE reduction uses the ratio of arithmetic mean squared errors against the clean reference.

Per-slice metric matrices, heatmaps, image comparisons, sigma maps and sampling-offset visualizations are saved alongside the summary.

## Requirements

In addition to the base Filter2Noise dependencies (`requirements.txt` in the parent directory), this code requires:

- `h5py`
- `loguru`

Use a PyTorch version supporting `torch.autocast` and bfloat16 CUDA execution (PyTorch 2.x recommended).
