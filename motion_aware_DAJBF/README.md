# Motion-Aware Deformable Attention-Guided Joint Bilateral Filter (DA-JBF) for 4D CT Denoising

This folder contains the code for our motion-aware extension of the Filter2Noise framework, designed for 4D (3D + time) low-dose CT denoising. The method combines deformable attention-guided joint bilateral filtering with a self-supervised training strategy that exploits temporal motion information.

## Method

The pipeline consists of three main components:

1. **Demons-based Motion Field Estimation**: Multi-scale diffeomorphic registration computes dense displacement fields between consecutive temporal frames, providing motion-compensated neighboring frames and motion magnitude maps as auxiliary inputs.

2. **Deformable Attention-Guided Joint Bilateral Filter (DA-JBF)**: Extends the original Filter2Noise bilateral filter with:
   - A deformable attention-based sigma predictor (`DLKASigmaPredictor`) that uses multi-scale deformable attention blocks to predict spatially varying filter parameters
   - Learnable sampling offsets that adapt the filter kernel shape to local image content and motion patterns
   - Joint bilateral filtering that incorporates both spatial and learned feature-space range kernels

3. **Regularized Self-Supervised Loss**: A self-supervised training objective that uses motion-compensated temporal averages as pseudo-targets, with a regularization term that progressively enforces consistency between denoised outputs.

## Input Format

The code expects 4D volumes stored as HDF5 files with a `volume` dataset of shape `(T, Z, X, Y)` (internally transposed to `(X, Y, Z, T)`).

## Usage

```bash
python filter2noise_4d_dattn_n2n.py
```

By default, the script looks for:
- `case2_4d_clean_norm_leap.h5` - clean reference volume (for evaluation)
- `case2_4d_noisy_projdomain_leap.h5` - noisy input volume

Outputs:
- `denoised_4d_dattn_n2n_leap.h5` - denoised 4D volume
- `enhanced_output_dattn_n2n_leap/` - visualizations, sigma maps, offset maps, and per-slice metrics

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRAIN_FRACTION` | 0.10 | Fraction of (z, t) pairs used for training |
| `epochs` | 100 | Number of training epochs |
| `hidden_dim` | 16 | Hidden dimension for the sigma predictor |
| `num_sample_points` | 25 | Number of sampling points in the bilateral filter kernel |
| `gamma_max` | 2.0 | Maximum regularization weight for the self-supervised loss |
| `num_scales` | 3 | Number of pyramid scales for Demons registration |

## Requirements

In addition to the base Filter2Noise dependencies (`requirements.txt` in the parent directory), this code requires:

- `h5py`
- `loguru`
