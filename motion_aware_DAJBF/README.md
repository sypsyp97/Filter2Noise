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
python filter2noise_4d_dajbf.py \
    --clean path/to/clean_volume.h5 \
    --noisy path/to/noisy_volume.h5 \
    --output denoised_4d.h5 \
    --output-dir output_4d
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--clean` | *(required)* | Path to clean 4D volume (HDF5, for evaluation) |
| `--noisy` | *(required)* | Path to noisy 4D volume (HDF5) |
| `--output` | `denoised_4d.h5` | Output denoised volume path |
| `--output-dir` | `output_4d` | Output directory for figures and metrics |
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 16 | Training batch size |
| `--lr` | 0.001 | Learning rate |
| `--gamma-max` | 2.0 | Maximum regularization weight for the self-supervised loss |
| `--train-fraction` | 0.10 | Fraction of (z, t) pairs used for training |
| `--seed` | 77 | Random seed |

## Requirements

In addition to the base Filter2Noise dependencies (`requirements.txt` in the parent directory), this code requires:

- `h5py`
- `loguru`
