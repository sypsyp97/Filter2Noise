# Filter2Noise: Interpretable Self-Supervised Single Low-Dose CT Image Denoising

## Abstract

Effective denoising is crucial in low-dose CT to enhance subtle structures and low-contrast lesions while preventing diagnostic errors. Supervised methods struggle with limited paired datasets, and self-supervised approaches often require multiple noisy images and rely on deep networks like U-Net, offering little insight into the denoising mechanism. To address these challenges, we propose an interpretable self-supervised single-image denoising framework---Filter2Noise (F2N). Our approach introduces an Attention-Guided Bilateral Filter that adapted to each noisy input through a lightweight module that predicts spatially varying filter parameters, which can be visualized and adjusted post-training for user-controlled denoising in specific regions of interest. To enable single-image training, we introduce a novel downsampling shuffle strategy with a new self-supervised loss function that extends the concept of Noise2Noise to a single image and addresses spatially correlated noise. On the Mayo Clinic 2016 low-dose CT dataset, F2N outperforms the leading self-supervised single-image method (ZS-N2N) by 4.59 dB PSNR while improving transparency, user control, and parametric efficiency. These features provide key advantages for medical applications that require precise and interpretable noise reduction. Our code is demonstrated in the following [Google Colab notebook](https://colab.research.google.com/drive/1vIfsWU32BxEBAanjSFhfiy4zIY8OIeZL?usp=sharing).

![Method Overview](method.png)

## Key Features

- **Interpretable Denoising**: Visualizable filter parameters (σx, σy, σr) for understanding the denoising behavior
- **Lightweight Architecture**: Only 1.8k parameters for single-stage (F2N-S1) and 3.6k for two-stage (F2N-S2)
- **Self-Supervised Learning**: No clean reference images needed for training
- **User-Controlled Denoising**: Post-training adjustment of filter parameters for region-specific denoising
- **Superior Performance**: Outperforms ZS-N2N by 4.59 dB PSNR on Mayo Clinic 2016 low-dose CT dataset

## Codebase Structure

- **filter2noise.py**: Core implementation of the Filter2Noise framework
  - `SigmaPredictor`: Lightweight attention module that predicts filter parameters
  - `AGBF`: Attention-Guided Bilateral Filter implementation
  - `DenoisingPipeline`: Multi-stage denoising pipeline
  - `LossFunction`: Self-supervised loss combining multi-scale consistency with edge preservation
- **demo.py**: Interactive Gradio interface for visualizing and adjusting filter parameters
- **requirements.txt**: Required dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python filter2noise.py
```

### Interactive Demo

```bash
python demo.py
```

The interactive demo provides a Gradio interface that allows you to:

- Upload and denoise your own CT images, better saved in `.npy` with shape (H, W)
- Visualize the predicted filter parameters
- Adjust filter parameters in specific regions
- See real-time updates of the denoised result

## Important Parameters

### Lambda Parameter (λ)

The lambda parameter (λ) in the loss function balances noise reduction and edge preservation:

```python
L_total = L_rec + λ * L_reg
```

- **Low λ**: More aggressive noise reduction but potential blurring
- **High λ**: Better edge preservation but possibly more residual noise
- **Recommended**: Start with λ=350 and adjust based on your specific needs

### Other Key Parameters

- `patch_size`: Controls the granularity of adaptive filtering (default: 8)
- `num_stages`: Number of AGBF stages (1 or 2)
- `in_channels`: Number of input channels (default: 1 for CT images)

## License

MIT License, see [LICENSE](LICENSE).
