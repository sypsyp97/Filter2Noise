[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13bIhK2inxzjLfUUoLoltnAodFKb_YzXb?usp=sharing) [![arXiv](https://img.shields.io/badge/arXiv-2504.13519-b31b1b.svg)](https://arxiv.org/abs/2504.13519)
[![Hugging Face Paper](https://img.shields.io/badge/🤗%20Hugging%20Face-Paper-yellow)](https://huggingface.co/papers/2504.13519)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sypsyp97/Filter2Noise)

# Filter2Noise: A Framework for Interpretable and Zero-Shot Low-Dose CT Image Denoising

<!-- Optional: Add a Table of Contents here if desired -->

## Table of Contents

- [Abstract](#abstract)
- [News](#news)
- [Method Overview](#method-overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Important Parameters](#important-parameters)
- [Citation](#citation)
- [License](#license)

## Abstract

>Noise in low-dose computed tomography (LDCT) can obscure important diagnostic details. While deep learning offers powerful denoising, supervised methods require impractical paired data, and self-supervised alternatives often use opaque, parameter-heavy networks that limit clinical trust. We propose Filter2Noise (F2N), a novel self-supervised framework for interpretable, zero-shot denoising from a single LDCT image. Instead of a black-box network, its core is an Attention-Guided Bilateral Filter, a transparent, content-aware mathematical operator. A lightweight attention module predicts spatially varying filter parameters, making the process transparent and allowing interactive radiologist control. To learn from a single image with correlated noise, we introduce a multi-scale self-supervised loss coupled with Euclidean Local Shuffle (ELS) to disrupt noise patterns while preserving anatomical integrity. On the Mayo Clinic LDCT Challenge, F2N achieves state-of-the-art results, outperforming competing zero-shot methods by up to 3.68 dB in PSNR. It accomplishes this with only 3.6k parameters, orders of magnitude fewer than competing models, which accelerates inference and simplifies deployment. By combining high performance with transparency, user control, and high parameter efficiency, F2N offers a trustworthy solution for LDCT enhancement. We further demonstrate its applicability by validating it on clinical photon-counting CT data.
## News

📢 **2025-07**: We have implemented a Mamba-based variant of Filter2Noise as an experimental alternative to the attention mechanism. However, preliminary results show that:

- **More parameters**: The Mamba implementation uses significantly more parameters (~8x increase) compared to the lightweight attention-based approach
- **Worse performance**: Initial testing indicates inferior denoising performance compared to the simple attention-based mechanism

## Method Overview

![Method Overview](method.png)
*Figure 1: (a) The F2N denoising pipeline. (b) The downsampling strategy, following [ZS-N2N](https://openaccess.thecvf.com/content/CVPR2023/papers/Mansour_Zero-Shot_Noise2Noise_Efficient_Image_Denoising_Without_Any_Data_CVPR_2023_paper.pdf). (c) Our proposed Euclidean Local Shuffle (ELS).*

## Key Features

- ✨ **Interpretable Denoising**: Visualizable filter parameters (σx, σy, σr) for understanding the denoising behavior.
- 🚀 **Lightweight Architecture**: Only 1.8k parameters for single-stage (F2N-S1) and 3.6k for two-stage (F2N-S2).
- 💡 **Self-Supervised Learning**: No clean reference images needed for training.
- 🖱️ **User-Controlled Denoising**: Post-training adjustment of filter parameters for region-specific denoising.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Denoising

**Important:** Ensure your input CT image is normalized to the range [0, 1] before processing.
Run the main script to apply the denoising pipeline:

```bash
python filter2noise.py
```

### Interactive Demo

Launch the Gradio interface for interactive denoising and parameter adjustment:

```bash
python demo.py
```

The interactive demo allows you to:

- Upload and denoise your own CT images.
- Visualize the predicted filter parameters (σx, σy, σr).
- Adjust filter parameters in specific regions using interactive controls.
- See real-time updates of the denoised result.

## Important Parameters

### Lambda (λ)

The parameter `λ` in the loss function balances noise reduction (`L_rec`) and edge preservation (`L_reg`):

```math
L_\text{total} = L_\text{rec} + \lambda \cdot L_\text{reg}
```

- **Low λ**: More aggressive noise reduction, potentially leading to blurring of fine details.
- **High λ**: Better preservation of edges and structures, but may leave more residual noise.
- **Recommended Starting Value**: `λ=350`. Adjust this value based on the specific image characteristics and desired denoising outcome.

### Other Parameters

- `patch_size`: Controls the granularity of adaptive filtering (default: `8`). Smaller patches adapt more locally.
- `num_stages`: Number of Attention-Guided Bilateral Filter (AGBF) stages (1 or 2). More stages can improve results but increase computation.
- `in_channels`: Number of input image channels (default: `1` for grayscale CT images).

## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@misc{sun2025filter2noise,
      title={Filter2Noise: Interpretable Self-Supervised Single-Image Denoising for Low-Dose CT with Attention-Guided Bilateral Filtering}, 
      author={Yipeng Sun and Linda-Sophie Schneider and Mingxuan Gu and Siyuan Mei and Chengze Ye and Fabian Wagner and Siming Bayer and Andreas Maier},
      year={2025},
      eprint={2504.13519},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2504.13519}, 
}
```

## License

This project is open source and available under the Creative Commons Attribution-NonCommercial 4.0 International License. See the [`LICENSE`](LICENSE) file for more details.
