# High-Resolution-Face-Restoration-with-Hybrid-Context-Encoder
Deep learning pipeline for restoring damaged celebrity portraits using U-Net Autoencoder, GAN, and Hybrid Context Encoder. Includes masking pipeline, custom losses, and evaluation metrics.

This repository demonstrates a complete **deep learning workflow** for restoring high-resolution damaged celebrity portraits using:

- **Part-A:** U-Net Autoencoder for pixel-wise reconstruction.
- **Part-B:** GAN (DCGAN) for perceptual realism.
- **Part-C:** Hybrid Context Encoder combining reconstruction + adversarial loss for state-of-the-art inpainting.

---

## Dataset

We use **CelebA-HQ** dataset (resized to 128×128×3) for training.  
You can download from Kaggle: [CelebA-HQ Resized](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256)

---

## Features

- Custom `tf.data` pipeline with **random rectangular masking** (10-25% area) per epoch.
- U-Net Autoencoder with **latent dense vector bottleneck**.
- GAN with **PatchGAN discriminator** and transposed convolutions.
- Hybrid model for combining **reconstruction + adversarial training**.
- Evaluation metrics: **SSIM, PSNR**, and side-by-side visual comparisons.
- Training curves for generator, discriminator, and reconstruction loss.

---

## Usage

```bash
# Clone repo
git clone https://github.com/fatimasood/high-resolution-face-restoration-with-hybrid-context-encoder.git
cd face-restoration-hybrid

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook High-Resolution Face Restoration with Hybrid Context Encoder.ipynb
