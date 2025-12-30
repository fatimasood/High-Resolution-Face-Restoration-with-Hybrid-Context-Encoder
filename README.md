# High-Resolution Face Restoration with Hybrid Context Encoder

Deep learning pipeline for restoring damaged celebrity portraits using **U-Net Autoencoder**, **GAN**, and **Hybrid Context Encoder**. Includes masking pipeline, custom losses, and evaluation metrics.

---

## Case Study: High-Resolution Face Restoration

### 1. The Scenario
A digital restoration archive provided a massive dataset of damaged high-resolution portraits. Traditional interpolation methods (nearest neighbor, bicubic) fail to capture semantic details, resulting in featureless blobs where eyes or mouths should be.

**Task:** Design, implement, and rigorously compare two deep learning architectures to restore these images:

1. **Convolutional Autoencoder (CAE)** focused on pure reconstruction.  
2. **Hybrid Context Encoder (GAN)** focused on perceptual realism.

### 2. Dataset & Preprocessing
- **Source:** CelebA-HQ (Resized to 128×128×3)  
- **Custom Data Generator:** Implemented using a `tf.data` pipeline.  
- **Masking Logic:**  
  - Random rectangular mask per batch, covering 10–25% of the image area.  
  - Mask location randomized every epoch.  
  - Target remains the original, unmasked image.

### 3. Technical Requirements

#### Part A: The Reconstructor (U-Net Autoencoder)
- U-Net inspired architecture with skip connections.  
- Bottleneck as a **flattened dense vector** to force semantic compression.  
- Custom combined loss:  

\[
L_{total} = MAE + \lambda \cdot (1 - SSIM)
\]

#### Part B: The Hallucinator (DCGAN Implementation)
- Generator uses transposed convolutions; input is a noise vector \(z \sim N(0,1)\)  
- Discriminator: standard CNN binary classifier  
- Custom training loop via `tf.GradientTape`, **no `model.fit()`**

#### Part C: The Hybrid (Context Encoder)
- Generator: U-Net Autoencoder from Part A  
- Discriminator: CNN predicting Original vs Inpainted  
- Adversarial Loss:  

\[
L_{hybrid} = \lambda_{recon} ||x - \hat{x}||^2 + \lambda_{adv} \log(1 - D(\hat{x}))
\]

**Recommended hyperparameters:**  
λ_recon = 0.99, λ_adv = 0.01


---

## Features

- Custom `tf.data` pipeline with **random rectangular masking** (10–25% per epoch)  
- U-Net Autoencoder with **latent dense vector bottleneck**  
- GAN with **PatchGAN discriminator** and transposed convolutions  
- Hybrid model combining **reconstruction + adversarial loss**  
- Evaluation metrics: **SSIM, PSNR**  
- Training curves and side-by-side visual comparisons

---

## Dataset

Download pre-resized CelebA-HQ from Kaggle:

[Kaggle: CelebA-HQ Resized 256×256](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256)

---

## Usage

```bash
# Clone repository
git clone https://github.com/fatimasood/high-resolution-face-restoration-with-hybrid-context-encoder.git
cd face-restoration-hybrid

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook "High-Resolution Face Restoration with Hybrid Context Encoder.ipynb"
