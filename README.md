# GeoRefine: Geometry-Aware Rendering Enhancement Framework

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> A deep learning system designed to improve the perceptual and structural quality of rendered 2D and 3D assets by explicitly conditioning image enhancement on geometric signals such as depth and surface normals.

## 📖 Overview

**GeoRefine** is a geometry-aware rendering enhancement framework that bridges the gap between traditional image enhancement methods and rendering-aware structural constraints. Unlike conventional image enhancement networks that operate purely in RGB space, GeoRefine integrates geometric conditioning to prevent structural artifacts commonly introduced during enhancement of rendered content.

Traditional image enhancement networks are effective for denoising or super-resolution but often fail when applied to rendered content because they ignore scene geometry. This project addresses that limitation by incorporating depth and surface normal information into the enhancement pipeline.

### Key Features

- 🎯 **UNet-based image enhancement model** with residual blocks
- 🔍 **Geometry conditioning** using depth and normal maps
- 📐 **Structural consistency losses** to preserve scene coherence
- 👁️ **Perceptual loss** for human-aligned quality assessment
- 📊 **Comprehensive evaluation** (PSNR, SSIM, LPIPS)
- 🎮 **Rendering pipeline integration** compatible with G-buffer outputs

## 🎯 Motivation

Rendering pipelines in games, simulations, and creative workflows frequently suffer from quality issues including:

- ⚡ **Temporal instability** across frames
- 🌫️ **Edge blurring** around object boundaries
- 📦 **Compression artifacts** from lossy encoding
- 🔇 **Denoising oversmoothing** that removes fine details
- 🎨 **Super-resolution hallucinations** that violate geometry

Pure image-space networks optimize pixel error but may distort geometry, especially around occlusion boundaries. This framework enforces structural alignment between depth gradients and surface orientation to preserve scene coherence.

**Inspired by advances in:**
- Differentiable rendering
- Multi-view consistency
- Geometry-guided learning
- Perceptual quality modeling

## 🏗️ Architecture

### 1. Input Representation

The model accepts a multi-channel input concatenating:

- **RGB image** (3 channels) - The rendered image to enhance
- **Depth map** (1 channel) - Scene depth information
- **Surface normals** (3 channels) - Surface orientation vectors

**Total input channels:** 7

Depth and normals can be derived from:
- Precomputed **G-buffer outputs** (if available from renderer)
- **Estimated depth** (MiDaS integration possible)
- **Screen-space depth proxy** (current stub implementation)

### 2. Enhancement Backbone

The enhancement network uses a **UNet-style encoder-decoder architecture** with the following components:

```
Input (7 channels) 
    ↓
Downsampling Encoder (global context extraction)
    ↓
Residual Bottleneck (feature processing)
    ↓
Upsampling Decoder with Skip Connections
    ↓
Sigmoid Output (RGB reconstruction, 3 channels)
```

UNet was chosen due to its proven performance in restoration tasks ([Ronneberger et al., 2015](#references)).

**Architecture highlights:**
- Skip connections preserve fine details
- Residual blocks facilitate training deep networks
- Multi-scale processing captures both local and global context

### 3. Loss Functions

The total training loss is a weighted combination of three components:

#### 1️⃣ L1 Reconstruction Loss
Encourages pixel-wise fidelity between enhanced and target images.

```
L_recon = ||I_enhanced - I_target||_1
```

#### 2️⃣ Perceptual Loss
Uses **VGG feature activations** to align high-level representations, ensuring enhanced images are perceptually similar to targets. Inspired by [Johnson et al., 2016](#references).

```
L_perceptual = ||φ(I_enhanced) - φ(I_target)||_2
```

where φ represents VGG feature extractor.

#### 3️⃣ Geometry Consistency Loss
Enforces alignment between depth gradients and surface normals to maintain structural coherence:

```
L_geo = ||∇D - f(N)||_1
```

where:
- ∇D = depth gradients
- N = surface normals
- f = transformation function

This loss prevents geometry-breaking hallucinations and preserves occlusion boundaries.

**Total loss:**
```
L_total = λ_recon × L_recon + λ_perceptual × L_perceptual + λ_geo × L_geo
```

## 📊 Evaluation Metrics

The framework evaluates enhancement quality using three complementary metrics:

| Metric | Purpose | Range |
|--------|---------|-------|
| **PSNR** | Peak Signal-to-Noise Ratio | Higher is better (dB) |
| **SSIM** | Structural Similarity Index ([Wang et al., 2004](#references)) | 0-1 (higher is better) |
| **LPIPS** | Learned Perceptual Image Patch Similarity ([Zhang et al., 2018](#references)) | 0-1 (lower is better) |

**LPIPS** aligns particularly well with human perceptual quality judgments, making it a crucial metric for this framework.

## 📁 Dataset Format

The framework expects paired training data organized as follows:

```
data/
├── train/
│   ├── input/      # Degraded or low-quality renders
│   └── target/     # High-quality ground truth renders
└── val/
    ├── input/      # Validation degraded renders
    └── target/     # Validation ground truth renders
```

### Data Preparation

Create synthetic degradation for training pairs using:
- **Gaussian blur** - simulate out-of-focus rendering
- **Downsampling** - simulate lower resolution rendering
- **Compression artifacts** - simulate JPEG/video compression
- **Noise injection** - simulate Monte Carlo noise

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0+
- CUDA-capable GPU (recommended for training)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/ramatoulaye02/GeoRefine.git
cd GeoRefine
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Usage

#### Training

Train the model on your dataset:

```bash
python scripts/train.py --config configs/default.yaml
```

**Key training parameters** (edit `configs/default.yaml`):
- `batch_size`: Number of samples per batch
- `learning_rate`: Optimizer learning rate
- `num_epochs`: Total training epochs
- `loss_weights`: λ values for loss components

#### Evaluation

Evaluate a trained model on validation data:

```bash
python scripts/eval.py --checkpoint outputs/checkpoints/model_ep10.pt
```

This computes PSNR, SSIM, and LPIPS metrics on the validation set.

#### Inference

Enhance a single image:

```bash
python scripts/infer.py \
  --ckpt outputs/checkpoints/model_ep10.pt \
  --input path/to/image.png \
  --output enhanced.png
```

For batch processing, use glob patterns:
```bash
python scripts/infer.py \
  --ckpt outputs/checkpoints/model_ep10.pt \
  --input "renders/*.png" \
  --output enhanced/
```

## 📂 Project Structure

```
GeoRefine/
├── configs/              # Configuration files
│   └── default.yaml     # Default training configuration
├── data/                # Dataset directory
│   ├── train/           # Training data
│   └── val/             # Validation data
├── scripts/             # Executable scripts
│   ├── train.py         # Training script
│   ├── eval.py          # Evaluation script
│   └── infer.py         # Inference script
├── src/                 # Source code
│   ├── data/            # Dataset loading and transforms
│   ├── models/          # Neural network architectures
│   │   ├── unet.py      # UNet backbone
│   │   └── conditioning.py  # Geometry conditioning
│   ├── losses/          # Loss functions
│   │   ├── perceptual.py    # VGG perceptual loss
│   │   └── geo_losses.py    # Geometry consistency loss
│   ├── geometry/        # Geometric processing
│   │   ├── depth_midas.py   # Depth estimation
│   │   └── normals.py       # Normal map computation
│   ├── engine/          # Training and evaluation loops
│   │   ├── trainer.py   # Training logic
│   │   └── evaluator.py # Evaluation logic
│   └── utils/           # Utility functions
│       ├── metrics.py   # PSNR, SSIM, LPIPS
│       ├── io.py        # File I/O operations
│       └── viz.py       # Visualization utilities
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🔬 Related Work

This project operates at the intersection of **computer vision**, **neural rendering**, **graphics pipelines**, and **perceptual image quality**. It is conceptually inspired by recent advances in geometry-aware learning and perceptual restoration.

### Neural Rendering & Geometry Conditioning

- **NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis**  
  *Mildenhall et al., 2020*  
  Introduced continuous neural scene representations conditioned on spatial structure.

- **Deep Shading: Convolutional Neural Networks for Screen-Space Shading**  
  *Nalbach et al., 2017*  
  Demonstrated CNN-based screen-space shading using G-buffer information.

- **Deferred Neural Rendering**  
  *Thies et al., 2019*  
  Combined traditional rendering buffers with neural refinement networks.

### Geometry-Aware Super-Resolution & Enhancement

- **Deep Image Prior**  
  *Ulyanov et al., 2018*  
  Showed structural biases in CNNs useful for image restoration.

- **Learning a Deep Convolutional Network for Image Super-Resolution (SRCNN)**  
  *Dong et al., 2014*  
  Early CNN-based super-resolution architecture.

- **Enhanced Deep Residual Networks for Single Image Super-Resolution (EDSR)**  
  *Lim et al., 2017*  
  State-of-the-art super-resolution with deep residual learning.

### Perceptual and Structural Losses

- **Perceptual Losses for Real-Time Style Transfer and Super-Resolution**  
  *Johnson et al., 2016*  
  Introduced perceptual loss using pre-trained VGG features.

- **The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS)**  
  *Zhang et al., 2018*  
  Proposed learned perceptual similarity metric aligned with human judgment.

- **Image Quality Assessment: From Error Visibility to Structural Similarity (SSIM)**  
  *Wang et al., 2004*  
  Classic structural similarity metric for image quality.

## 🔮 Future Extensions

Potential improvements and extensions for the framework:

- [ ] **MiDaS depth estimation** - Replace depth stub with robust monocular depth estimation
- [ ] **Temporal consistency** - Integrate temporal reprojection for video enhancement
- [ ] **Multi-view enforcement** - Support multi-view geometry constraints
- [ ] **GAN-based refinement** - Add adversarial training for enhanced realism
- [ ] **Pipeline integration** - Integrate into Blender, Unity, or Unreal Engine
- [ ] **Real-time inference** - Optimize for real-time rendering pipelines
- [ ] **Transformer backbone** - Experiment with vision transformers
- [ ] **3D scene understanding** - Incorporate 3D scene priors

## 💡 Why This Matters

Game engines and creative pipelines increasingly rely on neural post-processing for real-time quality enhancement. However, **geometry-agnostic models risk breaking scene structure**, introducing visual artifacts that violate physical plausibility.

By conditioning enhancement on depth and normals, **GeoRefine**:

✅ **Reduces structural artifacts** in enhanced images  
✅ **Preserves occlusion boundaries** and sharp edges  
✅ **Improves perceptual realism** through geometry-aware processing  
✅ **Aligns vision enhancement with rendering principles**

This work enables:
- **Real-time quality enhancement** in game engines
- **Post-processing pipelines** for animations and VFX
- **Quality recovery** from compressed or degraded renders
- **Neural rendering** with structural guarantees

## 📚 References

1. **Dong, C., Loy, C. C., He, K., & Tang, X.** (2014). Learning a Deep Convolutional Network for Image Super-Resolution. *ECCV*.

2. **Johnson, J., Alahi, A., & Fei-Fei, L.** (2016). Perceptual Losses for Real-Time Style Transfer and Super-Resolution. *ECCV*.

3. **Lim, B., Son, S., Kim, H., Nah, S., & Lee, K. M.** (2017). Enhanced Deep Residual Networks for Single Image Super-Resolution. *CVPR Workshop*.

4. **Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., & Ng, R.** (2020). NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. *ECCV*.

5. **Nalbach, O., Arabadzhiyska, E., Mehta, D., Seidel, H. P., & Ritschel, T.** (2017). Deep Shading: Convolutional Neural Networks for Screen-Space Shading. *Computer Graphics Forum*.

6. **Ronneberger, O., Fischer, P., & Brox, T.** (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI*.

7. **Thies, J., Zollhöfer, M., & Nießner, M.** (2019). Deferred Neural Rendering: Image Synthesis using Neural Textures. *ACM Transactions on Graphics*.

8. **Ulyanov, D., Vedaldi, A., & Lempitsky, V.** (2018). Deep Image Prior. *CVPR*.

9. **Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.** (2004). Image Quality Assessment: From Error Visibility to Structural Similarity. *IEEE Transactions on Image Processing*.

10. **Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O.** (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. *CVPR*.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.

## 🙏 Acknowledgments

This project builds upon foundational work in neural rendering, perceptual losses, and geometry-aware learning. We thank the authors of the referenced papers for their contributions to the field.

---

**Built with ❤️ for the computer graphics and vision community**
