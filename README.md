# Mamba-UNet++: Hybrid CNN–SSM Architecture for Diabetic Foot Ulcer Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Official implementation of the Mamba-UNet++ segmentation architecture — a hybrid CNN–State Space Model that integrates UNet++ dense skip connections with Mamba state-space blocks to capture long-range spatial dependencies in medical images.

## Paper Summary

**Hybrid Mamba and UNet plus plus framework for robust diabetic foot ulcer segmentation**

Diabetic foot ulcer segmentation is difficult because wounds have irregular borders, varied tissue appearance, and complex clinical backgrounds.
This work combines U-Net++ multi-scale feature fusion with bidirectional Mamba blocks to model long-range spatial context, trained with BCE and Focal Tversky loss.
On FUSC2021, it achieves a Dice score of 0.9140 and IoU of 0.8416; on DFUC2022, it achieves a Dice score of 0.8427 and IoU of 0.7281.
The results outperform standard U-Net and U-Net++ baselines, supporting state-space modeling for automated diabetic wound analysis.

## Overview

Diabetic foot ulcer (DFU) segmentation presents extreme challenges: lesions often occupy less than 5% of image area with highly irregular boundaries. Mamba-UNet++ addresses this by combining the multi-scale feature fusion of UNet++ with the global context modeling of Mamba state-space blocks, achieving state-of-the-art performance on the DFUC2022 benchmark.

### Key Components

- **EfficientNet-B4 Encoder** — Pretrained backbone providing strong multi-scale feature extraction
- **UNet++ Dense Skip Connections** — Nested decoder pathways (L⁰–L⁴) with deep supervision
- **Mamba State-Space Context Blocks** — Bidirectional 2D scanning with selective state-space modeling replacing conventional attention
- **Atrous Spatial Pyramid Pooling (ASPP)** — Multi-scale dilated convolutions at the bottleneck
- **Attention Gates** — Gated skip connections filtering irrelevant encoder features
- **Scheduled Multi-Objective Loss** — Phase-aware weighting of BCE, Dice, Tversky (FN-heavy), and Lovász-Hinge losses
- **Test-Time Augmentation** — Logit-space averaging across geometric flips

## Results

### DFUC2022 Dataset

| Model | Dice ↑ | IoU ↑ | Precision | Recall | Boundary Score ↑ |
|-------|--------|-------|-----------|--------|--------------------|
| U-Net | 0.5079 | 0.3466 | — | — | — |
| U-Net++ | 0.6009 | 0.4366 | — | — | — |
| **Mamba-UNet++ (Ours)** | **0.8427** | **0.7281** | **0.8188** | **0.8680** | **0.3768** |

The proposed architecture improves Dice by **+32.3%** over U-Net and **+23.0%** over U-Net++ on DFUC2022.

### FUSC2020 Dataset

| Model | Dice ↑ | IoU ↑ | Precision | Recall | Boundary Score ↑ |
|-------|--------|-------|-----------|--------|--------------------|
| U-Net | 0.6325 | 0.4657 | — | — | — |
| U-Net++ | 0.6487 | 0.4828 | — | — | — |
| **Mamba-UNet++ (Ours)** | **0.9140** | **0.8416** | **0.8958** | **0.9329** | **0.5515** |

### Ablation Study (DFUC2022)

| Configuration | Dice ↑ | IoU ↑ | Precision | Recall |
|--------------|--------|-------|-----------|--------|
| Full model (Mamba + DeepSup + BiDir + LossSched) | **0.7800** | **0.6922** | 0.8854 | 0.7847 |
| w/o Mamba (CBAM only) | 0.7667 | 0.6755 | 0.9066 | 0.7490 |
| w/o Deep Supervision | 0.7723 | 0.6787 | 0.9026 | 0.7563 |
| w/o Bidirectional Scan | 0.7974 | 0.7110 | 0.8868 | 0.8028 |
| w/o Loss Schedule | 0.8052 | 0.7196 | 0.8612 | 0.8369 |

### Model Efficiency

| Metric | Value |
|--------|-------|
| Parameters | 28.02 M |
| FLOPs | 239.67 G |
| FPS | 36.19 |
| Inference (single image) | 27.63 ms |

## Repository Structure

```
├── txt.py                  # Full training pipeline (Mamba-UNet++ v4)
├── unetseg.ipynb           # Jupyter notebook — training & evaluation
├── unetsegold.ipynb        # Previous iteration notebook
├── latest/
│   ├── metrics.csv              # Ablation study results
│   ├── metrics_history_full.csv  # Per-epoch training metrics
│   ├── training_metrics_full.png # Full training dashboard
│   └── training_metrics_v4_dfuc.png
├── training_metrics.png     # Training curves
├── training_metrics_DFUC.png
├── training_metrics_FUSC.png
├── inference_results_DFUC.png   # Qualitative segmentation samples
├── inference_results_FUSC.png
└── LICENSE
```

## Installation

```bash
# Create conda environment
conda create -n mamba-unet python=3.10 -y
conda activate mamba-unet

# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Mamba SSM
pip install mamba-ssm

# Install dependencies
pip install opencv-python albumentations timm scipy matplotlib tqdm
```

## Usage

### Training

```python
python txt.py
```

Place your dataset in `datasets/DFUC2022_train_release/` with the structure:
```
datasets/
└── DFUC2022_train_release/
    ├── DFUC2022_train_images/
    │   ├── subject001.jpg
    │   └── ...
    └── DFUC2022_train_masks/
        ├── subject001.png
        └── ...
```

### Inference

```python
from txt import MambaUNetPPv4, sliding_window_inference
import torch

model = MambaUNetPPv4(deep_supervision=False, pretrained=False)
model.load_state_dict(torch.load("best_mamba_unetpp_v4_dfuc.pth"))
model.eval().cuda()

# Run sliding-window inference with TTA
logit_map = sliding_window_inference(model, image_np, device="cuda", use_tta=True)
prediction = (1 / (1 + np.exp(-logit_map)) > 0.5).astype(np.float32)
```

### Key Training Configurations

| Setting | Value |
|---------|-------|
| Train patch size | 512×512 (ulcer-centred crops) |
| Inference stride | 384 (128 px overlap) |
| Batch size | 8 |
| Epochs | 150 |
| Optimizer | AdamW (encoder lr=3e-5, decoder lr=2e-4) |
| Scheduler | OneCycleLR (cosine anneal) |
| EMA decay | 0.9995 |
| Gradient accumulation | 2 steps |
| Early stopping patience | 30 epochs |

## Citation

If you use this work, please cite:

> Velladurai, D.W., Venkatesan, R., Debnath, S. et al. Hybrid Mamba and UNet plus plus framework for robust diabetic foot ulcer segmentation. *Discover Applied Sciences* **8**, 1135 (2026). https://doi.org/10.1007/s42452-026-09128-5

```bibtex
@article{velladurai2026hybrid,
  title   = {Hybrid Mamba and UNet plus plus framework for robust diabetic foot ulcer segmentation},
  author  = {Velladurai, D. W. and Venkatesan, R. and Debnath, S. and others},
  journal = {Discover Applied Sciences},
  volume  = {8},
  pages   = {1135},
  year    = {2026},
  doi     = {10.1007/s42452-026-09128-5}
}
```


## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
