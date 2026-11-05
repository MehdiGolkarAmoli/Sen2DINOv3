<div align="center">

# 🛰️ Sen2DINOv3

### Global building-footprint mapping from 10 m Sentinel-2 imagery

**A multispectral DINOv3 adaptation with LoRA and UPerNet**  
Research code · S12OpenBuilding benchmark · Interactive geospatial demo

[![Explore live demo](https://img.shields.io/badge/Explore-Live%20Demo-16803c?style=for-the-badge)](https://01a1a747-710b-3809-a63e-588361d71177.share.connect.posit.cloud/)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c?style=for-the-badge)
![Satellite imagery](https://img.shields.io/badge/Imagery-Sentinel--2-2563eb?style=for-the-badge)

[Overview](#overview) · [Results](#results) · [Models](#model-zoo) · [Quick start](#quick-start) · [Demo](#interactive-demo)

</div>

<p align="center">
  <a href="assets/figure17_qualitative_results.webp">
    <img src="assets/figure17_qualitative_results.webp" alt="Building-mask predictions compared with high-resolution reference imagery" width="94%">
  </a>
  <br><sub>Qualitative building-footprint predictions from the study. Select the image to enlarge.</sub>
</p>

## Overview

**Sen2DINOv3** adapts a satellite-pretrained DINOv3 vision transformer to **12-band Sentinel-2** imagery for binary building-footprint segmentation. A learned spectral adapter bridges the multispectral input, LoRA makes encoder adaptation parameter-efficient, and a UPerNet decoder produces spatial masks.

**Reported in the manuscript:** **80.37% Dice** · **67.15% IoU** · **84.47% recall** · approximately **14M trainable parameters**.

`12 Sentinel-2 bands` → `Spectral adapter (12 → 36 → 3)` → `DINOv3 + LoRA` → `UPerNet` → `Building mask`

### S12OpenBuilding dataset

The companion dataset is described as **111,709 image patches** across **1,492 regions on four continents**. Each patch is 224 × 224 pixels, with 12 Sentinel-2 bands represented at 10 m resolution. Building labels are based on confidence-filtered Google Open Buildings V3 footprints; the study describes a region-level 80:20 train/test split.

<p align="center">
  <a href="assets/figure3_dataset_distribution.webp">
    <img src="assets/figure3_dataset_distribution.webp" alt="Geographic coverage of the S12OpenBuilding dataset" width="86%">
  </a>
  <br><sub>Geographic coverage and sample regions from S12OpenBuilding.</sub>
</p>

## Results

| Architecture | Dice ↑ | IoU ↑ | Recall ↑ |
|:--|--:|--:|--:|
| **Sen2DINOv3 (ours)** | **80.37%** | **67.15%** | **84.47%** |
| Panopticon-LoRA | 77.72% | 63.54% | 82.04% |
| DOFA-LoRA | 77.31% | 63.02% | 81.39% |
| U-Net++ | 77.05% | 64.29% | 80.02% |
| DINOv3-LoRA | 75.12% | 60.16% | 80.27% |
| MAE-SSL4EO-LoRA | 73.05% | 57.58% | 77.60% |
| SegFormer | 72.65% | 57.35% | 80.46% |
| DeepLabV3+ | 61.48% | 44.59% | 62.38% |

<sub>Figures reproduce selected **manuscript-reported** results, not results rerun from this repository. The study does not supply repeated-trial statistics here; interpret model differences accordingly.</sub>

## Model zoo

This release includes the supplied **architecture implementations and training utilities**—not pretrained model checkpoint files.

| Family | Implementation |
|:--|:--|
| **Sen2DINOv3 / DINOv3-LoRA** | [`Sen2DINOv3.py`](models/Sen2DINOv3.py) |
| **DOFA-LoRA** | [`DOFA.py`](models/DOFA.py) |
| **Panopticon-LoRA** | [`Panopticon.py`](models/Panopticon.py) |
| **MAE-SSL4EO / MAE-SSL4EO-LoRA** | [MAE](models/MAE_SSL4EO.py) · [MAE + LoRA](models/MAE_SSL4EO_LORA.py) |
| **DeepLabV3+ / SegFormer / U-Net++** | [DeepLabV3+](models/DeepLabV3Plus.py) · [SegFormer](models/SegFormer.py) · [U-Net++](models/UNetPlusPlus.py) |

The `models/` package also contains shared [spectral adapters](models/adapters.py), [UPerNet components](models/upernet.py), and [Sentinel-2 wavelength metadata](models/wavelengths.py). Nine experiment presets—including the separate 224-pixel DINOv3-LoRA preset—are defined in [`configs.py`](configs.py).

<details>
<summary><strong>Repository layout</strong></summary>

~~~text
Sen2DINOv3/
├── models/              # Architectures, adapters, UPerNet, wavelengths
├── assets/              # Figures and demo screenshot
├── configs.py           # Experiment presets and hyperparameters
├── experiments.py       # Backbone loading and optimizer setup
├── train.py             # Training and validation loops
├── losses.py            # Dice + binary cross-entropy
├── early_stopping.py    # MAE early-stopping utility
├── requirements.txt     # Python dependencies
├── .editorconfig        # Editor formatting conventions
├── .gitignore           # Local outputs excluded from Git
└── README.md
~~~

</details>

## Quick start

**1 · Install** (Python 3.11+ recommended)

~~~bash
git clone https://github.com/MehdiGolkarAmoli/Sen2DINOv3.git
cd Sen2DINOv3
pip install -r requirements.txt
~~~

**2 · Choose an experiment and train** using your own Sentinel-2 dataloaders:

~~~python
from configs import EXPERIMENTS
from experiments import build_model, configure_training
from train import train_model

name = "Sen2DINOv3"
config = EXPERIMENTS[name]
model = build_model(name)
training = configure_training(model, config)

# Supply train_loader and val_loader from your dataset pipeline.
history = train_model(
    model, train_loader, val_loader,
    checkpoint_dir="checkpoints/Sen2DINOv3",
    config=config, **training,
)
~~~

**Input contract:** images should be `[B, 12, H, W]`; binary masks should be `[B, 1, 224, 224]` for the documented experiments. The configuration's `input_size` describes the image size expected from your loader; training aligns prediction logits to the target mask grid. The Sen2DINOv3 preset uses 512-pixel imagery, while the DINOv3-LoRA preset uses 224-pixel imagery.

> **Before running:** This code does not include a dataset download/preprocessing pipeline, geographic split files, or trained segmentation checkpoints. Backbone weights are loaded via Transformers/TorchGeo and may require network access and appropriate model permissions. The dependency list is not a fully locked CUDA environment.

## Interactive demo

Explore detections by selecting a location and imagery date in the companion app.

<p align="center">
  <a href="https://01a1a747-710b-3809-a63e-588361d71177.share.connect.posit.cloud/">
    <img src="assets/Demo.png" alt="Interactive Sen2DINOv3 map-based demo" width="90%">
  </a>
  <br><a href="https://01a1a747-710b-3809-a63e-588361d71177.share.connect.posit.cloud/"><strong>Launch interactive demo</strong></a>
</p>

## Research & attribution

**S12OpenBuilding: Global Building Footprint Mapping via Sen2DINOv3, a Parameter-Efficient Adaptation of DINOv3 to Sentinel-2 Imagery**

Mehdi Golkar Amoli · Mahdi Hasanlou · Mahdi Panahi · Saeid Homayouni

This repository shares the supplied research implementations for inspection and adaptation. The reported manuscript metrics have **not been independently reproduced** with the files included here. In particular, data preparation and original downstream checkpoint weights are not provided; pretrained backbone weights are separate downloads.

