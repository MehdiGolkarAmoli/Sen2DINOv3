# Sen2DINOv3

**S12OpenBuilding: Global Building Footprint Mapping via Sen2DINOv3, a Parameter-Efficient Adaptation of DINOv3 to Sentinel-2 Imagery**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open%20App-2ea44f?style=for-the-badge)](https://01a1a747-710b-3809-a63e-588361d71177.share.connect.posit.cloud/)

<p align="center">
  <img src="assets/figure3_dataset_distribution.jpg" alt="Global distribution of the S12OpenBuilding dataset and regional examples" width="90%">
</p>

<p align="center"><em>Figure 3. Global distribution and regional detail of the S12OpenBuilding dataset.</em></p>

---

## Overview

This repository accompanies the research article **“S12OpenBuilding: Global Building Footprint Mapping via Sen2DINOv3, a Parameter-Efficient Adaptation of DINOv3 to Sentinel-2 Imagery.”** The study investigates building-footprint segmentation directly from freely available, medium-resolution Sentinel-2 imagery and introduces two complementary contributions:

- **S12OpenBuilding:** a geographically diverse dataset containing **111,709 Sentinel-2 image patches** and corresponding binary building masks, collected from **1,492 geographic regions across four continents**.
- **Sen2DINOv3:** a parameter-efficient framework that adapts the satellite vision foundation model **DINOv3-sat493M** to 12-band Sentinel-2 imagery for building segmentation.

The aim is to improve large-scale building mapping using Sentinel-2's 10 m imagery, making building information more accessible for applications such as urban analysis, land-use monitoring, and disaster management.

## S12OpenBuilding dataset

S12OpenBuilding was constructed by pairing Sentinel-2 Level-2A imagery with building-footprint labels derived from **Google Open Buildings V3**. The source footprints were confidence-filtered, converted to binary masks, and downsampled to the Sentinel-2 10 m grid. The resulting samples have the following characteristics:

| Characteristic | Description |
|---|---|
| Number of samples | 111,709 |
| Geographic coverage | 1,492 regions across four continents |
| Image size | 224 × 224 pixels |
| Input data | 12 Sentinel-2 spectral bands (Band 10 excluded) |
| Target resolution | 10 m |
| Labels | Binary building / non-building masks |
| Source footprint data | Google Open Buildings V3 |

The dataset partition described in the article is geographically separated at the region level, so patches from a region are not shared between the training and test subsets.

**Upstream label source:** [Google Open Buildings V3 – Google Research](https://sites.research.google/gr/open-buildings/) · [Earth Engine dataset catalog](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_Research_open-buildings_v3_polygons)

> **Dataset availability:** S12OpenBuilding is the dataset introduced in the accompanying article. Its download instructions and hosting link will be added here when the dataset release is available. The Google Open Buildings links above refer to the upstream source data, not to a download of S12OpenBuilding.

## Sen2DINOv3 framework

Sen2DINOv3 is designed to address the spectral-domain gap between 12-band Sentinel-2 imagery and the three-channel input expected by the pretrained DINOv3 encoder. The framework consists of three main stages:

1. **Spectral-adaptation interface.** A learned interface adaptively reweights the Sentinel-2 spectral bands and then applies a nonlinear cross-band projection to produce a three-channel representation for the encoder.
2. **DINOv3-sat493M with LoRA.** The pretrained DINOv3 encoder extracts visual representations. Low-Rank Adaptation (LoRA) is applied to selected transformer projections to allow parameter-efficient task adaptation without full encoder fine-tuning.
3. **UPerNet-based decoder.** Selected transformer token features are reshaped and aligned for multi-scale dense prediction, producing the final building mask.

The complete configuration reported in the article updates approximately **14 million trainable parameters**.

## Quantitative results

Table 3 reproduces the segmentation results reported in the article for the S12OpenBuilding evaluation. Dice, IoU, and Recall are reported as percentages; trainable parameters are reported in millions.

| Model | Dice (%) | IoU (%) | Recall (%) | Trainable parameters (M) |
|---|---:|---:|---:|---:|
| U-Net++ | 77.05 | 64.29 | 80.02 | 49 |
| DeepLabv3 | 51.41 | 35.02 | 51.99 | 40 |
| DeepLabv3+ | 61.48 | 44.59 | 62.38 | 27 |
| PSPNet | 74.76 | 59.68 | 80.24 | 25 |
| SegFormer | 72.65 | 57.35 | 80.46 | 25 |
| Mask2Former | 72.49 | 56.88 | 80.20 | 216 |
| MAE-SSL4EO | 70.11 | 54.12 | 70.28 | 10 |
| MAE-SSL4EO-LoRA | 73.05 | 57.58 | 77.60 | 12 |
| Panopticon-LoRA | 77.72 | 63.54 | 82.04 | 10 |
| DOFA-LoRA | 77.31 | 63.02 | 81.39 | 14 |
| DINOv3-Frozen | 62.08 | 45.13 | 65.98 | 10 |
| DINOv3-LoRA | 75.12 | 60.16 | 80.27 | 14 |
| **Sen2DINOv3 (ours)** | **80.37** | **67.15** | **84.47** | **14** |

**Key result:** Sen2DINOv3 achieves the highest reported scores among the evaluated configurations: **80.37% Dice, 67.15% IoU, and 84.47% Recall**. It exceeds the U-Net++ baseline by 3.32 percentage points in Dice while using 14M trainable parameters compared with 49M for U-Net++.

*Note: These values reproduce Table 3 of the manuscript. The reported model comparisons are single-run evaluations; repeated-trial statistical analysis is not reported in the article.*

## Qualitative examples

The following figure presents detailed visual comparisons of Sen2DINOv3 building-mask predictions with Google Earth very-high-resolution (VHR) imagery across representative suburban, urban, and desert/industrial settings.

<p align="center">
  <img src="assets/figure17_qualitative_results.jpg" alt="Qualitative comparison of Sen2DINOv3 building predictions with Google Earth VHR imagery" width="95%">
</p>

<p align="center"><em>Figure 17. Detailed visual comparison of building masks generated by Sen2DINOv3 with Google Earth VHR imagery across representative suburban, urban, and desert/industrial regions.</em></p>

This figure provides a qualitative visual assessment; it should not be interpreted as an additional quantitative accuracy evaluation. The article also discusses remaining challenges, including mixed pixels and uncertain building boundaries in dense urban areas, omission of very small buildings after mask downsampling, and possible false detections over bright soil or artificial surfaces.

## Live demo

Explore the worldwide building-detection demonstration powered by Sen2DINOv3:

**[Launch the Sen2DINOv3 Demo](https://01a1a747-710b-3809-a63e-588361d71177.share.connect.posit.cloud/)**

The demo is provided as an interactive showcase of the model's building-detection capability. Availability of the demo may depend on the hosting service.

## Repository status and planned releases

This repository is the companion repository for the research article. At this stage, it provides the project introduction, manuscript figures, reported evaluation results, and the live demonstration link.

The following materials will be added as they are prepared for release:

- Training and inference source code
- Pretrained model weights and setup instructions
- S12OpenBuilding dataset access and download instructions

Until those materials are released, please use the live demo to explore the model.

## Article

**Title:** S12OpenBuilding: Global Building Footprint Mapping via Sen2DINOv3, a Parameter-Efficient Adaptation of DINOv3 to Sentinel-2 Imagery  
**Authors:** Mehdi Golkar Amoli, Mahdi Hasanlou, Mahdi Panahi, and Saeid Homayouni

This repository accompanies the article and will be updated with code and data availability information.

---

**Project repository:** [MehdiGolkarAmoli/Sen2DINOv3](https://github.com/MehdiGolkarAmoli/Sen2DINOv3)  
**Live demo:** [Open Sen2DINOv3](https://01a1a747-710b-3809-a63e-588361d71177.share.connect.posit.cloud/)
