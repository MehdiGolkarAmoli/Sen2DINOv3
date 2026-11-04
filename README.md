# Sen2DINOv3

**Global building-footprint mapping from 10 m Sentinel-2 imagery using a parameter-efficient adaptation of DINOv3.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open%20App-2ea44f?style=for-the-badge)](https://01a1a747-710b-3809-a63e-588361d71177.share.connect.posit.cloud/)

<p align="center">
  <a href="assets/figure3_dataset_distribution.webp">
    <img src="assets/figure3_dataset_distribution.webp" alt="Global distribution and regional examples of S12OpenBuilding" width="92%">
  </a>
</p>

<p align="center"><sub>Click the figure to open the full-resolution image.</sub></p>

<p align="center"><em>Figure 3. Global distribution and regional detail of the S12OpenBuilding dataset.</em></p>

## Project summary

This repository accompanies the research article **“S12OpenBuilding: Global Building Footprint Mapping via Sen2DINOv3, a Parameter-Efficient Adaptation of DINOv3 to Sentinel-2 Imagery.”** It presents a dataset and a deep-learning framework for detecting building footprints directly from medium-resolution Sentinel-2 satellite images.

The work addresses two closely related challenges: the lack of geographically diverse building labels designed for Sentinel-2, and the difficulty of transferring large vision foundation models from RGB imagery to multispectral satellite data.

### Main contributions

- **S12OpenBuilding dataset:** 111,709 image patches from 1,492 geographic regions across four continents, with corresponding building masks aligned to the Sentinel-2 10 m grid.
- **Sen2DINOv3 framework:** a parameter-efficient adaptation of the DINOv3-sat493M satellite vision encoder to 12-band Sentinel-2 imagery.
- **Global demonstration:** an interactive app that lets users explore the building-detection capability of Sen2DINOv3 in different parts of the world.

## S12OpenBuilding dataset

S12OpenBuilding pairs Sentinel-2 Level-2A imagery with building footprints derived from **Google Open Buildings V3**. The source footprints are confidence-filtered, converted to binary masks, and downsampled to the 10 m Sentinel-2 grid.

| Dataset property | Description |
|---|---|
| Image patches | **111,709** |
| Geographic coverage | **1,492 regions**, four continents |
| Patch dimensions | 224 × 224 pixels |
| Input channels | 12 Sentinel-2 bands (Band 10 excluded) |
| Spatial resolution | 10 m |
| Labels | Binary building / non-building masks |
| Label source | Google Open Buildings V3 |

The article uses a region-level 80:20 training/test partition, with no geographic region shared between the two subsets. As the original footprints are downsampled to 10 m, very small buildings may not remain represented in the resulting masks.

**Upstream label source:** [Google Open Buildings V3](https://sites.research.google/gr/open-buildings/) · [Earth Engine data catalog](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_Research_open-buildings_v3_polygons)

*The links above refer to the upstream footprint source. S12OpenBuilding is the dataset introduced in the article; its download link will be added when the dataset is released.*

## Sen2DINOv3 at a glance

Sen2DINOv3 adapts the RGB-pretrained DINOv3-sat493M encoder to the 12-band Sentinel-2 input domain. Its main components are:

1. **Learned spectral adaptation:** adaptive spectral reweighting followed by nonlinear cross-band projection maps the 12 Sentinel-2 bands to the three-channel representation expected by DINOv3.
2. **Parameter-efficient encoder adaptation:** LoRA is applied to selected transformer projections, allowing the pretrained representation to adapt without full encoder fine-tuning.
3. **Dense prediction:** a UPerNet-based decoder aligns selected DINOv3 token features and generates the building segmentation mask.

This design combines a learned multispectral input interface with limited adaptation of the pretrained encoder. The final reported configuration uses approximately **14 million trainable parameters**.

## Quantitative results

The following table reproduces **Table 3** from the article. Dice, IoU, and Recall are reported in percent; trainable parameters are reported in millions.

| Model | Dice (%) | IoU (%) | Recall (%) | Trainable params (M) |
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

**Best reported result:** Sen2DINOv3 achieves **80.37% Dice, 67.15% IoU, and 84.47% Recall**. Its Dice score is 3.32 percentage points higher than the U-Net++ baseline, while the reported number of trainable parameters is 14M compared with 49M for U-Net++.

*The manuscript reports single-run model comparisons; repeated-trial statistical analysis is not provided.*

## Qualitative examples

Figure 17 compares Sen2DINOv3 building-mask predictions with Google Earth very-high-resolution imagery across suburban, urban, and desert/industrial environments. This is a visual comparison rather than an additional quantitative evaluation.

<p align="center">
  <a href="assets/figure17_qualitative_results.webp">
    <img src="assets/figure17_qualitative_results.webp" alt="Sen2DINOv3 building predictions compared with Google Earth VHR imagery" width="96%">
  </a>
</p>

<p align="center"><sub>Click the figure to open the full-resolution image.</sub></p>

<p align="center"><em>Figure 17. Detailed visual comparison of building masks generated by Sen2DINOv3 with Google Earth VHR imagery across representative suburban, urban, and desert/industrial regions.</em></p>

The examples illustrate the model's ability to identify both dispersed buildings and broader urban patterns. The article also notes limitations associated with the native 10 m resolution, including mixed pixels, uncertain boundaries in dense settlements, and possible false detections over bright soil or artificial surfaces.

## Live demo

Explore the interactive Sen2DINOv3 building-detection demonstration:

<p align="center">
  <a href="https://01a1a747-710b-3809-a63e-588361d71177.share.connect.posit.cloud/"><strong>Launch the Sen2DINOv3 Demo</strong></a>
</p>

The demo provides an interactive showcase of the model's building-detection capability over different geographic areas.

## Code, weights, and data availability

This repository currently provides the project overview, manuscript figures, reported evaluation results, and the live demo link. Training/inference code, pretrained weights, and S12OpenBuilding download instructions will be added here when they are ready for public release.

## Article

**S12OpenBuilding: Global Building Footprint Mapping via Sen2DINOv3, a Parameter-Efficient Adaptation of DINOv3 to Sentinel-2 Imagery**

**Authors:** Mehdi Golkar Amoli, Mahdi Hasanlou, Mahdi Panahi, and Saeid Homayouni.

---

**Repository:** [MehdiGolkarAmoli/Sen2DINOv3](https://github.com/MehdiGolkarAmoli/Sen2DINOv3) · **Demo:** [Open the app](https://01a1a747-710b-3809-a63e-588361d71177.share.connect.posit.cloud/)
