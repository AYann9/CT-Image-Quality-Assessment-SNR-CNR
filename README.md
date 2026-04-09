# CT Image Quality Assessment: SNR & CNR Quantitative Analysis

A Python-based CT image quality evaluation project using phantom experiments, focusing on **Signal-to-Noise Ratio (SNR)** and **Contrast-to-Noise Ratio (CNR)** quantitative analysis with fan-beam FBP reconstruction.

## 📌 Project Overview
This project implements a complete pipeline for optical CT image reconstruction and quality assessment:
- Cone-shaped finger phantom scan data processing
- Flat-field correction & sinogram generation
- Fan-beam FBP reconstruction (ASTRA Toolkit)
- ROI-based automatic SNR / CNR calculation
- Analysis of radiation dose vs image quality relationship

## 🧮 Key Metrics
### SNR
\[SNR=\frac{\mu_{o}}{\sigma_{o}}\]
### CNR
\[CNR=\frac{\mu_{o}-\mu_{B}}{\sqrt{\sigma_{o}^{2}+\sigma_{B}^{2}}}\]
- \(\mu_o\): Mean attenuation of object ROI
- \(\sigma_o\): Standard deviation of object ROI
- \(\mu_B\): Mean attenuation of background ROI
- \(\sigma_B\): Standard deviation of background ROI

## 📁 Dataset Notice (Important)
**The original projection dataset is NOT included in this repository**  
Due to the large file size of raw projection images (hundreds of MB), the `ScanData/`, `ScanRef/`, `calibration.xml`, and `info.xml` files cannot be uploaded to GitHub.

To run this project, you need to prepare your own data with the same structure:
- `ScanData/`: Projection images with phantom
- `ScanRef/`: Reference background projections
- `calibration.xml`: Scanner geometry parameters
- `info.xml`: Projection count information

## 🧩 Pipeline
1. Geometry calibration & projection loading
2. Flat-field correction & log transformation
3. Sinogram construction
4. FBP reconstruction (Hamming filter)
5. ROI selection & statistical analysis
6. SNR / CNR calculation & visualization

## 🛠️ Requirements
numpy
matplotlib
pandas
astra-toolbox
scipy


## Usage
1. Place projection data in `ScanData/` and `ScanRef/`
2. Prepare `calibration.xml` and `info.xml`
3. Run reconstruction & analysis:
```bash
python main.py
