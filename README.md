# CT Image Quality Assessment: SNR & CNR Quantitative Analysis

A Python-based CT image quality evaluation project using phantom experiments, focusing on **Signal-to-Noise Ratio (SNR)** and **Contrast-to-Noise Ratio (CNR)** quantitative analysis with fan-beam FBP reconstruction.

## Overview
This project implements a complete CT imaging quality assessment pipeline:
- Cone-shaped finger phantom scan data processing
- Flat-field correction & sinogram generation
- Fan-beam FBP reconstruction (ASTRA Toolkit)
- ROI-based SNR / CNR automatic calculation
- Radiation dose vs image quality relationship analysis

## Key Metrics
### SNR
\[SNR=\frac{\mu_{o}}{\sigma_{o}}\]
### CNR
\[CNR=\frac{\mu_{o}-\mu_{B}}{\sqrt{\sigma_{o}^{2}+\sigma_{B}^{2}}}\]
- \(\mu_o\): Mean attenuation of object ROI
- \(\sigma_o\): Standard deviation of object ROI
- \(\mu_B\): Mean attenuation of background ROI
- \(\sigma_B\): Standard deviation of background ROI

## Pipeline
1. Data acquisition (DeskCAT scanner)
2. Geometry calibration & projection loading
3. Flat-field correction & log transformation
4. Sinogram construction
5. FBP reconstruction (Hamming filter)
6. ROI selection & statistical calculation
7. SNR / CNR output & visualization

## Requirements
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
