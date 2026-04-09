import os
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import astra
from scipy.ndimage import gaussian_filter
import pandas as pd

# ------------------------------
# Configuration
scan_data_dir = 'ScanData'
scan_ref_dir = 'ScanRef'
calib_path = 'calibration.xml'
info_path = 'info.xml'
header_bytes = 54

# BMP image dimensions
bmp_height = 480  # number of rows
bmp_width = 640  # number of detector channels


# ------------------------------
def parse_xml(calib_file, info_file):
    """Parse XML and return geometric parameters."""
    calib = ET.parse(calib_file).getroot()
    info = ET.parse(info_file).getroot()

    # distances in mm
    d_sod_mm = float(calib.find('SourceToAxis').text)
    d_odd_mm = float(calib.find('AxisToDetector').text)
    px_sz_mm = float(calib.find('HorizPixelSize').text)
    axis_off_mm = float(calib.find('AxisOfRotationOffset').text)
    num_proj = int(info.find('DataProjections').text)

    # convert axis offset to pixels
    axis_off_px = axis_off_mm / px_sz_mm

    return d_sod_mm, d_odd_mm, px_sz_mm, axis_off_px, num_proj


def load_projection_line(fp):
    """Load a 16-bit BMP, reshape to (480,640), return its central row."""
    with open(fp, 'rb') as f:
        buf = f.read()
    arr = np.frombuffer(buf, dtype=np.uint16, offset=header_bytes)
    arr = arr[:bmp_height * bmp_width]
    img = arr.reshape((bmp_height, bmp_width))
    return img[bmp_height // 2, :].astype(np.float32)


def calculate_snr_and_cnr(image, roi_coords, reference_roi):
    """Calculate SNR and CNR for given ROIs."""
    snr_values = []
    cnr_values = []
    roi_means = []
    roi_stds = []

    # Ensure ROIs are within image bounds
    height, width = image.shape
    ref_x, ref_y, ref_w, ref_h = reference_roi
    ref_x = max(0, min(ref_x, width - 1))
    ref_y = max(0, min(ref_y, height - 1))
    ref_w = min(ref_w, width - ref_x)
    ref_h = min(ref_h, height - ref_y)

    if ref_w <= 0 or ref_h <= 0:
        raise ValueError(f"Reference ROI dimensions invalid: w={ref_w}, h={ref_h}")

    # Extract reference ROI (background)
    ref_roi = image[ref_y:ref_y + ref_h, ref_x:ref_x + ref_w]
    mu_b = np.mean(ref_roi)
    sigma_b = np.std(ref_roi)

    print(f"Reference ROI ({ref_x},{ref_y},{ref_w},{ref_h}): mean = {mu_b:.4f}, std = {sigma_b:.4f}")

    for i, roi in enumerate(roi_coords):
        x, y, w, h = roi
        # Ensure ROI is within image bounds
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = min(w, width - x)
        h = min(h, height - y)

        if w <= 0 or h <= 0:
            print(f"Warning: ROI {i + 1} dimensions invalid, skipping")
            snr_values.append(float('nan'))
            cnr_values.append(float('nan'))
            roi_means.append(float('nan'))
            roi_stds.append(float('nan'))
            continue

        roi_data = image[y:y + h, x:x + w]
        mu_o = np.mean(roi_data)
        sigma_o = np.std(roi_data)

        roi_means.append(mu_o)
        roi_stds.append(sigma_o)

        print(f"ROI {i + 1} ({x},{y},{w},{h}): mean = {mu_o:.4f}, std = {sigma_o:.4f}")

        # Calculate SNR
        snr = mu_o / sigma_o if sigma_o > 0 else float('inf')
        snr_values.append(snr)

        # Calculate CNR
        cnr = abs(mu_o - mu_b) / sigma_b if sigma_b > 0 else float('inf')
        cnr_values.append(cnr)

    return snr_values, cnr_values, roi_means, roi_stds


def get_centered_roi(image, center_x, center_y, size=30):
    """Get ROI coordinates centered at a point"""
    h, w = image.shape
    half_size = size // 2
    x = max(0, min(center_x - half_size, w - size))
    y = max(0, min(center_y - half_size, h - size))
    return (x, y, size, size)


def main():
    # 1. Parse geometry
    d_sod_mm, d_odd_mm, px_sz_mm, axis_off_px, num_proj = parse_xml(calib_path, info_path)
    print(
        f"Projections={num_proj}, SOD={d_sod_mm}mm, ODD={d_odd_mm}mm, px_sz={px_sz_mm}mm, axis_off={axis_off_px:.1f}px")

    # 2. Build flat-field line
    ref_files = sorted([f for f in os.listdir(scan_ref_dir) if f.lower().endswith('.bmp')])[:num_proj]
    flats = [load_projection_line(os.path.join(scan_ref_dir, f)) for f in ref_files]
    flat_line = np.mean(flats, axis=0)
    flat_line = np.clip(flat_line, 1e-6, None)

    # 3. Load projections, correct and log-transform → sinogram
    data_files = sorted([f for f in os.listdir(scan_data_dir) if f.lower().endswith('.bmp')])[:num_proj]
    sinogram = np.zeros((num_proj, bmp_width), dtype=np.float32)
    for i, fname in enumerate(data_files):
        proj = load_projection_line(os.path.join(scan_data_dir, fname))
        proj = np.clip(proj, 1e-6, None)
        corr = proj / flat_line
        corr = np.clip(corr, 1e-8, None)
        sinogram[i] = -np.log(corr)

    # visualize sinogram
    plt.figure(figsize=(8, 6))
    plt.imshow(sinogram, cmap='gray', aspect='auto')
    plt.title('Sinogram (Full-scan Fan-beam)')
    plt.xlabel('Detector pixel')
    plt.ylabel('Projection index')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('sinogram_fullscan.png', dpi=300)

    # 4. Set up ASTRA full-scan fan-beam geometry
    angles = np.linspace(0, 2 * np.pi, num_proj, endpoint=False)
    proj_geom = astra.create_proj_geom('fanflat',
                                       px_sz_mm,  # detector spacing [mm]
                                       bmp_width,
                                       angles,
                                       d_sod_mm,
                                       d_odd_mm)

    # 5. Create volume geometry with physical extents in mm
    xmin = -0.5 * bmp_width * px_sz_mm
    xmax = 0.5 * bmp_width * px_sz_mm
    ymin = -0.5 * bmp_height * px_sz_mm
    ymax = 0.5 * bmp_height * px_sz_mm
    vol_geom = astra.create_vol_geom(bmp_height, bmp_width,
                                     xmin, xmax,
                                     ymin, ymax)

    # 6. Create ASTRA data and run FBP
    sino_id = astra.data2d.create('-sino', proj_geom, sinogram)
    rec_id = astra.data2d.create('-vol', vol_geom, 0)

    try:
        cfg = astra.astra_dict('FBP_CUDA')
        print("Using GPU FBP_CUDA")
    except:
        cfg = astra.astra_dict('FBP')
        print("Using CPU FBP")

    cfg.update(ProjectionDataId=sino_id,
               ReconstructionDataId=rec_id,
               option={'FilterType': 'Hamming',
                       'ParkerWeighting': False})

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    rec = astra.data2d.get(rec_id)

    # 7. Optional denoising
    denoised = gaussian_filter(rec, sigma=1.0)

    # 8. Display reconstructions
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(rec, cmap='gray', vmin=0, vmax=np.percentile(rec, 99))
    axes[0].set_title('Raw Reconstruction')
    axes[0].axis('off')
    axes[1].imshow(denoised, cmap='gray', vmin=0, vmax=np.percentile(denoised, 99))
    axes[1].set_title('Denoised (σ=1)')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig('reconstruction_fullscan.png', dpi=300)

    # 打印图像尺寸，便于设置ROI
    print(f"Image shape: {denoised.shape}")

    # 9. Define ROIs based on image center
    height, width = denoised.shape
    center_x, center_y = width // 2, height // 2

    # 选择合适的ROI位置（围绕图像中心）
    roi_size = 80
    roi_coords = [
        # 白色样本ROI (根据实际图像调整坐标)
        get_centered_roi(denoised, center_x - width//20, center_y + height*9//40, roi_size),
        get_centered_roi(denoised, center_x - width//15, center_y - height*9//40, roi_size),
        get_centered_roi(denoised, center_x + width*9.5//60, center_y + height //7, roi_size),
        get_centered_roi(denoised, center_x - width*11//60, center_y + height//100, roi_size),
        get_centered_roi(denoised, center_x + width*9//60, center_y - height*9//60, roi_size),
        ]

    # 背景ROI选择在没有样本的区域
    reference_roi = get_centered_roi(denoised, center_x, center_y, roi_size)

    # 10. Calculate SNR and CNR with clear error reporting
    print("\n===== SNR & CNR Calculation =====")
    try:
        snr_values, cnr_values, roi_means, roi_stds = calculate_snr_and_cnr(denoised, roi_coords, reference_roi)

        # 直接打印关键结果，确保值显示出来
        print("\n----- Raw SNR & CNR Values -----")
        for i in range(len(roi_coords)):
            print(f"ROI {i + 1}: SNR = {snr_values[i]:.4f}, CNR = {cnr_values[i]:.4f}")

        # 11. Print results in table format
        print("\n===== Results Table =====")
        results = pd.DataFrame({
            'ROI': [f'ROI {i + 1}' for i in range(len(roi_coords))],
            'Mean': roi_means,
            'Std Dev': roi_stds,
            'SNR': snr_values,
            'CNR': cnr_values
        })
        print(results)

        # 保存结果到CSV
        results.to_csv('snr_cnr_results.csv', index=False)
        print("Saved results to snr_cnr_results.csv")

        # 12. Display results with ROIs
        plt.figure(figsize=(10, 8))
        plt.imshow(denoised, cmap='gray', vmin=0, vmax=np.percentile(denoised, 99))

        # 添加ROI标记
        for i, (x, y, w, h) in enumerate(roi_coords):
            plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor='red', facecolor='none', lw=2))
            plt.text(x, y - 5, f"ROI {i + 1}\nSNR={snr_values[i]:.2f}\nCNR={cnr_values[i]:.2f}",
                     color='red', fontsize=9, verticalalignment='bottom')

        # 添加参考ROI标记
        ref_x, ref_y, ref_w, ref_h = reference_roi
        plt.gca().add_patch(plt.Rectangle((ref_x, ref_y), ref_w, ref_h, edgecolor='blue', facecolor='none', lw=2))
        plt.text(ref_x, ref_y - 5, "Reference ROI", color='blue', fontsize=9, verticalalignment='bottom')

        plt.title("ROIs with SNR & CNR Values")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('rois_snr_cnr.png', dpi=300)

    except Exception as e:
        import traceback
        print(f"Error calculating SNR/CNR: {e}")
        traceback.print_exc()

        # 尝试使用更简单的方法计算和显示
        print("\n----- Attempting simplified SNR & CNR calculation -----")
        # 使用简化的计算方法
        try:
            # 重新定义一个更简单的ROI中心位置
            simple_roi = [(width // 4, height // 4, 20, 20)]  # 简单的ROI位置
            simple_ref = (3 * width // 4, 3 * height // 4, 20, 20)  # 简单的参考ROI

            # 提取ROI区域
            roi_data = denoised[height // 4:height // 4 + 20, width // 4:width // 4 + 20]
            ref_data = denoised[3 * height // 4:3 * height // 4 + 20, 3 * width // 4:3 * width // 4 + 20]

            # 计算
            roi_mean = np.mean(roi_data)
            roi_std = np.std(roi_data)
            ref_mean = np.mean(ref_data)
            ref_std = np.std(ref_data)

            # SNR和CNR
            simple_snr = roi_mean / roi_std if roi_std > 0 else float('inf')
            simple_cnr = abs(roi_mean - ref_mean) / ref_std if ref_std > 0 else float('inf')

            print(f"Simple ROI: mean = {roi_mean:.4f}, std = {roi_std:.4f}")
            print(f"Simple Ref: mean = {ref_mean:.4f}, std = {ref_std:.4f}")
            print(f"Simple SNR = {simple_snr:.4f}, CNR = {simple_cnr:.4f}")

            # 确保这些值在命令行中打印出来
            import sys
            sys.stdout.flush()
        except Exception as e2:
            print(f"Simplified calculation also failed: {e2}")

    # 13. Save and clean up
    np.save('reconstruction_fullscan.npy', rec)
    print("Saved reconstruction_fullscan.npy and .png outputs")

    # 这里再打印一次SNR/CNR结果以确保它们显示出来
    try:
        print("\n===== FINAL SNR & CNR VALUES =====")
        for i in range(len(roi_coords)):
            print(f"ROI {i + 1}: SNR = {snr_values[i]:.4f}, CNR = {cnr_values[i]:.4f}")
    except:
        print("Could not print final SNR & CNR values")

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(sino_id)
    astra.data2d.delete(rec_id)


if __name__ == '__main__':
    main()