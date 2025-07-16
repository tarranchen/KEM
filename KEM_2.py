import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
from scipy.fft import fft
import matplotlib.patches as patches
import random

# --- 參數設定 ---
FILENAME = '5.tif'
PIXEL_SIZE_UM = 1.3
OVERSAMPLING_FACTOR = 7  # 超解析度取樣倍率 (建議 4 或 8)
ROI_WIDTH = 50  # 設定感興趣區域(ROI)的寬度 (像素)


def ransac_line_fitting(x, y, n_iterations=100, distance_threshold=1.0):
    """
    使用 RANSAC 演算法來擬合一條直線，以抵抗離群點的影響。

    Args:
        x (np.array): x 座標點。
        y (np.array): y 座標點。
        n_iterations (int): 迭代次數。
        distance_threshold (float): 判斷一個點是否為內群點的距離閾值。

    Returns:
        tuple: (最佳斜率, 最佳截距, 內群點的索引)
    """
    best_inliers = []
    best_model = (0, 0)
    data = np.column_stack((x, y))

    for i in range(n_iterations):
        # 1. 隨機選取兩個點
        sample = data[random.sample(range(len(data)), 2)]
        p1, p2 = sample[0], sample[1]

        # 避免兩點重合或形成垂直線
        if p1[0] == p2[0]:
            continue

        # 2. 根據這兩點建立一條線 (y = mx + c)
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        c = p1[1] - m * p1[0]

        # 3. 計算所有點到這條線的垂直距離
        distances = np.abs(m * data[:, 0] - data[:, 1] + c) / np.sqrt(m ** 2 + 1)

        # 4. 找出內群點 (inliers)
        inliers_idx = np.where(distances < distance_threshold)[0]

        # 5. 如果這次的內群點數量比之前多，就更新最佳模型
        if len(inliers_idx) > len(best_inliers):
            best_inliers = inliers_idx
            best_model = (m, c)

    # 6. 使用所有找到的最佳內群點，進行一次最終的最小二乘法擬合
    if len(best_inliers) > 2:
        inlier_data = data[best_inliers]
        final_coeffs = np.polyfit(inlier_data[:, 0], inlier_data[:, 1], 1)
        best_model = (final_coeffs[0], final_coeffs[1])

    return best_model[0], best_model[1], best_inliers


def analyze_slanted_edge_resolution(filename, pixel_size):
    """
    使用斜邊法 (ISO 12233) 分析影像以計算 MTF 解析度。
    """
    try:
        with Image.open(filename) as img:
            image_data_original = np.array(img.convert('L'), dtype=float)
    except FileNotFoundError:
        print(f"Error: File not found at '{filename}'. Please place it in the same directory as the script.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    image_dims = image_data_original.shape  # (height, width)

    # --- 步驟 0.1: 建立僅供視覺化與偵測輔助用的增強影像 ---
    min_val = np.min(image_data_original)
    max_val = np.max(image_data_original)
    if max_val > min_val:
        image_data_normalized = (image_data_original - min_val) * (255.0 / (max_val - min_val))
    else:
        image_data_normalized = image_data_original.copy()

    p_low = np.percentile(image_data_normalized, 2)
    p_high = np.percentile(image_data_normalized, 98)
    image_high_contrast = np.clip(image_data_normalized, p_low, p_high)
    if p_high > p_low:
        image_high_contrast = (image_high_contrast - p_low) * (255.0 / (p_high - p_low))

    # --- 1. 自動偵測邊緣與角度 (使用高對比影像) ---
    dx = sobel(image_high_contrast, axis=1)
    gradient_mag = np.abs(dx)

    column_gradient_sum = np.sum(gradient_mag, axis=0)
    center_col = np.argmax(column_gradient_sum)

    roi_start_col = max(0, center_col - ROI_WIDTH // 2)
    roi_end_col = min(image_data_original.shape[1], center_col + ROI_WIDTH // 2)

    # 步驟 1.2: 在 ROI 內使用中位灰階值法精確定位邊緣點
    edge_points_y = []
    edge_points_x_roi = []

    roi_to_find_edge = image_data_normalized[:, roi_start_col:roi_end_col]
    for i in range(roi_to_find_edge.shape[0]):
        row_data = roi_to_find_edge[i, :]
        min_row_val, max_row_val = np.min(row_data), np.max(row_data)
        if max_row_val <= min_row_val: continue
        median_val = (min_row_val + max_row_val) / 2.0
        try:
            above_indices = np.where(row_data > median_val)[0]
            if not len(above_indices): continue
            idx2 = above_indices[0]
            if idx2 == 0: continue
            idx1 = idx2 - 1
            val1, val2 = row_data[idx1], row_data[idx2]
            if val2 <= val1: continue
            interp = (median_val - val1) / (val2 - val1)
            sub_pixel_x = idx1 + interp
            edge_points_y.append(i)
            edge_points_x_roi.append(sub_pixel_x)
        except IndexError:
            continue

    edge_points_y = np.array(edge_points_y)
    edge_points_x_roi = np.array(edge_points_x_roi)

    if len(edge_points_x_roi) < 10:
        print("Error: Could not detect a clear edge within the ROI.")
        return

    # 步驟 1.3: 使用 RANSAC 演算法進行穩健的線性擬合
    slope, intercept_roi, inlier_indices = ransac_line_fitting(edge_points_x_roi, edge_points_y)

    angle = np.rad2deg(np.arctan(slope))

    # --- 2. 超解析度取樣以建立 ESF (使用 **原始** 影像數據) ---
    image_roi_original = image_data_original[:, roi_start_col:roi_end_col]

    x_roi, y_roi = np.meshgrid(np.arange(image_roi_original.shape[1]), np.arange(image_roi_original.shape[0]))
    distances = (slope * x_roi - y_roi + intercept_roi) / np.sqrt(slope ** 2 + 1)

    distances_flat = distances.ravel()
    intensities_flat = image_roi_original.ravel()

    sort_indices = np.argsort(distances_flat)
    distances_sorted = distances_flat[sort_indices]
    intensities_sorted = intensities_flat[sort_indices]

    min_dist, max_dist = distances_sorted[0], distances_sorted[-1]
    num_bins = int((max_dist - min_dist) * OVERSAMPLING_FACTOR)
    bin_edges = np.linspace(min_dist, max_dist, num_bins + 1)

    sum_in_bin, _ = np.histogram(distances_sorted, bins=bin_edges, weights=intensities_sorted)
    count_in_bin, _ = np.histogram(distances_sorted, bins=bin_edges)

    non_empty_bins = count_in_bin > 0
    esf_oversampled = np.full_like(sum_in_bin, np.nan)
    esf_oversampled[non_empty_bins] = sum_in_bin[non_empty_bins] / count_in_bin[non_empty_bins]

    esf_oversampled = np.interp(np.arange(len(esf_oversampled)),
                                np.where(~np.isnan(esf_oversampled))[0],
                                esf_oversampled[~np.isnan(esf_oversampled)])

    # --- 3. 計算 LSF ---
    window = np.hanning(11)
    esf_smooth = np.convolve(esf_oversampled, window / window.sum(), mode='valid')
    lsf = np.diff(esf_smooth)

    # --- 4. 計算 MTF ---
    lsf_windowed = lsf * np.hanning(len(lsf))
    mtf = np.abs(fft(lsf_windowed))
    mtf = mtf / mtf[0]

    freq = np.fft.fftfreq(len(mtf), d=1.0 / OVERSAMPLING_FACTOR)

    positive_freq_mask = freq >= 0
    freq = freq[positive_freq_mask]
    mtf = mtf[positive_freq_mask]

    # --- 5. 計算 MTF50 ---
    mtf50_lp_per_mm = float('nan')
    resolution_um = float('nan')
    try:
        mtf50_indices = np.where(mtf < 0.5)[0]
        idx1 = mtf50_indices[0] - 1
        idx2 = mtf50_indices[0]
        interp = (0.5 - mtf[idx1]) / (mtf[idx2] - mtf[idx1])
        mtf50_freq_cycles_per_pixel = freq[idx1] + interp * (freq[idx2] - freq[idx1])

        mtf50_lp_per_mm = mtf50_freq_cycles_per_pixel * 1000 / pixel_size
        if mtf50_lp_per_mm > 0:
            resolution_um = 1000 / (2 * mtf50_lp_per_mm)

    except IndexError:
        pass

    # --- 6. 輸出結果 ---
    print("\n--- Slanted-Edge Resolution Analysis Results ---")
    print(f"File: {os.path.basename(filename)}")
    print(f"Image Dimensions: {image_dims[1]}x{image_dims[0]} pixels")
    print(f"Detected Edge Angle: {angle:.2f} degrees")
    print(f"Pixel Size: {pixel_size:.2f} µm/pixel")
    print(f"Oversampling Factor: {OVERSAMPLING_FACTOR}x")
    print("-" * 30)
    if not np.isnan(mtf50_lp_per_mm):
        print(f"Resolution (MTF50): {mtf50_lp_per_mm:.2f} lp/mm")
        print(f"Equivalent Resolvable Size: {resolution_um:.3f} µm")
    else:
        print("Resolution (MTF50): Could not be calculated (MTF curve does not drop below 0.5)")
    print("--- --- --- --- --- --- --- ---\n")

    # --- 7. 視覺化 ---
    image_roi_to_plot = image_data_normalized[:, roi_start_col:roi_end_col]
    plot_slanted_edge_results(image_roi_to_plot,
                              edge_points_x_roi, edge_points_y, inlier_indices,
                              slope, intercept_roi, esf_oversampled, lsf, freq, mtf,
                              filename, pixel_size, image_dims, resolution_um)


def plot_slanted_edge_results(image_roi, edge_x_roi, edge_points_y, inlier_indices,
                              slope, intercept_roi, esf, lsf, freq, mtf,
                              filename, pixel_size, image_dims, resolution_um):
    """繪製斜邊法分析的所有結果圖，並儲存檔案。"""
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Slanted-Edge MTF Analysis (RANSAC)', fontsize=16, y=0.98)

    # 在主標題下方新增一個資訊列
    info_text_line1 = (
        f"File: {os.path.basename(filename)}  |  "
        f"Dimensions: {image_dims[1]}x{image_dims[0]} pixels  |  "
        f"Pixel Size: {pixel_size} µm"
    )
    info_text_line2 = (
        f"Resolvable Size (MTF50): {resolution_um:.3f} µm" if not np.isnan(
            resolution_um) else "Resolvable Size (MTF50): N/A"
    )
    fig.text(0.5, 0.94, info_text_line1, ha='center', va='top', fontsize=10, color='gray')
    fig.text(0.5, 0.91, info_text_line2, ha='center', va='top', fontsize=10, color='blue')

    # 圖 1: ROI 影像與偵測到的邊緣
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(image_roi, cmap='gray')

    x_fit = np.array([0, image_roi.shape[1]])
    y_fit = slope * x_fit + intercept_roi
    ax1.plot(x_fit, y_fit, 'b-', linewidth=2, label=f'RANSAC Fit (Angle: {np.rad2deg(np.arctan(slope)):.2f}°)')

    all_indices = np.arange(len(edge_x_roi))
    outlier_indices = np.setdiff1d(all_indices, inlier_indices)

    ax1.scatter(edge_x_roi[inlier_indices], edge_points_y[inlier_indices],
                c='green', s=8, alpha=0.7, label='Inliers')
    ax1.scatter(edge_x_roi[outlier_indices], edge_points_y[outlier_indices],
                c='red', s=8, alpha=0.7, marker='x', label='Outliers')

    ax1.set_title('Region of Interest (ROI) & Detected Edge')
    ax1.set_aspect('auto', adjustable='box')
    ax1.set_xlim(0, image_roi.shape[1])
    ax1.set_ylim(image_roi.shape[0], 0)
    ax1.legend()

    # 圖 2: 超取樣的 ESF
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(esf, 'b.-', markersize=2)
    ax2.set_title(f'Oversampled Edge Spread Function ({OVERSAMPLING_FACTOR}x ESF)')
    ax2.set_xlabel('Oversampled Pixel Position')
    ax2.set_ylabel('Intensity')
    ax2.grid(True, linestyle=':')

    # 圖 3: 線擴散函數
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(lsf, 'r-')
    ax3.set_title('Line Spread Function (LSF)')
    ax3.set_xlabel('Oversampled Pixel Position')
    ax3.set_ylabel('Derivative')
    ax3.grid(True, linestyle=':')

    # 圖 4: 調制轉換函數
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(freq, mtf, 'g-')
    ax4.axhline(0.5, color='orange', linestyle='--', label='MTF50 (50% Contrast)')
    ax4.set_title('Modulation Transfer Function (MTF)')
    ax4.set_xlabel('Spatial Frequency (cycles/pixel)')
    ax4.set_ylabel('Contrast')
    ax4.set_xlim(0, OVERSAMPLING_FACTOR / 2)
    ax4.set_ylim(0, 1.05)
    ax4.grid(True, which='both', linestyle=':')
    ax4.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.90])

    # 儲存圖檔
    base_name = os.path.splitext(filename)[0]
    output_filename = f"{base_name}.png"
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Analysis plot saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    plt.show()


if __name__ == '__main__':
    analyze_slanted_edge_resolution(FILENAME, PIXEL_SIZE_UM)
