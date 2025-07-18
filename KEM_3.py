import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, gaussian_filter
from scipy.fft import fft
from scipy.optimize import curve_fit
from scipy.special import erf
import matplotlib.patches as patches
import matplotlib.ticker as mticker
import random

# --- 參數設定 ---
FILENAME = '50X20KB1_DCM.tif'
PIXEL_SIZE_UM = 0.13
OVERSAMPLING_FACTOR = 4
MANUAL_ROI = (1000, 1000, 1400, 2000)  # (x_start, y_start, x_end, y_end)
CHANNEL_TO_ANALYZE = 0  # 0=Red, 1=Green, 2=Blue
GAUSSIAN_SIGMA = 5.0
RANSAC_THRESHOLD = 5.0


# --- 模組化函式 ---

def edge_model(x, amplitude, center, width, offset):
    """ 描述理想邊緣的數學模型 (基於高斯誤差函數 erf)。 """
    return offset + (amplitude / 2.0) * (1 + erf((x - center) / (width + 1e-6)))


def load_image(filename, channel):
    """ 讀取影像檔案並選取指定的顏色通道。 """
    try:
        with Image.open(filename) as img:
            if img.mode == 'RGB' and channel in [0, 1, 2]:
                print(f"RGB image detected. Analyzing channel {channel} (0=R, 1=G, 2=B).")
                return np.array(img.split()[channel], dtype=float)
            else:
                return np.array(img.convert('L'), dtype=float)
    except FileNotFoundError:
        print(f"Error: File not found at '{filename}'.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def detect_edge_points(image_roi):
    """ 使用模型擬合法 (Profile Fitting) 在 ROI 內找出所有候選的邊緣點。 """
    edge_points_y, edge_points_x_roi = [], []
    image_roi_smoothed = gaussian_filter(image_roi, sigma=GAUSSIAN_SIGMA)

    for i in range(image_roi_smoothed.shape[0]):
        row_data = image_roi_smoothed[i, :]
        x_data = np.arange(len(row_data))

        # --- NEW: 自動偵測邊緣方向 ---
        width = len(row_data)
        q_width = max(1, width // 4)
        mean_left = np.mean(row_data[:q_width])
        mean_right = np.mean(row_data[-q_width:])

        # 如果是左白右黑，則將數據反轉以符合S形上升模型
        if mean_left > mean_right:
            row_data_to_fit = np.max(row_data) - row_data
        else:
            row_data_to_fit = row_data

        min_val, max_val = np.min(row_data_to_fit), np.max(row_data_to_fit)
        if max_val <= min_val: continue

        # 根據 "row_data_to_fit" 計算初始猜測值
        amplitude_guess = max_val - min_val
        offset_guess = min_val
        median_val = (max_val + min_val) / 2.0
        center_guess = np.argmin(np.abs(row_data_to_fit - median_val))
        width_guess = 5.0
        initial_guesses = [amplitude_guess, center_guess, width_guess, offset_guess]

        lower_bounds = [0, 0, 0.1, 0]
        upper_bounds = [amplitude_guess * 2, len(row_data) - 1, len(row_data), max_val * 2]
        bounds = (lower_bounds, upper_bounds)

        try:
            # 使用 "row_data_to_fit" 進行擬合
            popt, _ = curve_fit(edge_model, x_data, row_data_to_fit, p0=initial_guesses, bounds=bounds, maxfev=5000)
            center_fit = popt[1]
            if 0 < center_fit < len(row_data) - 1:
                edge_points_y.append(i)
                edge_points_x_roi.append(center_fit)
        except (RuntimeError, ValueError):
            continue

    return np.array(edge_points_x_roi), np.array(edge_points_y)


def filter_and_fit_edge(x_points, y_points, n_iterations=100, distance_threshold=3.0):
    """ 對候選點進行預濾波，並使用 RANSAC 演算法進行穩健的線性擬合。 """
    # 預濾波
    q1 = np.percentile(x_points, 25)
    q3 = np.percentile(x_points, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    mask = (x_points > lower_bound) & (x_points < upper_bound)
    x_filtered, y_filtered = x_points[mask], y_points[mask]

    if len(x_filtered) < 10:
        return None, None, None, None

    # RANSAC
    best_inliers_idx = []
    best_model = (0, 0)
    data = np.column_stack((x_filtered, y_filtered))

    for i in range(n_iterations):
        if len(data) < 2: continue
        sample_indices = random.sample(range(len(data)), 2)
        sample = data[sample_indices]
        p1, p2 = sample[0], sample[1]
        if abs(p1[0] - p2[0]) < 1e-6: continue
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        c = p1[1] - m * p1[0]
        distances = np.abs(m * data[:, 0] - data[:, 1] + c) / np.sqrt(m ** 2 + 1)
        inliers_idx = np.where(distances < distance_threshold)[0]
        if len(inliers_idx) > len(best_inliers_idx):
            best_inliers_idx = inliers_idx
            best_model = (m, c)

    if len(best_inliers_idx) > 2:
        inlier_data = data[best_inliers_idx]
        x_inliers, y_inliers = inlier_data[:, 0], inlier_data[:, 1]
        x_range, y_range = np.ptp(x_inliers), np.ptp(y_inliers)
        if y_range > x_range:
            coeffs = np.polyfit(y_inliers, x_inliers, 1)
            m_inv, c_inv = coeffs[0], coeffs[1]
            if abs(m_inv) < 1e-9: m_inv = 1e-9
            slope, intercept = 1.0 / m_inv, -c_inv / m_inv
        else:
            slope, intercept = np.polyfit(x_inliers, y_inliers, 1)
        best_model = (slope, intercept)

    return best_model[0], best_model[1], (x_filtered, y_filtered), best_inliers_idx


def calculate_mtf_from_edge(image_roi, slope, intercept, oversampling_factor, pixel_size):
    """ 根據擬合線，計算 ESF, LSF, MTF 及解析度。 """
    x_roi, y_roi = np.meshgrid(np.arange(image_roi.shape[1]), np.arange(image_roi.shape[0]))
    distances = (slope * x_roi - y_roi + intercept) / np.sqrt(slope ** 2 + 1)

    intensities_flat = image_roi.ravel()
    distances_flat = distances.ravel()

    sort_indices = np.argsort(distances_flat)
    distances_sorted, intensities_sorted = distances_flat[sort_indices], intensities_flat[sort_indices]

    min_dist, max_dist = distances_sorted[0], distances_sorted[-1]
    num_bins = int((max_dist - min_dist) * oversampling_factor)
    if num_bins < 1: num_bins = 1
    bin_edges = np.linspace(min_dist, max_dist, num_bins + 1)

    sum_in_bin, _ = np.histogram(distances_sorted, bins=bin_edges, weights=intensities_sorted)
    count_in_bin, _ = np.histogram(distances_sorted, bins=bin_edges)

    non_empty_bins = count_in_bin > 0
    esf = np.full_like(sum_in_bin, np.nan)
    esf[non_empty_bins] = sum_in_bin[non_empty_bins] / count_in_bin[non_empty_bins]
    esf = np.interp(np.arange(len(esf)), np.where(~np.isnan(esf))[0], esf[~np.isnan(esf)])

    window = np.hanning(11)
    esf_smooth = np.convolve(esf, window / window.sum(), mode='valid')
    lsf = np.diff(esf_smooth)

    lsf_windowed = lsf * np.hanning(len(lsf))
    mtf = np.abs(fft(lsf_windowed))
    mtf = mtf / mtf[0]
    freq = np.fft.fftfreq(len(mtf), d=1.0 / oversampling_factor)
    positive_freq_mask = freq >= 0
    freq, mtf = freq[positive_freq_mask], mtf[positive_freq_mask]

    mtf50_lp_per_mm, resolution_um = float('nan'), float('nan')
    try:
        mtf50_indices = np.where(mtf < 0.5)[0]
        idx1, idx2 = mtf50_indices[0] - 1, mtf50_indices[0]
        interp = (0.5 - mtf[idx1]) / (mtf[idx2] - mtf[idx1])
        mtf50_freq_cycles_per_pixel = freq[idx1] + interp * (freq[idx2] - freq[idx1])
        mtf50_lp_per_mm = mtf50_freq_cycles_per_pixel * 1000 / pixel_size
        if mtf50_lp_per_mm > 0:
            resolution_um = 1000 / (2 * mtf50_lp_per_mm)
    except IndexError:
        pass

    return esf, lsf, mtf, freq, mtf50_lp_per_mm, resolution_um



def plot_analysis_results(image_roi, filtered_points, inlier_indices,
                          slope, intercept_roi, mtf_results,
                          filename, pixel_size, manual_roi):
    """ 繪製所有分析結果圖並儲存。 """
    esf, lsf, mtf, freq, _, resolution_um = mtf_results
    edge_x_roi, edge_points_y = filtered_points

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Slanted-Edge MTF Analysis (Modular Version)', fontsize=16, y=0.98)

    roi_dims = image_roi.shape
    info_text_line1 = (f"File: {os.path.basename(filename)}  |  "
                       f"Analyzed ROI Dimensions: {roi_dims[1]}x{roi_dims[0]} pixels  |  "
                       f"Pixel Size: {pixel_size} µm")
    info_text_line2 = (f"Resolvable Size (MTF50): {resolution_um:.3f} µm" if not np.isnan(
        resolution_um) else "Resolvable Size (MTF50): N/A")
    fig.text(0.5, 0.94, info_text_line1, ha='center', va='top', fontsize=10, color='gray')
    fig.text(0.5, 0.91, info_text_line2, ha='center', va='top', fontsize=10, color='blue')

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(image_roi, cmap='gray', aspect='auto')
    x_fit = np.array([0, roi_dims[1]])
    y_fit = slope * x_fit + intercept_roi
    ax1.plot(x_fit, y_fit, 'b-', linewidth=2, label=f'RANSAC Fit (Angle: {np.rad2deg(np.arctan(slope)):.2f}°)')

    all_indices_filtered = np.arange(len(edge_x_roi))
    outlier_indices = np.setdiff1d(all_indices_filtered, inlier_indices)

    ax1.scatter(edge_x_roi[inlier_indices], edge_points_y[inlier_indices], c='green', s=8, alpha=0.7, label='Inliers')
    ax1.scatter(edge_x_roi[outlier_indices], edge_points_y[outlier_indices], c='red', s=8, alpha=0.7, marker='x',
                label='Outliers')

    x_start, y_start, _, _ = manual_roi

    def x_formatter(x, pos):
        return f"{int(x + x_start)}"

    def y_formatter(y, pos):
        return f"{int(y + y_start)}"

    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(x_formatter))
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(y_formatter))
    ax1.set_title('Region of Interest (ROI) & Detected Edge')
    ax1.set_xlim(0, roi_dims[1])
    ax1.set_ylim(roi_dims[0], 0)
    ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(esf, 'b.-', markersize=2)
    ax2.set_title(f'Oversampled Edge Spread Function ({OVERSAMPLING_FACTOR}x ESF)')
    ax2.set_xlabel('Oversampled Pixel Position')
    ax2.set_ylabel('Intensity')
    ax2.grid(True, linestyle=':')

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(lsf, 'r-')
    ax3.set_title('Line Spread Function (LSF)')
    ax3.set_xlabel('Oversampled Pixel Position')
    ax3.set_ylabel('Derivative')
    ax3.grid(True, linestyle=':')

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

    base_name = os.path.splitext(filename)[0]
    output_filename = f"{base_name}.png"
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Analysis plot saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    plt.show()


def main():
    """ 主執行函式 """
    image_data = load_image(FILENAME, CHANNEL_TO_ANALYZE)
    if image_data is None:
        return

    if MANUAL_ROI is None:
        print("Error: Manual ROI is required.")
        return

    x_start, y_start, x_end, y_end = MANUAL_ROI
    x_end_clipped = min(x_end, image_data.shape[1])
    y_end_clipped = min(y_end, image_data.shape[0])
    MANUAL_ROI_CLIPPED = (x_start, y_start, x_end_clipped, y_end_clipped)

    image_roi = image_data[y_start:y_end_clipped, x_start:x_end_clipped]

    x_points, y_points = detect_edge_points(image_roi)
    if len(x_points) < 20:
        print("Error: Not enough edge points detected.")
        return

    slope, intercept, filtered_points, inlier_indices = filter_and_fit_edge(x_points, y_points,
                                                                            distance_threshold=RANSAC_THRESHOLD)
    if slope is None:
        print("Error: RANSAC fitting failed.")
        return

    mtf_results = calculate_mtf_from_edge(image_roi, slope, intercept, OVERSAMPLING_FACTOR, PIXEL_SIZE_UM)

    # 輸出文字報告
    _, _, _, _, mtf50_lp_per_mm, resolution_um = mtf_results
    print("\n--- Slanted-Edge Resolution Analysis Results ---")
    print(f"File: {os.path.basename(FILENAME)}")
    print(f"Analyzed ROI Dimensions: {image_roi.shape[1]}x{image_roi.shape[0]} pixels")
    print(f"Detected Edge Angle (in ROI): {np.rad2deg(np.arctan(slope)):.2f} degrees")
    print(f"Pixel Size: {PIXEL_SIZE_UM:.2f} µm/pixel")
    if not np.isnan(mtf50_lp_per_mm):
        print(f"Resolution (MTF50): {mtf50_lp_per_mm:.2f} lp/mm")
        print(f"Equivalent Resolvable Size: {resolution_um:.3f} µm")
    else:
        print("Resolution (MTF50): Could not be calculated.")
    print("--- --- --- --- --- --- --- ---\n")

    # 視覺化
    min_roi, max_roi = np.min(image_roi), np.max(image_roi)
    image_roi_to_plot = (image_roi - min_roi) * (255.0 / (max_roi - min_roi)) if max_roi > min_roi else image_roi
    plot_analysis_results(image_roi_to_plot, filtered_points, inlier_indices,
                          slope, intercept, mtf_results,
                          FILENAME, PIXEL_SIZE_UM, MANUAL_ROI_CLIPPED)


if __name__ == '__main__':
    main()
