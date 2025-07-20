import os
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, gaussian_filter
from scipy.fft import fft
from scipy.optimize import curve_fit
from scipy.special import erf
import matplotlib.patches as patches
import matplotlib.ticker as mticker
import random
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

# --- 演算法微調參數 ---
OVERSAMPLING_FACTOR = 3
CHANNEL_TO_ANALYZE = 0  # 0=Red, 1=Green, 2=Blue
GAUSSIAN_SIGMA = 2.0
RANSAC_THRESHOLD = 3.0


# ==============================================================================
#  核心分析模組 (Core Analysis Modules)
# ==============================================================================

def edge_model(x, amplitude, center, width, offset):
    """
    [數學模型]
    目的：定義一個理想的、平滑的S形邊緣曲線。
    原理：此模型基於高斯誤差函數 (erf)，能精確描述由光學繞射和像差
          所造成的模糊邊緣輪廓。
    輸入 (Args):
        x (np.array): 水平像素座標。
        amplitude (float): 邊緣的總亮度變化幅度 (亮區 - 暗區)。
        center (float): 邊緣的中心位置 (即 ESF 的中點)。
        width (float): 邊緣的寬度，與模糊程度成正比。
        offset (float): 邊緣暗區的基底亮度。
    輸出 (Returns):
        np.array: 在給定 x 座標下，由模型計算出的對應亮度值。
    """
    return offset + (amplitude / 2.0) * (1 + erf((x - center) / (width + 1e-6)))


def load_image(filename, channel):
    """
    [資料載入模組]
    目的：從指定的檔案路徑安全地載入影像，並轉換為可供分析的Numpy陣列。
    輸入 (Args):
        filename (str): 影像的完整檔案路徑。
        channel (int): 對於RGB影像，指定要分析的顏色通道 (0=R, 1=G, 2=B)。
    輸出 (Returns):
        np.array: 代表影像灰階值的二維Numpy陣列。若載入失敗則回傳 None。
    """
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
    """
    [邊緣偵測模組]
    目的：在給定的ROI內，逐行掃描並找出所有可能的邊緣點。
    原理：採用「模型擬合法」(Profile Fitting)，將每一行的像素數據與理想的
          `edge_model` 進行數學擬合，找出最匹配的模型中心點作為該行的邊緣位置。
          此方法對雜訊有很強的抵抗能力。
    輸入 (Args):
        image_roi (np.array): 只包含感興趣區域 (ROI) 的影像數據陣列。
    輸出 (Returns):
        tuple: (np.array, np.array)，分別為所有偵測到的邊緣點的 x 和 y 座標。
    """
    edge_points_y, edge_points_x_roi = [], []
    image_roi_smoothed = gaussian_filter(image_roi, sigma=GAUSSIAN_SIGMA)

    for i in range(image_roi_smoothed.shape[0]):
        row_data = image_roi_smoothed[i, :]
        x_data = np.arange(len(row_data))

        # 自動偵測邊緣是「暗->亮」還是「亮->暗」
        width = len(row_data)
        q_width = max(1, width // 4)
        mean_left = np.mean(row_data[:q_width])
        mean_right = np.mean(row_data[-q_width:])

        # 若為「亮->暗」，則將數據垂直翻轉，以符合S形上升模型
        if mean_left > mean_right:
            row_data_to_fit = np.max(row_data) - row_data
        else:
            row_data_to_fit = row_data

        min_val, max_val = np.min(row_data_to_fit), np.max(row_data_to_fit)
        if max_val <= min_val: continue

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
            popt, _ = curve_fit(edge_model, x_data, row_data_to_fit, p0=initial_guesses, bounds=bounds, maxfev=5000)
            center_fit = popt[1]
            if 0 < center_fit < len(row_data) - 1:
                edge_points_y.append(i)
                edge_points_x_roi.append(center_fit)
        except (RuntimeError, ValueError):
            continue

    return np.array(edge_points_x_roi), np.array(edge_points_y)


def filter_and_fit_edge(x_points, y_points):
    """
    [邊緣擬合模組]
    目的：從大量候選邊緣點中，找出最能代表真實邊緣趨勢的直線。
    原理：採用「預濾波 + RANSAC」的兩階段策略。
          1. 預濾波：使用統計學方法(IQR)去除最極端的離群點。
          2. RANSAC：在清理過的數據上，透過迭代和投票，找出能獲得最多數點
             支持的最佳擬合線，能有效抵抗雜訊和局部瑕疵的干擾。
    輸入 (Args):
        x_points (np.array): 所有候選點的 x 座標。
        y_points (np.array): 所有候選點的 y 座標。
    輸出 (Returns):
        tuple: (斜率, 截距, (過濾後的x,y點), 內群點的索引)。若失敗則回傳 None。
    """
    q1 = np.percentile(x_points, 25)
    q3 = np.percentile(x_points, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    mask = (x_points > lower_bound) & (x_points < upper_bound)
    x_filtered, y_filtered = x_points[mask], y_points[mask]
    if len(x_filtered) < 10: return None, None, None, None
    slope, intercept, inlier_indices = ransac_line_fitting(x_filtered, y_filtered, distance_threshold=RANSAC_THRESHOLD)
    return slope, intercept, (x_filtered, y_filtered), inlier_indices


def ransac_line_fitting(x, y, n_iterations=100, distance_threshold=3.0):
    """ [輔助函式] RANSAC 演算法的具體實現。 """
    best_inliers = []
    best_model = (0, 0)
    data = np.column_stack((x, y))
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
        if len(inliers_idx) > len(best_inliers):
            best_inliers = inliers_idx
            best_model = (m, c)
    if len(best_inliers) > 2:
        inlier_data = data[best_inliers]
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
    return best_model[0], best_model[1], best_inliers


def calculate_mtf_from_edge(image_roi, slope, intercept, oversampling_factor, pixel_size):
    """
    [MTF 計算模組]
    目的：根據已確定的邊緣線，執行所有核心的數學運算。
    原理：
        1. 投影與分箱：計算ROI內所有像素到擬合線的垂直距離，並根據此距離
           將像素亮度值分箱平均，建構出超解析度的 ESF。
        2. 微分：對 ESF 進行微分，得到 LSF。
        3. 傅立葉轉換：對 LSF 進行傅立葉轉換，得到 MTF。
        4. 內插：在 MTF 曲線上，透過內插找出 MTF50 和 MTF30 的精確頻率值。
    輸入 (Args):
        image_roi (np.array): 原始 ROI 影像數據。
        slope, intercept (float): 擬合線的斜率與截距。
        oversampling_factor (int): 超取樣倍率。
        pixel_size (float): 像素的物理尺寸。
    輸出 (Returns):
        dict: 一個包含所有計算結果的字典 (esf, lsf, mtf, freq, 解析度數值等)。
    """
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
    results = {'esf': esf, 'lsf': lsf, 'mtf': mtf, 'freq': freq}
    for val in [50, 30]:
        key_lp, key_um = f'mtf{val}_lp_per_mm', f'resolution_um_{val}'
        results[key_lp], results[key_um] = float('nan'), float('nan')
        try:
            threshold = val / 100.0
            indices = np.where(mtf < threshold)[0]
            idx1, idx2 = indices[0] - 1, indices[0]
            interp = (threshold - mtf[idx1]) / (mtf[idx2] - mtf[idx1])
            freq_cycles_per_pixel = freq[idx1] + interp * (freq[idx2] - freq[idx1])
            lp_per_mm = freq_cycles_per_pixel * 1000 / pixel_size
            if lp_per_mm > 0:
                results[key_lp], results[key_um] = lp_per_mm, 1000 / (2 * lp_per_mm)
        except IndexError:
            continue
    return results


def print_results(filename, image_dims, roi_dims, slope, pixel_size, mtf_results):
    """ [報告輸出模組] 格式化並在終端機印出所有重要的分析結果。 """
    print("\n--- Slanted-Edge Resolution Analysis Results ---")
    print(f"File: {os.path.basename(filename)}")
    print(f"Original Dimensions: {image_dims[1]}x{image_dims[0]} pixels")
    print(f"Analyzed ROI Dimensions: {roi_dims[1]}x{roi_dims[0]} pixels")
    print(f"Detected Edge Angle (in ROI): {np.rad2deg(np.arctan(slope)):.2f} degrees")
    print(f"Pixel Size: {pixel_size:.2f} µm/pixel")
    mtf50_lp, res_um_50 = mtf_results['mtf50_lp_per_mm'], mtf_results['resolution_um_50']
    mtf30_lp, res_um_30 = mtf_results['mtf30_lp_per_mm'], mtf_results['resolution_um_30']
    if not np.isnan(mtf50_lp): print(f"Resolution (MTF50): {mtf50_lp:.2f} lp/mm  |  Equiv. Size: {res_um_50:.3f} µm")
    if not np.isnan(mtf30_lp): print(f"Resolution (MTF30): {mtf30_lp:.2f} lp/mm  |  Equiv. Size: {res_um_30:.3f} µm")
    print("--- --- --- --- --- --- --- ---\n")


def plot_analysis_results(image_roi, filtered_points, inlier_indices, slope, intercept_roi, mtf_results, filename,
                          pixel_size, manual_roi):
    """ [視覺化模組] 產生一個包含四個子圖的完整分析報告，並儲存為PNG檔案。 """
    esf, lsf, mtf, freq = mtf_results['esf'], mtf_results['lsf'], mtf_results['mtf'], mtf_results['freq']
    res_um_50, res_um_30 = mtf_results['resolution_um_50'], mtf_results['resolution_um_30']
    edge_x_roi, edge_points_y = filtered_points
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Slanted-Edge MTF Analysis', fontsize=16, y=0.98)
    roi_dims = image_roi.shape
    info_text_line1 = (f"File: {os.path.basename(filename)}  |  "
                       f"Analyzed ROI Dimensions: {roi_dims[1]}x{roi_dims[0]} pixels  |  "
                       f"Pixel Size: {pixel_size} µm")
    res_text_50 = f"{res_um_50:.3f} µm" if not np.isnan(res_um_50) else "N/A"
    res_text_30 = f"{res_um_30:.3f} µm" if not np.isnan(res_um_30) else "N/A"
    info_text_line2 = f"Resolvable Size (MTF50): {res_text_50} | (MTF30): {res_text_30}"
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
    ax4.axhline(0.3, color='cyan', linestyle=':', label='MTF30 (30% Contrast)')
    ax4.set_title('Modulation Transfer Function (MTF)')
    ax4.set_xlabel('Spatial Frequency (cycles/pixel)')
    ax4.set_ylabel('Contrast')
    ax4.set_xlim(0, OVERSAMPLING_FACTOR / OVERSAMPLING_FACTOR / 4)
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


# ==============================================================================
#  GUI 互動介面 (GUI Interface)
# ==============================================================================

class ROISelector:
    """ 一個 Tkinter 視窗，用於載入影像並讓使用者用滑鼠框選 ROI。 """

    def __init__(self, parent, image_path):
        self.top = tk.Toplevel(parent)
        self.top.title("用滑鼠左鍵拖曳以選取 ROI")
        self.image_path = image_path
        self.roi_coords = None
        self.rect = None
        self.start_x = None
        self.start_y = None

        self.pil_image = Image.open(self.image_path)
        self.original_width, self.original_height = self.pil_image.size
        self.scale_factor = 1.0

        screen_width = self.top.winfo_screenwidth()
        screen_height = self.top.winfo_screenheight()
        padding = 100

        if self.original_width > (screen_width - padding) or self.original_height > (screen_height - padding):
            ratio_w = (screen_width - padding) / self.original_width
            ratio_h = (screen_height - padding) / self.original_height
            self.scale_factor = min(ratio_w, ratio_h)

        self.display_width = int(self.original_width * self.scale_factor)
        self.display_height = int(self.original_height * self.scale_factor)

        display_image = self.pil_image.resize((self.display_width, self.display_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(display_image)

        self.status_var = tk.StringVar()
        self.status_label = tk.Label(self.top, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.TOP, fill=tk.X)

        self.canvas = tk.Canvas(self.top, width=self.display_width, height=self.display_height, cursor="cross")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def on_button_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        if self.rect: self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red',
                                                 width=2)
        self.status_var.set("開始框選...")

    def on_mouse_drag(self, event):
        cur_x, cur_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)
        inv_scale = 1.0 / self.scale_factor
        roi_width = int(abs(cur_x - self.start_x) * inv_scale)
        roi_height = int(abs(cur_y - self.start_y) * inv_scale)
        self.status_var.set(f"ROI 尺寸: {roi_width} x {roi_height} pixels")

    def on_button_release(self, event):
        end_x, end_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        inv_scale = 1.0 / self.scale_factor
        x1 = int(min(self.start_x, end_x) * inv_scale)
        y1 = int(min(self.start_y, end_y) * inv_scale)
        x2 = int(max(self.start_x, end_x) * inv_scale)
        y2 = int(max(self.start_y, end_y) * inv_scale)
        self.roi_coords = (max(0, x1), max(0, y1), min(self.original_width, x2), min(self.original_height, y2))
        print(f"ROI selected (original image coordinates): {self.roi_coords}")
        self.top.destroy()


class App:
    """ GUI 應用程式的主控制器。 """

    def __init__(self, root):
        self.root = root
        self.root.withdraw()
        self.run_full_process()

    def run_full_process(self):
        """ 執行從「選擇檔案」到「顯示結果」的完整流程。 """
        filepath = filedialog.askopenfilename(
            title="選擇 TIF 檔案",
            filetypes=(("TIF Files", "*.tif *.tiff"), ("All files", "*.*"))
        )
        if not filepath:
            self.root.destroy()
            return

        global FILENAME
        FILENAME = filepath

        roi_selector = ROISelector(self.root, filepath)
        self.root.wait_window(roi_selector.top)
        manual_roi = roi_selector.roi_coords

        if not manual_roi or (manual_roi[2] - manual_roi[0] < 10) or (manual_roi[3] - manual_roi[1] < 10):
            messagebox.showwarning("取消", "未選取有效的 ROI，分析已取消。")
            self.root.destroy()
            return

        pixel_size = simpledialog.askfloat("輸入參數", "請輸入 Pixel Size (µm):", parent=self.root, minvalue=0.001,
                                           initialvalue=0.13)

        if pixel_size is None:
            messagebox.showwarning("取消", "未輸入 Pixel Size，分析已取消。")
            self.root.destroy()
            return

        try:
            image_data = load_image(filepath, CHANNEL_TO_ANALYZE)
            if image_data is not None:
                # --- 流程總指揮 ---
                # 1. 偵測邊緣點
                x_points, y_points = detect_edge_points(
                    image_data[manual_roi[1]:manual_roi[3], manual_roi[0]:manual_roi[2]])
                if len(x_points) < 20:
                    messagebox.showerror("錯誤", "在選取的 ROI 內無法偵測到足夠的邊緣點。")
                    self.root.destroy()
                    return

                # 2. 擬合邊緣線
                slope, intercept, filtered_points, inlier_indices = filter_and_fit_edge(x_points, y_points)
                if slope is None:
                    messagebox.showerror("錯誤", "RANSAC 擬合失敗，請確認 ROI 選擇是否正確。")
                    self.root.destroy()
                    return

                # 3. 計算 MTF
                image_roi = image_data[manual_roi[1]:manual_roi[3], manual_roi[0]:manual_roi[2]]
                mtf_results = calculate_mtf_from_edge(image_roi, slope, intercept, OVERSAMPLING_FACTOR, pixel_size)

                # 4. 輸出報告與視覺化
                print_results(filepath, image_data.shape, image_roi.shape, slope, pixel_size, mtf_results)
                min_roi, max_roi = np.min(image_roi), np.max(image_roi)
                image_roi_to_plot = (image_roi - min_roi) * (
                            255.0 / (max_roi - min_roi)) if max_roi > min_roi else image_roi
                plot_analysis_results(image_roi_to_plot, filtered_points, inlier_indices,
                                      slope, intercept, mtf_results,
                                      filepath, pixel_size, manual_roi)

        except Exception as e:
            messagebox.showerror("分析錯誤", f"分析過程中發生錯誤:\n{e}")

        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
