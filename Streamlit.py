import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.morphology import skeletonize
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splprep, splev
import math
import io
import warnings
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

# ==========================================
# 1. 網頁基本設定 (設定為寬螢幕模式)
# ==========================================
st.set_page_config(page_title="PocketLab 2D Engine", layout="wide", page_icon="🗺️")

st.title("🗺️ PocketLab 2D Engine - 空間測繪與地籍分析系統")

# ==========================================
# 2. 側邊欄 (Sidebar) - 參數控制台
# ==========================================
with st.sidebar:
    st.header("🛠️ 測量環境設定")
    
    # 圖片上傳器
    uploaded_file = st.file_uploader("📂 請上傳地籍圖或波形圖 (JPG/PNG)", type=['jpg', 'png', 'jpeg'])
    
    # 軌道選擇
    target_type = st.selectbox("🎯 選擇分析模式", ['contour (封閉地籍區塊)', 'signal (連續訊號分析)'])
    target_mode = 'contour' if 'contour' in target_type else 'signal'
    
    st.divider()
    
    # 比例尺與校正設定
    st.header("📏 比例尺與單位映射")
    map_scale = st.number_input("工程比例尺分母 (如 1:5000 請輸入 5000)", min_value=1.0, value=1.0, step=100.0)
    
    st.markdown("##### 📐 相片解析度校準")
    calibration_mode = st.radio("校準方式", ["自動 (ArUco標籤)", "手動 (自訂像素長度)"])
    
    marker_real_size_mm = 50.0
    manual_scale_mm = None
    known_pixel_length = None
    
    if calibration_mode == "自動 (ArUco標籤)":
        marker_real_size_mm = st.number_input("ArUco 標籤實體大小 (mm)", value=50.0)
    else:
        manual_scale_mm = st.number_input("對照實體長度 (mm)", value=20.0)

    st.divider()
    
    # ------------------------------------------
    # 模式專屬設定區
    # ------------------------------------------
    if target_mode == 'contour':
        st.header("🛡️ 幾何過濾裝甲設定")
        
        preset_scene = st.selectbox(
            "📋 選擇地籍圖資情境預設組",
            ["自訂參數調整", "高解析電子原始檔 (嚴格阻絕文字雜訊)", "實體紙張拍照/老舊掃描檔 (平衡容錯)", "微型畸零地觀測 (寬鬆擷取)"]
        )
        
        if preset_scene == "高解析電子原始檔 (嚴格阻絕文字雜訊)":
            init_area, init_circ, init_border = 2000, 0.18, 10
        elif preset_scene == "實體紙張拍照/老舊掃描檔 (平衡容錯)":
            init_area, init_circ, init_border = 1200, 0.12, 20
        elif preset_scene == "微型畸零地觀測 (寬鬆擷取)":
            init_area, init_circ, init_border = 400, 0.04, 5
        else:
            init_area, init_circ, init_border = 1500, 0.15, 15

        min_area_px = st.slider("最小面積門檻 (px²)", 100, 5000, init_area, 50)
        circularity_threshold = st.slider("最小緊緻度 / 圓形度", 0.01, 1.0, init_circ, 0.01)
        border_margin = st.slider("邊界排除距離 (px)", 0, 100, init_border, 5)

    elif target_mode == 'signal':
        st.header("📈 訊號空間與採樣設定")
        
        st.markdown("##### ✏️ 自訂座標軸名稱")
        x_axis_name = st.text_input("X 軸名稱", value="水平距離 (Distance)")
        y_axis_name = st.text_input("Y 軸名稱", value="垂直偏轉 (Deflection)")
        
        st.markdown("##### 🎯 自訂原點與基準")
        origin_x = st.number_input("X 起始座標 (原點)", value=0.0, step=1.0, help="設定曲線起點的 X 座標")
        origin_y = st.number_input("Y 起始座標 (原點)", value=0.0, step=1.0, help="設定曲線起點的 Y 座標 (如基準高程)")
        
        st.markdown("##### ⏱️ 空間採集頻率")
        sample_step = st.number_input(
            "取樣間距 ΔX", 
            min_value=0.001, 
            value=0.5, 
            step=0.1,
            format="%.3f",
            help="決定 B-Spline 切割頻率。例如 0.5 代表每前進 0.5 單位擷取一個數據點。"
        )
        
        st.markdown("##### 🔍 繪圖座標軸顯示範圍")
        use_custom_limits = st.checkbox("啟用手動限制圖表邊界", value=False)
        if use_custom_limits:
            col1, col2 = st.columns(2)
            with col1:
                x_min_limit = st.number_input("X 軸最小", value=0.0)
                y_min_limit = st.number_input("Y 軸最小", value=-50.0)
            with col2:
                x_max_limit = st.number_input("X 軸最大", value=100.0)
                y_max_limit = st.number_input("Y 軸最大", value=50.0)
        else:
            x_min_limit, x_max_limit, y_min_limit, y_max_limit = None, None, None, None

    # 執行按鈕
    run_btn = st.button("🚀 開始啟動分析引擎", type="primary", use_container_width=True)

# ==========================================
# 3. 主畫面大圖點擊工作區 (含智慧縮放與座標映射)
# ==========================================
if calibration_mode == "手動 (自訂像素長度)":
    if uploaded_file is not None:
        st.subheader("🎯 互動式像素長度點擊校準區")
        st.caption("請直接在下方圖片中，點擊地籍圖上比例尺線段的「起點」與「終點」：")
        
        pil_img = Image.open(uploaded_file)
        orig_w, orig_h = pil_img.size

        ui_scale = 800.0 / max(orig_w, orig_h) if max(orig_w, orig_h) > 800 else 1.0
        ui_img = pil_img.resize((int(orig_w * ui_scale), int(orig_h * ui_scale)))
        
        engine_scale = 2000.0 / max(orig_w, orig_h) if max(orig_w, orig_h) > 2000 else 1.0
        mapping_ratio = engine_scale / ui_scale

        value = streamlit_image_coordinates(ui_img, key="main_canvas")

        if "points" not in st.session_state:
            st.session_state["points"] = []

        if value is not None:
            point = (value["x"] * mapping_ratio, value["y"] * mapping_ratio)
            if len(st.session_state["points"]) == 0 or point != st.session_state["points"][-1]:
                st.session_state["points"].append(point)

        if len(st.session_state["points"]) >= 2:
            p1 = st.session_state["points"][-2]
            p2 = st.session_state["points"][-1]
            calc_dist = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
            st.success(f"🎯 點擊測量成功！對應後端引擎的圖上距離為: {calc_dist:.2f} px")
            known_pixel_length = calc_dist
        else:
            st.info("💡 尚未完成兩點點擊，目前使用系統預設值（100 px）。")
            known_pixel_length = 100.0
    else:
        known_pixel_length = 100.0

# ==========================================
# 4. 核心引擎函式
# ==========================================
def run_pocketlab_engine(img_bytes, k_pixel):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    h, w = img.shape[:2]
    max_dim = 2000 
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.split(img)[1] 
    
    mm_per_pixel = None
    corners, ids = [], None

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if calibration_mode == "手動 (自訂像素長度)" and manual_scale_mm and k_pixel:
        mm_per_pixel = manual_scale_mm / k_pixel
        st.success(f"📊 後端確認採用手動校準解析度: 1px = {mm_per_pixel:.4f} mm")
    else:
        if ids is not None and len(corners) > 0:
            c = corners[0][0]
            pixel_edge_length = np.linalg.norm(c[0] - c[1])
            mm_per_pixel = marker_real_size_mm / pixel_edge_length
            st.success(f"📊 後端確認採用 ArUco 自動校準解析度: 1px = {mm_per_pixel:.4f} mm")
        else:
            st.warning("⚠️ 系統預設為「相對像素 (Pixel)」模式。")

    if mm_per_pixel is not None:
        if map_scale > 1.0:
            real_unit_per_pixel = (mm_per_pixel * map_scale) / 1000.0
            unit_str, area_unit = "m", "m²"
            st.info(f"🌍 GIS 映射成功：1px = {real_unit_per_pixel:.4f} 公尺")
        else:
            real_unit_per_pixel, unit_str, area_unit = mm_per_pixel, "mm", "mm²"
    else:
        real_unit_per_pixel, unit_str, area_unit = None, "Pixel", "px²"

    binary_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    if ids is not None and len(corners) > 0:
        for c in corners:
            pts = c[0]
            x_min, x_max = int(np.min(pts[:, 0])), int(np.max(pts[:, 0]))
            y_min, y_max = int(np.min(pts[:, 1])), int(np.max(pts[:, 1]))
            w_aruco, h_aruco = x_max - x_min, y_max - y_min
            pad_x, pad_y = int(w_aruco * 0.3), int(h_aruco * 0.3)
            cv2.rectangle(binary_mask, (max(0, x_min - pad_x), max(0, y_min - pad_y)), 
                                       (min(w, x_max + pad_x), min(h, y_max + pad_y)), 0, -1)

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    clean_mask = cv2.morphologyEx(cv2.dilate(binary_mask, kernel_dilate, iterations=1), cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    # ------------------------------------------
    # Contour 地籍圖模式
    # ------------------------------------------
    if target_mode == 'contour':
        contours, hierarchy = cv2.findContours(clean_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        valid_contours = []
        img_h, img_w = clean_mask.shape[:2]
        max_area_px = img_h * img_w * 0.5 
        
        if hierarchy is not None:
            for i, cnt in enumerate(contours):
                area_px = cv2.contourArea(cnt)
                if min_area_px < area_px < max_area_px:
                    x, y, w_box, h_box = cv2.boundingRect(cnt)
                    if x < border_margin or y < border_margin or (x + w_box) > (img_w - border_margin) or (y + h_box) > (img_h - border_margin):
                        continue 
                    vertices = len(cv2.approxPolyDP(cnt, 0.005 * cv2.arcLength(cnt, True), True))
                    if vertices < 3 or vertices > 25: continue
                        
                    perimeter_px = cv2.arcLength(cnt, True)
                    if perimeter_px == 0 or ((4 * np.pi * area_px) / (perimeter_px ** 2)) < circularity_threshold:
                        continue

                    parent_idx = hierarchy[0][i][3]
                    is_inner_edge = False
                    if parent_idx != -1:  
                        parent_area = cv2.contourArea(contours[parent_idx])
                        if parent_area > 0 and (area_px / parent_area) > 0.8: is_inner_edge = True
                            
                    if not is_inner_edge: valid_contours.append(cnt)

        if not valid_contours:
            st.error("❌ 綜合幾何防線審查後無剩餘有效區塊，請嘗試調降左側過濾器門檻。")
            return

        st.subheader("📊 引擎演算空間可視化成果")
        col1, col2, col3 = st.columns(3)
        
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        display_img = img_rgb.copy()
        cv2.drawContours(display_img, valid_contours, -1, (255, 0, 0), 3)
        ax1.imshow(display_img)
        ax1.axis("off")
        ax1.set_title("1. Contour Extraction")
        col1.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.invert_yaxis() 
        ax2.set_aspect('equal', adjustable='box')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.set_title(f"2. Digital Map ({unit_str})")

        contour_data = []
        colors = plt.cm.jet(np.linspace(0.1, 0.9, max(len(valid_contours), 1)))
        scale_factor = real_unit_per_pixel if real_unit_per_pixel is not None else 1.0

        for i, cnt in enumerate(valid_contours):
            M = cv2.moments(cnt)
            if M['m00'] != 0: cx, cy = (M['m10'] / M['m00']) * scale_factor, (M['m01'] / M['m00']) * scale_factor
            else: cx, cy = cnt[:, 0, 0].mean() * scale_factor, cnt[:, 0, 1].mean() * scale_factor
                
            area = cv2.contourArea(cnt) * (scale_factor**2)
            approx = cv2.approxPolyDP(cnt, 0.005 * cv2.arcLength(cnt, True), True)
            x_poly = np.append(approx[:, 0, 0].astype(float) * scale_factor, approx[0, 0, 0].astype(float) * scale_factor)
            y_poly = np.append(approx[:, 0, 1].astype(float) * scale_factor, approx[0, 0, 1].astype(float) * scale_factor)
                
            contour_data.append({"Block_ID": i+1, f"Area ({area_unit})": round(area, 2), f"Centroid_X ({unit_str})": round(cx, 2), f"Centroid_Y ({unit_str})": round(cy, 2), "Nodes": len(approx)})
            ax2.plot(x_poly, y_poly, color=colors[i], linewidth=2.5)
            ax2.text(cx, cy, f"#{i+1}", fontsize=10, weight='bold')

        col2.pyplot(fig2)

        fig3, ax3 = plt.subplots(figsize=(6, 6))
        blocks = [f"#{d['Block_ID']}" for d in contour_data]
        areas = [d[f"Area ({area_unit})"] for d in contour_data]
        ax3.bar(blocks, areas, color=colors, alpha=0.7)
        ax3.set_ylabel(f"Area ({area_unit})")
        ax3.set_title(f"3. Area Distribution")
        if len(valid_contours) > 15: ax3.set_xticks([])
        col3.pyplot(fig3)

        st.subheader("📋 數位化地籍土地物理報表")
        df = pd.DataFrame(contour_data)
        st.dataframe(df, use_container_width=True)
        
        csv_data = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(label="📥 匯出 GIS 數位地籍觀測 CSV", data=csv_data, file_name='digitized_contours.csv', mime='text/csv')

    # ------------------------------------------
    # Signal 連續訊號模式 (極致專業版)
    # ------------------------------------------
    elif target_mode == 'signal':
        st.info("📈 處理中：正在掃描連續訊號與套用自訂參數...")

        if clean_mask is None or cv2.countNonZero(clean_mask) == 0:
            st.warning("⚠️ 警告：幾何過濾參數設定過於嚴苛，目前圖面無有效訊號。")
            return
            
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
        if num_labels <= 1:
            st.error("❌ 找不到有效的訊號線！請檢查過濾參數。")
            return
            
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        clean_mask = (labels == largest_label).astype(np.uint8) * 255
        
        skeleton = skeletonize(clean_mask > 0)
        y_coords, x_coords = np.where(skeleton)
        
        if len(x_coords) == 0:
            st.error("❌ 無法萃取出有效曲線，請確認圖片清晰度。")
            return

        orig_x_coords, orig_y_coords = x_coords.copy(), y_coords.copy()
        
        coords_matrix = np.vstack((x_coords, y_coords))
        cov_matrix = np.cov(coords_matrix)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        primary_vector = eigenvectors[:, np.argmax(eigenvalues)]
        
        is_rotated = False
        if abs(primary_vector[1]) > abs(primary_vector[0]):
            x_math, y_math = y_coords, -x_coords
            is_rotated = True
        else:
            x_math, y_math = x_coords, -y_coords

        sort_idx = np.argsort(x_math)
        x_raw, y_raw = x_math[sort_idx], y_math[sort_idx]
        img_plot_x, img_plot_y = orig_x_coords[sort_idx], orig_y_coords[sort_idx]

        unique_x = np.unique(x_raw)
        y_raw = np.array([np.mean(y_raw[x_raw == x]) for x in unique_x])
        x_raw = unique_x

        # 單位轉換
        sig_unit_str = "Pixel"
        x_local, y_local = x_raw - x_raw[0], y_raw - y_raw[0]
        
        if real_unit_per_pixel is not None:
            x_local = x_local * real_unit_per_pixel
            y_local = y_local * real_unit_per_pixel
            sig_unit_str = unit_str
            
        # 套用使用者自訂原點
        x_local = x_local + origin_x
        y_local = y_local + origin_y

        prominence_val = max((y_local.max() - y_local.min()) * 0.1, 1)
        y_smooth = gaussian_filter1d(y_local, sigma=5)
        peaks, _ = find_peaks(y_smooth, prominence=prominence_val) 
        valleys, _ = find_peaks(-y_smooth, prominence=prominence_val)

        st.subheader("📊 連續訊號特徵萃取與數位孿生")
        fig = plt.figure(figsize=(16, 9)) 
        
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(np.hstack((img_rgb, cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2RGB))))
        ax1.plot(img_plot_x, img_plot_y, color='red', linewidth=3) 
        ax1.set_title("1. Extraction & Marker Detection")
        ax1.axis("off")

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(x_local, y_local, color='black', linewidth=3)
        ax2.set_title(f"2. Extrema Analysis")
        ax2.set_xlabel(f"{x_axis_name} ({sig_unit_str})")
        ax2.set_ylabel(f"{y_axis_name} ({sig_unit_str})")
        
        # 套用圖表顯示範圍
        if use_custom_limits:
            ax2.set_xlim(x_min_limit, x_max_limit)
            ax2.set_ylim(y_min_limit, y_max_limit)
        else:
            ax2.set_aspect('equal', adjustable='datalim')
            y_margin = max((y_local.max() - y_local.min()) * 0.25, 10)
            ax2.set_ylim(y_local.min() - y_margin, y_local.max() + y_margin)
            
        ax2.grid(True, linestyle='--', alpha=0.6)

        raw_points = [{"idx": 0, "name": "Start", "color": "green", "offset": (-20, 15)}, {"idx": len(x_local)-1, "name": "End", "color": "green", "offset": (20, 15)}]
        for p_idx in peaks: raw_points.append({"idx": p_idx, "name": "Max", "color": "red", "offset": (0, 20)})
        for v_idx in valleys: raw_points.append({"idx": v_idx, "name": "Min", "color": "blue", "offset": (0, -25)})

        final_annotations, dist_threshold = [], (x_local.max() - x_local.min()) * 0.05
        for p in raw_points:
            merged = False
            for existing in final_annotations:
                if np.hypot(x_local[p["idx"]] - x_local[existing["idx"]], y_local[p["idx"]] - y_local[existing["idx"]]) < dist_threshold:
                    existing["name"] += f" & {p['name']}"
                    if p["name"] in ["Max", "Min"]: existing["color"], existing["offset"] = p["color"], p["offset"]
                    merged = True; break
            if not merged: final_annotations.append(p)

        for ann in final_annotations:
            idx = ann["idx"]
            ax2.scatter(x_local[idx], y_local[idx], color=ann["color"], s=60, zorder=5)
            ax2.annotate(f"{ann['name']}\n({x_local[idx]:.1f}, {y_local[idx]:.1f})", (x_local[idx], y_local[idx]), textcoords="offset points", xytext=ann["offset"], ha='center', fontsize=10, color=ann["color"], weight='bold')

        ax3 = fig.add_subplot(2, 2, 3)
        best_deg, best_r2, best_coeffs, best_y_poly = 3, 0, None, None
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for deg in range(3, 16):
                coeffs = np.polyfit(x_local, y_local, deg)
                y_fit = np.poly1d(coeffs)(x_local)
                r2 = 1 - (np.sum((y_local - y_fit) ** 2) / np.sum((y_local - np.mean(y_local)) ** 2))
                if r2 > best_r2: best_r2, best_coeffs, best_deg, best_y_poly = r2, coeffs, deg, y_fit
                if r2 >= 0.985: break 

        tck, u = splprep([x_local, y_local], s=len(x_local)*(3), k=3)
        x_bspl, y_bspl = splev(np.linspace(0, 1, 1000), tck)

        ax3.plot(x_local, y_local, 'k.', markersize=2, alpha=0.3, label='Raw Data')
        ax3.plot(x_local, best_y_poly, 'r-', linewidth=2, label=f'Polynomial (Deg {best_deg})')
        ax3.plot(x_bspl, y_bspl, 'b--', linewidth=2, label='B-Spline Fit')
        ax3.set_title(f"3. Geometrical Tracking")
        ax3.set_xlabel(f"{x_axis_name}")
        ax3.set_ylabel(f"{y_axis_name}")
        
        if use_custom_limits:
            ax3.set_xlim(x_min_limit, x_max_limit)
            ax3.set_ylim(y_min_limit, y_max_limit)
            
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, linestyle='--')

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis("off")
        
        info_text = (
            f"=== MEASUREMENT STATUS ===\n"
            f"Scale Unit: {sig_unit_str}\n"
            f"Precision (R-sq) = {best_r2:.4f}\n"
        )
        ax4.text(0.1, 0.5, info_text, transform=ax4.transAxes, ha='left', va='center', fontsize=12, family='monospace')

        st.pyplot(fig)
        
        # 📋 CSV 自動均勻取樣與匯出 (套用自訂頻率 ΔX)
        step_size = sample_step  
        x_uniform = np.arange(x_local.min(), x_local.max() + step_size*0.001, step_size) 
        if len(x_uniform) > 0 and x_uniform[-1] > x_local.max(): x_uniform = x_uniform[:-1]
        y_uniform = np.interp(x_uniform, x_bspl, y_bspl)
        
        total_samples = len(x_uniform)
        st.subheader(f"📋 數位孿生 B-Spline 座標數據 (共 {total_samples} 點, ΔX = {step_size})")
        
        sig_df = pd.DataFrame({
            f"{x_axis_name} ({sig_unit_str})": x_uniform,
            f"{y_axis_name} ({sig_unit_str})": y_uniform
        })
        
        st.dataframe(sig_df, use_container_width=True)
        csv_data = sig_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(label="📥 匯出連續訊號 CSV", data=csv_data, file_name='digitized_signal.csv', mime='text/csv')

# ==========================================
# 5. 程式執行入口
# ==========================================
if run_btn:
    if uploaded_file is not None:
        with st.spinner('PocketLab 2D 核心引擎高速運算中...'):
            run_pocketlab_engine(uploaded_file.getvalue(), known_pixel_length)
    else:
        st.error("⚠️ 請先在左側控制台依序上傳圖片檔案！")
elif uploaded_file is None:
    st.info("👈 歡迎使用！請先在左側側邊欄控制台「上傳圖資照片」以啟動測繪戰情室。")
