import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splprep, splev
from scipy.spatial import distance  # 🌟 新增：用於自動誤差比對的函式庫
import math
import io
import warnings
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

# ==========================================
# 1. 網頁基本設定 (寬螢幕模式)
# ==========================================
st.set_page_config(page_title="PocketLab 2D Engine", layout="wide", page_icon="🗺️")

st.title("🗺️ PocketLab 2D Engine - 空間測繪與地籍分析系統")

# ==========================================
# 🌟 2. 專案展示與精度驗證區塊 (折疊面板)
# ==========================================
with st.expander("🎬 3S 創客競賽專案展示 & 圖根點精度驗證 (點擊展開)", expanded=False):
    st.subheader("🎥 系統操作展示影片")
    st.info("💡 評審您好，下方為本系統的核心功能展示影片：")
    
    # 📝 請將下方的網址換成你們錄好的 YouTube 影片網址
    st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ") 

    st.divider()

    st.subheader("🎯 基礎圖根點誤差計算機")
    st.markdown("快速驗證單一座標點誤差，數學模型：$Error = \sqrt{\Delta X^2 + \Delta Y^2}$")

    col1, col2 = st.columns(2)
    with col1:
        gt_x = st.number_input("📍 實際圖根點 X (m)", value=100.000, format="%.3f")
        gt_y = st.number_input("📍 實際圖根點 Y (m)", value=200.000, format="%.3f")
    with col2:
        ai_x = st.number_input("🤖 AI 辨識 X (m)", value=100.050, format="%.3f")
        ai_y = st.number_input("🤖 AI 辨識 Y (m)", value=199.960, format="%.3f")

    dx, dy = ai_x - gt_x, ai_y - gt_y
    error = math.hypot(dx, dy)

    st.markdown("#### 📊 計算結果")
    c1, c2, c3 = st.columns(3)
    c1.metric("ΔX (X軸偏差)", f"{dx:.3f} m")
    c2.metric("ΔY (Y軸偏差)", f"{dy:.3f} m")
    c3.metric("平面點位誤差", f"{error:.3f} m", delta_color="inverse")

# ==========================================
# 3. 側邊欄 (Sidebar) - 參數控制台
# ==========================================
with st.sidebar:
    st.header("🛠️ 測量環境設定")
    uploaded_file = st.file_uploader("📂 請上傳地籍圖或波形圖 (JPG/PNG)", type=['jpg', 'png', 'jpeg'])
    target_type = st.selectbox("🎯 選擇分析模式", ['contour (封閉地籍區塊)', 'signal (連續訊號分析)'])
    target_mode = 'contour' if 'contour' in target_type else 'signal'
    
    st.divider()
    st.header("📏 比例尺與單位映射")
    map_scale = st.number_input("工程比例尺分母 (如 1:5000 請輸入 5000)", min_value=1.0, value=1.0, step=100.0)
    
    calibration_mode = st.radio("校準方式", ["自動 (ArUco標籤)", "手動 (自訂像素長度)"])
    marker_real_size_mm, manual_scale_mm, known_pixel_length = 50.0, None, None
    marker_pad_ratio = 0.3 # 預設防護結界比例
    
    if calibration_mode == "自動 (ArUco標籤)":
        marker_real_size_mm = st.number_input("ArUco 標籤實體大小 (mm)", value=50.0)
        marker_pad_ratio = st.slider("🛡️ 標籤周圍防護結界大小 (%)", 0, 100, 30, 1) / 100.0
        st.caption("💡 若標籤離目標圖形太近被誤刪，可調小此數值(如 5%)。")
    else:
        manual_scale_mm = st.number_input("對照實體長度 (mm)", value=20.0)

    st.divider()
    
    if target_mode == 'contour':
        st.header("🛡️ 幾何過濾裝甲設定")
        preset_scene = st.selectbox(
            "📋 選擇地籍圖資情境預設組",
            ["自訂參數調整", "高解析電子原始檔", "實體紙張拍照/老舊掃描檔", "微型畸零地觀測"]
        )
        if preset_scene == "高解析電子原始檔": init_area, init_circ, init_border = 2000, 0.18, 10
        elif preset_scene == "實體紙張拍照/老舊掃描檔": init_area, init_circ, init_border = 1200, 0.12, 20
        elif preset_scene == "微型畸零地觀測": init_area, init_circ, init_border = 400, 0.04, 5
        else: init_area, init_circ, init_border = 1500, 0.15, 15

        min_area_px = st.slider("最小面積門檻 (px²)", 100, 5000, init_area, 50)
        circularity_threshold = st.slider("最小緊緻度 / 圓形度", 0.01, 1.0, init_circ, 0.01)
        border_margin = st.slider("邊界排除距離 (px)", 0, 100, init_border, 5)

    elif target_mode == 'signal':
        st.header("📈 訊號空間與採樣設定")
        x_axis_name = st.text_input("X 軸名稱", value="水平距離 (Distance)")
        y_axis_name = st.text_input("Y 軸名稱", value="垂直偏轉 (Deflection)")
        origin_x = st.number_input("X 起始座標 (原點)", value=0.0, step=1.0)
        origin_y = st.number_input("Y 起始座標 (原點)", value=0.0, step=1.0)
        sample_step = st.number_input("取樣間距 ΔX", min_value=0.001, value=0.5, step=0.1, format="%.3f")
        
        use_custom_limits = st.checkbox("啟用手動限制圖表邊界", value=False)
        if use_custom_limits:
            st.caption("💡 留白不輸入數字，代表該軸維持「自動縮放」。")
            col1, col2 = st.columns(2)
            with col1:
                x_min_limit = st.number_input("X 軸最小", value=None, placeholder="自動")
                y_min_limit = st.number_input("Y 軸最小", value=None, placeholder="自動")
            with col2:
                x_max_limit = st.number_input("X 軸最大", value=None, placeholder="自動")
                y_max_limit = st.number_input("Y 軸最大", value=None, placeholder="自動")
        else:
            x_min_limit, x_max_limit, y_min_limit, y_max_limit = None, None, None, None

    st.divider()
    st.header("⚙️ 引擎運算效能設定")
    engine_resolution = st.select_slider(
        "分析畫質解析度 (Max Pixels)",
        options=[800, 1200, 2000, 3000],
        value=2000,
        format_func=lambda x: "800px (極速/自動除噪)" if x==800 else "1200px (標準/平衡)" if x==1200 else "2000px (高畫質/精細)" if x==2000 else "3000px (超高解析/耗能)"
    )
    st.caption("💡 降低畫質可加快運算並過濾紙張微小雜訊；提高畫質可保留細小圖案。")

    run_btn = st.button("🚀 開始啟動分析引擎", type="primary", use_container_width=True)

# ==========================================
# 4. 手動大圖點擊校準區
# ==========================================
if calibration_mode == "手動 (自訂像素長度)" and uploaded_file is not None:
    st.subheader("🎯 互動式像素長度點擊校準區")
    st.caption("請直接在下方圖片中，點擊地籍圖上比例尺線段的「起點」與「終點」：")
    
    pil_img = Image.open(uploaded_file)
    orig_w, orig_h = pil_img.size
    ui_scale = 800.0 / max(orig_w, orig_h) if max(orig_w, orig_h) > 800 else 1.0
    ui_img = pil_img.resize((int(orig_w * ui_scale), int(orig_h * ui_scale)))
    mapping_ratio = (2000.0 / max(orig_w, orig_h) if max(orig_w, orig_h) > 2000 else 1.0) / ui_scale

    value = streamlit_image_coordinates(ui_img, key="main_canvas")

    if "points" not in st.session_state: st.session_state["points"] = []
    if value is not None:
        point = (value["x"] * mapping_ratio, value["y"] * mapping_ratio)
        if len(st.session_state["points"]) == 0 or point != st.session_state["points"][-1]:
            st.session_state["points"].append(point)

    if len(st.session_state["points"]) >= 2:
        p1, p2 = st.session_state["points"][-2], st.session_state["points"][-1]
        calc_dist = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
        st.success(f"🎯 點擊測量成功！對應圖上距離: {calc_dist:.2f} px")
        known_pixel_length = calc_dist
    else:
        st.info("💡 尚未完成兩點點擊，使用預設值 100 px。")
        known_pixel_length = 100.0

# ==========================================
# 5. 核心引擎
# ==========================================
# 🚀 CTO 修正：將 max_dim 作為參數傳入
def run_pocketlab_engine(img_bytes, k_pixel, pad_ratio, max_dim):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    h, w = img.shape[:2]
    # 這裡不再寫死 2000，而是套用使用者在網頁上選擇的 max_dim
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
            mm_per_pixel = marker_real_size_mm / np.linalg.norm(c[0] - c[1])
            st.success(f"📊 後端確認採用 ArUco 自動校準解析度: 1px = {mm_per_pixel:.4f} mm")
        else:
            st.warning("⚠️ 系統預設為相對像素 (Pixel) 模式。")

    if mm_per_pixel is not None:
        if map_scale > 1.0:
            real_unit_per_pixel, unit_str, area_unit = (mm_per_pixel * map_scale) / 1000.0, "m", "m²"
            st.info(f"🌍 GIS 映射成功：1px = {real_unit_per_pixel:.4f} 公尺")
        else:
            real_unit_per_pixel, unit_str, area_unit = mm_per_pixel, "mm", "mm²"
    else:
        real_unit_per_pixel, unit_str, area_unit = None, "Pixel", "px²"

    binary_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    if ids is not None and len(corners) > 0:
        for c in corners:
            pts = c[0]
            x_min, x_max, y_min, y_max = int(np.min(pts[:, 0])), int(np.max(pts[:, 0])), int(np.min(pts[:, 1])), int(np.max(pts[:, 1]))
            pad_x, pad_y = int((x_max - x_min) * pad_ratio), int((y_max - y_min) * pad_ratio)
            cv2.rectangle(binary_mask, (max(0, x_min - pad_x), max(0, y_min - pad_y)), (min(w, x_max + pad_x), min(h, y_max + pad_y)), 0, -1)

    clean_mask = cv2.morphologyEx(cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2)), iterations=1), cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    # --- Contour 模式 ---
    if target_mode == 'contour':
        contours, hierarchy = cv2.findContours(clean_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        valid_contours = []
        max_area_px = clean_mask.shape[0] * clean_mask.shape[1] * 0.5 
        
        if hierarchy is not None:
            for i, cnt in enumerate(contours):
                area_px = cv2.contourArea(cnt)
                if min_area_px < area_px < max_area_px:
                    x, y, w_box, h_box = cv2.boundingRect(cnt)
                    if x < border_margin or y < border_margin or (x + w_box) > (w - border_margin) or (y + h_box) > (h - border_margin): continue 
                    if not (3 <= len(cv2.approxPolyDP(cnt, 0.005 * cv2.arcLength(cnt, True), True)) <= 25): continue
                    perimeter_px = cv2.arcLength(cnt, True)
                    if perimeter_px == 0 or ((4 * np.pi * area_px) / (perimeter_px ** 2)) < circularity_threshold: continue
                    if hierarchy[0][i][3] != -1 and (area_px / cv2.contourArea(contours[hierarchy[0][i][3]])) > 0.8: continue
                    valid_contours.append(cnt)

        if not valid_contours:
            st.error("❌ 無剩餘有效區塊，請調降門檻。")
            return

        st.subheader("📊 引擎演算空間可視化成果")
        col1, col2, col3 = st.columns(3)
        
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        display_img = img_rgb.copy()
        cv2.drawContours(display_img, valid_contours, -1, (255, 0, 0), 3)
        ax1.imshow(display_img)
        ax1.axis("off")
        col1.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.invert_yaxis() 
        ax2.set_aspect('equal', adjustable='box')
        ax2.grid(True, linestyle='--', alpha=0.6)

        contour_data = []
        colors = plt.cm.jet(np.linspace(0.1, 0.9, max(len(valid_contours), 1)))
        scale_factor = real_unit_per_pixel if real_unit_per_pixel is not None else 1.0

        for i, cnt in enumerate(valid_contours):
            M = cv2.moments(cnt)
            cx, cy = ((M['m10'] / M['m00']) * scale_factor, (M['m01'] / M['m00']) * scale_factor) if M['m00'] != 0 else (cnt[:, 0, 0].mean() * scale_factor, cnt[:, 0, 1].mean() * scale_factor)
            area = cv2.contourArea(cnt) * (scale_factor**2)
            approx = cv2.approxPolyDP(cnt, 0.005 * cv2.arcLength(cnt, True), True)
            x_poly = np.append(approx[:, 0, 0].astype(float) * scale_factor, approx[0, 0, 0].astype(float) * scale_factor)
            y_poly = np.append(approx[:, 0, 1].astype(float) * scale_factor, approx[0, 0, 1].astype(float) * scale_factor)
            
            contour_data.append({"Block_ID": i+1, f"Area ({area_unit})": round(area, 2), f"Centroid_X ({unit_str})": round(cx, 2), f"Centroid_Y ({unit_str})": round(cy, 2)})
            ax2.plot(x_poly, y_poly, color=colors[i], linewidth=2.5)
            ax2.text(cx, cy, f"#{i+1}", fontsize=10, weight='bold')
        col2.pyplot(fig2)

        fig3, ax3 = plt.subplots(figsize=(6, 6))
        ax3.bar([f"#{d['Block_ID']}" for d in contour_data], [d[f"Area ({area_unit})"] for d in contour_data], color=colors, alpha=0.7)
        col3.pyplot(fig3)

        df = pd.DataFrame(contour_data)
        st.dataframe(df, use_container_width=True)
        st.download_button(label="📥 匯出 GIS 數位地籍觀測 CSV", data=df.to_csv(index=False).encode('utf-8-sig'), file_name='digitized_contours.csv', mime='text/csv')

        # 🌟 自動化精度驗證模組 (僅在地籍圖模式下啟用)
        st.divider()
        st.subheader("🎯 AI 辨識圖根點自動精度驗證 (Auto-Matching)")
        st.markdown("系統將透過 **最近鄰演算法 (Nearest Neighbor)**，自動比對實際量測座標與 AI 數位孿生座標，並計算 RMSE。")

        gt_file = st.file_uploader("📂 上傳標準圖根點座標 (CSV格式，需含 X_m, Y_m 欄位)", type=['csv'], key="gt_uploader")
        
        if gt_file is not None:
            gt_df = pd.read_csv(gt_file)
            if 'X_m' in gt_df.columns and 'Y_m' in gt_df.columns:
                gt_coords = gt_df[['X_m', 'Y_m']].values
                
                if f'Centroid_X ({unit_str})' in df.columns:
                    ai_coords = df[[f'Centroid_X ({unit_str})', f'Centroid_Y ({unit_str})']].values
                    
                    dist_matrix = distance.cdist(gt_coords, ai_coords, 'euclidean')
                    closest_ai_indices = np.argmin(dist_matrix, axis=1)
                    min_distances = np.min(dist_matrix, axis=1)
                    
                    error_report = []
                    for i, gt_pt in enumerate(gt_coords):
                        ai_pt = ai_coords[closest_ai_indices[i]]
                        error_report.append({
                            "ID": i + 1, "實際X": round(gt_pt[0], 3), "實際Y": round(gt_pt[1], 3),
                            "AI X": round(ai_pt[0], 3), "AI Y": round(ai_pt[1], 3),
                            "ΔX": round(ai_pt[0]-gt_pt[0], 3), "ΔY": round(ai_pt[1]-gt_pt[1], 3),
                            "點位誤差": round(min_distances[i], 3)
                        })
                        
                    st.dataframe(pd.DataFrame(error_report), use_container_width=True)
                    
                    e1, e2 = st.columns(2)
                    e1.metric("平均點位誤差", f"{np.mean(min_distances):.3f} {unit_str}", delta_color="inverse")
                    e2.metric("均方根誤差 (RMSE)", f"{np.sqrt(np.mean(min_distances**2)):.3f} {unit_str}", delta_color="inverse")
                else:
                    st.warning("⚠️ 需設定為公尺(m)單位映射，才能進行比對。")
            else:
                st.error("❌ CSV 缺少 'X_m' 或 'Y_m' 欄位。")

    # --- Signal 模式 ---
    elif target_mode == 'signal':
        st.info("📈 處理中：正在掃描連續訊號...")
        if clean_mask is None or cv2.countNonZero(clean_mask) == 0: return st.warning("⚠️ 圖面無有效訊號。")
            
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
        if num_labels <= 1: return st.error("❌ 找不到訊號線！")
            
        clean_mask = (labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8) * 255
        y_coords, x_coords = np.where(skeletonize(clean_mask > 0))
        if len(x_coords) == 0: return st.error("❌ 無法萃取曲線。")

        orig_x, orig_y = x_coords.copy(), y_coords.copy()
        evals, evecs = np.linalg.eig(np.cov(np.vstack((x_coords, y_coords))))
        primary_vector = evecs[:, np.argmax(evals)]
        
        x_math, y_math = (y_coords, -x_coords) if abs(primary_vector[1]) > abs(primary_vector[0]) else (x_coords, -y_coords)
        sort_idx = np.argsort(x_math)
        x_raw, y_raw = x_math[sort_idx], y_math[sort_idx]
        
        unique_x = np.unique(x_raw)
        y_raw = np.array([np.mean(y_raw[x_raw == x]) for x in unique_x])
        x_raw = unique_x

        sig_unit_str = "Pixel"
        x_local, y_local = x_raw - x_raw[0], y_raw - y_raw[0]
        
        if real_unit_per_pixel is not None:
            x_local, y_local, sig_unit_str = x_local * real_unit_per_pixel, y_local * real_unit_per_pixel, unit_str
            
        x_local, y_local = x_local + origin_x, y_local + origin_y
        y_smooth = gaussian_filter1d(y_local, sigma=5)
        p_val = max((y_local.max() - y_local.min()) * 0.1, 1)
        peaks, _ = find_peaks(y_smooth, prominence=p_val) 
        valleys, _ = find_peaks(-y_smooth, prominence=p_val)

        st.subheader("📊 連續訊號特徵萃取與數位孿生")
        fig = plt.figure(figsize=(16, 9)) 
        
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(np.hstack((img_rgb, cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2RGB))))
        ax1.plot(orig_x[sort_idx], orig_y[sort_idx], color='red', linewidth=3) 
        ax1.axis("off")

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(x_local, y_local, color='black', linewidth=3)
        ax2.set_xlabel(f"{x_axis_name} ({sig_unit_str})"); ax2.set_ylabel(f"{y_axis_name} ({sig_unit_str})")
        
        if use_custom_limits and (x_min_limit is not None or x_max_limit is not None):
            ax2.set_xlim(left=x_min_limit, right=x_max_limit)
            
        if use_custom_limits and (y_min_limit is not None or y_max_limit is not None):
            ax2.set_ylim(bottom=y_min_limit, top=y_max_limit)
        else:
            y_margin = max((y_local.max() - y_local.min()) * 0.25, 10)
            ax2.set_ylim(y_local.min() - y_margin, y_local.max() + y_margin)
            
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        raw_points = [{"idx": 0, "name": "Start", "c": "green"}, {"idx": len(x_local)-1, "name": "End", "c": "green"}]
        for p in peaks: raw_points.append({"idx": p, "name": "Max", "c": "red"})
        for v in valleys: raw_points.append({"idx": v, "name": "Min", "c": "blue"})
        
        for p in raw_points:
            ax2.scatter(x_local[p["idx"]], y_local[p["idx"]], color=p["c"], s=60, zorder=5)

        ax3 = fig.add_subplot(2, 2, 3)
        
        # 🚀 CTO 修正 1：將平滑係數 (s) 降至 0.01，大幅提高波峰與波谷的貼合精準度 (解決切西瓜誤差)
        tck, u = splprep([x_local, y_local], s=len(x_local)*0.01, k=3)
        x_bspl, y_bspl = splev(np.linspace(0, 1, 1000), tck)

        ax3.plot(x_local, y_local, 'k.', alpha=0.3)
        ax3.plot(x_bspl, y_bspl, 'b--', linewidth=2)
        ax3.set_xlabel(f"{x_axis_name}"); ax3.set_ylabel(f"{y_axis_name}")
        
        if use_custom_limits and (x_min_limit is not None or x_max_limit is not None):
            ax3.set_xlim(left=x_min_limit, right=x_max_limit)
            
        if use_custom_limits and (y_min_limit is not None or y_max_limit is not None):
            ax3.set_ylim(bottom=y_min_limit, top=y_max_limit)
            
        ax3.grid(True, linestyle='--')
        
        st.pyplot(fig)
        
        x_uniform = np.arange(x_local.min(), x_local.max() + sample_step*0.001, sample_step) 
        if len(x_uniform) > 0 and x_uniform[-1] > x_local.max(): x_uniform = x_uniform[:-1]
        y_uniform = np.interp(x_uniform, x_bspl, y_bspl)
        
        sig_df = pd.DataFrame({f"{x_axis_name} ({sig_unit_str})": x_uniform, f"{y_axis_name} ({sig_unit_str})": y_uniform})
        
        # 🚀 CTO 修正 2：萃取 B-Spline 控制節點 (Control Nodes)
        ctrl_df = pd.DataFrame({
            f"Node_X ({sig_unit_str})": tck[1][0], 
            f"Node_Y ({sig_unit_str})": tck[1][1]
        })

        st.subheader(f"📋 數位孿生與數學建模數據下載")
        
        # 使用 Streamlit 的 columns 來並排顯示兩個下載區塊，讓 UI 更專業
        dl_col1, dl_col2 = st.columns(2)
        
        with dl_col1:
            st.markdown(f"**① 均勻採樣序列 (共 {len(x_uniform)} 點)**")
            st.dataframe(sig_df, use_container_width=True)
            st.download_button("📥 匯出均勻連續訊號 CSV", data=sig_df.to_csv(index=False).encode('utf-8-sig'), file_name='digitized_signal.csv', mime='text/csv')
            
        with dl_col2:
            st.markdown(f"**② B-Spline 控制節點 (共 {len(tck[1][0])} 點)**")
            st.dataframe(ctrl_df, use_container_width=True)
            st.download_button("📥 匯出 B-Spline 控制節點 CSV", data=ctrl_df.to_csv(index=False).encode('utf-8-sig'), file_name='bspline_control_nodes.csv', mime='text/csv')

if run_btn:
    if uploaded_file is not None:
        # 🚀 CTO 修正：將 engine_resolution 傳給引擎
        with st.spinner('PocketLab 2D 核心引擎高速運算中...'): run_pocketlab_engine(uploaded_file.getvalue(), known_pixel_length, marker_pad_ratio, engine_resolution)
    else: st.error("⚠️ 請先上傳圖片！")
