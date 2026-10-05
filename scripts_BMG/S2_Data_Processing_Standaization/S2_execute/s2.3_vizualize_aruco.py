#!/usr/bin/env python3
"""
Trực quan hóa quỹ đạo SLAM so với gốc tọa độ ArUco Marker.

Chức năng:
  - Tự động nhận diện ArUco marker từ ảnh RGB để làm gốc tọa độ (0,0,0).
  - Ánh xạ quỹ đạo Camera và Tool (SLAM) về hệ tọa độ của Marker.
  - Hỗ trợ xử lý 1 dataset lẻ hoặc chạy hàng loạt.
  - Chế độ chạy hàng loạt tích hợp Tool đánh giá (Tốt/Xấu) qua phím bấm.
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import pandas as pd
import cv2
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('TkAgg') # Bắt buộc dùng TkAgg cho giao diện tương tác
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
S1_OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "S1_Data_Collection", "S1_output"))
S2_OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "S2_output_slam"))
CALIB_JSON_FILE = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "S0_Camera_Calibration", "S0_output", "Date_18092026", "calibration_matrices_RS_camera", "eye_in_hand_result.json"))
INTRINSICS_FILE = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "S0_Camera_Calibration", "S0_output", "Date_18092026", "calibration_matrices_RS_camera", "camera_intrinsics.json"))

MARKER_LENGTH = 0.100  # 100mm = 0.1m

ARUCO_DICTS = {
    "DICT_4X4_50": aruco.DICT_4X4_50,
    "DICT_5X5_250": aruco.DICT_5X5_250,
    "DICT_6X6_250": aruco.DICT_6X6_250,
    "DICT_7X7_1000": aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": aruco.DICT_ARUCO_ORIGINAL,
    "APRILTAG_16h5": getattr(aruco, 'DICT_APRILTAG_16h5', 0),
    "APRILTAG_25h9": getattr(aruco, 'DICT_APRILTAG_25h9', 0),
    "APRILTAG_36h10": getattr(aruco, 'DICT_APRILTAG_36h10', 0),
    "APRILTAG_36h11": getattr(aruco, 'DICT_APRILTAG_36h11', 0)
}

# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def load_json_matrix(file_path, key):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return np.array(data[key])

def load_camera_intrinsics(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    K = np.array(data["camera_matrix"])
    D = np.array(data["dist_coeffs"])[0]
    return K, D

def load_camera_data(file_path):
    data = np.loadtxt(file_path)
    timestamps = data[:, 0]
    T_list = []
    for row in data:
        t = row[1:4]
        q = row[4:8]
        rot = R.from_quat(q).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = t
        T_list.append(T)
    return timestamps, T_list

def load_rgb_mapping(file_path):
    mapping = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split()
            if len(parts) >= 2:
                ts = float(parts[0])
                mapping[ts] = parts[1]
    return mapping

def average_poses(T_list):
    """Lấy trung bình của nhiều ma trận 4x4"""
    if len(T_list) == 0: return np.eye(4)
    if len(T_list) == 1: return T_list[0]
        
    t_mean = np.mean([T[:3, 3] for T in T_list], axis=0)
    rots = R.from_matrix([T[:3, :3] for T in T_list])
    R_mean = rots.mean().as_matrix()
    
    T_mean = np.eye(4)
    T_mean[:3, :3] = R_mean
    T_mean[:3, 3] = t_mean
    return T_mean

def detect_marker_pose(image_path, K, D):
    frame = cv2.imread(image_path)
    if frame is None: return None
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    is_cv_v47 = hasattr(aruco, 'ArucoDetector')
    
    for _, dict_id in ARUCO_DICTS.items():
        try:
            dictionary = aruco.getPredefinedDictionary(dict_id)
            if is_cv_v47:
                detector_params = aruco.DetectorParameters()
                detector = aruco.ArucoDetector(dictionary, detector_params)
                corners, ids, rejected = detector.detectMarkers(gray)
            else:
                detector_params = aruco.DetectorParameters_create()
                corners, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=detector_params)
                
            if ids is not None and len(ids) > 0:
                L = MARKER_LENGTH / 2.0
                obj_points = np.array([
                    [-L,  L, 0], [ L,  L, 0], [ L, -L, 0], [-L, -L, 0]
                ], dtype=np.float32)
                
                corner = corners[0][0]
                ret, rvec, tvec = cv2.solvePnP(obj_points, corner, K, D)
                if ret:
                    R_cam_marker, _ = cv2.Rodrigues(rvec)
                    T_cam_to_marker = np.eye(4)
                    T_cam_to_marker[:3, :3] = R_cam_marker
                    T_cam_to_marker[:3, 3] = tvec.squeeze()
                    return T_cam_to_marker
        except Exception:
            continue
    return None

# ═══════════════════════════════════════════════════════════════
# SINGLE DATASET PROCESSOR
# ═══════════════════════════════════════════════════════════════
def process_single_dataset(rel_path, num_frames=100, no_show=False):
    camera_traj_file = os.path.join(S2_OUTPUT_DIR, rel_path, "SmoothedCameraTrajectory.txt")
    rgb_mapping_file = os.path.join(S1_OUTPUT_DIR, rel_path, "rgb.txt")
    dataset_dir = os.path.join(S1_OUTPUT_DIR, rel_path)
    output_dir = os.path.join(S2_OUTPUT_DIR, rel_path)
    dataset_name = os.path.basename(rel_path)
    
    if not os.path.exists(camera_traj_file):
        print(f"[-] Bỏ qua {rel_path}: Chưa có SmoothedCameraTrajectory.txt")
        return False
        
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "Mapped_SLAM_Trajectory_Marker.csv")
    out_img = os.path.join(output_dir, "SLAM_in_Marker_Plot.png")

    # Nếu đã có CSV thì chỉ cần load và vẽ (tiết kiệm thời gian)
    if os.path.exists(out_csv):
        print(f"[+] Dữ liệu {dataset_name} đã được xử lý từ trước.")
        if not no_show:
            _plot_from_csv(out_csv, out_img, dataset_name, no_show)
        return True

    try:
        K, D = load_camera_intrinsics(INTRINSICS_FILE)
        T_cam2tool = load_json_matrix(CALIB_JSON_FILE, "T_cam_to_tool")
        cam_ts, T_cam_list = load_camera_data(camera_traj_file)
        rgb_map = load_rgb_mapping(rgb_mapping_file)
    except Exception as e:
        print(f"[-] Lỗi tải dữ liệu cho {dataset_name}: {e}")
        return False
        
    print(f"  [>] Tìm ArUco trong {num_frames} frames đầu của {dataset_name}...")
    T_slam0_to_marker_list = []
    
    frames_checked = 0
    for ts, T_slam0_to_cam in zip(cam_ts, T_cam_list):
        if frames_checked >= num_frames: break
        
        # Lấy ảnh có timestamp gần nhất
        if not rgb_map: break
        closest_rgb_ts = min(rgb_map.keys(), key=lambda k: abs(k - ts))
        if abs(closest_rgb_ts - ts) < 0.05:
            img_path = os.path.join(dataset_dir, rgb_map[closest_rgb_ts])
            T_cam_to_marker = detect_marker_pose(img_path, K, D)
            if T_cam_to_marker is not None:
                T_slam0_to_marker = T_slam0_to_cam @ T_cam_to_marker
                T_slam0_to_marker_list.append(T_slam0_to_marker)
        frames_checked += 1
        
    if not T_slam0_to_marker_list:
        print(f"  [!] Lỗi: Không tìm thấy ArUco Marker trong {dataset_name}!")
        return False
        
    T_slam0_to_marker = average_poses(T_slam0_to_marker_list)
    T_marker_to_slam0 = np.linalg.inv(T_slam0_to_marker)
    
    slam_points_in_marker, slam_tool_points_in_marker = [], []
    T_tool2cam = np.linalg.inv(T_cam2tool)
    
    for T_slam0_to_cam in T_cam_list:
        T_marker_to_cam = T_marker_to_slam0 @ T_slam0_to_cam
        slam_points_in_marker.append(T_marker_to_cam[:3, 3])
        
        T_marker_to_tool = T_marker_to_cam @ T_tool2cam
        slam_tool_points_in_marker.append(T_marker_to_tool[:3, 3])
        
    slam_points = np.array(slam_points_in_marker)
    tool_points = np.array(slam_tool_points_in_marker)
    
    df_out = pd.DataFrame({
        "timestamp": cam_ts,
        "marker_cam_x": slam_points[:, 0], "marker_cam_y": slam_points[:, 1], "marker_cam_z": slam_points[:, 2],
        "marker_tool_x": tool_points[:, 0], "marker_tool_y": tool_points[:, 1], "marker_tool_z": tool_points[:, 2]
    })
    df_out.to_csv(out_csv, index=False)
    print(f"  [+] Đã lưu CSV: {out_csv}")
    
    _plot_from_csv(out_csv, out_img, dataset_name, no_show)
    return True

def _plot_from_csv(csv_path, out_img, title, no_show):
    df = pd.read_csv(csv_path)
    cam_points = df[['marker_cam_x', 'marker_cam_y', 'marker_cam_z']].to_numpy()
    tool_points = df[['marker_tool_x', 'marker_tool_y', 'marker_tool_z']].to_numpy()
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(0, 0, 0, color='black', s=200, label='Marker Origin (0,0,0)', marker='*')
    ax.plot([0, 0.05], [0, 0], [0, 0], color='red', linewidth=3, label='Marker X-axis') 
    ax.plot([0, 0], [0, 0.05], [0, 0], color='green', linewidth=3, label='Marker Y-axis') 
    ax.plot([0, 0], [0, 0], [0, 0.05], color='blue', linewidth=3, label='Marker Z-axis') 
    
    ax.plot(cam_points[:, 0], cam_points[:, 1], cam_points[:, 2], label='Camera Trajectory', color='green', linewidth=1)
    ax.plot(tool_points[:, 0], tool_points[:, 1], tool_points[:, 2], label='SLAM Predicted (Tool)', color='red', linewidth=2)
            
    ax.scatter(*tool_points[0], color='cyan', s=100, label='Start (Tool)', marker='o', edgecolor='black', zorder=5)
    ax.scatter(*cam_points[0], color='yellow', s=100, label='Start (Camera)', marker='s', edgecolor='black', zorder=5)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'SLAM Trajectory in ArUco Marker Frame - {title}')
    ax.legend()
    
    # Scale tỉ lệ bằng nhau
    max_range = np.array([cam_points[:, 0].max()-cam_points[:, 0].min(), 
                          cam_points[:, 1].max()-cam_points[:, 1].min(), 
                          cam_points[:, 2].max()-cam_points[:, 2].min()]).max() / 2.0
    mid_x = (cam_points[:, 0].max()+cam_points[:, 0].min()) * 0.5
    mid_y = (cam_points[:, 1].max()+cam_points[:, 1].min()) * 0.5
    mid_z = (cam_points[:, 2].max()+cam_points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.savefig(out_img, dpi=300)
    if not no_show:
        plt.show()
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════
# INTERACTIVE LABELER (Bult-in)
# ═══════════════════════════════════════════════════════════════
def interactive_labeler(date_dir_path):
    date_dir = os.path.join(S2_OUTPUT_DIR, date_dir_path)
    datasets = [d for d in sorted(glob.glob(os.path.join(date_dir, "dataset_*"))) if os.path.isdir(d)]
    
    if not datasets:
        print(f"[-] Không có dataset nào trong {date_dir}")
        return
        
    print(f"[+] Chế độ Đánh giá Hàng loạt. Tìm thấy {len(datasets)} datasets.")
    eval_csv_path = os.path.join(date_dir, "dataset_evaluation.csv")
    evaluations = {}
    if os.path.exists(eval_csv_path):
        df_eval = pd.read_csv(eval_csv_path, header=None, names=["dataset", "status"])
        evaluations = dict(zip(df_eval["dataset"], df_eval["status"]))
        
    def save_evals():
        with open(eval_csv_path, 'w') as f:
            for ds_path in datasets:
                ds = os.path.basename(ds_path)
                f.write(f"{ds},{evaluations.get(ds, 'chưa đánh giá')}\n")
    save_evals()

    print("\n[!] Hướng dẫn:")
    print("  Mũi tên Trái/Phải : Chuyển dataset")
    print("  G                 : Đánh dấu TỐT (Good)")
    print("  B                 : Đánh dấu XẤU (Bad)")
    print("  Q / Esc           : Thoát\n")

    current_idx = 0
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    def update_plot(idx):
        if idx < 0 or idx >= len(datasets): return
        ax.clear()
        
        ds_path = datasets[idx]
        ds_name = os.path.basename(ds_path)
        csv_path = os.path.join(ds_path, "Mapped_SLAM_Trajectory_Marker.csv")
        
        # Tạo dữ liệu ngầm nếu chưa có
        if not os.path.exists(csv_path):
            print(f"[*] Đang khởi tạo dữ liệu cho {ds_name} (chạy ngầm)...")
            rel_path = os.path.join(date_dir_path, ds_name)
            process_single_dataset(rel_path, no_show=True)
            
        status = evaluations.get(ds_name, "Chưa đánh giá")
        print(f"[>] Đang xem: {ds_name} ({idx+1}/{len(datasets)}) - Trạng thái: {status.upper()}")
        
        if not os.path.exists(csv_path):
            ax.text2D(0.5, 0.5, f"ERROR / MISSING DATA: {ds_name}", transform=ax.transAxes, ha="center")
            ax.set_title(f"[{idx+1}/{len(datasets)}] {ds_name} - ERROR")
            fig.canvas.draw()
            return
            
        df = pd.read_csv(csv_path)
        cam = df[['marker_cam_x', 'marker_cam_y', 'marker_cam_z']].to_numpy()
        tool = df[['marker_tool_x', 'marker_tool_y', 'marker_tool_z']].to_numpy()
        
        ax.scatter(0, 0, 0, color='black', s=200, marker='*')
        ax.plot([0, 0.05], [0, 0], [0, 0], color='red', linewidth=3)
        ax.plot([0, 0], [0, 0.05], [0, 0], color='green', linewidth=3)
        ax.plot([0, 0], [0, 0], [0, 0.05], color='blue', linewidth=3)
        ax.plot(cam[:, 0], cam[:, 1], cam[:, 2], color='green', linewidth=1, label="Camera")
        ax.plot(tool[:, 0], tool[:, 1], tool[:, 2], color='red', linewidth=2, label="Tool")

        if len(tool) > 0:
            ax.scatter(*tool[0], color='cyan', s=100, marker='o', edgecolor='black', zorder=5, label='Start')

        ax.set_title(f"[{idx+1}/{len(datasets)}] {ds_name}\nStatus: {status.upper()}")
        ax.legend()
        
        if len(cam) > 0:
            max_r = np.max([cam[:,0].ptp(), cam[:,1].ptp(), cam[:,2].ptp()]) / 2.0
            mx, my, mz = cam[:,0].mean(), cam[:,1].mean(), cam[:,2].mean()
            ax.set_xlim(mx - max_r, mx + max_r)
            ax.set_ylim(my - max_r, my + max_r)
            ax.set_zlim(mz - max_r, mz + max_r)
            
        fig.canvas.draw()

    def on_key(event):
        nonlocal current_idx
        if event.key == 'right':
            current_idx = min(len(datasets) - 1, current_idx + 1)
            update_plot(current_idx)
        elif event.key == 'left':
            current_idx = max(0, current_idx - 1)
            update_plot(current_idx)
        elif event.key == 'g':
            ds = os.path.basename(datasets[current_idx])
            evaluations[ds] = "tốt"
            save_evals()
            print(f"  [v] Đã lưu: {ds} -> TỐT")
            update_plot(current_idx)
        elif event.key == 'b':
            ds = os.path.basename(datasets[current_idx])
            evaluations[ds] = "xấu"
            save_evals()
            print(f"  [x] Đã lưu: {ds} -> XẤU")
            update_plot(current_idx)
        elif event.key in ['escape', 'q']:
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    update_plot(current_idx)
    plt.show()

# ═══════════════════════════════════════════════════════════════
# MAIN ROUTING
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Trực quan hóa SLAM trong hệ tọa độ ArUco Marker")
    parser.add_argument("--path", required=True, help="Đường dẫn đến dataset hoặc thư mục Date")
    parser.add_argument("--no_show", action="store_true", help="Không hiển thị ảnh (chỉ lưu)")
    parser.add_argument("--num_frames", type=int, default=100, help="Số frame đầu tiên để quét tìm Marker")
    args = parser.parse_args()

    target_path = os.path.join(S1_OUTPUT_DIR, args.path)
    
    if not os.path.exists(target_path):
        # Fallback thử kiểm tra S2_OUTPUT_DIR (do trước đó người dùng truyền nhầm đường dẫn)
        target_path_s2 = os.path.join(S2_OUTPUT_DIR, args.path)
        if os.path.exists(target_path_s2):
            target_path = target_path_s2
        else:
            print(f"[!] Lỗi: Không tìm thấy thư mục: {target_path}")
            sys.exit(1)

    # Nếu là 1 dataset
    if os.path.basename(target_path).startswith("dataset_"):
        print(f"\n>>> CHẾ ĐỘ: XỬ LÝ 1 DATASET <<<")
        process_single_dataset(args.path, args.num_frames, args.no_show)
    
    # Nếu là thư mục Date (chứa nhiều dataset) -> Bật chế độ đánh giá tương tác
    else:
        print(f"\n>>> CHẾ ĐỘ: ĐÁNH GIÁ TƯƠNG TÁC HÀNG LOẠT <<<")
        interactive_labeler(args.path)

    print("\n>>> HOÀN THÀNH! <<<")

if __name__ == "__main__":
    main()
