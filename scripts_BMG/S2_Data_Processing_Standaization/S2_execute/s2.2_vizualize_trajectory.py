#!/usr/bin/env python3
"""
Trực quan hóa Quỹ đạo SLAM so với Groundtruth.

Chức năng:
  - Vẽ quỹ đạo SLAM trong hệ tọa độ Base của Robot.
  - So sánh với quỹ đạo Groundtruth (robot_log.csv).
  - Hỗ trợ xử lý 1 dataset lẻ hoặc xử lý hàng loạt toàn bộ thư mục Date.
  - Hỗ trợ chế độ xem lại (Slideshow).
"""

import os
import sys
import json
import argparse
import glob
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg') # Tránh xung đột Qt/cv2 khi chạy tự động
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
S1_OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "S1_Data_Collection", "S1_output"))
S2_OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "S2_output_slam"))
CALIB_JSON_FILE = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "S0_Camera_Calibration", "S0_output", "Date_18092026", "calibration_matrices_R_camera", "eye_in_hand_result.json"))

# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def load_json_matrix(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return np.array(data["T_cam_to_tool"])

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

def load_robot_data(file_path):
    df = pd.read_csv(file_path)
    timestamps = df['timestamp'].values
    T_list = []
    for _, row in df.iterrows():
        t = np.array([row['x'], row['y'], row['z']]) / 1000.0
        r_euler = np.array([row['u'], row['v'], row['w']])
        rot = R.from_euler('xyz', r_euler, degrees=True).as_matrix() 
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = t
        T_list.append(T)
    return timestamps, T_list

# ═══════════════════════════════════════════════════════════════
# CORE PROCESSING
# ═══════════════════════════════════════════════════════════════
def process_single_dataset(rel_path):
    camera_traj_file = os.path.join(S2_OUTPUT_DIR, rel_path, "SmoothedCameraTrajectory.txt")
    robot_log_file = os.path.join(S1_OUTPUT_DIR, rel_path, "robot_log.csv")
    output_dir = os.path.join(S2_OUTPUT_DIR, rel_path)
    dataset_name = os.path.basename(rel_path)
    
    if not os.path.exists(camera_traj_file):
        print(f"[-] Bỏ qua {rel_path}: Không tìm thấy SmoothedCameraTrajectory.txt")
        return False
        
    has_robot = os.path.exists(robot_log_file)

    try:
        T_cam2tool = load_json_matrix(CALIB_JSON_FILE)
        cam_ts, T_cam_list = load_camera_data(camera_traj_file)
        if has_robot:
            rob_ts, T_rob_list = load_robot_data(robot_log_file)
        else:
            T_rob_list = []
    except Exception as e:
        print(f"[-] Lỗi khi tải dữ liệu {rel_path}: {e}")
        return False

    print(f"[+] Đang xử lý: {rel_path}")
    
    if has_robot:
        T_b_e0 = T_rob_list[0]
        T_base_to_SLAMWorld = T_b_e0 @ T_cam2tool
    else:
        T_base_to_SLAMWorld = np.eye(4)
    
    slam_points_in_base = []
    for T_W_c in T_cam_list:
        T_b_c = T_base_to_SLAMWorld @ T_W_c
        slam_points_in_base.append(T_b_c[:3, 3])
        
    slam_points_in_base = np.array(slam_points_in_base)
    robot_tool_points = np.array([T[:3, 3] for T in T_rob_list]) if has_robot else np.array([])
    
    T_tool2cam = np.linalg.inv(T_cam2tool)
    slam_tool_points = []
    for T_W_c in T_cam_list:
        T_b_c = T_base_to_SLAMWorld @ T_W_c
        T_b_tool = T_b_c @ T_tool2cam
        slam_tool_points.append(T_b_tool[:3, 3])
    slam_tool_points = np.array(slam_tool_points)

    _plot_and_save(dataset_name, output_dir, slam_tool_points, slam_points_in_base, robot_tool_points, has_robot)

    # Lưu CSV
    out_csv = os.path.join(output_dir, "Mapped_SLAM_Trajectory.csv")
    df_out = pd.DataFrame({
        "timestamp": cam_ts,
        "base_x": slam_points_in_base[:, 0],
        "base_y": slam_points_in_base[:, 1],
        "base_z": slam_points_in_base[:, 2]
    })
    df_out.to_csv(out_csv, index=False)
    return True

def _plot_and_save(dataset_name, output_dir, slam_tool_points, slam_points_in_base, robot_tool_points, has_robot):
    start_tool = slam_tool_points[0]
    start_cam = slam_points_in_base[0]

    # ĐỒ THỊ 1: Có hoặc Không có Robot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    if has_robot:
        ax.plot(robot_tool_points[:, 0], robot_tool_points[:, 1], robot_tool_points[:, 2], 
                label='Robot Kinematics (Tool)', color='blue', linewidth=3, alpha=0.6)
    
    ax.plot(slam_tool_points[:, 0], slam_tool_points[:, 1], slam_tool_points[:, 2], 
            label='SLAM Predicted (Tool)', color='red', linewidth=2, linestyle='--')
    ax.plot(slam_points_in_base[:, 0], slam_points_in_base[:, 1], slam_points_in_base[:, 2], 
            label='Camera Trajectory', color='green', linewidth=1)

    ax.scatter(*start_tool, color='cyan', s=100, label='Start (Tool)', marker='o', edgecolor='black', zorder=5)
    ax.scatter(*start_cam, color='yellow', s=100, label='Start (Camera)', marker='s', edgecolor='black', zorder=5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'SLAM Trajectory Mapped to Robot Base Frame - {dataset_name}')
    ax.legend()
    
    out_img = os.path.join(output_dir, f"{dataset_name}_in_Base_Plot.png")
    plt.savefig(out_img, dpi=300)
    plt.close(fig)
    
    # ĐỒ THỊ 2: SLAM Only
    fig2 = plt.figure(figsize=(12, 9))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot(slam_tool_points[:, 0], slam_tool_points[:, 1], slam_tool_points[:, 2], 
            label='SLAM Predicted (Tool)', color='red', linewidth=2)
    ax2.plot(slam_points_in_base[:, 0], slam_points_in_base[:, 1], slam_points_in_base[:, 2], 
            label='Camera Trajectory', color='green', linewidth=1)
    ax2.scatter(*start_tool, color='cyan', s=100, label='Start (Tool)', marker='o', edgecolor='black', zorder=5)
    ax2.scatter(*start_cam, color='yellow', s=100, label='Start (Camera)', marker='s', edgecolor='black', zorder=5)

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title(f'SLAM Trajectory ONLY - {dataset_name}')
    ax2.legend()
    
    out_img2 = os.path.join(output_dir, f"{dataset_name}_SLAM_Only_Plot.png")
    plt.savefig(out_img2, dpi=300)
    plt.close(fig2)

def slideshow_mode(date_folder):
    search_path = os.path.join(S2_OUTPUT_DIR, date_folder, "*", "*_in_Base_Plot.png")
    images = sorted(glob.glob(search_path))
    
    if not images:
        print(f"[!] Không tìm thấy ảnh nào trong {date_folder}.")
        return

    if HAS_CV2:
        print(">> Đang hiển thị ảnh bằng OpenCV...")
        print(">> Nhấn phím MŨI TÊN PHẢI/TRÁI để chuyển ảnh. Nhấn 'q' để thoát.")
        idx = 0
        while True:
            if idx < 0: idx = 0
            if idx >= len(images): idx = len(images) - 1
            
            img_path = images[idx]
            img = cv2.imread(img_path)
            if img is not None:
                height, width = img.shape[:2]
                scale = 0.5 if width > 1920 else 0.8
                img = cv2.resize(img, (int(width * scale), int(height * scale)))
                cv2.imshow(f"Slideshow - {date_folder} [{idx+1}/{len(images)}]", img)
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == 83 or key == ord('d'): # Right arrow / D
                idx += 1
            elif key == 81 or key == ord('a'): # Left arrow / A
                idx -= 1
            else:
                idx += 1
        cv2.destroyAllWindows()
    else:
        print(">> OpenCV không khả dụng. Hiển thị bằng Matplotlib...")
        print(">> Hãy TẮT CỬA SỔ ẢNH để chuyển sang ảnh tiếp theo.")
        import matplotlib.image as mpimg
        matplotlib.use('TkAgg') # Trả lại backend để hiển thị GUI
        for img_path in images:
            img = mpimg.imread(img_path)
            fig, ax = plt.subplots(figsize=(12, 9))
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(os.path.basename(img_path))
            plt.show()

# ═══════════════════════════════════════════════════════════════
# MAIN ROUTING
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Trực quan hóa quỹ đạo SLAM")
    parser.add_argument("--path", required=True, help="Đường dẫn đến dataset hoặc thư mục Date (VD: Date_27082026/dataset_104821)")
    parser.add_argument("--slideshow", action="store_true", help="Bật chế độ trình chiếu ảnh đã lưu")
    args = parser.parse_args()

    target_path = os.path.join(S2_OUTPUT_DIR, args.path)
    
    if args.slideshow:
        print("\n>>> CHẾ ĐỘ: TRÌNH CHIẾU SLIDESHOW <<<")
        # Phải là thư mục Date thì mới xem dạng slideshow hợp lý
        if os.path.basename(target_path).startswith("dataset_"):
            slideshow_mode(os.path.dirname(args.path))
        else:
            slideshow_mode(args.path)
        return

    if not os.path.exists(target_path):
        print(f"[!] Lỗi: Không tìm thấy thư mục: {target_path}")
        sys.exit(1)

    # Chạy 1 dataset
    if os.path.basename(target_path).startswith("dataset_"):
        print(f"\n>>> CHẾ ĐỘ: TRỰC QUAN HÓA 1 DATASET <<<")
        process_single_dataset(args.path)
    
    # Chạy hàng loạt
    else:
        datasets = [d for d in os.listdir(target_path) if os.path.isdir(os.path.join(target_path, d)) and d.startswith("dataset_")]
        datasets.sort()
        print(f"\n>>> CHẾ ĐỘ: TRỰC QUAN HÓA HÀNG LOẠT ({len(datasets)} datasets) <<<")
        
        for ds_name in datasets:
            rel_path = os.path.join(args.path, ds_name)
            process_single_dataset(rel_path)

    print("\n>>> HOÀN THÀNH! <<<")

if __name__ == "__main__":
    main()
