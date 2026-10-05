#!/usr/bin/env python3
"""
Tạo và Làm mượt Quỹ đạo SLAM.

Chức năng:
  - Làm mượt quỹ đạo SLAM (CameraTrajectory.txt) bằng bộ lọc Savitzky-Golay.
  - Ánh xạ quỹ đạo từ Camera sang Tool bằng ma trận Eye-in-Hand.
  - Lưu kết quả ra file SmoothedCameraTrajectory.txt và SmoothedToolTrajectory.txt.
  - Hỗ trợ chạy cho một dataset lẻ hoặc chạy hàng loạt cho toàn bộ các dataset trong 1 thư mục Date.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

from s2_menu import choose_path

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
S2_OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "S2_output_slam"))
CALIB_FILE = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "S0_Camera_Calibration", "S0_output", "Date_18092026", "calibration_matrices_RS_camera", "eye_in_hand_result.json"))

# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def normalize_quaternions(q):
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    return q / norms

def pose_to_matrix(pos, quat):
    mat = np.eye(4)
    mat[:3, :3] = R.from_quat(quat).as_matrix()
    mat[:3, 3] = pos
    return mat

def matrix_to_pose(mat):
    pos = mat[:3, 3]
    quat = R.from_matrix(mat[:3, :3]).as_quat()
    return pos, quat

def load_calibration_matrix():
    """Đọc Eye-in-Hand (Thực chất JSON lưu G_T_C). Trả về C_T_G (T_tool_to_cam)"""
    try:
        with open(CALIB_FILE, 'r') as f:
            T_cam_to_tool_json = np.array(json.load(f)["T_cam_to_tool"])
            # Lấy nghịch đảo để có C_T_G chuẩn
            return np.linalg.inv(T_cam_to_tool_json)
    except Exception as e:
        print(f"[!] Lỗi đọc {CALIB_FILE}: {e}. Mặc định T_tool_to_cam = Identity")
        return np.eye(4)

# ═══════════════════════════════════════════════════════════════
# CORE PROCESSING
# ═══════════════════════════════════════════════════════════════
def process_single_trajectory(input_file, output_file, T_tool_to_cam, window_size=51, poly_order=3):
    print(f"Đọc dữ liệu từ {input_file}...")
    try:
        data = np.loadtxt(input_file)
    except Exception as e:
        print(f"Lỗi đọc file: {e}")
        return False

    if len(data) < window_size:
        print("Cảnh báo: Dữ liệu quá ngắn so với window_size. Giảm window_size xuống.")
        window_size = len(data) if len(data) % 2 != 0 else len(data) - 1
        if window_size < 3:
            print("Không đủ dữ liệu để làm mượt.")
            return False

    timestamps = data[:, 0]
    translations = data[:, 1:4]
    quaternions = data[:, 4:8]

    print(f"Áp dụng bộ lọc Savitzky-Golay (Window: {window_size}, Bậc: {poly_order})...")
    
    # 1. Làm mượt Translation (x, y, z)
    smoothed_trans = np.zeros_like(translations)
    for i in range(3):
        smoothed_trans[:, i] = savgol_filter(translations[:, i], window_length=window_size, polyorder=poly_order)

    # 2. Làm mượt Quaternion (qx, qy, qz, qw)
    smoothed_quat = np.zeros_like(quaternions)
    for i in range(4):
        smoothed_quat[:, i] = savgol_filter(quaternions[:, i], window_length=window_size, polyorder=poly_order)
    
    smoothed_quat = normalize_quaternions(smoothed_quat)

    # 3. Tính quỹ đạo Tool Tương đối
    smoothed_tool_trans = np.zeros_like(smoothed_trans)
    smoothed_tool_quat = np.zeros_like(smoothed_quat)

    # Khởi tạo T_world_tool_0
    T_world_cam_0 = pose_to_matrix(smoothed_trans[0], smoothed_quat[0])
    T_world_tool_0 = T_world_cam_0 @ T_tool_to_cam
    T_world_tool_0_inv = np.linalg.inv(T_world_tool_0)

    for i in range(len(smoothed_trans)):
        T_world_cam_i = pose_to_matrix(smoothed_trans[i], smoothed_quat[i])
        # Đổi điểm: W_T_G = W_T_C * C_T_G
        T_world_tool_i = T_world_cam_i @ T_tool_to_cam
        
        # Chiếu về hệ tọa độ của Tool tại t=0 (Đổi neo)
        T_rel_tool_i = T_world_tool_0_inv @ T_world_tool_i
        
        t_pos, t_quat = matrix_to_pose(T_rel_tool_i)
        smoothed_tool_trans[i] = t_pos
        smoothed_tool_quat[i] = t_quat

    # 4. Gộp lại và lưu Camera Trajectory
    smoothed_data = np.hstack((timestamps.reshape(-1,1), smoothed_trans, smoothed_quat))
    np.savetxt(output_file, smoothed_data, fmt="%.6f")
    print(f"[+] Đã lưu quỹ đạo CAMERA mượt vào {output_file}")

    # 5. Gộp lại và lưu Tool Trajectory (Relative)
    tool_output_file = output_file.replace("SmoothedCameraTrajectory.txt", "SmoothedToolTrajectory.txt")
    smoothed_tool_data = np.hstack((timestamps.reshape(-1,1), smoothed_tool_trans, smoothed_tool_quat))
    np.savetxt(tool_output_file, smoothed_tool_data, fmt="%.6f")
    print(f"[+] Đã lưu quỹ đạo TOOL TƯƠNG ĐỐI vào {tool_output_file}")

    # 6. Vẽ so sánh
    _plot_comparison(translations, smoothed_trans, output_file)
    return True

def _plot_comparison(original, smoothed, output_file):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Quỹ đạo gốc & mượt
    ax.plot(original[:, 0], original[:, 1], original[:, 2], label='Original (Raw)', color='r', alpha=0.4, linewidth=1)
    ax.plot(smoothed[:, 0], smoothed[:, 1], smoothed[:, 2], label='Smoothed (Savitzky-Golay)', color='b', linewidth=2.5)

    # Start/End markers
    ax.scatter(smoothed[0, 0], smoothed[0, 1], smoothed[0, 2], color='g', s=100, label='Start')
    ax.scatter(smoothed[-1, 0], smoothed[-1, 1], smoothed[-1, 2], color='orange', s=100, label='End')

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('Trajectory Smoothing Comparison')
    
    # Scale tỉ lệ bằng nhau
    max_range = np.array([original[:,0].max()-original[:,0].min(), 
                          original[:,1].max()-original[:,1].min(), 
                          original[:,2].max()-original[:,2].min()]).max() / 2.0
    mid_x = (original[:,0].max()+original[:,0].min()) * 0.5
    mid_y = (original[:,1].max()+original[:,1].min()) * 0.5
    mid_z = (original[:,2].max()+original[:,2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend()
    plot_file = os.path.join(os.path.dirname(output_file), "smoothed_trajectory_plot.png")
    plt.savefig(plot_file, dpi=300)
    plt.close(fig)
    print(f"[+] Đã lưu hình ảnh so sánh vào {plot_file}")

# ═══════════════════════════════════════════════════════════════
# MAIN ROUTING
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Tạo và làm mượt quỹ đạo SLAM")
    parser.add_argument("--path", default=None, help="Relative path to a dataset OR a Date folder (e.g. Date_27082026/dataset_104821 OR Date_27082026). Bỏ trống để mở menu chọn")
    args = parser.parse_args()

    rel_path = args.path or choose_path(S2_OUTPUT_DIR)
    if not rel_path:
        print("[!] Không có lựa chọn hợp lệ!")
        sys.exit(1)

    target_path = os.path.join(S2_OUTPUT_DIR, rel_path)
    if not os.path.exists(target_path):
        print(f"[!] Lỗi: Không tìm thấy thư mục SLAM output: {target_path}")
        sys.exit(1)

    T_tool_to_cam = load_calibration_matrix()

    # Kiểm tra xem đường dẫn là 1 dataset cụ thể hay là thư mục chứa nhiều dataset
    if os.path.basename(target_path).startswith("dataset_"):
        datasets = [target_path]
        print(f"\n>>> CHẾ ĐỘ: LÀM MƯỢT 1 DATASET <<<")
    else:
        # Lấy danh sách các dataset
        datasets = [os.path.join(target_path, d) for d in os.listdir(target_path) 
                    if os.path.isdir(os.path.join(target_path, d)) and d.startswith("dataset_")]
        datasets.sort()
        print(f"\n>>> CHẾ ĐỘ: LÀM MƯỢT HÀNG LOẠT ({len(datasets)} datasets) <<<")

    if not datasets:
        print(f"[!] Không tìm thấy dataset nào trong {target_path}")
        sys.exit(1)

    for ds_path in datasets:
        ds_name = os.path.basename(ds_path)
        input_file = os.path.join(ds_path, "CameraTrajectory.txt")
        output_file = os.path.join(ds_path, "SmoothedCameraTrajectory.txt")
        
        print(f"\n" + "="*50)
        print(f" Đang xử lý: {ds_name}")
        print("="*50)

        if not os.path.exists(input_file):
            print(f"[-] Bỏ qua {ds_name}: Không có CameraTrajectory.txt (chưa chạy SLAM?)")
            continue
            
        try:
            process_single_trajectory(input_file, output_file, T_tool_to_cam, window_size=51, poly_order=3)
        except Exception as e:
            print(f"[!] Lỗi xử lý {ds_name}: {e}")

    print("\n>>> HOÀN THÀNH! <<<")

if __name__ == '__main__':
    main()
