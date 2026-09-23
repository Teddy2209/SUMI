"""
This file is used to visualize SLAM trajectory in Robot's base coordinate and compare with groundtruth trajectory (robot_log.csv)

Input:
    CameraTrajectory.txt: SLAM trajectory
    robot_log.csv: Groundtruth trajectory (robot_log.csv)
    eye_in_hand_result.json: Camera-to-tool transformation matrix

Output:
    SLAM_in_Base_Plot.png: 3D plot of SLAM trajectory in base coordinate
    Mapped_SLAM_Trajectory.csv: SLAM trajectory in base coordinate
    
"""
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import os
import json
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_JSON_FILE = os.path.join(BASE_DIR, "..", "..", "S0_Camera_Calibration","S0_output","Date_18092026","calibration_matrices_R_camera", "eye_in_hand_result.json")

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
        t = np.array([row['x'], row['y'], row['z']]) / 1000.0  # Đổi mm sang m
        # Neuromeka mặc định dùng ZYX (Yaw-Pitch-Roll) hoặc XYZ. 
        # Chúng ta giả định u, v, w tương ứng với Rz, Ry, Rx theo chuẩn ZYX:
        r_euler = np.array([row['u'], row['v'], row['w']])
        # Thử 'zyx' nếu sai thì đổi thành 'xyz'
        rot = R.from_euler('xyz', r_euler, degrees=True).as_matrix() 
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = t
        T_list.append(T)
    return timestamps, T_list

def main():
    parser = argparse.ArgumentParser(description="Visualize SLAM trajectory.")
    parser.add_argument("--path", required=True, help="Path to dataset, e.g. Date_09092026/dataset_110627")
    args = parser.parse_args()

    camera_traj_file = os.path.join(BASE_DIR, "..", "S2_output_slam", args.path, "SmoothedCameraTrajectory.txt")
    robot_log_file = os.path.join(BASE_DIR, "..", "..", "S1_Data_Collection", "S1_output", args.path, "robot_log.csv")
    output_dir = os.path.join(BASE_DIR, "..", "S2_output_slam", args.path)
    os.makedirs(output_dir, exist_ok=True)

    print("1. Đang tải ma trận chuẩn T_cam_to_tool...")
    T_cam2tool = load_json_matrix(CALIB_JSON_FILE)
    print(np.round(T_cam2tool, 4))
    
    print("\n2. Đang tải dữ liệu quỹ đạo...")
    cam_ts, T_cam_list = load_camera_data(camera_traj_file)
    
    has_robot = os.path.exists(robot_log_file)
    if has_robot:
        rob_ts, T_rob_list = load_robot_data(robot_log_file)
        # Bước 1: Lấy tọa độ Robot ở mốc t=0 (Điểm đầu tiên)
        # T_base_to_tool_0
        T_b_e0 = T_rob_list[0]
        
        # Bước 2: Tính mốc tọa độ World của SLAM trong hệ Base
        # Điểm xuất phát của SLAM (t=0) luôn là I (Identity).
        # Tại t=0, Camera nằm ở: T_b_c0 = T_b_e0 * T_cam2tool
        # Do đó Gốc tọa độ SLAM (World) so với Base chính là T_b_c0
        T_base_to_SLAMWorld = T_b_e0 @ T_cam2tool
    else:
        print("Không tìm thấy robot_log.csv (Chế độ SUMI). Dùng SLAM World làm gốc tọa độ.")
        T_base_to_SLAMWorld = np.eye(4)
        T_rob_list = []
    
    # Bước 3: Ánh xạ toàn bộ quỹ đạo SLAM về hệ Base
    slam_points_in_base = []
    for T_W_c in T_cam_list:
        # T_W_c là tọa độ camera trong SLAM World
        # T_b_c là tọa độ camera trong Robot Base
        T_b_c = T_base_to_SLAMWorld @ T_W_c
        slam_points_in_base.append(T_b_c[:3, 3])
        
    slam_points_in_base = np.array(slam_points_in_base)
    
    # Tọa độ Tool thực tế của Robot (để so sánh đối chiếu)
    if has_robot:
        robot_tool_points = np.array([T[:3, 3] for T in T_rob_list])
    else:
        robot_tool_points = np.array([])
    
    # Do camera và tool cách nhau một khoảng t_cam2tool, ta có thể tính quỹ đạo Tool từ SLAM
    # T_base_to_tool = T_base_to_cam * T_cam_to_tool^-1
    T_tool2cam = np.linalg.inv(T_cam2tool)
    slam_tool_points = []
    for T_W_c in T_cam_list:
        T_b_c = T_base_to_SLAMWorld @ T_W_c
        T_b_tool = T_b_c @ T_tool2cam
        slam_tool_points.append(T_b_tool[:3, 3])
    slam_tool_points = np.array(slam_tool_points)

    print("\n3. Đang vẽ biểu đồ 3D...")
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Vẽ quỹ đạo Robot (Thực tế của tay máy)
    if has_robot:
        ax.plot(robot_tool_points[:, 0], robot_tool_points[:, 1], robot_tool_points[:, 2], 
                label='Robot Kinematics (Tool)', color='blue', linewidth=3, alpha=0.6)
            
    # Vẽ quỹ đạo SLAM (dự đoán vị trí Tool)
    ax.plot(slam_tool_points[:, 0], slam_tool_points[:, 1], slam_tool_points[:, 2], 
            label='SLAM Predicted (Tool)', color='red', linewidth=2, linestyle='--')
            
    # Vẽ quỹ đạo Camera (từ SLAM)
    ax.plot(slam_points_in_base[:, 0], slam_points_in_base[:, 1], slam_points_in_base[:, 2], 
            label='Camera Trajectory', color='green', linewidth=1)

    # Đánh dấu điểm bắt đầu (0,0,0 của SLAM)
    start_tool = slam_tool_points[0]
    start_cam = slam_points_in_base[0]
    ax.scatter(*start_tool, color='cyan', s=100, label='Start (Tool)', marker='o', edgecolor='black', zorder=5)
    ax.scatter(*start_cam, color='yellow', s=100, label='Start (Camera)', marker='s', edgecolor='black', zorder=5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('SLAM Trajectory Mapped to Robot Base Frame')
    ax.legend()
    
    out_img = os.path.join(output_dir, "SLAM_in_Base_Plot.png")
    plt.savefig(out_img, dpi=300)
    print(f"\nĐã lưu ảnh vẽ 3D tại: {out_img}")
    
    out_csv = os.path.join(output_dir, "Mapped_SLAM_Trajectory.csv")
    df_out = pd.DataFrame({
        "timestamp": cam_ts,
        "base_x": slam_points_in_base[:, 0],
        "base_y": slam_points_in_base[:, 1],
        "base_z": slam_points_in_base[:, 2]
    })
    df_out.to_csv(out_csv, index=False)
    print(f"Đã lưu tọa độ SLAM trong hệ Base tại: {out_csv}")
    
    # ----------------------------------------------------
    # ĐỒ THỊ 2: CHỈ VẼ QUỸ ĐẠO SLAM (CAMERA VÀ TOOL)
    # ----------------------------------------------------
    fig2 = plt.figure(figsize=(12, 9))
    ax2 = fig2.add_subplot(111, projection='3d')
    
    # Vẽ quỹ đạo SLAM (dự đoán vị trí Tool) - Không vẽ nét đứt nữa để dễ nhìn
    ax2.plot(slam_tool_points[:, 0], slam_tool_points[:, 1], slam_tool_points[:, 2], 
            label='SLAM Predicted (Tool)', color='red', linewidth=2)
            
    # Vẽ quỹ đạo Camera (từ SLAM)
    ax2.plot(slam_points_in_base[:, 0], slam_points_in_base[:, 1], slam_points_in_base[:, 2], 
            label='Camera Trajectory', color='green', linewidth=1)

    ax2.scatter(*start_tool, color='cyan', s=100, label='Start (Tool)', marker='o', edgecolor='black', zorder=5)
    ax2.scatter(*start_cam, color='yellow', s=100, label='Start (Camera)', marker='s', edgecolor='black', zorder=5)

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title('SLAM Trajectory ONLY (Mapped to Robot Base Frame)')
    ax2.legend()
    
    out_img2 = os.path.join(output_dir, "SLAM_Only_Plot.png")
    plt.savefig(out_img2, dpi=300)
    print(f"Đã lưu ảnh vẽ 3D SLAM-only tại: {out_img2}")
    
    # Hiển thị cửa sổ 3D tương tác
    print("\nĐang mở cửa sổ 3D... Bạn có thể kéo thả để xem các góc độ.")
    plt.show()

if __name__ == "__main__":
    main()
