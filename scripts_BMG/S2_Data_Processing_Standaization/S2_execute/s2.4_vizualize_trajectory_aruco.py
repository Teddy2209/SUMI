"""
This file is used to visualize SLAM trajectory in the ArUco marker's coordinate system.

Input:
    CameraTrajectory.txt: SLAM trajectory
    rgb.txt & rgb/: RGB images from the dataset to detect the ArUco marker
    camera_intrinsics.json: RealSense intrinsic matrix
    eye_in_hand_result.json: Camera-to-tool transformation matrix

Output:
    SLAM_in_Marker_Plot.png: 3D plot of SLAM trajectory relative to ArUco marker
    Mapped_SLAM_Trajectory_Marker.csv: SLAM trajectory in marker coordinate
"""
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import os
import json
import cv2
import cv2.aruco as aruco
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Default paths (can be modified if paths change)
CALIB_JSON_FILE = os.path.join(BASE_DIR, "..", "..", "S0_Camera_Calibration", "S0_output", "Date_18092026", "calibration_matrices_RS_camera", "eye_in_hand_result.json")
INTRINSICS_FILE = os.path.join(BASE_DIR, "..", "..", "S0_Camera_Calibration", "S0_output", "Date_18092026", "calibration_matrices_RS_camera", "camera_intrinsics.json")

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
    """Average multiple 4x4 transformation matrices"""
    if len(T_list) == 0:
        return np.eye(4)
    if len(T_list) == 1:
        return T_list[0]
        
    t_mean = np.mean([T[:3, 3] for T in T_list], axis=0)
    rots = R.from_matrix([T[:3, :3] for T in T_list])
    R_mean = rots.mean().as_matrix()
    
    T_mean = np.eye(4)
    T_mean[:3, :3] = R_mean
    T_mean[:3, 3] = t_mean
    return T_mean

def detect_marker_pose(image_path, K, D):
    frame = cv2.imread(image_path)
    if frame is None:
        return None
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    is_cv_v47 = hasattr(aruco, 'ArucoDetector')
    
    for dict_name, dict_id in ARUCO_DICTS.items():
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
                    [-L,  L, 0],
                    [ L,  L, 0],
                    [ L, -L, 0],
                    [-L, -L, 0]
                ], dtype=np.float32)
                
                # Taking the first detected marker
                corner = corners[0][0]
                ret, rvec, tvec = cv2.solvePnP(obj_points, corner, K, D)
                if ret:
                    R_cam_marker, _ = cv2.Rodrigues(rvec)
                    T_cam_to_marker = np.eye(4)
                    T_cam_to_marker[:3, :3] = R_cam_marker
                    T_cam_to_marker[:3, 3] = tvec.squeeze()
                    return T_cam_to_marker
        except Exception as e:
            continue
    return None

def main():
    parser = argparse.ArgumentParser(description="Visualize SLAM trajectory relative to ArUco marker.")
    parser.add_argument("--path", required=True, help="Path to dataset, e.g. Date_09092026/dataset_110627")
    parser.add_argument("--no_show", action="store_true", help="Do not show plot")
    parser.add_argument("--num_frames", type=int, default=200, help="Number of frames to scan for marker averaging")
    args = parser.parse_args()

    camera_traj_file = os.path.join(BASE_DIR, "..", "S2_output_slam", args.path, "SmoothedCameraTrajectory.txt")
    rgb_mapping_file = os.path.join(BASE_DIR, "..", "..", "S1_Data_Collection", "S1_output", args.path, "rgb.txt")
    dataset_dir = os.path.join(BASE_DIR, "..", "..", "S1_Data_Collection", "S1_output", args.path)
    output_dir = os.path.join(BASE_DIR, "..", "S2_output_slam", args.path)
    
    os.makedirs(output_dir, exist_ok=True)

    print("1. Đang tải tham số Camera và T_cam_to_tool...")
    K, D = load_camera_intrinsics(INTRINSICS_FILE)
    T_cam2tool = load_json_matrix(CALIB_JSON_FILE, "T_cam_to_tool")
    
    print("2. Đang tải dữ liệu quỹ đạo và hình ảnh...")
    cam_ts, T_cam_list = load_camera_data(camera_traj_file)
    rgb_map = load_rgb_mapping(rgb_mapping_file)
    
    print(f"3. Đang quét {args.num_frames} frames đầu tiên để tìm ArUco Marker và tính trung bình...")
    T_slam0_to_marker_list = []
    
    # Try to find marker in the first matched frames
    frames_checked = 0
    for ts, T_slam0_to_cam in zip(cam_ts, T_cam_list):
        if frames_checked >= args.num_frames:
            break
            
        # Find closest rgb timestamp
        closest_rgb_ts = min(rgb_map.keys(), key=lambda k: abs(k - ts))
        if abs(closest_rgb_ts - ts) < 0.05: # threshold 50ms
            img_path = os.path.join(dataset_dir, rgb_map[closest_rgb_ts])
            
            T_cam_to_marker = detect_marker_pose(img_path, K, D)
            if T_cam_to_marker is not None:
                T_slam0_to_marker = T_slam0_to_cam @ T_cam_to_marker
                T_slam0_to_marker_list.append(T_slam0_to_marker)
                print(f"  [+] Đã tìm thấy marker ở frame {frames_checked + 1}")
        
        frames_checked += 1
        
    if not T_slam0_to_marker_list:
        print("[ERROR] Không tìm thấy bất kỳ ArUco Marker nào trong các frame đầu tiên!")
        return
        
    print(f"  => Tổng cộng detect thành công ở {len(T_slam0_to_marker_list)} frames. Đang tính trung bình ma trận...")
    T_slam0_to_marker = average_poses(T_slam0_to_marker_list)
    T_marker_to_slam0 = np.linalg.inv(T_slam0_to_marker)
    
    print("\n4. Đang ánh xạ quỹ đạo SLAM về hệ tọa độ Marker...")
    slam_points_in_marker = []
    slam_tool_points_in_marker = []
    T_tool2cam = np.linalg.inv(T_cam2tool)
    
    for T_slam0_to_cam in T_cam_list:
        # Camera trong hệ Marker
        T_marker_to_cam = T_marker_to_slam0 @ T_slam0_to_cam
        slam_points_in_marker.append(T_marker_to_cam[:3, 3])
        
        # Tool trong hệ Marker
        T_marker_to_tool = T_marker_to_cam @ T_tool2cam
        slam_tool_points_in_marker.append(T_marker_to_tool[:3, 3])
        
    slam_points_in_marker = np.array(slam_points_in_marker)
    slam_tool_points_in_marker = np.array(slam_tool_points_in_marker)
    
    print("5. Đang lưu file và vẽ đồ thị 3D...")
    out_csv = os.path.join(output_dir, "Mapped_SLAM_Trajectory_Marker.csv")
    df_out = pd.DataFrame({
        "timestamp": cam_ts,
        "marker_cam_x": slam_points_in_marker[:, 0],
        "marker_cam_y": slam_points_in_marker[:, 1],
        "marker_cam_z": slam_points_in_marker[:, 2],
        "marker_tool_x": slam_tool_points_in_marker[:, 0],
        "marker_tool_y": slam_tool_points_in_marker[:, 1],
        "marker_tool_z": slam_tool_points_in_marker[:, 2]
    })
    df_out.to_csv(out_csv, index=False)
    print(f"Đã lưu tọa độ vào: {out_csv}")
    
    # Vẽ đồ thị
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Vẽ gốc tọa độ Marker
    ax.scatter(0, 0, 0, color='black', s=200, label='Marker Origin (0,0,0)', marker='*')
    # Vẽ các trục của Marker (5cm)
    ax.plot([0, 0.05], [0, 0], [0, 0], color='red', linewidth=3, label='Marker X-axis') 
    ax.plot([0, 0], [0, 0.05], [0, 0], color='green', linewidth=3, label='Marker Y-axis') 
    ax.plot([0, 0], [0, 0], [0, 0.05], color='blue', linewidth=3, label='Marker Z-axis') 
    
    # Vẽ quỹ đạo Camera
    ax.plot(slam_points_in_marker[:, 0], slam_points_in_marker[:, 1], slam_points_in_marker[:, 2], 
            label='Camera Trajectory', color='green', linewidth=1)
            
    # Vẽ quỹ đạo Tool
    ax.plot(slam_tool_points_in_marker[:, 0], slam_tool_points_in_marker[:, 1], slam_tool_points_in_marker[:, 2], 
            label='SLAM Predicted (Tool)', color='red', linewidth=2)
            
    # Điểm bắt đầu
    ax.scatter(*slam_tool_points_in_marker[0], color='cyan', s=100, label='Start (Tool)', marker='o', edgecolor='black', zorder=5)
    ax.scatter(*slam_points_in_marker[0], color='yellow', s=100, label='Start (Camera)', marker='s', edgecolor='black', zorder=5)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('SLAM Trajectory in ArUco Marker Frame')
    ax.legend()
    
    # Equal aspect ratio trick cho 3D plot
    max_range = np.array([slam_points_in_marker[:, 0].max()-slam_points_in_marker[:, 0].min(), 
                          slam_points_in_marker[:, 1].max()-slam_points_in_marker[:, 1].min(), 
                          slam_points_in_marker[:, 2].max()-slam_points_in_marker[:, 2].min()]).max() / 2.0
    mid_x = (slam_points_in_marker[:, 0].max()+slam_points_in_marker[:, 0].min()) * 0.5
    mid_y = (slam_points_in_marker[:, 1].max()+slam_points_in_marker[:, 1].min()) * 0.5
    mid_z = (slam_points_in_marker[:, 2].max()+slam_points_in_marker[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    out_img = os.path.join(output_dir, "SLAM_in_Marker_Plot.png")
    plt.savefig(out_img, dpi=300)
    print(f"Đã lưu ảnh vẽ 3D tại: {out_img}")
    
    print("\nĐang hiển thị cửa sổ 3D...")
    if not args.no_show:
        plt.show()

if __name__ == "__main__":
    main()
