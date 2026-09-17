"""
This file is used to visualize SLAM trajectory in Robot's base coordinate and compare with groundtruth trajectory (robot_log.csv)

Input:
    CameraTrajectory.txt: SLAM trajectory
    robot_log.csv: Groundtruth trajectory (robot_log.csv)
    eye_in_hand_result.json: Camera-to-tool transformation matrix

Output:
    {dataset_name}_in_Base_Plot.png: 3D plot of SLAM trajectory in base coordinate
    {dataset_name}_SLAM_Only_Plot.png: 3D plot of SLAM trajectory only
    Mapped_SLAM_Trajectory.csv: SLAM trajectory in base coordinate
    
"""
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import os
import json
import matplotlib
matplotlib.use('Agg') # Đổi sang Agg để tránh xung đột Qt với cv2 khi chỉ cần lưu ảnh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_JSON_FILE = os.path.join(BASE_DIR, "..", "..", "..", "Data_calibration", "realsense_flange_louis", "eye_in_hand_result.json")

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
        r_euler = np.array([row['u'], row['v'], row['w']])
        rot = R.from_euler('xyz', r_euler, degrees=True).as_matrix() 
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = t
        T_list.append(T)
    return timestamps, T_list

def process_dataset(path_rel):
    camera_traj_file = os.path.join(BASE_DIR, "..", "S2_output_slam", path_rel, "SmoothedCameraTrajectory.txt")
    robot_log_file = os.path.join(BASE_DIR, "..", "..", "S1_Data_Collection", "S1_output", path_rel, "robot_log.csv")
    output_dir = os.path.join(BASE_DIR, "..", "S2_output_slam", path_rel)
    dataset_name = os.path.basename(path_rel)
    
    if not os.path.exists(camera_traj_file) or not os.path.exists(robot_log_file):
        print(f"[-] Bỏ qua {path_rel}: Không tìm thấy file dữ liệu (robot_log hoặc SmoothedCameraTrajectory).")
        return

    try:
        T_cam2tool = load_json_matrix(CALIB_JSON_FILE)
        cam_ts, T_cam_list = load_camera_data(camera_traj_file)
        rob_ts, T_rob_list = load_robot_data(robot_log_file)
    except Exception as e:
        print(f"[-] Lỗi khi tải dữ liệu {path_rel}: {e}")
        return

    print(f"[+] Đang xử lý: {path_rel}")
    
    T_b_e0 = T_rob_list[0]
    T_base_to_SLAMWorld = T_b_e0 @ T_cam2tool
    
    slam_points_in_base = []
    for T_W_c in T_cam_list:
        T_b_c = T_base_to_SLAMWorld @ T_W_c
        slam_points_in_base.append(T_b_c[:3, 3])
        
    slam_points_in_base = np.array(slam_points_in_base)
    robot_tool_points = np.array([T[:3, 3] for T in T_rob_list])
    
    T_tool2cam = np.linalg.inv(T_cam2tool)
    slam_tool_points = []
    for T_W_c in T_cam_list:
        T_b_c = T_base_to_SLAMWorld @ T_W_c
        T_b_tool = T_b_c @ T_tool2cam
        slam_tool_points.append(T_b_tool[:3, 3])
    slam_tool_points = np.array(slam_tool_points)

    # ĐỒ THỊ 1
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(robot_tool_points[:, 0], robot_tool_points[:, 1], robot_tool_points[:, 2], 
            label='Robot Kinematics (Tool)', color='blue', linewidth=3, alpha=0.6)
    ax.plot(slam_tool_points[:, 0], slam_tool_points[:, 1], slam_tool_points[:, 2], 
            label='SLAM Predicted (Tool)', color='red', linewidth=2, linestyle='--')
    ax.plot(slam_points_in_base[:, 0], slam_points_in_base[:, 1], slam_points_in_base[:, 2], 
            label='Camera Trajectory', color='green', linewidth=1)

    start_tool = slam_tool_points[0]
    start_cam = slam_points_in_base[0]
    ax.scatter(*start_tool, color='cyan', s=100, label='Start (Tool)', marker='o', edgecolor='black', zorder=5)
    ax.scatter(*start_cam, color='yellow', s=100, label='Start (Camera)', marker='s', edgecolor='black', zorder=5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'SLAM Trajectory Mapped to Robot Base Frame - {dataset_name}')
    ax.legend()
    
    out_img = os.path.join(output_dir, f"{dataset_name}_in_Base_Plot.png")
    plt.savefig(out_img, dpi=300)
    plt.close(fig) # Xóa bộ nhớ
    
    # Lưu CSV
    out_csv = os.path.join(output_dir, "Mapped_SLAM_Trajectory.csv")
    df_out = pd.DataFrame({
        "timestamp": cam_ts,
        "base_x": slam_points_in_base[:, 0],
        "base_y": slam_points_in_base[:, 1],
        "base_z": slam_points_in_base[:, 2]
    })
    df_out.to_csv(out_csv, index=False)
    
    # ĐỒ THỊ 2
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

def generate_all():
    slam_root = os.path.join(BASE_DIR, "..", "S2_output_slam")
    if not os.path.exists(slam_root):
        print(f"Không tìm thấy thư mục {slam_root}")
        return

    # Lấy toàn bộ các folder dataset
    for date_d in sorted(os.listdir(slam_root)):
        date_path = os.path.join(slam_root, date_d)
        if not os.path.isdir(date_path): continue
        for dataset_d in sorted(os.listdir(date_path)):
            if dataset_d.startswith("dataset_"):
                path_rel = os.path.join(date_d, dataset_d)
                process_dataset(path_rel)
    print("\nHoàn tất quá trình tạo và lưu ảnh cho toàn bộ dữ liệu.")

def slideshow():
    date_folder = input("Nhập tên thư mục Date (VD: Date_09092026): ").strip()
    search_path = os.path.join(BASE_DIR, "..", "S2_output_slam", date_folder, "*", "*_in_Base_Plot.png")
    images = sorted(glob.glob(search_path))
    
    if not images:
        print(f"Không tìm thấy ảnh nào trong {date_folder}. Bạn đã chạy tạo ảnh chưa?")
        return

    if HAS_CV2:
        print("Đang hiển thị ảnh bằng OpenCV...")
        print(">> Nhấn phím MŨI TÊN PHẢI (hoặc phím bất kỳ) để xem ảnh tiếp theo.")
        print(">> Nhấn 'q' hoặc 'ESC' để thoát trình chiếu.")
        for img_path in images:
            img = cv2.imread(img_path)
            if img is None: continue
            
            # Thu nhỏ một chút nếu ảnh quá to
            height, width = img.shape[:2]
            scale = 0.5 if width > 1920 else 0.8
            img = cv2.resize(img, (int(width * scale), int(height * scale)))
            
            cv2.imshow(f"Slideshow - {date_folder}", img)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q') or key == 27:
                break
        cv2.destroyAllWindows()
    else:
        import matplotlib.image as mpimg
        print("Đang hiển thị ảnh bằng Matplotlib...")
        print(">> Hãy TẮT CỬA SỔ ẢNH để chuyển sang ảnh tiếp theo.")
        for img_path in images:
            img = mpimg.imread(img_path)
            fig, ax = plt.subplots(figsize=(12, 9))
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(os.path.basename(img_path))
            plt.show() # Code sẽ dừng ở đây cho đến khi user tắt cửa sổ
            
def main():
    while True:
        print("\n" + "="*50)
        print("MENU XỬ LÝ ẢNH QUỸ ĐẠO")
        print("1: Tạo và lưu ảnh cho TOÀN BỘ tập dữ liệu")
        print("2: Trình chiếu bộ sưu tập ảnh của 1 Date cụ thể")
        print("3: Thoát")
        print("="*50)
        
        choice = input("Nhập lựa chọn của bạn (1-3): ").strip()
        
        if choice == '1':
            generate_all()
        elif choice == '2':
            slideshow()
        elif choice == '3':
            print("Thoát chương trình.")
            break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng nhập lại.")

if __name__ == "__main__":
    main()
