import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter
import sys
import os

def normalize_quaternions(q):
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    return q / norms

def smooth_trajectory(input_file, output_file, window_size=51, poly_order=3):
    print(f"Đọc dữ liệu từ {input_file}...")
    try:
        data = np.loadtxt(input_file)
    except Exception as e:
        print(f"Lỗi đọc file: {e}")
        return

    if len(data) < window_size:
        print("Cảnh báo: Dữ liệu quá ngắn so với window_size. Giảm window_size xuống.")
        window_size = len(data) if len(data) % 2 != 0 else len(data) - 1
        if window_size < 3:
            print("Không đủ dữ liệu để làm mượt.")
            return

    timestamps = data[:, 0]
    translations = data[:, 1:4]
    quaternions = data[:, 4:8]

    print(f"Áp dụng bộ lọc Savitzky-Golay (Window: {window_size}, Bậc: {poly_order})...")
    
    # Làm mượt Translation (x, y, z)
    smoothed_trans = np.zeros_like(translations)
    for i in range(3):
        smoothed_trans[:, i] = savgol_filter(translations[:, i], window_length=window_size, polyorder=poly_order)

    # Làm mượt Quaternion (qx, qy, qz, qw)
    smoothed_quat = np.zeros_like(quaternions)
    for i in range(4):
        smoothed_quat[:, i] = savgol_filter(quaternions[:, i], window_length=window_size, polyorder=poly_order)
    
    # Chuẩn hóa lại Quaternion để đảm bảo hợp lệ (norm = 1)
    smoothed_quat = normalize_quaternions(smoothed_quat)

    # Gộp lại và lưu
    smoothed_data = np.hstack((timestamps.reshape(-1,1), smoothed_trans, smoothed_quat))
    np.savetxt(output_file, smoothed_data, fmt="%.6f")
    print(f"Đã lưu quỹ đạo mượt vào {output_file}")

    # Vẽ so sánh
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Quỹ đạo gốc
    ax.plot(translations[:, 0], translations[:, 1], translations[:, 2], 
            label='Original (Raw)', color='r', alpha=0.4, linewidth=1)
    
    # Quỹ đạo đã làm mượt
    ax.plot(smoothed_trans[:, 0], smoothed_trans[:, 1], smoothed_trans[:, 2], 
            label='Smoothed (Savitzky-Golay)', color='b', linewidth=2.5)

    # Đánh dấu Start/End của quỹ đạo mượt
    ax.scatter(smoothed_trans[0, 0], smoothed_trans[0, 1], smoothed_trans[0, 2], color='g', s=100, label='Start')
    ax.scatter(smoothed_trans[-1, 0], smoothed_trans[-1, 1], smoothed_trans[-1, 2], color='orange', s=100, label='End')

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('Trajectory Smoothing Comparison')
    
    # Scale tỉ lệ bằng nhau
    max_range = np.array([translations[:,0].max()-translations[:,0].min(), 
                          translations[:,1].max()-translations[:,1].min(), 
                          translations[:,2].max()-translations[:,2].min()]).max() / 2.0
    mid_x = (translations[:,0].max()+translations[:,0].min()) * 0.5
    mid_y = (translations[:,1].max()+translations[:,1].min()) * 0.5
    mid_z = (translations[:,2].max()+translations[:,2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend()
    plot_file = os.path.join(os.path.dirname(output_file), "smoothed_trajectory_plot.png")
    plt.savefig(plot_file, dpi=300)
    print(f"Đã lưu hình ảnh so sánh vào {plot_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Relative path to dataset (e.g. Date_27082026/dataset_104821)")
    args = parser.parse_args()

    # Đường dẫn tương đối
    S2_EXEC_DIR = os.path.dirname(os.path.abspath(__file__))
    S2_OUTPUT_DIR = os.path.abspath(os.path.join(S2_EXEC_DIR, "..", "S2_output_slam"))
    
    target_dir = os.path.join(S2_OUTPUT_DIR, args.path)
    
    if not os.path.exists(target_dir):
        print(f"Lỗi: Không tìm thấy thư mục SLAM output: {target_dir}")
        return
        
    input_file = os.path.join(target_dir, "CameraTrajectory.txt")
    output_file = os.path.join(target_dir, "SmoothedCameraTrajectory.txt")
    
    if not os.path.exists(input_file):
        print(f"Không tìm thấy file quỹ đạo thô: {input_file}")
        print("Vui lòng chạy s2_run_slam.py trước để tạo quỹ đạo thô!")
        sys.exit(1)
        
    print(f"\n>>> LÀM MƯỢT QUỸ ĐẠO CHO: {args.path} <<<")
    smooth_trajectory(input_file, output_file, window_size=51, poly_order=3)

if __name__ == '__main__':
    main()
