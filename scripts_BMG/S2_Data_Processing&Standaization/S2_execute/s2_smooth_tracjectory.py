import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter
import sys
import os

# Đường dẫn thư mục output
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

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

def get_latest_dataset_dir():
    if not os.path.exists(OUTPUT_DIR): return None
    dirs = [d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d)) and d.startswith("dataset_tum")]
    if not dirs: return None
    dirs.sort(key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)), reverse=True)
    return os.path.join(OUTPUT_DIR, dirs[0])

if __name__ == '__main__':
    latest_dir = get_latest_dataset_dir()
    if latest_dir:
        default_input = os.path.join(latest_dir, "CameraTrajectory.txt")
        default_output = os.path.join(latest_dir, "SmoothedCameraTrajectory.txt")
    else:
        default_input = os.path.join(OUTPUT_DIR, "CameraTrajectory.txt")
        default_output = os.path.join(OUTPUT_DIR, "SmoothedCameraTrajectory.txt")

    input_file = default_input
    output_file = default_output
    
    # Cho phép ghi đè đường dẫn bằng đối số dòng lệnh nếu cần
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
        
    if not os.path.exists(input_file):
        print(f"Không tìm thấy file quỹ đạo thô: {input_file}")
        print("Vui lòng chạy s1.5_run_slam.py trước để tạo quỹ đạo thô!")
        sys.exit(1)
        
    print(f"\n>>> LÀM MƯỢT QUỸ ĐẠO CHO: {input_file} <<<")
    smooth_trajectory(input_file, output_file, window_size=51, poly_order=3)
