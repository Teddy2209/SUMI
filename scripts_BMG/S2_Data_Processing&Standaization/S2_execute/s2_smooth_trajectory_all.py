import os
import sys

# Thêm đường dẫn để import hàm từ s2_Smoothe_tracjectory.py
sys.path.append(os.path.dirname(__file__))
from s2_Smoothe_tracjectory import smooth_trajectory

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

def get_all_dataset_dirs():
    if not os.path.exists(OUTPUT_DIR): return []
    dirs = [d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d)) and d.startswith("dataset_tum")]
    dirs.sort() # Sắp xếp theo tên (thời gian)
    return [os.path.join(OUTPUT_DIR, d) for d in dirs]

def main():
    all_dirs = get_all_dataset_dirs()
    if not all_dirs:
        print("Không tìm thấy bộ dữ liệu nào trong output!")
        return
        
    print(f"Tìm thấy {len(all_dirs)} bộ dữ liệu. Bắt đầu xử lý làm mượt...")
    
    for dataset_dir in all_dirs:
        input_file = os.path.join(dataset_dir, "CameraTrajectory.txt")
        output_file = os.path.join(dataset_dir, "SmoothedCameraTrajectory.txt")
        
        if not os.path.exists(input_file):
            print(f"[-] Bỏ qua {os.path.basename(dataset_dir)}: Không có CameraTrajectory.txt (chưa chạy SLAM?)")
            continue
            
        print(f"\n>>> LÀM MƯỢT QUỸ ĐẠO CHO: {os.path.basename(dataset_dir)} <<<")
        try:
            smooth_trajectory(input_file, output_file, window_size=51, poly_order=3)
        except Exception as e:
            print(f"Lỗi làm mượt cho {os.path.basename(dataset_dir)}: {e}")

    print("\n>>> HOÀN THÀNH BATCH SMOOTHING! <<<")

if __name__ == "__main__":
    main()
