import argparse
import os
import sys
import importlib.util

# Import hàm smooth_trajectory từ file có ký tự đặc biệt (s2.1_...)
spec = importlib.util.spec_from_file_location("smooth_traj", os.path.join(os.path.dirname(__file__), "s2.1_smooth_tracjectory.py"))
smooth_traj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smooth_traj)
smooth_trajectory = smooth_traj.smooth_trajectory

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Relative path to Date folder (e.g. Date_27082026)")
    args = parser.parse_args()

    # Đường dẫn tương đối
    S2_EXEC_DIR = os.path.dirname(os.path.abspath(__file__))
    S2_OUTPUT_DIR = os.path.abspath(os.path.join(S2_EXEC_DIR, "..", "S2_output_slam"))
    
    target_date_dir = os.path.join(S2_OUTPUT_DIR, args.path)
    
    if not os.path.exists(target_date_dir):
        print(f"Lỗi: Không tìm thấy thư mục SLAM output: {target_date_dir}")
        return

    # Lấy danh sách các dataset đã được chạy SLAM trong thư mục Date
    datasets = [d for d in os.listdir(target_date_dir) if os.path.isdir(os.path.join(target_date_dir, d)) and d.startswith("dataset_")]
    datasets.sort()
    
    if not datasets:
        print(f"Không tìm thấy dataset nào trong {target_date_dir}")
        return
        
    print(f"Tìm thấy {len(datasets)} bộ dữ liệu trong {args.path}. Bắt đầu xử lý làm mượt...")
    
    for ds_name in datasets:
        dataset_dir = os.path.join(target_date_dir, ds_name)
        input_file = os.path.join(dataset_dir, "CameraTrajectory.txt")
        output_file = os.path.join(dataset_dir, "SmoothedCameraTrajectory.txt")
        
        if not os.path.exists(input_file):
            print(f"[-] Bỏ qua {ds_name}: Không có CameraTrajectory.txt (chưa chạy SLAM?)")
            continue
            
        print(f"\n=======================================================")
        print(f">>> LÀM MƯỢT QUỸ ĐẠO CHO: {args.path}/{ds_name} <<<")
        print(f"=======================================================")
        
        try:
            smooth_trajectory(input_file, output_file, window_size=51, poly_order=3)
        except Exception as e:
            print(f"Lỗi làm mượt cho {ds_name}: {e}")

    print("\n>>> HOÀN THÀNH BATCH SMOOTHING! <<<")

if __name__ == "__main__":
    main()
