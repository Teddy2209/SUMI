import subprocess
import os
import shutil

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
        
    print(f"Tìm thấy {len(all_dirs)} bộ dữ liệu. Bắt đầu xử lý...")
    
    for dataset_dir in all_dirs:
        camera_traj_dst = os.path.join(dataset_dir, "CameraTrajectory.txt")
        keyframe_traj_dst = os.path.join(dataset_dir, "KeyFrameTrajectory.txt")
        
        # Bỏ qua nếu đã có kết quả SLAM
        if os.path.exists(camera_traj_dst):
            print(f"[-] Bỏ qua {os.path.basename(dataset_dir)}: Đã có CameraTrajectory.txt")
            continue
            
        dataset_dir_rel = os.path.relpath(dataset_dir, BASE_DIR)
        print(f"\n>>> BẮT ĐẦU CHẠY ORB-SLAM3 CHO DATASET: {dataset_dir_rel} <<<")
        
        slam_cmd = [
            "./Examples/RGB-D/rgbd_tum",
            "Vocabulary/ORBvoc.txt",
            "Examples/RGB-D/RealSense_D435i_Custom.yaml",
            dataset_dir_rel,
            f"{dataset_dir_rel}/associations.txt"
        ]
        
        try:
            # Chạy SLAM ở thư mục gốc (BASE_DIR).
            subprocess.run(slam_cmd, cwd=BASE_DIR)
            
            # Di chuyển kết quả SLAM vào thư mục output của dataset hiện tại
            camera_traj_src = os.path.join(BASE_DIR, "CameraTrajectory.txt")
            keyframe_traj_src = os.path.join(BASE_DIR, "KeyFrameTrajectory.txt")
            
            if os.path.exists(camera_traj_src):
                shutil.move(camera_traj_src, camera_traj_dst)
                print(f"[+] Đã lưu quỹ đạo thô vào: {camera_traj_dst}")
            else:
                print(f"[!] Cảnh báo: Không tìm thấy CameraTrajectory.txt, SLAM thất bại cho {dataset_dir_rel}")
                
            if os.path.exists(keyframe_traj_src):
                shutil.move(keyframe_traj_src, keyframe_traj_dst)
                
        except Exception as e:
            print(f"Lỗi xử lý file cho dataset {dataset_dir_rel}: {e}")

    print("\n>>> HOÀN THÀNH CHẠY BATCH SLAM! <<<")

if __name__ == "__main__":
    main()
