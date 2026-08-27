import subprocess
import os
import shutil

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

def get_latest_dataset_dir():
    if not os.path.exists(OUTPUT_DIR): return None
    dirs = [d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d)) and d.startswith("dataset_tum")]
    if not dirs: return None
    dirs.sort(key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)), reverse=True)
    return os.path.join(OUTPUT_DIR, dirs[0])

def main():
    latest_dir_full = get_latest_dataset_dir()
    if not latest_dir_full:
        print("Không tìm thấy bộ dữ liệu nào trong output!")
        return
        
    latest_dir_rel = os.path.relpath(latest_dir_full, BASE_DIR)
    print(f"\n>>> BẮT ĐẦU CHẠY ORB-SLAM3 CHO DATASET: {latest_dir_rel} <<<")
    
    slam_cmd = [
        "./Examples/RGB-D/rgbd_tum",
        "Vocabulary/ORBvoc.txt",
        "Examples/RGB-D/RealSense_D435i_Custom.yaml",
        latest_dir_rel,
        f"{latest_dir_rel}/associations.txt"
    ]
    
    import time
    try:
        # Chạy SLAM. Dùng Popen để theo dõi log, tự động Force Kill khi đã lưu file xong
        # nhằm tránh lỗi treo (deadlock) của ORB-SLAM3 khi shutdown.
        process = subprocess.Popen(slam_cmd, cwd=BASE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            
            if line:
                print(line, end='')
                # ORB-SLAM3 có thể treo ngay sau khi in "Saving camera trajectory" nếu tracking fail
                if "Saving camera trajectory to CameraTrajectory.txt" in line or "Saving keyframe trajectory" in line:
                    # Chờ 2 giây để đảm bảo tiến trình C++ đã ghi file xong (việc ghi chỉ tốn < 0.1s)
                    time.sleep(2.0)
                    process.terminate() # Bắn tín hiệu Force Kill
                    break
                    
        process.wait()
        
        # Di chuyển kết quả SLAM vào thư mục output của dataset hiện tại
        camera_traj_src = os.path.join(BASE_DIR, "CameraTrajectory.txt")
        keyframe_traj_src = os.path.join(BASE_DIR, "KeyFrameTrajectory.txt")
        
        camera_traj_dst = os.path.join(latest_dir_full, "CameraTrajectory.txt")
        keyframe_traj_dst = os.path.join(latest_dir_full, "KeyFrameTrajectory.txt")
        
        if os.path.exists(camera_traj_src):
            shutil.move(camera_traj_src, camera_traj_dst)
            print(f"Đã lưu quỹ đạo thô vào: {camera_traj_dst}")
        else:
            print("Cảnh báo: Không tìm thấy CameraTrajectory.txt, có thể SLAM đã thất bại hoàn toàn.")
            
        if os.path.exists(keyframe_traj_src):
            shutil.move(keyframe_traj_src, keyframe_traj_dst)
            
        print(">>> HOÀN THÀNH CHẠY SLAM VÀ LƯU DỮ LIỆU! <<<")
        
    except Exception as e:
        print(f"Lỗi xử lý file: {e}")

if __name__ == "__main__":
    main()
