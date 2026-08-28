import argparse
import subprocess
import os
import shutil
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Relative path to dataset (e.g. Date_27082026/dataset_104821)")
    args = parser.parse_args()

    # Đường dẫn tương đối từ file script
    # S2_execute -> S2_Data_Processing -> scripts_BMG -> SUMI (chính là gốc của ORB-SLAM3)
    S2_EXEC_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # ROOT_DIR giờ là thư mục SUMI
    ORB_SLAM3_DIR = os.path.abspath(os.path.join(S2_EXEC_DIR, "..", "..", ".."))
    
    # S1_Data_Collection nằm trong scripts_BMG
    S1_OUTPUT_DIR = os.path.abspath(os.path.join(S2_EXEC_DIR, "..", "..", "S1_Data_Collection", "S1_output"))
    S2_OUTPUT_DIR = os.path.abspath(os.path.join(S2_EXEC_DIR, "..", "S2_output_slam"))
    
    input_dataset_dir = os.path.join(S1_OUTPUT_DIR, args.path)
    output_dataset_dir = os.path.join(S2_OUTPUT_DIR, args.path)
    
    if not os.path.exists(input_dataset_dir):
        print(f"Lỗi: Không tìm thấy thư mục input: {input_dataset_dir}")
        return

    os.makedirs(output_dataset_dir, exist_ok=True)
    
    print(f"\n>>> BẮT ĐẦU CHẠY ORB-SLAM3 CHO DATASET: {args.path} <<<")
    print(f"Input: {input_dataset_dir}")
    print(f"Output: {output_dataset_dir}")
    
    slam_cmd = [
        "./Examples/RGB-D/rgbd_tum",
        "Vocabulary/ORBvoc.txt",
        "Examples/RGB-D/RealSense_D435i_Custom.yaml",
        input_dataset_dir,
        os.path.join(input_dataset_dir, "associations.txt")
    ]
    
    try:
        # Thiết lập biến môi trường để trỏ tới file libORB_SLAM3.so
        env = os.environ.copy()
        lib_path = os.path.join(ORB_SLAM3_DIR, "lib")
        dbow2_path = os.path.join(ORB_SLAM3_DIR, "Thirdparty", "DBoW2", "lib")
        g2o_path = os.path.join(ORB_SLAM3_DIR, "Thirdparty", "g2o", "lib")
        extra_paths = f"{lib_path}:{dbow2_path}:{g2o_path}"
        if "LD_LIBRARY_PATH" in env:
            env["LD_LIBRARY_PATH"] = f"{extra_paths}:{env['LD_LIBRARY_PATH']}"
        else:
            env["LD_LIBRARY_PATH"] = extra_paths
            
        # Chạy SLAM. Dùng Popen để theo dõi log, tự động Force Kill khi đã lưu file xong
        process = subprocess.Popen(slam_cmd, cwd=ORB_SLAM3_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            
            if line:
                print(line, end='')
                if "Saving camera trajectory to CameraTrajectory.txt" in line or "Saving keyframe trajectory" in line:
                    time.sleep(2.0)
                    process.terminate()
                    break
                    
        process.wait()
        
        camera_traj_src = os.path.join(ORB_SLAM3_DIR, "CameraTrajectory.txt")
        keyframe_traj_src = os.path.join(ORB_SLAM3_DIR, "KeyFrameTrajectory.txt")
        
        camera_traj_dst = os.path.join(output_dataset_dir, "CameraTrajectory.txt")
        keyframe_traj_dst = os.path.join(output_dataset_dir, "KeyFrameTrajectory.txt")
        
        if os.path.exists(camera_traj_src):
            shutil.move(camera_traj_src, camera_traj_dst)
            print(f"\n[+] Đã lưu quỹ đạo thô vào: {camera_traj_dst}")
        else:
            print("\n[!] Cảnh báo: Không tìm thấy CameraTrajectory.txt, có thể SLAM đã thất bại hoàn toàn.")
            
        if os.path.exists(keyframe_traj_src):
            shutil.move(keyframe_traj_src, keyframe_traj_dst)
            
        print(">>> HOÀN THÀNH CHẠY SLAM VÀ LƯU DỮ LIỆU! <<<")
        
    except Exception as e:
        print(f"Lỗi chạy SLAM: {e}")

if __name__ == "__main__":
    main()
