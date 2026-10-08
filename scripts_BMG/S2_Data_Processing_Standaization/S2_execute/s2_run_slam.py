#!/usr/bin/env python3
"""
Chạy ORB-SLAM3 trên dữ liệu.

Chức năng:
  - Khởi chạy ORB-SLAM3 với file cấu hình và từ vựng.
  - Tự động dừng SLAM khi lưu xong quỹ đạo.
  - Hỗ trợ chạy 1 dataset lẻ hoặc chạy hàng loạt toàn bộ thư mục Date.
"""

import argparse
import subprocess
import os
import sys
import shutil
import time

from s2_menu import ask_overwrite, choose_path

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR giờ là thư mục SUMI
ORB_SLAM3_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))

S1_OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "S1_Data_Collection", "S1_output"))
S2_OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "S2_output_slam"))

# ═══════════════════════════════════════════════════════════════
# CORE PROCESSING
# ═══════════════════════════════════════════════════════════════
def run_slam_for_dataset(ds_rel_path):
    input_dataset_dir = os.path.join(S1_OUTPUT_DIR, ds_rel_path)
    output_dataset_dir = os.path.join(S2_OUTPUT_DIR, ds_rel_path)
    
    if not os.path.exists(input_dataset_dir):
        print(f"[-] Lỗi: Không tìm thấy thư mục input: {input_dataset_dir}")
        return False
        
    os.makedirs(output_dataset_dir, exist_ok=True)
    
    camera_traj_dst = os.path.join(output_dataset_dir, "CameraTrajectory.txt")
    keyframe_traj_dst = os.path.join(output_dataset_dir, "KeyFrameTrajectory.txt")

    print(f"\n" + "="*50)
    print(f">>> BẮT ĐẦU CHẠY ORB-SLAM3 CHO DATASET: {ds_rel_path} <<<")
    print(f"=======================================================")
    
    slam_cmd = [
        "./Examples/RGB-D/rgbd_tum",
        "Vocabulary/ORBvoc.txt",
        "Examples/RGB-D/RealSense_D435i_Custom.yaml",
        input_dataset_dir,
        os.path.join(input_dataset_dir, "associations.txt")
    ]
    
    try:
        env = os.environ.copy()
        lib_path = os.path.join(ORB_SLAM3_DIR, "lib")
        dbow2_path = os.path.join(ORB_SLAM3_DIR, "Thirdparty", "DBoW2", "lib")
        g2o_path = os.path.join(ORB_SLAM3_DIR, "Thirdparty", "g2o", "lib")
        extra_paths = f"{lib_path}:{dbow2_path}:{g2o_path}"
        
        if "LD_LIBRARY_PATH" in env:
            env["LD_LIBRARY_PATH"] = f"{extra_paths}:{env['LD_LIBRARY_PATH']}"
        else:
            env["LD_LIBRARY_PATH"] = extra_paths
            
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
        
        if os.path.exists(camera_traj_src):
            shutil.move(camera_traj_src, camera_traj_dst)
            print(f"\n[+] Đã lưu quỹ đạo thô vào: {camera_traj_dst}")
        else:
            print(f"\n[!] Cảnh báo: Không tìm thấy CameraTrajectory.txt, SLAM thất bại cho {ds_rel_path}")
            
        if os.path.exists(keyframe_traj_src):
            shutil.move(keyframe_traj_src, keyframe_traj_dst)
            
        return True
    except Exception as e:
        print(f"Lỗi chạy SLAM cho {ds_rel_path}: {e}")
        return False

# ═══════════════════════════════════════════════════════════════
# MAIN ROUTING
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Khởi chạy ORB-SLAM3")
    parser.add_argument("--path", default=None, help="Đường dẫn đến dataset hoặc thư mục Date (VD: Date_27082026/dataset_104821 OR Date_27082026). Bỏ trống để mở menu chọn")
    args = parser.parse_args()

    rel_path = args.path or choose_path(S1_OUTPUT_DIR)
    if not rel_path:
        print("[!] Không có lựa chọn hợp lệ!")
        sys.exit(1)

    target_path = os.path.join(S1_OUTPUT_DIR, rel_path)

    if not os.path.exists(target_path):
        print(f"[!] Lỗi: Không tìm thấy thư mục input: {target_path}")
        sys.exit(1)

    # 1 dataset hoặc cả thư mục Date (chứa nhiều dataset)
    if os.path.basename(target_path).startswith("dataset_"):
        print(f"\n>>> CHẾ ĐỘ: CHẠY SLAM CHO 1 DATASET <<<")
        ds_list = [rel_path]
    else:
        datasets = sorted(d for d in os.listdir(target_path) if os.path.isdir(os.path.join(target_path, d)) and d.startswith("dataset_"))
        if not datasets:
            print(f"[!] Không tìm thấy dataset nào trong {target_path}")
            sys.exit(1)
        print(f"\n>>> CHẾ ĐỘ: CHẠY HÀNG LOẠT ({len(datasets)} datasets) <<<")
        ds_list = [os.path.join(rel_path, d) for d in datasets]

    # Đã có dữ liệu → hỏi bỏ qua / ghi đè trước khi chạy
    done = [p for p in ds_list if os.path.exists(os.path.join(S2_OUTPUT_DIR, p, "CameraTrajectory.txt"))]
    if done and not ask_overwrite(done):
        ds_list = [p for p in ds_list if p not in done]
        print(f"[-] Bỏ qua {len(done)} dataset đã có dữ liệu.")

    for ds in ds_list:
        run_slam_for_dataset(ds)

    print("\n>>> HOÀN THÀNH CHẠY BATCH SLAM! <<<")

if __name__ == "__main__":
    main()
