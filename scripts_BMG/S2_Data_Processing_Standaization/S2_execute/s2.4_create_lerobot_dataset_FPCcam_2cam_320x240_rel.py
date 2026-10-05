#!/usr/bin/env python3
"""
Tạo LeRobot Dataset (Tương Đối) từ dữ liệu gốc.

Chức năng:
  - Đồng bộ hóa dữ liệu từ FPC Camera, Side Camera, Gripper và SLAM.
  - Nội suy (interpolation) dữ liệu để đạt chuẩn target FPS.
  - Lưu trạng thái Action và State dưới dạng tọa độ TƯƠNG ĐỐI (Relative to Frame 0).
  - Tích hợp ghi đè hoặc nối tiếp (append) các episodes.
"""

import os
import glob
import json
import shutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    print("[!] Không tìm thấy lerobot. Hãy chắc chắn bạn đang dùng venv có cài lerobot.")
    exit(1)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
S1_DIR = os.path.join(BASE_DIR, "..", "..", "S1_Data_Collection", "S1_output")
S2_SLAM_DIR = os.path.join(BASE_DIR, "..", "S2_output_slam")
CALIB_FILE = os.path.join(BASE_DIR, "..", "..", "S0_Camera_Calibration", "S0_output", "Date_18092026", "calibration_matrices_RS_camera", "eye_in_hand_result.json")

# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def get_all_dataset_dirs(date_str):
    dirs = []
    if not os.path.exists(S1_DIR):
        return dirs
    date_path = os.path.join(S1_DIR, date_str)
    if not os.path.isdir(date_path): 
        return dirs
    for d in os.listdir(date_path):
        if d.startswith("dataset_"):
            dirs.append(os.path.join(date_path, d))
    dirs.sort()
    return dirs

# ═══════════════════════════════════════════════════════════════
# CORE PROCESSING
# ═══════════════════════════════════════════════════════════════
def load_and_interpolate_data(dataset_dir, target_fps):
    date_folder = os.path.basename(os.path.dirname(dataset_dir))
    dataset_folder = os.path.basename(dataset_dir)
    slam_dir = os.path.join(S2_SLAM_DIR, date_folder, dataset_folder)

    fpc_file = os.path.join(dataset_dir, "fpc_rgb.txt")
    if not os.path.exists(fpc_file):
        fpc_file = os.path.join(dataset_dir, "web_rgb.txt")
        
    side_file = os.path.join(dataset_dir, "side_rgb.txt")
    gripper_file = os.path.join(dataset_dir, "gripper_log.csv")
    slam_file = os.path.join(slam_dir, "SmoothedCameraTrajectory.txt")

    if not all([os.path.exists(f) for f in [fpc_file, side_file, gripper_file, slam_file]]):
        return None, None, None, None

    print(f"  -> Đã tìm thấy đủ dữ liệu: fpc_rgb, side_rgb, gripper, SLAM")

    # Đọc dữ liệu
    fpc_data = pd.read_csv(fpc_file, sep=" ", comment="#", names=["timestamp", "filename"])
    cam_times = fpc_data["timestamp"].values
    img_files_fpc = fpc_data["filename"].values
    
    side_data = pd.read_csv(side_file, sep=" ", comment="#", names=["timestamp", "filename"])
    img_files_side = side_data["filename"].values

    if len(cam_times) < 2 or len(img_files_side) != len(cam_times):
        print("  -> Lỗi: Số lượng frame FPC và Side không khớp hoặc quá ít.")
        return None, None, None, None
        
    cam_duration = cam_times[-1] - cam_times[0]
    raw_cam_fps = (len(cam_times) - 1) / cam_duration if cam_duration > 0 else 0

    gripper_data = pd.read_csv(gripper_file)
    gripper_times = gripper_data["timestamp"].values
    gripper_states = gripper_data["gripper_state"].values
    
    if len(gripper_times) < 2:
        return None, None, None, None
        
    gripper_duration = gripper_times[-1] - gripper_times[0]
    raw_gripper_fps = (len(gripper_times) - 1) / gripper_duration if gripper_duration > 0 else 0
    
    # Kiểm tra target_fps
    max_fps = min(raw_cam_fps, raw_gripper_fps)
    if target_fps > (max_fps * 1.1):
        raise ValueError(f"LỖI: Target FPS là {target_fps}, nhưng dữ liệu gốc chỉ đạt tối đa {max_fps:.2f} FPS. Vui lòng hạ target_fps xuống!")

    slam_data = np.loadtxt(slam_file)
    if len(slam_data) == 0:
        return None, None, None, None
    slam_times = slam_data[:, 0]
    slam_poses = slam_data[:, 1:8]

    # Lọc frame hợp lệ
    valid_mask = (cam_times >= slam_times.min()) & (cam_times <= slam_times.max())
    cam_times = cam_times[valid_mask]
    img_files_fpc = img_files_fpc[valid_mask]
    img_files_side = img_files_side[valid_mask]

    if len(cam_times) == 0:
        print("  -> Lỗi: Không có frame camera nằm trong thời gian SLAM.")
        return None, None, None, None

    # SUBSAMPLING
    duration = cam_times[-1] - cam_times[0]
    num_target_frames = int(duration * target_fps) + 1
    
    if num_target_frames < len(cam_times):
        target_timestamps = np.linspace(cam_times[0], cam_times[-1], num_target_frames)
        selected_indices = [np.argmin(np.abs(cam_times - t)) for t in target_timestamps]
        selected_indices = sorted(list(set(selected_indices)))
        
        orig_len = len(cam_times)
        cam_times = cam_times[selected_indices]
        img_files_fpc = img_files_fpc[selected_indices]
        img_files_side = img_files_side[selected_indices]
        print(f"  -> Downsample từ {orig_len} xuống {len(cam_times)} frame ({target_fps} FPS)")
    else:
        print(f"  -> Giữ nguyên {len(cam_times)} frame")

    # NỘI SUY
    f_gripper = interp1d(gripper_times, gripper_states, bounds_error=False, fill_value="extrapolate")
    interp_gripper = f_gripper(cam_times)

    f_slam = interp1d(slam_times, slam_poses, axis=0, bounds_error=False, fill_value="extrapolate")
    interp_slam = f_slam(cam_times)

    # Load Extrinsics (T_cam_to_tool)
    with open(CALIB_FILE, 'r') as f:
        calib_data = json.load(f)
    T_cam_to_tool_json = np.array(calib_data["T_cam_to_tool"])
    T_tool_to_cam = np.linalg.inv(T_cam_to_tool_json)

    trans = interp_slam[:, :3]
    quats = interp_slam[:, 3:]
    rot_matrices = R.from_quat(quats).as_matrix()

    N = len(cam_times)
    states_10d = np.zeros((N, 10), dtype=np.float32)

    # Bước 1: Tính T_world_tool cho toàn bộ frames
    T_world_tool_list = []
    for i in range(N):
        T_world_cam = np.eye(4)
        T_world_cam[:3, :3] = rot_matrices[i]
        T_world_cam[:3, 3] = trans[i]
        T_world_tool = T_world_cam @ T_tool_to_cam
        T_world_tool_list.append(T_world_tool)
        
    # Bước 2: TƯƠNG ĐỐI HÓA - Lấy mốc frame 0 làm gốc
    T_world_tool_0_inv = np.linalg.inv(T_world_tool_list[0])

    for i in range(N):
        T_rel_tool = T_world_tool_0_inv @ T_world_tool_list[i]
        
        pos = T_rel_tool[:3, 3]
        rot_tool = T_rel_tool[:3, :3]
        u1 = rot_tool[:, 0]
        u2 = rot_tool[:, 1]
        rot_6d = np.concatenate([u1, u2])

        states_10d[i, :3] = pos
        states_10d[i, 3:9] = rot_6d
        states_10d[i, 9] = interp_gripper[i]

    # Action (Next state)
    actions = np.zeros_like(states_10d)
    actions[:-1] = states_10d[1:]
    actions[-1] = states_10d[-1]

    return img_files_fpc, img_files_side, states_10d, actions

# ═══════════════════════════════════════════════════════════════
# MAIN ROUTING
# ═══════════════════════════════════════════════════════════════
def main():
    if not os.path.exists(S1_DIR):
        print("[!] Không có thư mục S1_output!")
        return
        
    dates = [d for d in os.listdir(S1_DIR) if d.startswith("Date_") and os.path.isdir(os.path.join(S1_DIR, d))]
    dates.sort()
    
    if not dates:
        print("[!] Không tìm thấy thư mục Date_... nào trong S1_output!")
        return
        
    print("\n=== CHỌN THƯ MỤC DỮ LIỆU NGUỒN ===")
    for i, d in enumerate(dates):
        print(f"{i+1}. {d}")
    
    try:
        date_idx = int(input(f"Chọn số tương ứng (1-{len(dates)}): ")) - 1
    except ValueError:
        print("Lựa chọn không hợp lệ!")
        return
        
    if date_idx < 0 or date_idx >= len(dates):
        print("Lựa chọn không hợp lệ!")
        return
        
    source_date = dates[date_idx]
    
    try:
        target_fps = input("\nNhập FPS mong muốn (mặc định 10): ")
        target_fps = int(target_fps) if target_fps.strip() else 10
    except ValueError:
        print("Lựa chọn không hợp lệ!")
        return

    all_dirs = get_all_dataset_dirs(source_date)
    if not all_dirs:
        print(f"[!] Không có dataset nào trong S1_output/{source_date}!")
        return

    out_date_folder = datetime.now().strftime("Date_%d%m%Y")
    OUTPUT_DIR = os.path.join(BASE_DIR, "..", "S2_datasets_lerobot", out_date_folder, f"lerobot_dataset_fpccam_slam_{target_fps}fps_320x240_rel")

    print(f"\nBắt đầu tạo LeRobot Dataset (TƯƠNG ĐỐI) cho Webcam ({target_fps} FPS - 320x240) từ {len(all_dirs)} episodes trong {source_date}...")
    
    append_mode = False
    if os.path.exists(OUTPUT_DIR):
        choice = input(f"\nDataset {OUTPUT_DIR} ĐÃ TỒN TẠI.\nBạn muốn (1) Ghi đè mới hoàn toàn hay (2) Nối tiếp (Append) dữ liệu? [1/2]: ").strip()
        if choice == '1':
            print(f"Đang xóa dataset cũ tại {OUTPUT_DIR}...")
            shutil.rmtree(OUTPUT_DIR)
        else:
            print(f"Chế độ Nối tiếp (Append). Sẽ load dataset cũ và thêm dữ liệu mới vào.")
            append_mode = True

    features = {
        "observation.image": {"dtype": "video", "shape": (3, 240, 320), "names": ["c", "h", "w"]},
        "observation.image_side": {"dtype": "video", "shape": (3, 240, 320), "names": ["c", "h", "w"]},
        "observation.state": {"dtype": "float32", "shape": (10,), "names": ["x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6", "gripper"]},
        "action": {"dtype": "float32", "shape": (10,), "names": ["x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6", "gripper"]}
    }

    processed_log_file = os.path.join(OUTPUT_DIR, "processed_episodes.txt")
    processed_datasets = set()

    if append_mode:
        dataset = LeRobotDataset(repo_id=f"apicoo/sumi_webcam_slam_dataset_rel", root=OUTPUT_DIR)
        if os.path.exists(processed_log_file):
            with open(processed_log_file, "r") as f:
                processed_datasets = set(line.strip() for line in f if line.strip())
            print(f"Đã tìm thấy {len(processed_datasets)} episodes cũ. Sẽ bỏ qua các episodes này.")
    else:
        dataset = LeRobotDataset.create(
            repo_id=f"apicoo/sumi_webcam_slam_dataset_rel",
            fps=target_fps,
            features=features,
            root=OUTPUT_DIR,
            use_videos=True,
            vcodec="h264_nvenc",
            video_backend="ffmpeg",
            image_writer_processes=0,  
            image_writer_threads=16,    
            encoder_queue_maxsize=60,
            streaming_encoding=True
        )

    new_dirs = [d for d in all_dirs if os.path.basename(d) not in processed_datasets]
    if not new_dirs:
        print("\nKhông có dữ liệu mới nào để nối tiếp. Tất cả đã được xử lý từ trước!")
        return
        
    print(f"\nSẽ xử lý {len(new_dirs)} episodes mới...")

    for episode_idx, dataset_dir in enumerate(new_dirs):
        print(f"\n[{episode_idx+1}/{len(new_dirs)}] Đang xử lý: {dataset_dir}")

        try:
            img_files_fpc, img_files_side, states, actions = load_and_interpolate_data(dataset_dir, target_fps)
        except ValueError as e:
            print(f"[-] {e}")
            print("Ngừng quá trình xử lý do FPS đầu vào nhỏ hơn FPS bạn mong muốn.")
            return

        if img_files_fpc is None or len(img_files_fpc) == 0:
            print(f"[-] Bỏ qua episode này do thiếu dữ liệu hoặc frame quá ít.")
            continue

        def process_frame(i):
            img_fpc_path = os.path.join(dataset_dir, img_files_fpc[i])
            img_side_path = os.path.join(dataset_dir, img_files_side[i])
            if not os.path.exists(img_fpc_path) or not os.path.exists(img_side_path): return None

            img_fpc = Image.open(img_fpc_path).convert("RGB")
            if img_fpc.size != (320, 240): img_fpc = img_fpc.resize((320, 240))
            img_fpc_tensor = torch.tensor(np.array(img_fpc), dtype=torch.uint8).permute(2, 0, 1)

            img_side = Image.open(img_side_path).convert("RGB")
            if img_side.size != (320, 240): img_side = img_side.resize((320, 240))
            img_side_tensor = torch.tensor(np.array(img_side), dtype=torch.uint8).permute(2, 0, 1)

            return {
                "observation.image": img_fpc_tensor,
                "observation.image_side": img_side_tensor,
                "observation.state": torch.tensor(states[i], dtype=torch.float32),
                "action": torch.tensor(actions[i], dtype=torch.float32),
                "task": "pick_and_place"
            }

        valid_frames = 0
        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(executor.map(process_frame, range(len(img_files_fpc))))
            
        for frame_dict in results:
            if frame_dict is not None:
                dataset.add_frame(frame_dict)
                valid_frames += 1

        if valid_frames > 0:
            dataset.save_episode()
            print(f"  -> Lưu thành công episode (Length: {valid_frames} frames)")
            with open(processed_log_file, "a") as f:
                f.write(os.path.basename(dataset_dir) + "\n")
        else:
            print(f"  -> Bỏ qua episode (0 frames hợp lệ)")

    print("\nTất cả episodes đã được thêm. Đang finalize dataset...")
    dataset.finalize()
    print(f"Xong! Dataset được lưu tại: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
