import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
import shutil
import json
import argparse

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
S1_DIR = os.path.join(BASE_DIR, "..", "..", "S1_Data_Collection", "S1_output")
S2_SLAM_DIR = os.path.join(BASE_DIR, "..", "S2_output_slam")
# Đường dẫn CALIB_FILE thay đổi thành dạng tương đối
CALIB_FILE = os.path.join(BASE_DIR, "..", "..", "S0_Camera_Calibration", "S0_output", "intrinsics_matrixes_28082026", "extrinsic_matrixes", "fpccamera_to_tool.json")

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    print("Không tìm thấy lerobot. Hãy chắc chắn bạn đang dùng venv có cài lerobot.")
    exit(1)

def get_all_dataset_dirs():
    dirs = []
    if not os.path.exists(S1_DIR):
        return dirs
    for date_d in os.listdir(S1_DIR):
        date_path = os.path.join(S1_DIR, date_d)
        if not os.path.isdir(date_path): continue
        for d in os.listdir(date_path):
            if d.startswith("dataset_"):
                dirs.append(os.path.join(date_path, d))
    dirs.sort()
    return dirs

def load_and_interpolate_data(dataset_dir, target_fps):
    date_folder = os.path.basename(os.path.dirname(dataset_dir))
    dataset_folder = os.path.basename(dataset_dir)
    slam_dir = os.path.join(S2_SLAM_DIR, date_folder, dataset_folder)

    web_file = os.path.join(dataset_dir, "web_rgb.txt")
    gripper_file = os.path.join(dataset_dir, "gripper_log.csv")
    slam_file = os.path.join(slam_dir, "SmoothedCameraTrajectory.txt")

    if not all([os.path.exists(f) for f in [web_file, gripper_file, slam_file]]):
        return None, None, None

    print(f"  -> Đã tìm thấy đủ dữ liệu: web_rgb, gripper, SLAM")

    # Đọc Webcam làm Timestamp gốc (Master Clock)
    web_data = pd.read_csv(web_file, sep=" ", comment="#", names=["timestamp", "filename"])
    cam_times = web_data["timestamp"].values
    img_files = web_data["filename"].values

    if len(cam_times) < 2:
        return None, None, None
        
    # Tính toán tần số max của webcam
    cam_duration = cam_times[-1] - cam_times[0]
    raw_cam_fps = (len(cam_times) - 1) / cam_duration if cam_duration > 0 else 0

    # Đọc Gripper
    gripper_data = pd.read_csv(gripper_file)
    gripper_times = gripper_data["timestamp"].values
    gripper_states = gripper_data["gripper_state"].values
    
    if len(gripper_times) < 2:
        return None, None, None
        
    gripper_duration = gripper_times[-1] - gripper_times[0]
    raw_gripper_fps = (len(gripper_times) - 1) / gripper_duration if gripper_duration > 0 else 0
    
    # 1. Kiểm tra target_fps không được vượt quá max_fps (Thêm biên độ dao động 10% do nhiễu thời gian thực)
    max_fps = min(raw_cam_fps, raw_gripper_fps)
    if target_fps > (max_fps * 1.1):
        raise ValueError(f"LỖI: Bạn setup tần số tạo dataset là {target_fps} FPS, nhưng dữ liệu gốc chỉ đạt tối đa {max_fps:.2f} FPS. Vui lòng hạ target_fps xuống!")

    # Đọc SLAM
    slam_data = np.loadtxt(slam_file)
    if len(slam_data) == 0:
        return None, None, None
    slam_times = slam_data[:, 0]
    slam_poses = slam_data[:, 1:8] # x,y,z,qx,qy,qz,qw

    # Chỉ giữ lại các frame webcam nằm lọt trong khoảng thời gian mà SLAM tracking thành công
    valid_mask = (cam_times >= slam_times.min()) & (cam_times <= slam_times.max())
    cam_times = cam_times[valid_mask]
    img_files = img_files[valid_mask]

    if len(cam_times) == 0:
        print("  -> Lỗi: Không có frame webcam nào nằm trong khoảng thời gian Tracking của SLAM.")
        return None, None, None

    # SUBSAMPLING: Lọc bớt frame để đưa về đúng với target_fps
    duration = cam_times[-1] - cam_times[0]
    num_target_frames = int(duration * target_fps) + 1
    
    if num_target_frames < len(cam_times):
        # Nội suy đều các mốc thời gian để chọn index gốc gần nhất
        target_timestamps = np.linspace(cam_times[0], cam_times[-1], num_target_frames)
        selected_indices = []
        for t in target_timestamps:
            idx = np.argmin(np.abs(cam_times - t))
            selected_indices.append(idx)
            
        # Loại bỏ các frame trùng (nếu có) và sắp xếp
        selected_indices = sorted(list(set(selected_indices)))
        
        orig_len = len(cam_times)
        cam_times = cam_times[selected_indices]
        img_files = img_files[selected_indices]
        print(f"  -> Đã downsample từ {orig_len} frame xuống còn {len(cam_times)} frame ({target_fps} FPS)")
    else:
        print(f"  -> Giữ nguyên {len(cam_times)} frame (Không cần downsample do raw FPS gần bằng hoặc thấp hơn chút xíu target FPS)")

    # Nội suy Gripper (sau khi đã lọc frame)
    f_gripper = interp1d(gripper_times, gripper_states, bounds_error=False, fill_value="extrapolate")
    interp_gripper = f_gripper(cam_times)

    # Nội suy SLAM (sau khi đã lọc frame)
    f_slam = interp1d(slam_times, slam_poses, axis=0, bounds_error=False, fill_value="extrapolate")
    interp_slam = f_slam(cam_times)

    # Load Extrinsics
    with open(CALIB_FILE, 'r') as f:
        calib_data = json.load(f)
    T_cam_to_tool = np.array(calib_data["T_cam_to_tool"])

    trans = interp_slam[:, :3]
    quats = interp_slam[:, 3:]
    rot_matrices = R.from_quat(quats).as_matrix() # (N, 3, 3)

    N = len(cam_times)
    states_10d = np.zeros((N, 10), dtype=np.float32)

    for i in range(N):
        T_world_cam = np.eye(4)
        T_world_cam[:3, :3] = rot_matrices[i]
        T_world_cam[:3, 3] = trans[i]

        T_world_tool = T_world_cam @ T_cam_to_tool
        pos = T_world_tool[:3, 3]
        rot_tool = T_world_tool[:3, :3]

        u1 = rot_tool[:, 0]
        u2 = rot_tool[:, 1]
        rot_6d = np.concatenate([u1, u2])

        states_10d[i, :3] = pos
        states_10d[i, 3:9] = rot_6d
        states_10d[i, 9] = interp_gripper[i]

    # Action là state ở frame kế tiếp
    actions = np.zeros_like(states_10d)
    actions[:-1] = states_10d[1:]
    actions[-1] = states_10d[-1]

    return img_files, states_10d, actions

def main():
    parser = argparse.ArgumentParser(description="Tạo dataset LeRobot")
    parser.add_argument("--fps", type=int, default=10, help="Tần số FPS muốn chuyển đổi cho dataset")
    args = parser.parse_args()
    
    target_fps = args.fps

    all_dirs = get_all_dataset_dirs()
    if not all_dirs:
        print("Không có dataset nào trong S1_output!")
        return

    OUTPUT_DIR = os.path.join(BASE_DIR, "..","S2_datasets_lerobot", f"lerobot_dataset_fpccam_slam_{target_fps}fps")

    print(f"Bắt đầu tạo LeRobot Dataset cho Webcam ({target_fps} FPS) từ {len(all_dirs)} episodes...")
    
    if os.path.exists(OUTPUT_DIR):
        print(f"Xóa dataset cũ tại {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)

    features = {
        "observation.image": {
            "dtype": "video",
            "shape": (3, 480, 640),
            "names": ["c", "h", "w"]
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (10,),
            "names": ["x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6", "gripper"]
        },
        "action": {
            "dtype": "float32",
            "shape": (10,),
            "names": ["x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6", "gripper"]
        }
    }

    dataset = LeRobotDataset.create(
        repo_id=f"apicoo/sumi_webcam_slam_dataset",
        fps=target_fps,
        features=features,
        root=OUTPUT_DIR,
        use_videos=True
    )

    episode_idx = 0
    for dataset_dir in all_dirs:
        print(f"\n[{episode_idx+1}/{len(all_dirs)}] Đang xử lý: {dataset_dir}")

        try:
            img_files, states, actions = load_and_interpolate_data(dataset_dir, target_fps)
        except ValueError as e:
            print(f"[-] {e}")
            print("Ngừng quá trình xử lý do FPS đầu vào nhỏ hơn FPS bạn mong muốn.")
            return

        if img_files is None or len(img_files) == 0:
            print(f"[-] Bỏ qua episode này do thiếu dữ liệu hoặc frame quá ít.")
            continue

        valid_frames = 0
        for i in range(len(img_files)):
            img_path = os.path.join(dataset_dir, img_files[i])
            if not os.path.exists(img_path):
                continue

            img = Image.open(img_path).convert("RGB")
            if img.size != (640, 480):
                img = img.resize((640, 480))

            img_tensor = torch.tensor(np.array(img), dtype=torch.uint8).permute(2, 0, 1)

            frame_dict = {
                "observation.image": img_tensor,
                "observation.state": torch.tensor(states[i], dtype=torch.float32),
                "action": torch.tensor(actions[i], dtype=torch.float32),
                "task": "pick_and_place"
            }
            dataset.add_frame(frame_dict)
            valid_frames += 1

        if valid_frames > 0:
            dataset.save_episode()
            print(f"  -> Lưu thành công episode (Length: {valid_frames} frames)")
            episode_idx += 1
        else:
            print(f"  -> Bỏ qua episode (0 frames hợp lệ)")

    print("\nTất cả episodes đã được thêm. Đang finalize dataset...")
    dataset.finalize()
    print(f"Xong! Dataset được lưu tại: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
