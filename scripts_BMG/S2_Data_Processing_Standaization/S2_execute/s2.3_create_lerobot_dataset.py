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

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Import LeRobot
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    print("Không tìm thấy lerobot. Hãy chắc chắn bạn đang dùng venv có cài lerobot.")
    exit(1)

def get_all_dataset_dirs():
    if not os.path.exists(OUTPUT_DIR): return []
    dirs = [d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d)) and d.startswith("dataset_tum")]
    dirs.sort()
    return [os.path.join(OUTPUT_DIR, d) for d in dirs]

def downsample_times(times, files, target_hz=20.0):
    if len(times) == 0:
        return times, files
    
    interval = 1.0 / target_hz
    keep_indices = [0]
    last_t = times[0]
    
    for i in range(1, len(times)):
        if times[i] - last_t >= interval:
            keep_indices.append(i)
            last_t = times[i]
            
    return times[keep_indices], files[keep_indices]

def load_and_interpolate_data(dataset_dir, mode):
    rgb_file = os.path.join(dataset_dir, "rgb.txt")
    gripper_file = os.path.join(dataset_dir, "gripper_log.csv")
    
    if "slam" in mode:
        slam_file = os.path.join(dataset_dir, "SmoothedCameraTrajectory.txt")
        if not all([os.path.exists(f) for f in [rgb_file, slam_file, gripper_file]]):
            return None, None, None
    else:
        robot_file = os.path.join(dataset_dir, "robot_log.csv")
        if not all([os.path.exists(f) for f in [rgb_file, robot_file, gripper_file]]):
            return None, None, None

    # Đọc rgb
    rgb_data = pd.read_csv(rgb_file, sep=" ", comment="#", names=["timestamp", "filename"])
    cam_times = rgb_data["timestamp"].values
    img_files = rgb_data["filename"].values

    # Đọc Gripper
    gripper_data = pd.read_csv(gripper_file)
    gripper_times = gripper_data["timestamp"].values
    gripper_states = gripper_data["gripper_state"].values
    
    # Tính valid mask
    min_t = gripper_times.min()
    max_t = gripper_times.max()

    if "slam" in mode:
        slam_data = np.loadtxt(slam_file)
        if len(slam_data) == 0: return None, None, None
        slam_times = slam_data[:, 0]
        min_t = max(min_t, slam_times.min())
        max_t = min(max_t, slam_times.max())
    else:
        robot_data = pd.read_csv(robot_file).values
        robot_times = robot_data[:, 0]
        min_t = max(min_t, robot_times.min())
        max_t = min(max_t, robot_times.max())

    valid_mask = (cam_times >= min_t) & (cam_times <= max_t)
    cam_times = cam_times[valid_mask]
    img_files = img_files[valid_mask]

    if len(cam_times) == 0:
        return None, None, None

    # Nếu mode là diff (20Hz), downsample cam_times
    if "diff" in mode:
        cam_times, img_files = downsample_times(cam_times, img_files, target_hz=20.0)

    # Nội suy Gripper
    f_gripper = interp1d(gripper_times, gripper_states, bounds_error=False, fill_value="extrapolate")
    interp_gripper = f_gripper(cam_times)

    N = len(cam_times)
    states_10d = np.zeros((N, 10), dtype=np.float32)

    if "slam" in mode:
        slam_poses = slam_data[:, 1:8] # x,y,z,qx,qy,qz,qw
        f_slam = interp1d(slam_times, slam_poses, axis=0, bounds_error=False, fill_value="extrapolate")
        interp_slam = f_slam(cam_times)
        
        calib_file = os.path.join(BASE_DIR, "test", "eye_in_hand_result.json")
        with open(calib_file, 'r') as f:
            calib_data = json.load(f)
        T_cam_to_tool = np.array(calib_data["T_cam_to_tool"])
        
        trans = interp_slam[:, :3]
        quats = interp_slam[:, 3:]
        rot_matrices = R.from_quat(quats).as_matrix() # (N, 3, 3)
        
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
            
    elif "gt" in mode:
        # robot_data columns: 0=time, 1..6=q, 7=x, 8=y, 9=z, 10=u, 11=v, 12=w
        robot_poses = robot_data[:, 7:13]
        f_robot = interp1d(robot_times, robot_poses, axis=0, bounds_error=False, fill_value="extrapolate")
        interp_robot = f_robot(cam_times)
        
        trans = interp_robot[:, :3] / 1000.0 # mm to meters
        eulers = interp_robot[:, 3:] # degrees
        rot_matrices = R.from_euler('xyz', eulers, degrees=True).as_matrix()
        
        for i in range(N):
            pos = trans[i]
            rot_tool = rot_matrices[i]
            u1 = rot_tool[:, 0]
            u2 = rot_tool[:, 1]
            rot_6d = np.concatenate([u1, u2])
            
            states_10d[i, :3] = pos
            states_10d[i, 3:9] = rot_6d
            states_10d[i, 9] = interp_gripper[i]

    actions = np.zeros_like(states_10d)
    actions[:-1] = states_10d[1:]
    actions[-1] = states_10d[-1]
    
    return img_files, states_10d, actions

def main():
    parser = argparse.ArgumentParser(description="Tạo LeRobot Dataset với 4 Mode khác nhau")
    parser.add_argument("--mode", type=str, choices=["act_gt", "act_slam", "diff_gt", "diff_slam"], required=True, help="Chế độ xử lý dataset")
    args = parser.parse_args()

    mode = args.mode
    dataset_name = f"lerobot_dataset_{mode}"
    target_path = os.path.join(BASE_DIR, dataset_name)
    target_fps = 55 if "act" in mode else 20

    all_dirs = get_all_dataset_dirs()
    if not all_dirs:
        print("Không có dataset_tum nào!")
        return

    print(f"Bắt đầu tạo LeRobot dataset ({mode}) từ {len(all_dirs)} episodes...")
    print(f"Target FPS: {target_fps}")

    if os.path.exists(target_path):
        print(f"Xóa dataset cũ tại {target_path}...")
        shutil.rmtree(target_path)

    features = {
        "observation.image": {
            "dtype": "video",
            "shape": (3, 540, 960),
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
        repo_id=f"apicoo/robot_pick_place_{mode}",
        fps=target_fps,
        features=features,
        root=target_path,
        use_videos=True
    )

    episode_idx = 0
    for dataset_dir in all_dirs:
        print(f"Đang xử lý episode {episode_idx} từ {os.path.basename(dataset_dir)}...")

        img_files, states, actions = load_and_interpolate_data(dataset_dir, mode)

        if img_files is None or len(img_files) == 0:
            print(f"[-] Bỏ qua {os.path.basename(dataset_dir)}: Dữ liệu không đủ hoặc lỗi nội suy.")
            continue

        for i in range(len(img_files)):
            img_path = os.path.join(dataset_dir, img_files[i])
            if not os.path.exists(img_path):
                continue

            img = Image.open(img_path).convert("RGB")
            img_tensor = torch.tensor(np.array(img), dtype=torch.uint8).permute(2, 0, 1)

            frame_dict = {
                "observation.image": img_tensor,
                "observation.state": torch.tensor(states[i], dtype=torch.float32),
                "action": torch.tensor(actions[i], dtype=torch.float32),
                "task": "pick_and_place"
            }
            dataset.add_frame(frame_dict)

        dataset.save_episode()
        episode_idx += 1

    print("Tất cả episodes đã được thêm. Đang finalize dataset...")
    dataset.finalize()
    print(f"Xong! Dataset được lưu tại: {target_path}")

if __name__ == "__main__":
    main()
