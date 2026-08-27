import os
import glob
import pandas as pd
import numpy as np
import random
from scipy.spatial.transform import Rotation as R

def get_first_parquet(dataset_dir):
    pattern = os.path.join(dataset_dir, "data", "**", "*.parquet")
    files = glob.glob(pattern, recursive=True)
    if not files: return None
    files.sort()
    return files[0]

def extract_states(parquet_file):
    df = pd.read_parquet(parquet_file)
    # df['observation.state'] chứa mảng 10D: [x, y, z, r1..r6, gripper]
    states = np.stack(df['observation.state'].values)
    return states

def reconstruct_T(state):
    """
    Tái tạo lại ma trận 4x4 (Pose) từ state 10D.
    state = [x, y, z, r1, r2, r3, r4, r5, r6, gripper]
    """
    x, y, z = state[:3]
    v1 = state[3:6]
    v2 = state[6:9]
    v3 = np.cross(v1, v2) # Trục Z của Rot Matrix
    
    T = np.eye(4)
    T[:3, 0] = v1
    T[:3, 1] = v2
    T[:3, 2] = v3
    T[:3, 3] = [x, y, z]
    return T

def main():
    gt_dir = "lerobot_dataset_diff_gt"
    slam_dir = "lerobot_dataset_diff_slam"

    file_gt = get_first_parquet(gt_dir)
    file_slam = get_first_parquet(slam_dir)

    if not (file_gt and file_slam):
        print("Chưa tìm thấy file parquet. Có thể script sinh dataset chưa chạy xong!")
        return

    states_gt = extract_states(file_gt)
    states_slam = extract_states(file_slam)
    
    n_frames = min(len(states_gt), len(states_slam))
    
    # 1. Chọn ngẫu nhiên 6 frame
    random.seed(42) # Cố định seed để dễ debug
    sample_indices = random.sample(range(n_frames), 6)
    
    print("\n" + "="*70)
    print("1. TÍNH TOÁN MA TRẬN 'T_world_to_base' TỪ 6 CẶP ĐIỂM NGẪU NHIÊN")
    print("="*70)
    
    T_base_world_list = []
    
    for idx in sample_indices:
        # T_base_tool (từ GT)
        T_bt = reconstruct_T(states_gt[idx])
        # T_world_tool (từ SLAM)
        T_wt = reconstruct_T(states_slam[idx])
        
        # Công thức: T_bt = T_bw @ T_wt  =>  T_bw = T_bt @ inv(T_wt)
        T_bw = T_bt @ np.linalg.inv(T_wt)
        T_base_world_list.append(T_bw)
        
    # Tính giá trị trung bình của 6 ma trận T_bw này
    trans_avg = np.mean([T[:3, 3] for T in T_base_world_list], axis=0)
    rotations = R.from_matrix([T[:3, :3] for T in T_base_world_list])
    rot_avg = rotations.mean().as_matrix()
    
    T_bw_avg = np.eye(4)
    T_bw_avg[:3, :3] = rot_avg
    T_bw_avg[:3, 3] = trans_avg
    
    print(f"Đã lấy trung bình từ 6 frame: {sample_indices}")
    print("Ma trận Alignment (T_world_to_base) tính được:")
    print(np.round(T_bw_avg, 4))
    
    print("\n" + "="*70)
    print("2. KIỂM CHỨNG: DÙNG MA TRẬN NÀY CHIẾU LẠI 5 ĐIỂM BẤT KỲ KHÁC")
    print("="*70)
    
    # Chọn 5 frame khác hoàn toàn để test
    test_indices = random.sample([i for i in range(n_frames) if i not in sample_indices], 5)
    
    errors = []
    for idx in test_indices:
        T_bt_real = reconstruct_T(states_gt[idx])
        T_wt = reconstruct_T(states_slam[idx])
        
        # CHIẾU: SLAM World -> Robot Base
        T_bt_proj = T_bw_avg @ T_wt
        
        pos_real = T_bt_real[:3, 3]
        pos_proj = T_bt_proj[:3, 3]
        
        # Tính khoảng cách sai số Euclidean (m)
        error = np.linalg.norm(pos_real - pos_proj)
        errors.append(error)
        
        print(f"--- Frame {idx:04d} ---")
        print(f"  + Tọa độ Tool thực tế (GT)   : [{pos_real[0]:8.4f}, {pos_real[1]:8.4f}, {pos_real[2]:8.4f}]")
        print(f"  + Tọa độ Tool (SLAM chiếu về): [{pos_proj[0]:8.4f}, {pos_proj[1]:8.4f}, {pos_proj[2]:8.4f}]")
        print(f"  => Lệch: {error*1000:.2f} mm\n")
        
    print(f"Sai số trung bình trên các tập Test: {np.mean(errors)*1000:.2f} mm")

if __name__ == "__main__":
    main()
