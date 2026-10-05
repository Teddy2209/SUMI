import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

# ===================================================================
# 0. HÀM TIỆN ÍCH
# ===================================================================
def pose_to_matrix(tx, ty, tz, qx, qy, qz, qw):
    mat = np.eye(4)
    mat[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    mat[0, 3] = tx
    mat[1, 3] = ty
    mat[2, 3] = tz
    return mat

# ===================================================================
# 1. LOAD DỮ LIỆU THỰC TẾ
# ===================================================================
CALIB_FILE = "/media/apicoo-ai/5511010c-3660-41c3-b501-36e739767b6a/SUMI/scripts_BMG/S0_Camera_Calibration/S0_output/Date_18092026/calibration_matrices_RS_camera/eye_in_hand_result.json"
TRAJECTORY_FILE = "/media/apicoo-ai/5511010c-3660-41c3-b501-36e739767b6a/SUMI/scripts_BMG/S2_Data_Processing_Standaization/S2_output_slam/Date_23092026/dataset_181031/CameraTrajectory.txt"

# Đọc ma trận Eye-in-Hand thực tế
with open(CALIB_FILE, 'r') as f:
    calib_data = json.load(f)
T_cam_to_tool = np.array(calib_data["T_cam_to_tool"])

# Đọc 2 dòng đầu tiên của CameraTrajectory.txt (t=0 và t=1)
with open(TRAJECTORY_FILE, 'r') as f:
    lines = f.readlines()
    line_0 = lines[0].strip().split()
    line_1 = lines[1].strip().split()

# Dòng 0 (Thời điểm bắt đầu SLAM)
tx0, ty0, tz0, qx0, qy0, qz0, qw0 = map(float, line_0[1:8])
T_world_cam_0 = pose_to_matrix(tx0, ty0, tz0, qx0, qy0, qz0, qw0)

# Dòng 1 (Điểm mục tiêu tiếp theo)
tx1, ty1, tz1, qx1, qy1, qz1, qw1 = map(float, line_1[1:8])
T_world_cam_1 = pose_to_matrix(tx1, ty1, tz1, qx1, qy1, qz1, qw1)


# ===================================================================
# 2. TÍNH TOÁN THEO CÁCH CŨ (DỮ LIỆU THỰC TẾ CỦA BẠN)
# ===================================================================
T_tool_to_cam = np.linalg.inv(T_cam_to_tool)  # JSON lưu G_T_C → đảo để có C_T_G
T_world_tool_0 = T_world_cam_0 @ T_tool_to_cam
T_world_tool_1 = T_world_cam_1 @ T_tool_to_cam

print("="*70)
print("DỮ LIỆU CŨ: TỌA ĐỘ TOOL TRONG HỆ SLAM WORLD (TUYỆT ĐỐI)")
print("="*70)
print(f"Ma trận Eye-in-Hand gốc (từ file JSON):")
print(f"X: {T_cam_to_tool[0,3]*1000:.1f}mm, Y: {T_cam_to_tool[1,3]*1000:.1f}mm, Z: {T_cam_to_tool[2,3]*1000:.1f}mm")

print(f"\nTọa độ Tool ở t=0 (RAW PREDICT MÀ AI NHẢ RA LÚC BẮT ĐẦU):")
print(f"X: {T_world_tool_0[0,3]*1000:.1f}mm, Y: {T_world_tool_0[1,3]*1000:.1f}mm, Z: {T_world_tool_0[2,3]*1000:.1f}mm")

print(f"\nTọa độ Tool ở t=1 (Điểm AI nhả ra tiếp theo):")
print(f"X: {T_world_tool_1[0,3]*1000:.1f}mm, Y: {T_world_tool_1[1,3]*1000:.1f}mm, Z: {T_world_tool_1[2,3]*1000:.1f}mm")


# ===================================================================
# 3. TÍNH TOÁN THEO CÁCH CỦA UMI (HỆ TỌA ĐỘ TOOL TƯƠNG ĐỐI)
# ===================================================================
T_world_tool_0_inv = np.linalg.inv(T_world_tool_0)

# Trừ đi mốc T (Đổi hệ quy chiếu)
T_rel_0 = T_world_tool_0_inv @ T_world_tool_0
T_rel_1 = T_world_tool_0_inv @ T_world_tool_1

print("\n" + "="*70)
print("CÁCH LÀM CỦA UMI: CHIẾU TỌA ĐỘ LÊN HỆ TOOL TƯƠNG ĐỐI")
print("="*70)
print(f"Tọa độ Tool ở t=0 (Mốc hiện tại):")
print(f"X: {T_rel_0[0,3]*1000:.1f}mm, Y: {T_rel_0[1,3]*1000:.1f}mm, Z: {T_rel_0[2,3]*1000:.1f}mm")

print(f"\nTọa độ Tool ở t=1 (Độ dời thực tế mà Tool phải đi so với chính nó):")
print(f"X: {T_rel_1[0,3]*1000:.3f}mm, Y: {T_rel_1[1,3]*1000:.3f}mm, Z: {T_rel_1[2,3]*1000:.3f}mm")

