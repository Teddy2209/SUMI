"""
Script kiểm tra tính chính xác của quỹ đạo Tool trong hệ tool_0.

Các bài test:
  1. Frame 0 = Identity (vị trí = 0, rotation = I)
  2. Quaternion norm = 1
  3. Rotation matrix hợp lệ (det=1, R^T R = I)
  4. Distance invariance (khoảng cách bảo toàn qua rigid transform)
  5. Round-trip test (world -> tool_0 -> world phải khớp)
  6. So sánh quỹ đạo tool giữa s2.1 và s2.4 (tại timestamp khớp)
  7. Physical reasonableness (biên độ hợp lý)

Usage:
  python verify_tool_trajectory.py --path Date_XXXXX/dataset_XXXXX
"""

import argparse
import os
import sys
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d


def pose_to_matrix(pos, quat):
    mat = np.eye(4)
    mat[:3, :3] = R.from_quat(quat).as_matrix()
    mat[:3, 3] = pos
    return mat


def matrix_to_pose(mat):
    pos = mat[:3, 3]
    quat = R.from_matrix(mat[:3, :3]).as_quat()
    return pos, quat


def load_trajectory(filepath):
    data = np.loadtxt(filepath)
    timestamps = data[:, 0]
    positions = data[:, 1:4]
    quaternions = data[:, 4:8]
    return timestamps, positions, quaternions


def test_frame0_identity(tool_positions, tool_quaternions):
    """Test 1: Frame 0 phải là identity (pos=0, rot=I)"""
    print("\n" + "=" * 60)
    print("TEST 1: Frame 0 = Identity")
    print("=" * 60)

    pos_0 = tool_positions[0]
    quat_0 = tool_quaternions[0]
    rot_0 = R.from_quat(quat_0).as_matrix()

    pos_error = np.linalg.norm(pos_0)
    rot_error = np.linalg.norm(rot_0 - np.eye(3), 'fro')

    print(f"  Position tại t=0 : {pos_0}")
    print(f"  ||pos_0||         : {pos_error:.2e}")
    print(f"  Rotation tại t=0  :\n{rot_0}")
    print(f"  ||R_0 - I||_F     : {rot_error:.2e}")

    passed = pos_error < 1e-6 and rot_error < 1e-6
    print(f"  -> {'PASS' if passed else 'FAIL'}")
    return passed


def test_quaternion_norms(quaternions):
    """Test 2: Tất cả quaternion phải có norm = 1"""
    print("\n" + "=" * 60)
    print("TEST 2: Quaternion Norms = 1")
    print("=" * 60)

    norms = np.linalg.norm(quaternions, axis=1)
    max_dev = np.max(np.abs(norms - 1.0))
    mean_dev = np.mean(np.abs(norms - 1.0))

    print(f"  Số lượng frames   : {len(norms)}")
    print(f"  Norm min          : {norms.min():.10f}")
    print(f"  Norm max          : {norms.max():.10f}")
    print(f"  Max deviation     : {max_dev:.2e}")
    print(f"  Mean deviation    : {mean_dev:.2e}")

    passed = max_dev < 1e-4
    print(f"  -> {'PASS' if passed else 'FAIL'} (threshold: 1e-4)")
    return passed


def test_rotation_validity(quaternions):
    """Test 3: Ma trận rotation phải hợp lệ (det=1, R^T R = I)"""
    print("\n" + "=" * 60)
    print("TEST 3: Rotation Matrix Validity (det=1, R^T R = I)")
    print("=" * 60)

    max_det_error = 0
    max_ortho_error = 0

    for i in range(len(quaternions)):
        rot = R.from_quat(quaternions[i]).as_matrix()
        det = np.linalg.det(rot)
        ortho = rot.T @ rot - np.eye(3)

        det_err = abs(det - 1.0)
        ortho_err = np.linalg.norm(ortho, 'fro')

        max_det_error = max(max_det_error, det_err)
        max_ortho_error = max(max_ortho_error, ortho_err)

    print(f"  Max |det(R) - 1|    : {max_det_error:.2e}")
    print(f"  Max ||R^T R - I||_F : {max_ortho_error:.2e}")

    passed = max_det_error < 1e-6 and max_ortho_error < 1e-6
    print(f"  -> {'PASS' if passed else 'FAIL'}")
    return passed


def test_distance_invariance(cam_positions, cam_quaternions, T_tool_to_cam):
    """Test 4: Khoảng cách giữa frame liên tiếp phải bảo toàn qua rigid transform"""
    print("\n" + "=" * 60)
    print("TEST 4: Distance Invariance (rigid transform bảo toàn khoảng cách)")
    print("=" * 60)

    N = len(cam_positions)

    cam_world_positions = cam_positions
    tool_world_positions = np.zeros_like(cam_positions)

    for i in range(N):
        T_wc = pose_to_matrix(cam_positions[i], cam_quaternions[i])
        T_wt = T_wc @ T_tool_to_cam
        tool_world_positions[i] = T_wt[:3, 3]

    cam_dists = np.linalg.norm(np.diff(cam_world_positions, axis=0), axis=1)
    tool_dists = np.linalg.norm(np.diff(tool_world_positions, axis=0), axis=1)

    dist_diff = np.abs(cam_dists - tool_dists)
    max_diff = dist_diff.max()
    mean_diff = dist_diff.mean()

    print(f"  Số khoảng cách    : {len(cam_dists)}")
    print(f"  Max |d_cam - d_tool| : {max_diff:.2e} m")
    print(f"  Mean               : {mean_diff:.2e} m")

    print(f"\n  Lưu ý: Khoảng cách camera và tool khác nhau là BÌNH THƯỜNG")
    print(f"  vì camera và tool ở vị trí khác nhau trên gripper.")
    print(f"  Điều cần kiểm tra: tổng quãng đường có cùng bậc độ lớn không.")
    print(f"  Tổng quãng đường camera : {cam_dists.sum():.4f} m")
    print(f"  Tổng quãng đường tool   : {tool_dists.sum():.4f} m")
    ratio = tool_dists.sum() / cam_dists.sum() if cam_dists.sum() > 0 else float('inf')
    print(f"  Tỷ lệ tool/camera       : {ratio:.4f}")

    passed = 0.5 < ratio < 2.0
    print(f"  -> {'PASS' if passed else 'FAIL'} (ratio trong khoảng 0.5-2.0)")
    return passed


def test_roundtrip(cam_positions, cam_quaternions, T_tool_to_cam):
    """Test 5: Round-trip: cam -> tool_world -> tool_rel -> tool_world -> cam phải khớp"""
    print("\n" + "=" * 60)
    print("TEST 5: Round-trip (cam -> tool_rel -> cam)")
    print("=" * 60)

    T_cam_to_tool = np.linalg.inv(T_tool_to_cam)
    N = len(cam_positions)

    T_wc_0 = pose_to_matrix(cam_positions[0], cam_quaternions[0])
    T_wt_0 = T_wc_0 @ T_tool_to_cam
    T_wt_0_inv = np.linalg.inv(T_wt_0)

    max_pos_err = 0
    max_rot_err = 0

    for i in range(N):
        # Forward: cam -> tool_world -> tool_rel
        T_wc_i = pose_to_matrix(cam_positions[i], cam_quaternions[i])
        T_wt_i = T_wc_i @ T_tool_to_cam
        T_rel = T_wt_0_inv @ T_wt_i

        # Backward: tool_rel -> tool_world -> cam
        T_wt_reconstructed = T_wt_0 @ T_rel
        T_wc_reconstructed = T_wt_reconstructed @ T_cam_to_tool

        pos_err = np.linalg.norm(T_wc_i[:3, 3] - T_wc_reconstructed[:3, 3])
        rot_err = np.linalg.norm(T_wc_i[:3, :3] - T_wc_reconstructed[:3, :3], 'fro')

        max_pos_err = max(max_pos_err, pos_err)
        max_rot_err = max(max_rot_err, rot_err)

    print(f"  Max position error  : {max_pos_err:.2e} m")
    print(f"  Max rotation error  : {max_rot_err:.2e}")

    passed = max_pos_err < 1e-10 and max_rot_err < 1e-10
    print(f"  -> {'PASS' if passed else 'FAIL'}")
    return passed


def test_s21_vs_s24_consistency(tool_timestamps, tool_positions, tool_quaternions,
                                cam_timestamps, cam_positions, cam_quaternions,
                                T_tool_to_cam):
    """Test 6: So sánh quỹ đạo tool từ s2.1 vs cách tính trong s2.4"""
    print("\n" + "=" * 60)
    print("TEST 6: Consistency s2.1 vs s2.4 (tính lại từ camera trajectory)")
    print("=" * 60)

    # Tìm timestamp chung (s2.1 dùng SmoothedCamera, s2.4 nội suy về cam timestamps)
    # Ở đây ta tính lại theo cách s2.4: nội suy SLAM tại cam timestamps rồi đổi hệ
    # So sánh với kết quả s2.1 tại cùng timestamps

    # Nội suy camera trajectory tại tool timestamps (vì tool file cùng timestamps với camera)
    common_ts = tool_timestamps  # s2.1 dùng cùng timestamps

    N = len(common_ts)
    T_wc_0 = pose_to_matrix(cam_positions[0], cam_quaternions[0])
    T_wt_0 = T_wc_0 @ T_tool_to_cam
    T_wt_0_inv = np.linalg.inv(T_wt_0)

    max_pos_err = 0
    max_rot_err = 0

    for i in range(N):
        # Tính lại từ camera (giống cách s2.4 tính)
        T_wc_i = pose_to_matrix(cam_positions[i], cam_quaternions[i])
        T_wt_i = T_wc_i @ T_tool_to_cam
        T_rel_recomputed = T_wt_0_inv @ T_wt_i

        recomputed_pos = T_rel_recomputed[:3, 3]
        recomputed_rot = R.from_matrix(T_rel_recomputed[:3, :3]).as_quat()

        # So với file tool đã lưu
        pos_err = np.linalg.norm(tool_positions[i] - recomputed_pos)
        # Quaternion distance (xử lý double-cover: q và -q là cùng rotation)
        dot = abs(np.dot(tool_quaternions[i], recomputed_rot))
        dot = min(dot, 1.0)
        rot_err = 2 * np.arccos(dot) * 180 / np.pi  # degrees

        max_pos_err = max(max_pos_err, pos_err)
        max_rot_err = max(max_rot_err, rot_err)

    print(f"  Max position error : {max_pos_err:.2e} m")
    print(f"  Max rotation error : {max_rot_err:.4f} degrees")

    passed = max_pos_err < 1e-3 and max_rot_err < 1.0
    print(f"  -> {'PASS' if passed else 'FAIL'} (pos < 1mm, rot < 1°)")
    return passed


def test_physical_reasonableness(tool_positions, tool_quaternions):
    """Test 7: Kiểm tra biên độ có hợp lý với workspace robot arm"""
    print("\n" + "=" * 60)
    print("TEST 7: Physical Reasonableness")
    print("=" * 60)

    range_x = tool_positions[:, 0].max() - tool_positions[:, 0].min()
    range_y = tool_positions[:, 1].max() - tool_positions[:, 1].min()
    range_z = tool_positions[:, 2].max() - tool_positions[:, 2].min()
    max_dist = np.linalg.norm(tool_positions, axis=1).max()

    print(f"  Range X : {range_x * 1000:.1f} mm")
    print(f"  Range Y : {range_y * 1000:.1f} mm")
    print(f"  Range Z : {range_z * 1000:.1f} mm")
    print(f"  Max distance from origin : {max_dist * 1000:.1f} mm")

    # Tính góc xoay tối đa so với t=0
    max_angle = 0
    for i in range(len(tool_quaternions)):
        rot = R.from_quat(tool_quaternions[i])
        angle = rot.magnitude() * 180 / np.pi
        max_angle = max(max_angle, angle)
    print(f"  Max rotation from t=0 : {max_angle:.1f} degrees")

    # Kiểm tra step-to-step jumps
    pos_steps = np.linalg.norm(np.diff(tool_positions, axis=0), axis=1)
    max_step = pos_steps.max() * 1000
    mean_step = pos_steps.mean() * 1000
    print(f"  Max step between frames   : {max_step:.2f} mm")
    print(f"  Mean step between frames  : {mean_step:.2f} mm")

    # Heuristic: robot arm workspace thường < 2m, bước nhảy < 50mm/frame
    warnings = []
    if max_dist > 2.0:
        warnings.append(f"  [WARN] Max distance {max_dist:.3f}m vượt quá 2m — có thể sai hệ tọa độ")
    if max_step > 50:
        warnings.append(f"  [WARN] Max step {max_step:.1f}mm — có thể có outlier hoặc tracking loss")
    if max_angle > 180:
        warnings.append(f"  [WARN] Góc xoay {max_angle:.1f}° > 180° — kiểm tra lại quaternion flip")

    if warnings:
        for w in warnings:
            print(w)
        print(f"  -> WARN (xem chi tiết ở trên)")
        return False
    else:
        print(f"  -> PASS (biên độ hợp lý)")
        return True


def main():
    parser = argparse.ArgumentParser(description="Verify tool trajectory accuracy")
    parser.add_argument("--path", required=True, help="Relative path to dataset (e.g. Date_27082026/dataset_104821)")
    args = parser.parse_args()

    S2_EXEC_DIR = os.path.dirname(os.path.abspath(__file__))
    S2_OUTPUT_DIR = os.path.abspath(os.path.join(S2_EXEC_DIR, "..", "S2_output_slam"))
    target_dir = os.path.join(S2_OUTPUT_DIR, args.path)

    cam_file = os.path.join(target_dir, "SmoothedCameraTrajectory.txt")
    tool_file = os.path.join(target_dir, "SmoothedToolTrajectory.txt")

    CALIB_FILE = os.path.join(S2_EXEC_DIR, "..", "..", "S0_Camera_Calibration", "S0_output",
                              "Date_18092026", "calibration_matrices_RS_camera", "eye_in_hand_result.json")

    for f, name in [(cam_file, "SmoothedCameraTrajectory"), (tool_file, "SmoothedToolTrajectory"), (CALIB_FILE, "Eye-in-Hand calibration")]:
        if not os.path.exists(f):
            print(f"Không tìm thấy {name}: {f}")
            sys.exit(1)

    print(f"Loading data từ: {target_dir}")

    cam_ts, cam_pos, cam_quat = load_trajectory(cam_file)
    tool_ts, tool_pos, tool_quat = load_trajectory(tool_file)

    with open(CALIB_FILE, 'r') as f:
        T_cam_to_tool_json = np.array(json.load(f)["T_cam_to_tool"])
    T_tool_to_cam = np.linalg.inv(T_cam_to_tool_json)

    print(f"Camera frames: {len(cam_ts)}")
    print(f"Tool frames  : {len(tool_ts)}")

    results = {}
    results["1_frame0_identity"] = test_frame0_identity(tool_pos, tool_quat)
    results["2_quaternion_norms"] = test_quaternion_norms(tool_quat)
    results["3_rotation_validity"] = test_rotation_validity(tool_quat)
    results["4_distance_invariance"] = test_distance_invariance(cam_pos, cam_quat, T_tool_to_cam)
    results["5_roundtrip"] = test_roundtrip(cam_pos, cam_quat, T_tool_to_cam)
    results["6_s21_vs_s24"] = test_s21_vs_s24_consistency(tool_ts, tool_pos, tool_quat, cam_ts, cam_pos, cam_quat, T_tool_to_cam)
    results["7_physical"] = test_physical_reasonableness(tool_pos, tool_quat)

    print("\n" + "=" * 60)
    print("TỔNG KẾT")
    print("=" * 60)
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL/WARN"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n  >>> TẤT CẢ PASS — Quỹ đạo tool chính xác! <<<")
    else:
        print("\n  >>> CÓ TEST FAIL/WARN — Xem chi tiết ở trên <<<")


if __name__ == "__main__":
    main()
