#!/usr/bin/env python3
import time
import threading
import queue
import json
import os
import argparse
from pathlib import Path
import numpy as np
import cv2
import pyrealsense2 as rs
from pymodbus.client import ModbusSerialClient
import torch
from scipy.spatial.transform import Rotation as R
from neuromeka import IndyDCP3

from lerobot.policies.act.modeling_act import ACTPolicy

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_POLICY_PATH = os.path.join(BASE_DIR, "output_trained", "act_checkpoints_act_slam_v2", "checkpoints", "last", "pretrained_model")
CALIB_FILE = os.path.join(BASE_DIR, "test", "eye_in_hand_result.json")

GRIPPER_OPEN_MM = 120.0
GRIPPER_CLOSE_MM = 30.0

H, W = 540, 960
FPS = 55
ROBOT_IP = "192.168.2.100"

# ============================================================
# HARDWARE CLASSES
# ============================================================
class robot:
    def __init__(self):
        self.robot = IndyDCP3(ROBOT_IP)
        self.latest_p = np.zeros(6, dtype=np.float32)
        
        self.running = True
        self.lock = threading.Lock()
        self._cmd_queue = queue.Queue()
        self._last_read_ts = time.time()
        
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        time.sleep(1.0)

    def _poll(self):
        while self.running:
            cmd = None
            try:
                cmd = self._cmd_queue.get_nowait()
            except queue.Empty:
                pass

            if cmd is not None:
                try:
                    self.robot.movetelel_abs(
                        tpos=list(np.asarray(cmd, dtype=np.float32)),
                        vel_ratio=0.1,  # Vận tốc an toàn
                        acc_ratio=1.0
                    )
                except Exception as e:
                    print(f"Robot command error: {e}")
            try:
                p = self.robot.get_robot_data()["p"]
                with self.lock:
                    self.latest_p = np.array(p, dtype=np.float32)
                    self._last_read_ts = time.time()
            except Exception as e:
                print(f"Lỗi đọc data robot: {e}")
            
            time.sleep(0.01)

    def get_robot_data(self):
        with self.lock:
            return self.latest_p.copy()

    def start_teleop(self):
        try:
            self.robot.stop_teleop()
        except Exception:
            pass
        time.sleep(0.1)
        self.robot.start_teleop(method=0)

    def stop_teleop(self):
        self.robot.stop_teleop()
        self.robot.stop_motion()

    def send_task_target(self, p_target):
        self._cmd_queue.put(p_target)

    def stop(self):
        self.running = False
        try:
            self.thread.join(timeout=0.2)
        except Exception:
            pass

class camera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, W, H, rs.format.rgb8, 60)
        
        profile = self.pipeline.start(self.config)
        
        try:
            sensors = profile.get_device().query_sensors()
            for sensor in sensors:
                if sensor.supports(rs.option.frames_queue_size):
                    sensor.set_option(rs.option.frames_queue_size, 1)
        except Exception:
            pass

        self.latest_frame = None
        self.running = True
        self.lock = threading.Lock()
        
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        time.sleep(1.0)

    def _poll(self):
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=100)
                color_frame = frames.get_color_frame()
                if color_frame:
                    img = np.array(color_frame.get_data())
                    with self.lock:
                        self.latest_frame = img
            except Exception:
                pass
            time.sleep(0.01)

    def get_images(self):
        with self.lock:
            if self.latest_frame is None:
                return np.zeros((H, W, 3), dtype=np.uint8)
            return self.latest_frame.copy()

    def stop(self):
        self.running = False
        try:
            self.thread.join(timeout=0.2)
        except Exception:
            pass
        try:
            self.pipeline.stop()
        except Exception:
            pass

class gripper:
    def __init__(self):
        self.client = ModbusSerialClient(
            port='/dev/ttyUSB0', baudrate=115200, stopbits=1,
            bytesize=8, parity='N', timeout=0.2, retries=0, handle_local_echo=False
        )
        self.client.connect()
        self.current_state = 1.0
        
        self._lock = threading.Lock()
        self._cmd_queue = queue.Queue()
        self._running = True

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            cmd = None
            try:
                cmd = self._cmd_queue.get_nowait()
            except queue.Empty:
                pass

            if cmd is not None:
                try:
                    self.client.write_register(1, cmd, device_id=1)
                except Exception:
                    pass
            else:
                try:
                    res = self.client.read_input_registers(address=1, count=1, device_id=1)
                    if not res.isError():
                        pos_mm = float(res.registers[0])
                        norm = np.clip((pos_mm - GRIPPER_CLOSE_MM) / (GRIPPER_OPEN_MM - GRIPPER_CLOSE_MM), 0.0, 1.0)
                        with self._lock:
                            self.current_state = float(norm)
                except Exception:
                    pass
            time.sleep(0.01)

    def close(self):
        self._cmd_queue.put(int(GRIPPER_CLOSE_MM))

    def open(self):
        self._cmd_queue.put(int(GRIPPER_OPEN_MM))
    
    def move(self, norm_pos):
        # Đảm bảo giá trị nằm trong khoảng an toàn [0.0, 1.0]
        norm_pos = float(np.clip(norm_pos, 0.0, 1.0))
        # norm_pos (0.0 đến 1.0) -> pos_mm
        pos_mm = norm_pos * (GRIPPER_OPEN_MM - GRIPPER_CLOSE_MM) + GRIPPER_CLOSE_MM
        self._cmd_queue.put(int(pos_mm))


    def get_gripper_state(self):
        with self._lock:
            return self.current_state

    def stop(self):
        self._running = False
        try:
            self._thread.join(timeout=0.5)
        except Exception:
            pass
        try:
            self.client.close()
        except Exception:
            pass

# ============================================================
# KINEMATICS & SAFETY
# ============================================================
def clamp_task_target(current_p, target_p, max_trans=20.0, max_rot=5.0):
    """
    Giới hạn bước di chuyển tối đa trong không gian Descartes để an toàn.
    """
    current_p = np.asarray(current_p, dtype=np.float32)
    target_p = np.asarray(target_p, dtype=np.float32)
    
    # 1. Tịnh tiến (Translation in mm)
    delta_trans = target_p[:3] - current_p[:3]
    max_abs_trans = np.max(np.abs(delta_trans))
    if max_abs_trans > max_trans:
        scale_t = max_trans / max_abs_trans
        scaled_trans = delta_trans * scale_t
    else:
        scaled_trans = delta_trans
        
    # 2. Xoay (Rotation in deg)
    delta_rot = (target_p[3:6] - current_p[3:6] + 180.0) % 360.0 - 180.0
    max_abs_rot = np.max(np.abs(delta_rot))
    if max_abs_rot > max_rot:
        scale_r = max_rot / max_abs_rot
        scaled_rot = delta_rot * scale_r
    else:
        scaled_rot = delta_rot
        
    safe_p = current_p.copy()
    safe_p[:3] += scaled_trans
    safe_p[3:6] = (current_p[3:6] + scaled_rot + 180.0) % 360.0 - 180.0
    return safe_p

def tcp_to_matrix(p):
    T = np.eye(4)
    r = R.from_euler('xyz', p[3:6], degrees=True)
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = p[:3]
    return T

def angular_dist(a, b):
    diff = (a - b + 180.0) % 360.0 - 180.0
    return np.linalg.norm(diff)

def get_closest_euler(target_eulers, current_eulers):
    """
    Tìm bộ Euler (xyz) gần nhất với góc xoay hiện tại của Robot để tránh Gimbal Lock.
    """
    alt_eulers = np.array([
        target_eulers[0] + 180.0,
        180.0 - target_eulers[1],
        target_eulers[2] + 180.0
    ])
    
    dist1 = angular_dist(target_eulers, current_eulers)
    dist2 = angular_dist(alt_eulers, current_eulers)
    
    if dist2 < dist1:
        return (alt_eulers + 180.0) % 360.0 - 180.0
    return (target_eulers + 180.0) % 360.0 - 180.0

# ============================================================
# MAIN INFERENCE LOOP
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", type=str, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--execute", action="store_true", help="Execute on physical robot")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🚀 ACT POLICY INFERENCE (CARTESIAN)")
    print("=" * 70)

    device = torch.device(args.device)
    
    # Load Hand-Eye Calibration
    with open(CALIB_FILE, 'r') as f:
        calib_data = json.load(f)
    T_cam_to_tcp = np.array(calib_data["T_cam_to_tool"])
    T_cam_to_tcp[:3, 3] *= 1000.0 # Convert to mm
    T_cam_to_tcp_inv = np.linalg.inv(T_cam_to_tcp)

    # Load Policy
    print(f"Loading ACT model from: {args.policy_path}")
    policy = ACTPolicy.from_pretrained(args.policy_path, local_files_only=True)
    policy.config.temporal_ensemble_coeff = 0.01
    policy.config.n_action_steps = 1
    from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
    policy.temporal_ensembler = ACTTemporalEnsembler(0.01, policy.config.chunk_size)
    policy.reset()
    policy.to(device)
    policy.eval()

    print("Loading Preprocessors...")
    from lerobot.policies.factory import make_pre_post_processors
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=args.policy_path)

    # Initialize Threads
    print("Connecting to Indy7...")
    indy = robot()
    print("Connecting to SusGrip...")
    susgrip = gripper()
    print("Initializing Realsense Camera...")
    cam = camera()

    # Lấy vị trí ban đầu của Robot khi bắt đầu inference
    time.sleep(1.0) # Đợi các luồng thu thập dữ liệu
    p_tcp_0 = indy.get_robot_data()
    print(f"Initial Robot Pose: {np.round(p_tcp_0, 2)}")
    T_tcp_0 = tcp_to_matrix(p_tcp_0)
    T_tcp_0_inv = np.linalg.inv(T_tcp_0)
    
    # Ma trận SLAM ban đầu (Tương ứng với gốc tọa độ)
    T_slam_0 = np.eye(4)
    T_slam_0_inv = np.linalg.inv(T_slam_0)

    # Hàm quy đổi T_tcp_curr sang AI Observation State (10D SLAM Pose)
    def get_obs_dict(p_tcp_curr, g_state, rgb_img):
        T_tcp_curr = tcp_to_matrix(p_tcp_curr)
        
        # Inverse Kinematics: delta_T_cam = T_cam_to_tcp_inv @ T_tcp_0_inv @ T_tcp_curr @ T_cam_to_tcp
        delta_T_cam = T_cam_to_tcp_inv @ T_tcp_0_inv @ T_tcp_curr @ T_cam_to_tcp
        
        T_world_cam = T_slam_0 @ delta_T_cam
        T_world_tool = T_world_cam @ T_cam_to_tcp
        
        tx = T_world_tool[0, 3] / 1000.0 # Chuyển mm -> m
        ty = T_world_tool[1, 3] / 1000.0
        tz = T_world_tool[2, 3] / 1000.0
        
        R_tool = T_world_tool[:3, :3]
        r1, r2, r3 = R_tool[:, 0]
        r4, r5, r6 = R_tool[:, 1]
        
        obs_state = np.array([tx, ty, tz, r1, r2, r3, r4, r5, r6, g_state], dtype=np.float32)
        obs_state_t = torch.from_numpy(obs_state).unsqueeze(0).to(device)

        chw = np.transpose(rgb_img, (2, 0, 1)).astype(np.float32) / 255.0
        img_t = torch.from_numpy(chw).unsqueeze(0).to(device)

        return {
            "observation.state": obs_state_t,
            "observation.image": img_t,
        }

    # Model CUDA Graph Warmup
    print("Warming up CUDA Graph...")
    dummy_obs = get_obs_dict(indy.get_robot_data(), susgrip.get_gripper_state(), cam.get_images())
    dummy_obs = preprocessor(dummy_obs)
    with torch.inference_mode():
        for _ in range(3):
            policy.select_action(dummy_obs)
    print("Warmup complete.")
    
    if hasattr(policy, "reset"):
        policy.reset()

    if args.execute:
        print("\n⚠️ [WARNING] EXECUTE MODE ACTIVE!")
        confirm = input("Type 'YES' to transmit commands: ").strip()
        if confirm != "YES":
            return
            
    dt = 1.0 / FPS
    print(f"Starting inference loop...\n")

    if args.execute:
        print("Enabling Indy Teleop (Method 0)...")
        indy.start_teleop()
        # Wait for TELE_OP transition
        transition_success = False
        print("Waiting for TELE_OP transition...")
        for _ in range(30):
            state = indy.robot.get_robot_data().get("op_state", 0)
            if state == 17:
                transition_success = True
                print("✅ Robot successfully entered TELE_OP mode!")
                break
            time.sleep(0.02)
    
        if not transition_success:
            print(f"❌ CRITICAL ERROR: Robot failed to enter TELE_OP mode (Mode {state}). Aborting.")
            return

        p_curr = indy.get_robot_data()
        
        # Keep-alive buffering
        for _ in range(10):
            indy.send_task_target(p_curr)
            time.sleep(0.01)

    frame_idx = 0
    last_g_cmd = "open"

    try:
        while True:
            loop_start = time.perf_counter()
            frame_idx += 1

            # 1. Lấy trạng thái vật lý thực tế
            p_phys_curr = indy.get_robot_data()
            g_state = susgrip.get_gripper_state()
            rgb_img = cam.get_images() # Đã là RGB từ RS config

            # 2. Quy đổi ra SLAM Pose (Ảo) để nạp vào AI
            obs_dict = get_obs_dict(p_phys_curr, g_state, rgb_img)
            obs_dict = preprocessor(obs_dict)

            # 3. Dự đoán Action (Tọa độ SLAM đích)
            with torch.inference_mode():
                action_pred = policy.select_action(obs_dict)
                
            action_pred = postprocessor(action_pred)
            
            # Action của LeRobot Diffusion có thể là [action_dim] hoặc [1, action_dim]
            if isinstance(action_pred, dict):
                act_vec = action_pred["action"].squeeze(0).cpu().numpy()
            else:
                act_vec = action_pred.squeeze(0).cpu().numpy()
                
            # Đảm bảo act_vec có shape là (10,)
            if len(act_vec.shape) > 1 and act_vec.shape[0] == 1:
                act_vec = act_vec[0]

            target_tx, target_ty, target_tz = act_vec[0], act_vec[1], act_vec[2]
            v1 = act_vec[3:6]
            v2 = act_vec[6:9]
            g_cmd_target = act_vec[9]
            
            v3 = np.cross(v1, v2)
            R_mat = np.column_stack((v1, v2, v3))

            # 4. Quy đổi ngược từ SLAM Target ra TCP Target
            T_world_tool_target = np.eye(4)
            T_world_tool_target[:3, :3] = R_mat
            T_world_tool_target[:3, 3] = np.array([target_tx, target_ty, target_tz]) * 1000.0 # m -> mm
            
            T_world_cam_target = T_world_tool_target @ T_cam_to_tcp_inv
            delta_T_cam = T_slam_0_inv @ T_world_cam_target
            
            # TCP_target = TCP_start * (TCP->CAM) * Delta_CAM * (CAM->TCP)
            T_tcp_target = T_tcp_0 @ T_cam_to_tcp @ delta_T_cam @ T_cam_to_tcp_inv
            
            target_x, target_y, target_z = T_tcp_target[:3, 3]
            target_eulers = R.from_matrix(T_tcp_target[:3, :3]).as_euler('xyz', degrees=True)
            target_eulers = get_closest_euler(target_eulers, p_phys_curr[3:6])
            target_u, target_v, target_w = target_eulers[0], target_eulers[1], target_eulers[2]
            
            p_target_phys = np.array([target_x, target_y, target_z, target_u, target_v, target_w], dtype=np.float32)

            # 5. Khóa an toàn
            p_safe_phys = clamp_task_target(p_phys_curr, p_target_phys, max_trans=20.0, max_rot=5.0)

            print(f"Target: {np.round(p_target_phys, 1)} | Safe: {np.round(p_safe_phys, 1)} | Grip: {g_cmd_target:.2f}")

            if args.execute:
                try:
                    indy.send_task_target(p_safe_phys)
                    if g_cmd_target > 0.5: new_g_cmd = "open"
                    else: new_g_cmd = "close"
                    
                    if new_g_cmd != last_g_cmd:
                        if new_g_cmd == "open": susgrip.open()
                        else: susgrip.close()
                        last_g_cmd = new_g_cmd
                    #usgrip.move(g_cmd_target)
                except Exception as e:
                    print(f"Transmission error: {e}")
                    break

            elapsed = time.perf_counter() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\nStopped inference loop.")
    finally:
        print("Stopping hardware threads...")
        try: indy.stop()
        except: pass
        if args.execute:
            print("Disabling Teleop...")
            try: indy.stop_teleop()
            except: pass
        susgrip.stop()
        cam.stop()
        print("Hardware disconnected securely.")

if __name__ == "__main__":
    main()
