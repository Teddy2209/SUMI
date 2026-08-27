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

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_POLICY_PATH = os.path.join(BASE_DIR, "output_trained", "diffusion_checkpoints_diff_gt", "checkpoints", "last", "pretrained_model")
DEFAULT_POLICY_PATH = os.path.join(BASE_DIR, "output_trained", "diffusion_checkpoints_diff_gt", "checkpoints", "last", "pretrained_model")

GRIPPER_OPEN_MM = 120.0
GRIPPER_CLOSE_MM = 30.0

H, W = 540, 960
AI_FREQ = 10
ROBOT_FREQ = 25
EMA_ALPHA =  0.4   # Hệ số nội suy (0.0 -> 1.0), càng nhỏ càng mượt nhưng bám mục tiêu chậm hơn
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
            time.sleep(0.02)

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
    Tìm bộ Euler (xyz) gần nhất với góc xoay hiện tại của Robot để tránh Gimbal Lock (xoay 180 độ).
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
    print("🚀 DIFFUSION POLICY INFERENCE (CARTESIAN)")
    print("=" * 70)

    device = torch.device(args.device)
    
    # No calibration needed for Ground Truth (Base frame directly)

    # Load Policy
    print(f"Loading Diffusion model from: {args.policy_path}")
    policy = DiffusionPolicy.from_pretrained(args.policy_path, local_files_only=True)
    
    # Speed up inference by reducing denoising steps (default is usually 100)
    # Lưu ý: phải trỏ đúng vào policy.diffusion (DiffusionModel) chứ không phải policy (PreTrainedPolicy)
    policy.diffusion.num_inference_steps = 20

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
    # Không cần các biến trung gian T_tcp_0 hay T_slam_0 vì data train trực tiếp trên hệ Base của Robot
    
    # Hàm quy đổi T_tcp_curr sang AI Observation State (Ground Truth Pose)
    def get_obs_dict(p_tcp_curr, g_state, rgb_img):
        # Chuyển đổi vị trí từ mm sang m
        tx = p_tcp_curr[0] / 1000.0 
        ty = p_tcp_curr[1] / 1000.0
        tz = p_tcp_curr[2] / 1000.0
        
        # Chuyển đổi u, v, w (deg) thành Rotation Matrix
        R_mat = R.from_euler('xyz', p_tcp_curr[3:6], degrees=True).as_matrix()
        r1, r2, r3 = R_mat[:, 0]
        r4, r5, r6 = R_mat[:, 1]
        
        # Vector 10 chiều (3 vị trí, 6 góc quay, 1 kẹp)
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
        print(f"Starting async execution loop at {ROBOT_FREQ} Hz (AI runs at {AI_FREQ} Hz)...")
        indy.start_teleop()
    
    # Biến dùng chung (Thread-safe)
    shared_data_lock = threading.Lock()
    latest_ai_p_target = indy.get_robot_data()
    latest_ai_g_target = susgrip.get_gripper_state()
    ai_running = True

    def ai_worker():
        nonlocal latest_ai_p_target, latest_ai_g_target, ai_running
        ai_dt = 1.0 / AI_FREQ
        
        while ai_running:
            loop_start = time.perf_counter()
            
            # 1. Lấy trạng thái vật lý thực tế
            p_phys_curr = indy.get_robot_data()
            g_state = susgrip.get_gripper_state()
            rgb_img = cam.get_images()

            # 2. Quy đổi ra SLAM Pose
            obs_dict = get_obs_dict(p_phys_curr, g_state, rgb_img)
            obs_dict = preprocessor(obs_dict)

            # 3. Dự đoán Action
            with torch.inference_mode():
                inf_start = time.perf_counter()
                action_pred = policy.select_action(obs_dict)
                inf_end = time.perf_counter()
            
            inf_time = inf_end - inf_start
            if inf_time > 0.01:
                print(f"[AI Worker] Inference Time: {inf_time:.3f} s")
                
            action_pred = postprocessor(action_pred)
            
            if isinstance(action_pred, dict):
                act_vec = action_pred["action"].squeeze(0).cpu().numpy()
            else:
                act_vec = action_pred.squeeze(0).cpu().numpy()
                
            if len(act_vec.shape) > 1 and act_vec.shape[0] == 1:
                act_vec = act_vec[0]

            target_tx, target_ty, target_tz = act_vec[0], act_vec[1], act_vec[2]
            v1 = act_vec[3:6]
            v2 = act_vec[6:9]
            g_cmd_target = float(act_vec[9])

            v3 = np.cross(v1, v2)
            R_mat = np.column_stack((v1, v2, v3))
            
            target_x = target_tx * 1000.0
            target_y = target_ty * 1000.0
            target_z = target_tz * 1000.0
            
            target_eulers = R.from_matrix(R_mat).as_euler('xyz', degrees=True)
            target_eulers = get_closest_euler(target_eulers, p_phys_curr[3:6])
            target_u, target_v, target_w = target_eulers[0], target_eulers[1], target_eulers[2]
            
            p_target_phys = np.array([target_x, target_y, target_z, target_u, target_v, target_w], dtype=np.float32)

            with shared_data_lock:
                latest_ai_p_target = p_target_phys.copy()
                latest_ai_g_target = g_cmd_target

            elapsed = time.perf_counter() - loop_start
            if elapsed < ai_dt:
                time.sleep(ai_dt - elapsed)
            else: print(elapsed)

    # Khởi động luồng AI
    ai_thread = threading.Thread(target=ai_worker, daemon=True)
    ai_thread.start()

    # Vòng lặp Robot 
    dt = 1.0 / ROBOT_FREQ
    current_cmd_p = indy.get_robot_data()
    current_cmd_g = susgrip.get_gripper_state()

    try:
        while True:
            loop_start = time.perf_counter()

            with shared_data_lock:
                target_p = latest_ai_p_target.copy()
                target_g = latest_ai_g_target

            p_phys_curr = indy.get_robot_data()

            # Lọc EMA (Exponential Moving Average) để tạo các điểm nội suy mượt mà
            current_cmd_p = current_cmd_p * (1.0 - EMA_ALPHA) + target_p * EMA_ALPHA
            current_cmd_g = current_cmd_g * (1.0 - EMA_ALPHA) + target_g * EMA_ALPHA

            # Khóa an toàn
            p_safe_phys = clamp_task_target(p_phys_curr, current_cmd_p, max_trans=20.0, max_rot=5.0)

            print(f"Target(AI): {np.round(target_p, 1)} | Smooth(EMA): {np.round(current_cmd_p, 1)} | Grip: {current_cmd_g:.2f}")

            if args.execute:
                try:
                    indy.send_task_target(p_safe_phys)
                    
                    # if g_cmd_target > 0.5: new_g_cmd = "open"
                    # else: new_g_cmd = "close"
                    
                    # if new_g_cmd != last_g_cmd:
                    #     if new_g_cmd == "open": susgrip.open()
                    #     else: susgrip.close()
                    #     last_g_cmd = new_g_cmd
                    
                    # Điều khiển gripper theo toạ độ liên tục thay vì đóng mở cứng ngắc
                    # target_g được suy luận ra nằm trong khoảng 0.0 đến 1.0, current_cmd_g là giá trị đã làm mượt
                    susgrip.move(current_cmd_g)
                    
                except Exception as e:
                    print(f"Transmission error: {e}")
                    break

            elapsed = time.perf_counter() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
            else:
                print(f"[Robot Loop] Loop too slow: {elapsed:.3f} s")

    except KeyboardInterrupt:
        print("\nStopped execution loop.")
        ai_running = False
        ai_thread.join(timeout=1.0)
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
