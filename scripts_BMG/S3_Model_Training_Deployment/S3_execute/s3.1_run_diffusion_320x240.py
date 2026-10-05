#!/usr/bin/env python3
import time
import threading
import queue
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#DEFAULT_POLICY_PATH = os.path.join(BASE_DIR, "S3_output", "Date_01102026", "diffusion_checkpoints_lerobot_dataset_fpccam_slam_10fps_320x240", "checkpoints", "last", "pretrained_model")
DEFAULT_POLICY_PATH = os.path.join(
    BASE_DIR, "S3_output", "Date_01102026",
    "diffusion_umi_rel_lerobot_dataset_fpccam_slam_10fps_320x240",
    "checkpoints", "last", "pretrained_model"
)

GRIPPER_OPEN_MM = 100.0
GRIPPER_CLOSE_MM = 10.0

H, W = 540, 960
TRAIN_FREQ = 10
CHUNK_SIZE = 8
INTERP_STEPS = 80
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

class fpc_camera:
    def __init__(self, target_fps=30):
        self.cam_id = find_webcam_id()
        self.stream = None
        self.frame = None
        self.frame_id = 0
        self.grabbed = False
        self.stopped = False
        self.lock = threading.Lock()
        
        if self.cam_id is not None:
            self.stream = cv2.VideoCapture(self.cam_id, cv2.CAP_V4L2)
            self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.stream.set(cv2.CAP_PROP_FPS, 30)
            self.stream.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            (self.grabbed, self.frame) = self.stream.read()
            self.start()

    def start(self):
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        return self
        
    def update(self):
        while not self.stopped and self.stream is not None:
            (grabbed, frame) = self.stream.read()
            if grabbed:
                with self.lock:
                    self.grabbed = True
                    self.frame = frame
                    self.frame_id += 1
            else:
                self.grabbed = False
                print("[WARNING] FPC Camera dropped a frame!")
        
    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy(), self.frame_id
            return None, -1

    def get_images(self):
        with self.lock:
            if self.frame is not None:
                # Resize to 320x240
                resized_frame = cv2.resize(self.frame, (320, 240))
                # Convert BGR (from OpenCV) to RGB (for policy input)
                #return cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            #return np.zeros((480, 640, 3), dtype=np.uint8)
                return cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            return np.zeros((240, 320, 3), dtype=np.uint8)
            
    def stop(self):
        self.stopped = True
        try:
            if hasattr(self, 'thread'):
                self.thread.join(timeout=0.2)
        except Exception:
            pass
        if self.stream is not None:
            self.stream.release()

class rs_camera:
    def __init__(self, target_fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        self.frame = None
        self.frame_id = 0
        self.grabbed = False
        self.stopped = False
        self.lock = threading.Lock()
        
        # Đợi camera ổn định và lấy frame đầu tiên
        for _ in range(10):
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            if color_frame:
                self.frame = np.asanyarray(color_frame.get_data())
                self.grabbed = True
                break
        self.start()

    def start(self):
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        return self
        
    def update(self):
        while not self.stopped:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                if not color_frame:
                    continue
                frame = np.asanyarray(color_frame.get_data())
                with self.lock:
                    self.grabbed = True
                    self.frame = frame
                    self.frame_id += 1
            except Exception as e:
                pass
                
    # def get_images(self):
    #     with self.lock:
    #         if self.frame is not None:
    #             return cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
    #         return np.zeros((480, 640, 3), dtype=np.uint8)


    def get_images(self):
        with self.lock:
            if self.frame is not None:
                # Resize to 320x240
                resized_frame = cv2.resize(self.frame, (320, 240))
                # Convert BGR (from OpenCV) to RGB (for policy input)
                #return cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            #return np.zeros((480, 640, 3), dtype=np.uint8)
                return cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            return np.zeros((240, 320, 3), dtype=np.uint8)
            

    def stop(self):
        self.stopped = True
        try:
            if hasattr(self, 'thread'):
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

def find_webcam_id():
    import glob
    video_paths = glob.glob('/sys/class/video4linux/video*')
    video_paths.sort(key=lambda x: int(os.path.basename(x).replace('video', '')))
    for path in video_paths:
        try:
            with open(os.path.join(path, 'name'), 'r') as f:
                name = f.read().strip()
                if "RealSense" not in name and "Metadata" not in name:
                    idx = int(os.path.basename(path).replace('video', ''))
                    print(f"[INFO] Tự động nhận diện Webcam: '{name}' tại /dev/video{idx}")
                    return idx
        except Exception:
            continue
    print("[WARNING] Không tìm thấy Webcam ngoài! Vui lòng kiểm tra cáp cắm.")
    return None

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
    
    # 3. An toàn mặt bàn (Z >= 7.5mm)
    if safe_p[2] < 10:
        safe_p[2] = 10
        
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
    print("🚀 DIFFUSION POLICY INFERENCE (CARTESIAN SLAM)")
    print("=" * 70)

    device = torch.device(args.device)
    
    # Load Policy
    print(f"Loading Diffusion model from: {args.policy_path}")
    policy = DiffusionPolicy.from_pretrained(args.policy_path, local_files_only=True)
    
    # Speed up inference by reducing denoising steps
    policy.diffusion.num_inference_steps = 16
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
    print("Initializing FPC Camera (Front)...")
    cam_fpc = fpc_camera(target_fps=30)
    print("Initializing Realsense Camera (Side)...")
    cam_rs = rs_camera(target_fps=30)

    # Lấy vị trí ban đầu của Robot khi bắt đầu inference
    time.sleep(1.0) # Đợi các luồng thu thập dữ liệu
    p_tcp_0 = indy.get_robot_data()
    print(f"Initial Robot Pose: {np.round(p_tcp_0, 2)}")
    T_tcp_0 = tcp_to_matrix(p_tcp_0)
    T_tcp_0_inv = np.linalg.inv(T_tcp_0)

    # Hàm quy đổi T_tcp_curr sang AI Observation State (10D relative tool pose)
    # Công thức: T_rel = inv(T_tcp_0) @ T_tcp_curr
    # Dữ liệu training đã ở dạng relative tool trong hệ tool_0, robot báo TCP trực tiếp
    def get_obs_dict(p_tcp_curr, g_state, rgb_img, rgb_img_side):
        T_tcp_curr = tcp_to_matrix(p_tcp_curr)
        T_rel = T_tcp_0_inv @ T_tcp_curr

        pos = T_rel[:3, 3] / 1000.0  # mm -> m
        R_rel = T_rel[:3, :3]
        r1, r2, r3 = R_rel[:, 0]
        r4, r5, r6 = R_rel[:, 1]

        obs_state = np.array([pos[0], pos[1], pos[2], r1, r2, r3, r4, r5, r6, g_state], dtype=np.float32)
        obs_state_t = torch.from_numpy(obs_state).unsqueeze(0).to(device)

        chw = np.transpose(rgb_img, (2, 0, 1)).astype(np.float32) / 255.0
        img_t = torch.from_numpy(chw).unsqueeze(0).to(device)

        chw_side = np.transpose(rgb_img_side, (2, 0, 1)).astype(np.float32) / 255.0
        img_side_t = torch.from_numpy(chw_side).unsqueeze(0).to(device)

        return {
            "observation.state": obs_state_t,
            "observation.image": img_t,
            "observation.image_side": img_side_t,
        }

    # Model CUDA Graph Warmup
    print("Warming up CUDA Graph...")
    dummy_obs = get_obs_dict(indy.get_robot_data(), susgrip.get_gripper_state(), cam_fpc.get_images(), cam_rs.get_images())
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
    last_g_cmd_time = time.perf_counter()

    try:
        while True:
            # 1. Force replan bằng cách xóa hàng đợi action (giữ nguyên lịch sử observation)
            if hasattr(policy, "policy") and hasattr(policy.policy, "_queues"):
                if "action" in policy.policy._queues:
                    policy.policy._queues["action"].clear()
                    
            chunk_p_phys = []
            chunk_g_cmd = []
            
            inf_start = time.perf_counter()
            
            for step_i in range(CHUNK_SIZE):
                p_phys_curr = indy.get_robot_data()
                g_state = susgrip.get_gripper_state()
                rgb_img = cam_fpc.get_images() 
                rgb_img_side = cam_rs.get_images() 
                
                obs_dict = get_obs_dict(p_phys_curr, g_state, rgb_img, rgb_img_side)
                obs_dict = preprocessor(obs_dict)
                
                with torch.inference_mode():
                    action_pred = policy.select_action(obs_dict)
                    
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
                g_cmd_target = act_vec[9]
                
                v1 = v1 / np.linalg.norm(v1)
                v3 = np.cross(v1, v2)
                v3 = v3 / np.linalg.norm(v3)
                v2 = np.cross(v3, v1)
                
                R_mat = np.column_stack((v1, v2, v3))

                # Quy đổi ngược: T_tcp_target = T_tcp_0 @ T_rel_predicted
                T_rel_target = np.eye(4)
                T_rel_target[:3, :3] = R_mat
                T_rel_target[:3, 3] = np.array([target_tx, target_ty, target_tz]) * 1000.0  # m -> mm

                T_tcp_target = T_tcp_0 @ T_rel_target

                target_x, target_y, target_z = T_tcp_target[:3, 3]
                target_eulers = R.from_matrix(T_tcp_target[:3, :3]).as_euler('xyz', degrees=True)
                target_eulers = get_closest_euler(target_eulers, p_phys_curr[3:6])
                target_u, target_v, target_w = target_eulers[0], target_eulers[1], target_eulers[2]

                p_target_phys = np.array([target_x, target_y, target_z, target_u, target_v, target_w], dtype=np.float32)
                
                chunk_p_phys.append(p_target_phys)
                chunk_g_cmd.append(g_cmd_target)
                
            inf_time = time.perf_counter() - inf_start
            print(f"[AI] Inference & extraction of {CHUNK_SIZE} points took: {inf_time:.3f} s")
            
            # 2. Nội suy quỹ đạo từ CHUNK_SIZE lên INTERP_STEPS
            chunk_p_phys = np.array(chunk_p_phys)
            chunk_g_cmd = np.array(chunk_g_cmd)
            
            p_phys_curr_neo = indy.get_robot_data()
            g_state_neo = susgrip.get_gripper_state()
            
            chunk_p_phys = np.vstack([p_phys_curr_neo, chunk_p_phys])
            chunk_g_cmd = np.insert(chunk_g_cmd, 0, g_state_neo)
            
            orig_t = np.linspace(0, 1, CHUNK_SIZE + 1)
            interp_t = np.linspace(0, 1, INTERP_STEPS)
            
            interp_p_phys = np.zeros((INTERP_STEPS, 6), dtype=np.float32)
            # 1. Dịch chuyển (X, Y, Z)
            for i in range(3):
                interp_p_phys[:, i] = np.interp(interp_t, orig_t, chunk_p_phys[:, i])
            # 2. Góc xoay (U, V, W) - Unwrap góc trước khi nội suy
            for i in range(3, 6):
                angles = chunk_p_phys[:, i]
                diffs = np.diff(angles)
                diffs = (diffs + 180.0) % 360.0 - 180.0
                unwrapped_angles = np.zeros_like(angles)
                unwrapped_angles[0] = angles[0]
                for j in range(1, len(angles)):
                    unwrapped_angles[j] = unwrapped_angles[j-1] + diffs[j-1]
                interp_unwrapped = np.interp(interp_t, orig_t, unwrapped_angles)
                interp_p_phys[:, i] = (interp_unwrapped + 180.0) % 360.0 - 180.0
                
            interp_g_cmd = np.interp(interp_t, orig_t, chunk_g_cmd)
            
            # 3. Gửi lần lượt các điểm đã nội suy xuống robot
            chunk_duration = CHUNK_SIZE / TRAIN_FREQ
            exec_dt = 0.02
            a = time.perf_counter()
            SKIP_POINTS = 0
            for i in range(SKIP_POINTS, INTERP_STEPS):
                loop_start = time.perf_counter()
                
                target_p = interp_p_phys[i]
                target_g = interp_g_cmd[i]
                
                p_phys_curr = indy.get_robot_data()
                
                p_safe_phys = clamp_task_target(p_phys_curr, target_p, max_trans=20.0, max_rot=5.0)
                
                print(f"Executing {i+1}/{INTERP_STEPS} | Target: {np.round(target_p, 1)} | Grip: {target_g:.2f}")
                
                if args.execute:
                    try:
                        indy.send_task_target(p_safe_phys)
                        if target_g < 0.5:
                            new_g_cmd = "close"
                        else:
                            new_g_cmd = "open"
                            
                        # Lọc theo thời gian tồn tại của trạng thái thực tế (Cooldown Filter)
                        if new_g_cmd != last_g_cmd:
                            current_time = time.perf_counter()
                            # Kiểm tra xem trạng thái cũ đã tồn tại đủ 0.75s chưa
                            if (current_time - last_g_cmd_time) >= 1.0:
                                if new_g_cmd == "close":
                                    susgrip.close()
                                else:
                                    susgrip.open()
                                # Lật trạng thái và reset đồng hồ đếm
                                last_g_cmd = new_g_cmd
                                last_g_cmd_time = current_time
                    except Exception as e:
                        print(f"Transmission error: {e}")
                        break
                        
                elapsed = time.perf_counter() - loop_start
                if elapsed < exec_dt:
                    time.sleep(exec_dt - elapsed)
            print(f"Execution time: {time.perf_counter()-a}")      
            # 4. Chờ robot di chuyển tới điểm cuối của quỹ đạo (so sánh sai số)
            if args.execute:
                final_p = interp_p_phys[-1]
                print(f"Waiting for robot to reach target... {np.round(final_p, 1)}")
                
                wait_start = time.perf_counter()
                while True:
                    p_phys_curr = indy.get_robot_data()
                    
                    err_trans = np.linalg.norm(p_phys_curr[:3] - final_p[:3])
                    err_rot = np.linalg.norm((p_phys_curr[3:6] - final_p[3:6] + 180) % 360 - 180)
                    
                    if err_trans < 1.0 and err_rot < 1.0: # Đã nới lỏng: 3mm, 2 độ
                        print(f"Target reached! Err_Trans: {err_trans:.1f}mm, Err_Rot: {err_rot:.1f}deg")
                        break
                        
                    # Thêm timeout chống kẹt (Tối đa 0.5s)
                    if time.perf_counter() - wait_start > 0.5:
                        print(f"Wait timeout! Moving to next prediction (Err: {err_trans:.1f}mm, {err_rot:.1f}deg)")
                        break
                    
                    time.sleep(0.01)

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
        cam_fpc.stop()
        cam_rs.stop()
        print("Hardware disconnected securely.")

if __name__ == "__main__":
    main()
