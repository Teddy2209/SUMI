#!/usr/bin/env python3
"""
Diffusion Policy Inference trên Robot Indy7 (Cartesian SLAM - Tuyệt đối).

Kiến trúc:
  - Indy7Robot     → Điều khiển robot
  - FPCCamera      → Camera đầu tay máy (Front view)
  - RealSenseCamera→ Camera hông (Side view)
  - ModbusGripper  → Điều khiển tay kẹp
  - Inference Loop → Dự đoán mục tiêu SLAM (Chunk 8 steps) và nội suy quỹ đạo trơn tru.

Thuật toán:
  - Sử dụng Diffusion Policy dự đoán Chunk (n_action_steps = 8).
  - Tọa độ mục tiêu được tính toán trong hệ quy chiếu tuyệt đối (Base Frame) dựa vào SLAM.
  - Quỹ đạo được nội suy mềm mại thành 80 điểm (INTERP_STEPS) trước khi gửi tới robot.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import threading
import queue
import argparse

import numpy as np
import cv2
import pyrealsense2 as rs
from pymodbus.client import ModbusSerialClient
import torch
from scipy.spatial.transform import Rotation as R
from neuromeka import IndyDCP3
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_POLICY_PATH = os.path.join(      # Checkpoint ABSOLUTE (s3_train_diffusion.py), không dùng ckpt umi_rel
    BASE_DIR, "S3_output", "Date_25092026",
    "diffusion_checkpoints_lerobot_dataset_fpccam_slam_10fps_320x240",
    "checkpoints", "last", "pretrained_model"
)

# Robot
ROBOT_IP = "192.168.2.100"

# Inference
TRAIN_FREQ = 10              # Tần số khi train mô hình
CHUNK_SIZE = 8               # Kích thước chunk
INTERP_STEPS = 80            # Số lượng điểm nội suy để đi cho mượt
NUM_DENOISE_STEPS = 16       # Giảm số bước denoise để tăng tốc

# Camera
IMG_SIZE = (320, 240)        # (width, height) resize cho model

# An toàn
MAX_TRANS_PER_STEP = 20.0    # mm
MAX_ROT_PER_STEP = 5.0       # deg
MIN_Z_HEIGHT = 10.0          # mm

# Gripper
GRIPPER_PORT = "/dev/ttyUSB0"
GRIPPER_OPEN_MM = 100.0
GRIPPER_CLOSE_MM = 10.0
GRIPPER_COOLDOWN = 1.0       # giây

# ═══════════════════════════════════════════════════════════════
# HARDWARE: Robot
# ═══════════════════════════════════════════════════════════════
class Indy7Robot:
    def __init__(self, ip=ROBOT_IP):
        self.robot = IndyDCP3(ip)
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
                        vel_ratio=0.1,  
                        acc_ratio=1.0
                    )
                except Exception as e:
                    print(f"[Robot] Command error: {e}")
            try:
                p = self.robot.get_robot_data()["p"]
                with self.lock:
                    self.latest_p = np.array(p, dtype=np.float32)
                    self._last_read_ts = time.time()
            except Exception:
                pass
            
            time.sleep(0.01)

    def get_pose(self):
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

    def send_target(self, p_target):
        self._cmd_queue.put(p_target)

    def wait_teleop_ready(self, timeout=0.6):
        """Đợi robot chuyển sang TELE_OP mode."""
        for _ in range(int(timeout / 0.02)):
            state = self.robot.get_robot_data().get("op_state", 0)
            if state == 17:
                return True
            time.sleep(0.02)
        return False

    def stop(self):
        self.running = False
        try:
            self.thread.join(timeout=0.2)
        except Exception:
            pass

# ═══════════════════════════════════════════════════════════════
# HARDWARE: Camera
# ═══════════════════════════════════════════════════════════════
class FPCCamera:
    def __init__(self, target_fps=30):
        self.w, self.h = IMG_SIZE
        self.cam_id = self._find_webcam_id()
        self.stream = None
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        
        if self.cam_id is not None:
            self.stream = cv2.VideoCapture(self.cam_id, cv2.CAP_V4L2)
            self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.stream.set(cv2.CAP_PROP_FPS, 30)
            self.stream.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            ok, frame = self.stream.read()
            if ok:
                self.frame = frame
            
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()

    def _update(self):
        while not self.stopped and self.stream is not None:
            ok, frame = self.stream.read()
            if ok:
                with self.lock:
                    self.frame = frame
            else:
                print("[WARNING] FPC Camera dropped a frame!")

    def get_images(self):
        with self.lock:
            if self.frame is not None:
                resized = cv2.resize(self.frame, (self.w, self.h))
                return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            return np.zeros((self.h, self.w, 3), dtype=np.uint8)
            
    def stop(self):
        self.stopped = True
        try:
            if hasattr(self, 'thread'):
                self.thread.join(timeout=0.2)
        except Exception:
            pass
        if self.stream is not None:
            self.stream.release()

    @staticmethod
    def _find_webcam_id():
        import glob
        video_paths = sorted(glob.glob('/sys/class/video4linux/video*'), 
                           key=lambda x: int(os.path.basename(x).replace('video', '')))
        for path in video_paths:
            try:
                with open(os.path.join(path, 'name'), 'r') as f:
                    name = f.read().strip()
                    if "RealSense" not in name and "Metadata" not in name:
                        idx = int(os.path.basename(path).replace('video', ''))
                        print(f"[INFO] Found FPC Webcam: '{name}' at /dev/video{idx}")
                        return idx
            except Exception:
                continue
        print("[WARNING] FPC Webcam not found!")
        return None


class RealSenseCamera:
    def __init__(self, target_fps=30):
        self.w, self.h = IMG_SIZE
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        
        for _ in range(10):
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            if color_frame:
                self.frame = np.asanyarray(color_frame.get_data())
                break

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                if color_frame:
                    with self.lock:
                        self.frame = np.asanyarray(color_frame.get_data())
            except Exception:
                pass
            time.sleep(1/60.0)

    def get_images(self):
        with self.lock:
            if self.frame is not None:
                resized = cv2.resize(self.frame, (self.w, self.h))
                return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            return np.zeros((self.h, self.w, 3), dtype=np.uint8)
            
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

# ═══════════════════════════════════════════════════════════════
# HARDWARE: Gripper
# ═══════════════════════════════════════════════════════════════
class ModbusGripper:
    def __init__(self, port=GRIPPER_PORT, open_mm=GRIPPER_OPEN_MM, close_mm=GRIPPER_CLOSE_MM):
        self.open_mm = open_mm
        self.close_mm = close_mm
        
        self.client = ModbusSerialClient(
            port=port, baudrate=115200, stopbits=1,
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
                        norm = np.clip((pos_mm - self.close_mm) / (self.open_mm - self.close_mm), 0.0, 1.0)
                        with self._lock:
                            self.current_state = float(norm)
                except Exception:
                    pass
            time.sleep(0.01)

    def close(self):
        self._cmd_queue.put(int(self.close_mm))

    def open(self):
        self._cmd_queue.put(int(self.open_mm))
    
    def move(self, norm_pos):
        norm_pos = float(np.clip(norm_pos, 0.0, 1.0))
        pos_mm = norm_pos * (self.open_mm - self.close_mm) + self.close_mm
        self._cmd_queue.put(int(pos_mm))

    def get_state(self):
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

# ═══════════════════════════════════════════════════════════════
# KINEMATICS & SAFETY
# ═══════════════════════════════════════════════════════════════
class Kinematics:
    @staticmethod
    def clamp_target(current_p, target_p, max_trans=MAX_TRANS_PER_STEP, max_rot=MAX_ROT_PER_STEP):
        current_p = np.asarray(current_p, dtype=np.float32)
        target_p = np.asarray(target_p, dtype=np.float32)
        
        # Translation
        delta_trans = target_p[:3] - current_p[:3]
        max_abs_trans = np.max(np.abs(delta_trans))
        if max_abs_trans > max_trans:
            scaled_trans = delta_trans * (max_trans / max_abs_trans)
        else:
            scaled_trans = delta_trans
            
        # Rotation
        delta_rot = (target_p[3:6] - current_p[3:6] + 180.0) % 360.0 - 180.0
        max_abs_rot = np.max(np.abs(delta_rot))
        if max_abs_rot > max_rot:
            scaled_rot = delta_rot * (max_rot / max_abs_rot)
        else:
            scaled_rot = delta_rot
            
        safe_p = current_p.copy()
        safe_p[:3] += scaled_trans
        safe_p[3:6] = (current_p[3:6] + scaled_rot + 180.0) % 360.0 - 180.0
        
        if safe_p[2] < MIN_Z_HEIGHT:
            safe_p[2] = MIN_Z_HEIGHT
            
        return safe_p

    @staticmethod
    def tcp_to_matrix(p):
        T = np.eye(4)
        r = R.from_euler('xyz', p[3:6], degrees=True)
        T[:3, :3] = r.as_matrix()
        T[:3, 3] = p[:3]
        return T

    @staticmethod
    def get_closest_euler(target_eulers, current_eulers):
        alt_eulers = np.array([
            target_eulers[0] + 180.0,
            180.0 - target_eulers[1],
            target_eulers[2] + 180.0
        ])
        
        def angular_dist(a, b):
            return np.linalg.norm((a - b + 180.0) % 360.0 - 180.0)
            
        dist1 = angular_dist(target_eulers, current_eulers)
        dist2 = angular_dist(alt_eulers, current_eulers)
        
        chosen = alt_eulers if dist2 < dist1 else target_eulers
        return (chosen + 180.0) % 360.0 - 180.0

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Diffusion Policy Inference (Cartesian SLAM Absolute)")
    parser.add_argument("--policy_path", type=str, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--execute", action="store_true", help="Execute on physical robot")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🚀 DIFFUSION POLICY INFERENCE (CARTESIAN SLAM ABSOLUTE)")
    print("=" * 70)

    device = torch.device(args.device)
    
    # ── 1. Load Model ─────────────────────────────────────────
    print(f"\n[1/3] Loading Diffusion model: {args.policy_path}")
    policy = DiffusionPolicy.from_pretrained(args.policy_path, local_files_only=True)
    policy.diffusion.num_inference_steps = NUM_DENOISE_STEPS
    policy.to(device)
    policy.eval()

    from lerobot.policies.factory import make_pre_post_processors
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=args.policy_path)

    # ── 2. Khởi tạo Hardware ──────────────────────────────────
    print("\n[2/3] Khởi tạo Hardware...")
    indy = Indy7Robot()
    susgrip = ModbusGripper()
    cam_fpc = FPCCamera()
    cam_rs = RealSenseCamera()

    # Lấy vị trí ban đầu
    time.sleep(1.0)
    p_tcp_0 = indy.get_pose()
    print(f"  [>] Initial Robot Pose: {np.round(p_tcp_0, 2)}")
    T_tcp_0 = Kinematics.tcp_to_matrix(p_tcp_0)
    T_tcp_0_inv = np.linalg.inv(T_tcp_0)

    def get_obs_dict(p_tcp_curr, g_state, rgb_img, rgb_img_side):
        """Hàm quy đổi T_tcp_curr sang AI Observation State (10D relative tool pose tuyệt đối)"""
        T_tcp_curr = Kinematics.tcp_to_matrix(p_tcp_curr)
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

    # Warmup
    print("  [>] Warming up CUDA Graph...")
    dummy_obs = get_obs_dict(indy.get_pose(), susgrip.get_state(), cam_fpc.get_images(), cam_rs.get_images())
    dummy_obs = preprocessor(dummy_obs)
    with torch.inference_mode():
        for _ in range(3):
            policy.select_action(dummy_obs)
            
    if hasattr(policy, "reset"):
        policy.reset()

    # ── 3. Mode & Bật Teleop ──────────────────────────────────
    if args.execute:
        print("\n[3/3] Chuẩn bị chạy...")
        print("⚠️ [WARNING] EXECUTE MODE - Robot sẽ di chuyển thật!")
        if input("Gõ 'YES' để tiếp tục: ").strip() != "YES":
            print("Hủy.")
            return
            
        indy.start_teleop()
        if not indy.wait_teleop_ready():
            print("❌ Lỗi: Robot không thể vào chế độ TELE_OP.")
            return
        print("✅ TELE_OP Ready!")

        p_curr = indy.get_pose()
        for _ in range(10):
            indy.send_target(p_curr)
            time.sleep(0.01)
    else:
        print("\n[3/3] Chế độ DRY-RUN (chỉ log tọa độ, không chạy robot)")

    # ── Inference Loop ────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"INFERENCE & INTERPOLATION LOOP (Chunk={CHUNK_SIZE})")
    print("Ctrl+C để dừng")
    print("=" * 60 + "\n")

    last_g_cmd = "open"
    last_g_cmd_time = time.perf_counter()

    try:
        while True:
            # Xóa hàng đợi action để force replan
            if hasattr(policy, "policy") and hasattr(policy.policy, "_queues"):
                if "action" in policy.policy._queues:
                    policy.policy._queues["action"].clear()
                    
            chunk_p_phys = []
            chunk_g_cmd = []
            
            inf_start = time.perf_counter()
            
            # ─── BƯỚC 1: DỰ ĐOÁN CHUNK 8 STEPS ───
            for _ in range(CHUNK_SIZE):
                p_phys_curr = indy.get_pose()
                g_state = susgrip.get_state()
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
                v1, v2 = act_vec[3:6], act_vec[6:9]
                g_cmd_target = act_vec[9]
                
                v3 = np.cross(v1, v2)
                R_mat = np.column_stack((v1, v2, v3))

                # Quy đổi ngược
                T_rel_target = np.eye(4)
                T_rel_target[:3, :3] = R_mat
                T_rel_target[:3, 3] = np.array([target_tx, target_ty, target_tz]) * 1000.0  # m -> mm

                T_tcp_target = T_tcp_0 @ T_rel_target
                target_x, target_y, target_z = T_tcp_target[:3, 3]
                target_eulers = R.from_matrix(T_tcp_target[:3, :3]).as_euler('xyz', degrees=True)
                target_eulers = Kinematics.get_closest_euler(target_eulers, p_phys_curr[3:6])

                p_target_phys = np.array([target_x, target_y, target_z, *target_eulers], dtype=np.float32)
                
                chunk_p_phys.append(p_target_phys)
                chunk_g_cmd.append(g_cmd_target)
                
            inf_time = time.perf_counter() - inf_start
            print(f"[AI] Inference {CHUNK_SIZE} steps: {inf_time:.3f} s")
            
            # ─── BƯỚC 2: NỘI SUY (INTERPOLATION) ───
            chunk_p_phys = np.array(chunk_p_phys)
            chunk_g_cmd = np.array(chunk_g_cmd)
            
            p_phys_curr_neo = indy.get_pose()
            g_state_neo = susgrip.get_state()
            
            chunk_p_phys = np.vstack([p_phys_curr_neo, chunk_p_phys])
            chunk_g_cmd = np.insert(chunk_g_cmd, 0, g_state_neo)
            
            orig_t = np.linspace(0, 1, CHUNK_SIZE + 1)
            interp_t = np.linspace(0, 1, INTERP_STEPS)
            
            interp_p_phys = np.zeros((INTERP_STEPS, 6), dtype=np.float32)
            
            # Nội suy Translation
            for i in range(3):
                interp_p_phys[:, i] = np.interp(interp_t, orig_t, chunk_p_phys[:, i])
                
            # Nội suy Rotation (Unwrap)
            for i in range(3, 6):
                angles = chunk_p_phys[:, i]
                diffs = (np.diff(angles) + 180.0) % 360.0 - 180.0
                unwrapped_angles = np.zeros_like(angles)
                unwrapped_angles[0] = angles[0]
                for j in range(1, len(angles)):
                    unwrapped_angles[j] = unwrapped_angles[j-1] + diffs[j-1]
                interp_unwrapped = np.interp(interp_t, orig_t, unwrapped_angles)
                interp_p_phys[:, i] = (interp_unwrapped + 180.0) % 360.0 - 180.0
                
            interp_g_cmd = np.interp(interp_t, orig_t, chunk_g_cmd)
            
            # ─── BƯỚC 3: GỬI LỆNH LẦN LƯỢT ───
            exec_dt = 0.02
            a = time.perf_counter()
            for i in range(INTERP_STEPS):
                loop_start = time.perf_counter()
                
                target_p = interp_p_phys[i]
                target_g = interp_g_cmd[i]
                
                p_phys_curr = indy.get_pose()
                p_safe_phys = Kinematics.clamp_target(p_phys_curr, target_p)
                
                print(f"  Executing {i+1:02d}/{INTERP_STEPS} | Target: {np.round(target_p[:3], 1)} | Grip: {target_g:.2f}")
                
                if args.execute:
                    try:
                        indy.send_target(p_safe_phys)
                        new_g_cmd = "close" if target_g < 0.5 else "open"
                            
                        # Cooldown Filter Gripper
                        if new_g_cmd != last_g_cmd:
                            current_time = time.perf_counter()
                            if (current_time - last_g_cmd_time) >= GRIPPER_COOLDOWN:
                                susgrip.close() if new_g_cmd == "close" else susgrip.open()
                                last_g_cmd = new_g_cmd
                                last_g_cmd_time = current_time
                    except Exception as e:
                        print(f"Transmission error: {e}")
                        break
                        
                elapsed = time.perf_counter() - loop_start
                if elapsed < exec_dt:
                    time.sleep(exec_dt - elapsed)
            print(f"  [>] Execution time: {time.perf_counter()-a:.3f} s")
            
            # ─── BƯỚC 4: WAIT FOR TARGET ───
            if args.execute:
                final_p = interp_p_phys[-1]
                wait_start = time.perf_counter()
                while True:
                    p_phys_curr = indy.get_pose()
                    err_trans = np.linalg.norm(p_phys_curr[:3] - final_p[:3])
                    err_rot = np.linalg.norm((p_phys_curr[3:6] - final_p[3:6] + 180) % 360 - 180)
                    
                    if err_trans < 1.0 and err_rot < 1.0:
                        print(f"  [v] Target reached! Err: {err_trans:.1f}mm, {err_rot:.1f}deg")
                        break
                        
                    if time.perf_counter() - wait_start > 0.5:
                        print(f"  [x] Wait timeout! Err: {err_trans:.1f}mm, {err_rot:.1f}deg")
                        break
                    
                    time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopped inference loop.")
    finally:
        print("Stopping hardware threads...")
        try: indy.stop()
        except: pass
        if args.execute:
            try: indy.stop_teleop()
            except: pass
        susgrip.stop()
        cam_fpc.stop()
        cam_rs.stop()
        print("Hardware disconnected securely.")

if __name__ == "__main__":
    main()
