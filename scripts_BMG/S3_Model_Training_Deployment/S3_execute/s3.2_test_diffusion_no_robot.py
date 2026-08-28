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

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_POLICY_PATH = os.path.join(BASE_DIR, "output_trained", "diffusion_checkpoints_diff_slam", "checkpoints", "last", "pretrained_model")

GRIPPER_OPEN_MM = 120.0
GRIPPER_CLOSE_MM = 30.0

H, W = 540,960
FPS = 20

# ============================================================
# HARDWARE CLASSES (NO ROBOT)
# ============================================================
class camera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, W, H, rs.format.rgb8, 30)
        
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
            time.sleep(0.016)

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
            time.sleep(0.03)

    def close(self):
        self._cmd_queue.put(int(GRIPPER_CLOSE_MM))

    def open(self):
        self._cmd_queue.put(int(GRIPPER_OPEN_MM))

    def move(self, norm_pos):
        norm_pos = float(np.clip(norm_pos, 0.0, 1.0))
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
# MAIN INFERENCE LOOP
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", type=str, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🚀 MOCK DIFFUSION INFERENCE (HANDHELD CAMERA TEST)")
    print("=" * 70)

    device = torch.device(args.device)

    # Load Policy
    print(f"Loading Diffusion model from: {args.policy_path}")
    policy = DiffusionPolicy.from_pretrained(args.policy_path, local_files_only=True)
    policy.diffusion.num_inference_steps = 16  # Rút ngắn số bước để chạy kịp 20Hz
    policy.to(device)
    policy.eval()

    print("Loading Preprocessors...")
    from lerobot.policies.factory import make_pre_post_processors
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=args.policy_path)

    # Initialize Threads
    print("Connecting to SusGrip...")
    susgrip = gripper()
    print("Initializing Realsense Camera...")
    cam = camera()
    
    time.sleep(1.0) # Đợi camera

    # Khởi tạo tọa độ SLAM ảo ban đầu
    # 9D (Tịnh tiến 3D + Xoay 6D)
    mock_slam_state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    def get_obs_dict(slam_9dof, g_state, rgb_img):
        obs_state = np.array([*slam_9dof, g_state], dtype=np.float32)
        obs_state_t = torch.from_numpy(obs_state).unsqueeze(0).to(device)

        chw = np.transpose(rgb_img, (2, 0, 1)).astype(np.float32) / 255.0
        img_t = torch.from_numpy(chw).unsqueeze(0).to(device)

        return {
            "observation.state": obs_state_t,
            "observation.image": img_t,
        }

    # Model CUDA Graph Warmup
    print("Warming up CUDA Graph...")
    dummy_obs = get_obs_dict(mock_slam_state, susgrip.get_gripper_state(), cam.get_images())
    dummy_obs = preprocessor(dummy_obs)
    with torch.inference_mode():
        for _ in range(3):
            policy.select_action(dummy_obs)
    print("Warmup complete.")
    
    if hasattr(policy, "reset"):
        policy.reset()
            
    dt = 1.0 / FPS
    print(f"Bắt đầu vòng lặp test...\n")

    frame_idx = 0
    last_g_cmd = "open"

    try:
        while True:
            loop_start = time.perf_counter()
            frame_idx += 1

            # 1. Đọc trạng thái
            g_state = susgrip.get_gripper_state()
            rgb_img = cam.get_images() 

            # 2. Đóng gói Observation (Dùng tọa độ SLAM ảo)
            obs_dict = get_obs_dict(mock_slam_state, g_state, rgb_img)
            obs_dict = preprocessor(obs_dict)

            # 3. Dự đoán Action (8 Chiều)
            with torch.inference_mode():
                action_pred = policy.select_action(obs_dict)
                
            action_pred = postprocessor(action_pred)
            
            if isinstance(action_pred, dict):
                act_vec = action_pred["action"].squeeze(0).cpu().numpy()
            else:
                act_vec = action_pred.squeeze(0).cpu().numpy()
                
            if len(act_vec.shape) > 1 and act_vec.shape[0] == 1:
                act_vec = act_vec[0]

            # 4. In ra thông số 10 chiều
            target_tx, target_ty, target_tz = act_vec[0], act_vec[1], act_vec[2]
            v1 = act_vec[3:6]
            v2 = act_vec[6:9]
            g_cmd_target = float(act_vec[9])

            print(f"Predicted SLAM 10D Action: [X:{target_tx:5.2f}, Y:{target_ty:5.2f}, Z:{target_tz:5.2f}, V1:[{v1[0]:4.2f},{v1[1]:4.2f},{v1[2]:4.2f}], V2:[{v2[0]:4.2f},{v2[1]:4.2f},{v2[2]:4.2f}], Grip:{g_cmd_target:4.2f}]")

            # 5. Cập nhật trạng thái SLAM ảo bằng chính dự đoán của mạng AI (Closed-loop Hallucination)
            # Điều này giúp AI tự cảm giác rằng nó đang "tiến tới" mục tiêu trong không gian toán học!
            mock_slam_state = act_vec[:9]

            # 6. Kích hoạt kìm liên tục
            susgrip.move(g_cmd_target)

            elapsed = time.perf_counter() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\nStopped test loop.")
    finally:
        print("Stopping hardware threads...")
        susgrip.stop()
        cam.stop()
        print("Hardware disconnected securely.")

if __name__ == "__main__":
    main()
