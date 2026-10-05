#!/usr/bin/env python3
"""
Diffusion Policy Inference trên Robot Indy7.

Kiến trúc (dễ thay thế hardware):
  - Indy7Robot     → thay bằng class robot khác, giữ interface: get_pose, send_target, start/stop_teleop
  - FPCCamera      → thay bằng camera USB khác, giữ interface: get_images, stop
  - RealSenseCamera→ thay bằng camera khác,     giữ interface: get_images, stop
  - ModbusGripper  → thay bằng gripper khác,    giữ interface: get_state, open, close, stop
  - PoseConverter  → chuyển đổi TCP ↔ model state 10D
  - PolicyWrapper  → load model + inference

Pipeline:
  Observation:  2 frame TCP gần nhất, biểu diễn trong hệ frame hiện tại (UMI relative) → 10D state
  Action:       chunk 8 action 10D, neo vào pose TCP lúc suy luận → PoseConverter.action_to_tcp() → TCP (mm, deg)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import threading
import queue
import glob
import argparse
from collections import deque
import numpy as np
import cv2
import pyrealsense2 as rs
import torch
from pymodbus.client import ModbusSerialClient
from scipy.spatial.transform import Rotation as Rot
from neuromeka import IndyDCP3
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import OBS_STATE, OBS_IMAGES


# ═══════════════════════════════════════════════════════════════
# CONFIG - Tập trung tất cả tham số ở đây
# ═══════════════════════════════════════════════════════════════

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_POLICY_PATH = os.path.join(
    BASE_DIR, "S3_output", "Date_01102026",
    "diffusion_umi_rel_lerobot_dataset_fpccam_slam_10fps_320x240",
    "checkpoints", "last", "pretrained_model"
)

# Robot
ROBOT_IP = "192.168.2.100"
ROBOT_CMD_HZ = 50          # Tần số gửi lệnh EMA (Hz)
EMA_ALPHA = 0.1            # Hệ số mượt EMA [0..1], càng nhỏ càng mượt

# Inference
INFERENCE_FPS = 10           # Tần số vòng lặp suy luận (Hz)
CHUNK_SIZE = 8               # Số bước mỗi chunk (model predict 1 lần, lấy queue CHUNK_SIZE-1 lần)
NUM_DENOISE_STEPS = 16      # Số bước denoising (giảm từ mặc định để tăng tốc)

# Camera
IMG_SIZE = (320, 240)        # (width, height) resize cho model input

# An toàn
MAX_TRANS_PER_STEP = 20.0    # mm - giới hạn tịnh tiến mỗi bước
MAX_ROT_PER_STEP = 5.0       # deg - giới hạn xoay mỗi bước
MIN_Z_HEIGHT = 10.0           # mm - sàn an toàn (không cho tool xuống thấp hơn)

# Gripper
GRIPPER_PORT = "/dev/ttyUSB0"
GRIPPER_OPEN_MM = 100.0
GRIPPER_CLOSE_MM = 10.0
GRIPPER_COOLDOWN = 0.75      # seconds - thời gian chờ tối thiểu giữa 2 lệnh đổi trạng thái


# ═══════════════════════════════════════════════════════════════
# HARDWARE: Robot
# Interface cần giữ khi thay robot khác:
#   get_pose()         → np.ndarray (6,): [x, y, z, u, v, w] (mm, deg)
#   send_target(6D)    → gửi target pose
#   start_teleop()     → bật chế độ điều khiển teleop
#   stop_teleop()      → tắt teleop
#   wait_teleop_ready()→ đợi robot sẵn sàng
#   stop()             → dừng thread, đóng kết nối
# ═══════════════════════════════════════════════════════════════

class Indy7Robot:
    """
    Điều khiển Indy7 qua IndyDCP3.
    Thread riêng chạy ở ROBOT_CMD_HZ (50Hz):
      - Đọc pose thực tế từ robot
      - Gửi lệnh EMA-smoothed (low-pass filter) để robot di chuyển mượt
      - Ghi log CSV
    """

    def __init__(self, ip, enable_execute=True, log_csv=False):
        self.driver = IndyDCP3(ip)
        self.enable_execute = enable_execute

        self._pose = np.zeros(6, dtype=np.float32)
        self._target = None
        self._ema_cmd = None
        self._lock = threading.Lock()
        self._running = True

        # CSV log
        self._csv = None
        if log_csv:
            path = f"robot_pose_log_{int(time.time())}.csv"
            self._csv = open(path, 'w')
            self._csv.write(
                "timestamp,x,y,z,u,v,w,"
                "target_x,target_y,target_z,target_u,target_v,target_w\n"
            )
            print(f"[Robot] CSV log: {path}")

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        time.sleep(1.0)

    def _loop(self):
        dt = 1.0 / ROBOT_CMD_HZ
        while self._running:
            with self._lock:
                target = self._target.copy() if self._target is not None else None

            if target is not None:
                if self._ema_cmd is None:
                    self._ema_cmd = self._pose.copy()

                a = EMA_ALPHA

                # Translation: EMA tuyến tính
                self._ema_cmd[:3] += a * (target[:3] - self._ema_cmd[:3])

                # Rotation: EMA theo đường ngắn nhất trên vòng tròn 360°
                # (tránh giật khi chuyển 179° → -179°)
                dr = (target[3:6] - self._ema_cmd[3:6] + 180) % 360 - 180
                self._ema_cmd[3:6] = (self._ema_cmd[3:6] + a * dr + 180) % 360 - 180

                if self.enable_execute:
                    try:
                        self.driver.movetelel_abs(
                            tpos=list(self._ema_cmd.astype(np.float32)),
                            vel_ratio=0.1,
                            acc_ratio=1.0,
                        )
                    except Exception as e:
                        print(f"[Robot] Command error: {e}")

            # Đọc pose thực tế
            try:
                p = self.driver.get_robot_data()["p"]
                with self._lock:
                    self._pose = np.array(p, dtype=np.float32)
            except Exception:
                pass

            # Ghi CSV
            if self._csv is not None:
                rp = self._pose
                tp = self._ema_cmd if self._ema_cmd is not None else rp
                self._csv.write(
                    f"{time.perf_counter():.3f},"
                    f"{rp[0]:.2f},{rp[1]:.2f},{rp[2]:.2f},"
                    f"{rp[3]:.2f},{rp[4]:.2f},{rp[5]:.2f},"
                    f"{tp[0]:.2f},{tp[1]:.2f},{tp[2]:.2f},"
                    f"{tp[3]:.2f},{tp[4]:.2f},{tp[5]:.2f}\n"
                )

            time.sleep(dt)

    def get_pose(self):
        with self._lock:
            return self._pose.copy()

    def send_target(self, p_target):
        with self._lock:
            self._target = np.array(p_target, dtype=np.float32)

    def start_teleop(self):
        try:
            self.driver.stop_teleop()
        except Exception:
            pass
        time.sleep(0.1)
        self.driver.start_teleop(method=0)

    def stop_teleop(self):
        self.driver.stop_teleop()
        self.driver.stop_motion()

    def wait_teleop_ready(self, timeout=0.6):
        """Đợi robot chuyển sang TELE_OP mode (op_state=17)."""
        for _ in range(int(timeout / 0.02)):
            state = self.driver.get_robot_data().get("op_state", 0)
            if state == 17:
                return True
            time.sleep(0.02)
        return False

    def stop(self):
        self._running = False
        try:
            self._thread.join(timeout=0.3)
        except Exception:
            pass
        if self._csv is not None:
            self._csv.close()


# ═══════════════════════════════════════════════════════════════
# HARDWARE: Cameras
# Interface cần giữ khi thay camera khác:
#   get_images() → np.ndarray (H, W, 3) dtype=uint8, RGB
#   stop()
# ═══════════════════════════════════════════════════════════════

class FPCCamera:
    """USB Webcam gắn trên tool (first-person view). Thread capture liên tục."""

    def __init__(self, img_size=IMG_SIZE):
        self._w, self._h = img_size
        self._frame = None
        self._lock = threading.Lock()
        self._stopped = False

        cam_id = self._find_webcam_id()
        if cam_id is None:
            raise RuntimeError("Không tìm thấy USB Webcam! Kiểm tra cáp.")

        self._cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        ok, frame = self._cap.read()
        if ok:
            self._frame = frame

        self._thread = threading.Thread(target=self._capture, daemon=True)
        self._thread.start()

    def _capture(self):
        while not self._stopped:
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = frame
            else:
                print("[FPC] Dropped frame!")

    def get_images(self):
        with self._lock:
            if self._frame is not None:
                resized = cv2.resize(self._frame, (self._w, self._h))
                return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return np.zeros((self._h, self._w, 3), dtype=np.uint8)

    def stop(self):
        self._stopped = True
        try:
            self._thread.join(timeout=0.3)
        except Exception:
            pass
        if self._cap is not None:
            self._cap.release()

    @staticmethod
    def _find_webcam_id():
        """Tự động tìm USB webcam (bỏ qua RealSense)."""
        paths = sorted(
            glob.glob('/sys/class/video4linux/video*'),
            key=lambda x: int(os.path.basename(x).replace('video', ''))
        )
        for path in paths:
            try:
                with open(os.path.join(path, 'name')) as f:
                    name = f.read().strip()
                if "RealSense" not in name and "Metadata" not in name:
                    idx = int(os.path.basename(path).replace('video', ''))
                    print(f"[FPC] Found: '{name}' at /dev/video{idx}")
                    return idx
            except Exception:
                continue
        return None


class RealSenseCamera:
    """Intel RealSense (side view). Thread capture liên tục."""

    def __init__(self, img_size=IMG_SIZE):
        self._w, self._h = img_size
        self._frame = None
        self._lock = threading.Lock()
        self._stopped = False

        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self._pipeline.start(config)
        self._align = rs.align(rs.stream.color)

        # Đợi camera ổn định
        for _ in range(10):
            frames = self._pipeline.wait_for_frames()
            color = self._align.process(frames).get_color_frame()
            if color:
                self._frame = np.asanyarray(color.get_data())
                break

        self._thread = threading.Thread(target=self._capture, daemon=True)
        self._thread.start()

    def _capture(self):
        while not self._stopped:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                color = self._align.process(frames).get_color_frame()
                if color:
                    with self._lock:
                        self._frame = np.asanyarray(color.get_data())
            except Exception:
                pass
            time.sleep(1/60)

    def get_images(self):
        with self._lock:
            if self._frame is not None:
                resized = cv2.resize(self._frame, (self._w, self._h))
                return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return np.zeros((self._h, self._w, 3), dtype=np.uint8)

    def stop(self):
        self._stopped = True
        try:
            self._thread.join(timeout=0.3)
        except Exception:
            pass
        try:
            self._pipeline.stop()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
# HARDWARE: Gripper
# Interface cần giữ khi thay gripper khác:
#   get_state()       → float [0=đóng, 1=mở]
#   open() / close()
#   move(norm_pos)    → di chuyển đến vị trí [0..1]
#   stop()
# ═══════════════════════════════════════════════════════════════

class ModbusGripper:
    """Gripper qua Modbus RTU. Thread riêng đọc/ghi ~100Hz."""

    def __init__(self, port=GRIPPER_PORT, open_mm=GRIPPER_OPEN_MM, close_mm=GRIPPER_CLOSE_MM):
        self.open_mm = open_mm
        self.close_mm = close_mm
        self._range = open_mm - close_mm

        self._client = ModbusSerialClient(
            port=port, baudrate=115200, stopbits=1,
            bytesize=8, parity='N', timeout=0.2,
            retries=0, handle_local_echo=False,
        )
        self._client.connect()

        self._state = 1.0
        self._lock = threading.Lock()
        self._cmd_queue = queue.Queue()
        self._running = True

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            try:
                cmd = self._cmd_queue.get_nowait()
            except queue.Empty:
                cmd = None

            if cmd is not None:
                try:
                    self._client.write_register(1, cmd, device_id=1)
                except Exception:
                    pass
            else:
                try:
                    res = self._client.read_input_registers(address=1, count=1, device_id=1)
                    if not res.isError():
                        pos_mm = float(res.registers[0])
                        norm = np.clip((pos_mm - self.close_mm) / self._range, 0.0, 1.0)
                        with self._lock:
                            self._state = float(norm)
                except Exception:
                    pass

            time.sleep(0.01)

    def get_state(self):
        with self._lock:
            return self._state

    def move(self, norm_pos):
        norm_pos = float(np.clip(norm_pos, 0.0, 1.0))
        self._cmd_queue.put(int(norm_pos * self._range + self.close_mm))

    def open(self):
        self._cmd_queue.put(int(self.open_mm))

    def close(self):
        self._cmd_queue.put(int(self.close_mm))

    def stop(self):
        self._running = False
        try:
            self._thread.join(timeout=0.5)
        except Exception:
            pass
        try:
            self._client.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
# POSE CONVERTER: TCP (mm, deg) ↔ Model State 10D
#
# State 10D = [x, y, z, r1..r6, gripper]
#   - xyz    : vị trí tool (meters)
#   - r1..r6 : 6D rotation = 2 cột đầu của rotation matrix
#   - gripper: [0=đóng, 1=mở]
#
# UMI relative-to-current (khớp s3_umi_relative_wrapper.py lúc train):
#   Mỗi lần suy luận, lấy pose TCP hiện tại làm gốc T_cur.
#   Observation (n_obs_steps=2 frame): state_i = inv(T_cur) @ T_i
#       → frame hiện tại luôn = [0,0,0, 1,0,0, 0,1,0], frame t-1 = dịch chuyển so với hiện tại.
#   Action (8 bước của chunk): T_target = T_cur @ T_rel_predicted
#       → cả 8 action neo vào CÙNG pose T_cur tại thời điểm suy luận.
#   Robot báo trực tiếp TCP (đã gồm tool offset) → không cần ma trận Eye-in-Hand.
# ═══════════════════════════════════════════════════════════════

class PoseConverter:
    """Chuyển đổi giữa TCP robot và model state 10D (UMI relative-to-current)."""

    @staticmethod
    def relative_state(T_ref_inv, T_i, gripper):
        """Pose T_i (4x4, mm) biểu diễn trong hệ T_ref → state 10D (m)."""
        T_rel = T_ref_inv @ T_i
        pos = T_rel[:3, 3] / 1000.0  # mm → m (training data dùng meters)
        R_rel = T_rel[:3, :3]
        rot_6d = np.concatenate([R_rel[:, 0], R_rel[:, 1]])
        return np.array([*pos, *rot_6d, gripper], dtype=np.float32)

    @staticmethod
    def action_to_tcp(action, T_anchor, current_euler):
        """
        Model action 10D (relative so với T_anchor) → robot TCP target 6D (mm, deg).
        Công thức: T_tcp_target = T_anchor @ T_rel_predicted
        T_anchor = pose TCP lúc model suy luận ra chunk này.
        """
        # Gram-Schmidt: 6D → orthogonal rotation matrix
        v1 = action[3:6].copy()
        v2 = action[6:9].copy()
        v1 /= np.linalg.norm(v1)
        v3 = np.cross(v1, v2)
        v3 /= np.linalg.norm(v3)
        v2 = np.cross(v3, v1)

        T_rel = np.eye(4)
        T_rel[:3, :3] = np.column_stack((v1, v2, v3))
        T_rel[:3, 3] = action[:3] * 1000.0  # m → mm

        T_tcp_target = T_anchor @ T_rel

        pos = T_tcp_target[:3, 3]
        euler = Rot.from_matrix(T_tcp_target[:3, :3]).as_euler('xyz', degrees=True)
        euler = PoseConverter._closest_euler(euler, current_euler)

        return np.array([*pos, *euler], dtype=np.float32)

    @staticmethod
    def clamp_target(current, target, max_trans=MAX_TRANS_PER_STEP,
                     max_rot=MAX_ROT_PER_STEP, min_z=MIN_Z_HEIGHT):
        """Giới hạn bước di chuyển tối đa mỗi step (an toàn)."""
        current = np.asarray(current, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        safe = current.copy()

        # Translation
        dt = target[:3] - current[:3]
        max_dt = np.max(np.abs(dt))
        if max_dt > max_trans:
            dt *= max_trans / max_dt
        safe[:3] += dt

        # Rotation (shortest path)
        dr = (target[3:6] - current[3:6] + 180) % 360 - 180
        max_dr = np.max(np.abs(dr))
        if max_dr > max_rot:
            dr *= max_rot / max_dr
        safe[3:6] = (current[3:6] + dr + 180) % 360 - 180

        # Z safety floor
        if safe[2] < min_z:
            safe[2] = min_z

        return safe

    @staticmethod
    def tcp_to_matrix(p):
        """TCP 6D [x,y,z,u,v,w] (mm, deg) → 4x4 homogeneous matrix."""
        T = np.eye(4)
        T[:3, :3] = Rot.from_euler('xyz', p[3:6], degrees=True).as_matrix()
        T[:3, 3] = p[:3]
        return T

    @staticmethod
    def _closest_euler(target, current):
        """Chọn biểu diễn Euler xyz gần nhất với góc hiện tại (tránh gimbal jump)."""
        alt = np.array([target[0] + 180, 180 - target[1], target[2] + 180])

        d_target = np.linalg.norm((target - current + 180) % 360 - 180)
        d_alt = np.linalg.norm((alt - current + 180) % 360 - 180)

        chosen = alt if d_alt < d_target else target
        return (chosen + 180) % 360 - 180


# ═══════════════════════════════════════════════════════════════
# POLICY WRAPPER: Load model + inference
# ═══════════════════════════════════════════════════════════════

class PolicyWrapper:
    """
    Load Diffusion Policy và chạy inference theo UMI relative-to-current.

    Tự quản lý:
      - history  : n_obs_steps quan sát gần nhất (pose TCP tuyệt đối + gripper + ảnh)
      - actions  : queue các action của chunk hiện tại
      - anchor   : pose TCP lúc suy luận chunk (gốc của cả chunk action)
    Không dùng model.select_action vì queue nội bộ của nó lưu state đã tính tại các thời điểm
    khác nhau → không thể re-anchor frame t-1 về frame hiện tại.
    """

    IMAGE_KEYS = ("observation.image", "observation.image_side")  # (front/FPC, side/RealSense)

    def __init__(self, policy_path, device):
        print(f"[Policy] Loading: {policy_path}")
        self.device = device

        self.model = DiffusionPolicy.from_pretrained(policy_path, local_files_only=True)
        self.model.diffusion.num_inference_steps = NUM_DENOISE_STEPS
        self.model.to(device)
        self.model.eval()

        self.pre, self.post = make_pre_post_processors(
            self.model.config, pretrained_path=policy_path
        )
        self.n_obs = self.model.config.n_obs_steps
        self.image_keys = list(self.model.config.image_features)
        self.last_prev_obs_mm = np.zeros(3)  # dịch chuyển frame t-1 so với hiện tại (để log)
        self.reset()
        print(f"[Policy] Loaded. Denoise steps = {NUM_DENOISE_STEPS} | "
              f"n_obs={self.n_obs} | n_action_steps={self.model.config.n_action_steps}")

    def reset(self):
        self._history = deque(maxlen=self.n_obs)
        self._actions = deque()
        self._anchor = None

    def _push_obs(self, p_tcp, gripper, img_front, img_side):
        obs = (PoseConverter.tcp_to_matrix(p_tcp), float(gripper),
               {"observation.image": img_front, "observation.image_side": img_side})
        # Lần đầu: nhân bản quan sát đầu cho đủ n_obs_steps (giống populate_queues của LeRobot)
        while len(self._history) < self.n_obs - 1:
            self._history.append(obs)
        self._history.append(obs)
        return obs[0]

    def _build_batch(self):
        """Quan sát n_obs_steps frame, biểu diễn trong hệ pose frame hiện tại (mới nhất)."""
        T_cur_inv = np.linalg.inv(self._history[-1][0])
        states = np.stack([PoseConverter.relative_state(T_cur_inv, T, g) for T, g, _ in self._history])
        self.last_prev_obs_mm = states[0, :3] * 1000.0

        def to_tensor(img):
            return torch.from_numpy(np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0)

        batch = {"observation.state": torch.from_numpy(states).unsqueeze(0).to(self.device)}
        for key in self.IMAGE_KEYS:
            batch[key] = torch.stack([to_tensor(h[2][key]) for h in self._history]).unsqueeze(0).to(self.device)
        return batch

    def _infer_chunk(self):
        """Chạy diffusion 1 lần → (n_action_steps, 10) action đã unnormalize, relative so với anchor."""
        batch = self.pre(self._build_batch())
        model_in = {
            OBS_STATE: batch[OBS_STATE],
            OBS_IMAGES: torch.stack([batch[k] for k in self.image_keys], dim=-4),
        }
        t0 = time.perf_counter()
        with torch.inference_mode():
            actions = self.model.diffusion.generate_actions(model_in)  # (1, n_action_steps, 10)
        inf_time = time.perf_counter() - t0

        actions = self.post(actions)
        if isinstance(actions, dict):
            actions = actions["action"]
        return actions.squeeze(0).cpu().numpy(), inf_time

    def warmup(self, p_tcp, gripper, img_front, img_side):
        """CUDA warmup (3 passes) để inference sau nhanh hơn."""
        print("[Policy] CUDA warmup...")
        self._push_obs(p_tcp, gripper, img_front, img_side)
        for _ in range(3):
            self._infer_chunk()
        self.reset()
        print("[Policy] Warmup done.")

    def step(self, p_tcp, gripper, img_front, img_side):
        """
        Gọi mỗi vòng lặp. Khi queue rỗng → suy luận chunk mới, anchor = pose TCP hiện tại.
        Returns:
            action   : np.ndarray (10,) — action relative so với anchor
            T_anchor : np.ndarray (4,4) — pose TCP (mm) lúc suy luận chunk chứa action này
            inf_time : float — thời gian inference (0 nếu lấy từ queue)
            is_new   : bool — True nếu vừa suy luận chunk mới
        """
        T_cur = self._push_obs(p_tcp, gripper, img_front, img_side)

        is_new, inf_time = False, 0.0
        if not self._actions:
            self._anchor = T_cur
            chunk, inf_time = self._infer_chunk()
            self._actions.extend(chunk)
            is_new = True

        return self._actions.popleft(), self._anchor, inf_time, is_new


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Diffusion Policy Inference on Robot")
    parser.add_argument("--policy_path", type=str, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--execute", action="store_true", help="Thực thi lệnh trên robot thật")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("\n" + "=" * 60)
    print("DIFFUSION POLICY INFERENCE")
    print(f"  Device : {device}")
    print(f"  Execute: {args.execute}")
    print(f"  Policy : {args.policy_path}")
    print("=" * 60)

    # ── 1. Khởi tạo Hardware ──────────────────────────────────
    print("\n[1/4] Khởi tạo hardware...")

    robot = Indy7Robot(ROBOT_IP, enable_execute=args.execute)
    print("  Robot OK")

    grip = ModbusGripper()
    print("  Gripper OK")

    cam_front = FPCCamera()
    print("  FPC Camera OK")

    cam_side = RealSenseCamera()
    print("  RealSense OK")

    # ── 2. Load Model ─────────────────────────────────────────
    print("\n[2/4] Load model...")
    time.sleep(1.0)
    policy = PolicyWrapper(args.policy_path, device)

    # Warmup với dữ liệu thật
    policy.warmup(robot.get_pose(), grip.get_state(), cam_front.get_images(), cam_side.get_images())

    # ── 3. Mode ───────────────────────────────────────────────
    print("\n[3/4] UMI relative-to-current: gốc tọa độ = pose TCP tại mỗi lần suy luận")

    # ── 4. Bật Teleop & xác nhận ──────────────────────────────
    if args.execute:
        print("\n[4/4] Chuẩn bị chạy...")
        print("[WARNING] EXECUTE MODE - Robot sẽ di chuyển thật!")
        if input("Gõ 'YES' để tiếp tục: ").strip() != "YES":
            print("Hủy.")
            return

        robot.start_teleop()
        if not robot.wait_teleop_ready():
            print("[ERROR] Robot không vào được TELE_OP mode!")
            return
        print("[OK] TELE_OP ready.")

        # Keep-alive buffer
        p_curr = robot.get_pose()
        for _ in range(10):
            robot.send_target(p_curr)
            time.sleep(0.01)
    else:
        print("\n[4/4] Chế độ DRY-RUN (không gửi lệnh robot)")

    # Keyboard listener: nhấn 'p' để đánh dấu thời điểm jerk
    from pynput import keyboard
    def on_press(key):
        try:
            if key.char in ('p', 'P'):
                print(f"\n[MARK] Jerk at t={time.perf_counter():.3f}s\n")
        except AttributeError:
            pass
    keyboard.Listener(on_press=on_press).start()

    # ── Inference Loop ────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"INFERENCE LOOP ({INFERENCE_FPS} Hz, chunk={CHUNK_SIZE})")
    print("Ctrl+C để dừng")
    print("=" * 60 + "\n")

    step_idx = 0
    last_sent = None
    chunk_start = time.perf_counter()
    last_grip_cmd = "open"
    last_grip_time = time.perf_counter()
    loop_dt = 1.0 / INFERENCE_FPS

    try:
        while True:
            t_start = time.perf_counter()

            # --- Thu thập dữ liệu ---
            p_curr = robot.get_pose()
            g_state = grip.get_state()
            img_f = cam_front.get_images()
            img_s = cam_side.get_images()

            # --- Inference (obs relative-to-current được dựng bên trong policy.step) ---
            action, T_anchor, inf_time, is_new = policy.step(p_curr, g_state, img_f, img_s)

            if is_new:
                track_err = np.linalg.norm(p_curr[:3] - last_sent[:3]) if last_sent is not None else 0.0
                po = policy.last_prev_obs_mm
                print(f"[AI] Inference: {inf_time:.3f}s | "
                      f"OBS[t-1 vs t]=({po[0]:.1f}, {po[1]:.1f}, {po[2]:.1f}) mm | "
                      f"ACT[0]=({action[0]*1000:.1f}, {action[1]*1000:.1f}, {action[2]*1000:.1f}) mm | "
                      f"Grip={action[9]:.2f} | Robot-vs-lastTarget={track_err:.1f} mm")

            # --- Action: 10D (relative anchor) → TCP target ---
            target_tcp = PoseConverter.action_to_tcp(action, T_anchor, p_curr[3:6])
            safe_tcp = PoseConverter.clamp_target(p_curr, target_tcp)

            # --- Chunk tracking ---
            step_idx = (step_idx % CHUNK_SIZE) + 1
            if step_idx == 1:
                chunk_start = t_start

            print(f"  [{step_idx}/{CHUNK_SIZE}] Target: {np.round(target_tcp[:3], 1)} mm | "
                  f"Euler: {np.round(target_tcp[3:6], 1)} deg")

            if step_idx == CHUNK_SIZE:
                print(f"  Chunk {CHUNK_SIZE} steps: {time.perf_counter() - chunk_start:.3f}s")

            # --- Gửi lệnh robot ---
            robot.send_target(safe_tcp)
            last_sent = safe_tcp

            # --- Điều khiển gripper (cooldown filter) ---
            if args.execute:
                new_cmd = "close" if action[9] < 0.5 else "open"
                now = time.perf_counter()
                if new_cmd != last_grip_cmd and (now - last_grip_time) >= GRIPPER_COOLDOWN:
                    grip.close() if new_cmd == "close" else grip.open()
                    last_grip_cmd = new_cmd
                    last_grip_time = now
                    print(f"  [Gripper] {new_cmd.upper()}")

            # --- Giữ đúng tần số ---
            elapsed = time.perf_counter() - t_start
            if elapsed < loop_dt:
                time.sleep(loop_dt - elapsed)

    except KeyboardInterrupt:
        print("\n\nDừng inference loop.")

    finally:
        print("Ngắt kết nối hardware...")
        robot.stop()
        if args.execute:
            try:
                robot.stop_teleop()
            except Exception:
                pass
        grip.stop()
        cam_front.stop()
        cam_side.stop()
        print("Done. Hardware disconnected.")


if __name__ == "__main__":
    main()
