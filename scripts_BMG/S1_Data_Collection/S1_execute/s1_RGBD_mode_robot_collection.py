'''
Using robot and scripts to collect data, two camera are mounted on robot arm
Two cameras are used: an Intel RealSense D435i captures RGB-D data for SLAM at 55 FPS,
and a webcam records 20 FPS RGB data for training a diffusion model.
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass
import time
import threading
import queue
import shutil
import inspect
import queue

import warnings
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image
import pyrealsense2 as rs
from pymodbus.client import ModbusSerialClient

from neuromeka import IndyDCP3, OpState

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Camera Setup
H, W = 540,960
FPS = 55  
CAMERA_SERIAL = "317222074902"
# Robot Setup
robot_ip = "192.168.2.100"
# AI setup
MODEL_PATH = "/media/apicoo-ai/5511010c-3660-41c3-b501-36e739767b6a/SUMI/.files/mobile_sam.pt"
# Gripper Calibration
GRIPPER_OPEN_MM = 90.0
GRIPPER_CLOSE_MM =10.0

class Rate:
    def __init__(self, hz, name: str = ""):
        self.dt = 1.0 / hz
        self.last_time = time.perf_counter()
        self.name = name
        
    def sleep(self):
        elapsed = time.perf_counter() - self.last_time
        if elapsed < self.dt:
            time.sleep(self.dt - elapsed)
        else:
            print(f"Loop is too slow: {self.name} {elapsed}")
        self.last_time = time.perf_counter()

class robot:
    def __init__(self):
        self.robot = IndyDCP3(robot_ip, 0)
        self.pick_point = None
        self.via_point = None
        self.place_point = [222.8, -642.5, 78.5, 37.5, 179.0, 90.0]
        self.home = [-23.887644,-470.4143,209.37445,32.759556,178.70001,92.19685]

        self.latest_q = np.zeros(6, dtype=np.float32)
        self.latest_p = np.zeros(6, dtype=np.float32)
        self.latest_qdot = np.zeros(6, dtype=np.float32)
        self.latest_pdot = np.zeros(6, dtype=np.float32)
        self.latest_op_state = 0
        self.full_data_cache = None
        self.lock = threading.Lock()
        self.socket_lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        time.sleep(0.5)

    def _poll(self):
        rate = Rate(60, name="robot")
        while self.running:
            try:
                with self.socket_lock:
                    data = self.robot.get_robot_data()
                with self.lock:
                    self.full_data_cache = data
                    self.latest_q = np.array(data["q"], dtype=np.float32)
                    self.latest_p = np.array(data["p"], dtype=np.float32)
                    self.latest_qdot = np.array(data["qdot"], dtype=np.float32)
                    self.latest_pdot = np.array(data["pdot"], dtype=np.float32)
                    self.latest_op_state = int(data.get("op_state", 0))
            except Exception:
                pass
            rate.sleep()

    def get_robot_data(self):
        with self.lock:
            return self.latest_q.copy()

    def get_robot_pose(self):
        with self.lock:
            return self.latest_p.copy()

    def get_full_data(self):
        with self.lock:
            return self.full_data_cache

    def wait_for_op_state_idle(self, timeout=10.0):
        # Poll from local RAM state cache to prevent blocking the socket lock during robot moves
        # Sleep briefly first to allow the robot to start moving and transition out of IDLE
        time.sleep(0.15)
        start_time = time.perf_counter()
        while time.perf_counter() - start_time < timeout:
            with self.lock:
                current_state = self.latest_op_state
            if current_state == OpState.IDLE:
                return True
            time.sleep(0.01)
        print("Timeout waiting for robot to reach IDLE state!")
        return False

    def stop(self):
        self.running = False
        try:
            self.thread.join(timeout=0.2)
        except Exception:
            pass

    def move_to_pick(self):
        print("Moving to Pick position!", self.pick_point)
        with self.socket_lock:
            self.robot.movel_time(self.pick_point, move_time = 4.0)

    def move_to_place(self):
        with self.socket_lock:
            self.robot.movec(self.via_point, self.place_point,vel_ratio=25,acc_ratio=20)

    def movel(self, target, vel_ratio=25, acc_ratio=25):
        with self.socket_lock:
            self.robot.movel(target, vel_ratio=vel_ratio, acc_ratio=acc_ratio)

    def stop_motion(self):
        with self.socket_lock:
            self.robot.stop_motion()

    def calculate_points(self):
        pick_arr = np.array(self.pick_point, dtype=np.float32)
        place_arr = np.array(self.place_point, dtype=np.float32)

        place_arr[3:6] = pick_arr[3:6]

        via_arr = pick_arr.copy()
        via_arr[0] = (pick_arr[0] + place_arr[0]) / 2.0
        via_arr[1] = (pick_arr[1] + place_arr[1]) / 2.0
        via_arr[2] = 200.0  # Do cao an toan khi luon

        self.pick_point = pick_arr.tolist()
        self.place_point = place_arr.tolist()
        self.via_point = via_arr.tolist()


class camera:
    def __init__(self, serial_number, use_depth=False):
        self.serial_number = serial_number
        self.use_depth = use_depth
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(self.serial_number)
        # Color chạy ở 960x540 @ 60FPS
        self.config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 60)

        if self.use_depth:
            # Depth BẮT BUỘC phải là 848x480 để hỗ trợ 60FPS
            self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
            # Hàm align này sẽ tự động phóng to/thu nhỏ (resize & crop) ảnh Depth 848x480 
            # cho khớp TỪNG PIXEL một với ảnh Màu 960x540
            self.align = rs.align(rs.stream.color)
        else:
            self.align = None

        # Start pipeline
        profile = self.pipeline.start(self.config)

        if self.use_depth:
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
        else:
            self.depth_scale = 0.001

        # Optimize frame queue size to 1 to avoid frame accumulation/lag
        try:
            sensors = profile.get_device().query_sensors()
            for sensor in sensors:
                if sensor.supports(rs.option.frames_queue_size):
                    sensor.set_option(rs.option.frames_queue_size, 1)
        except Exception as e:
            print(f"Warning: could not set frames_queue_size for camera {self.serial_number}: {e}")

        self.latest_frame = None
        self.latest_depth_data = None
        self.running = True
        self.lock = threading.Lock()

        # Start background polling thread
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

        # Wait for first frames to arrive
        time.sleep(1.0)

    def _poll(self):
        rate = Rate(58, name="camera")
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=100)
                if self.use_depth:
                    aligned_frames = self.align.process(frames)
                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()
                    if color_frame and depth_frame:
                        img = np.asanyarray(color_frame.get_data())
                        depth_arr = np.asanyarray(depth_frame.get_data())
                        with self.lock:
                            self.latest_frame = img
                            self.latest_depth_data = depth_arr
                else:
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        img = np.asanyarray(color_frame.get_data())
                        with self.lock:
                            self.latest_frame = img
            except Exception:
                pass
            rate.sleep()

    def get_images(self):
        with self.lock:
            if self.latest_frame is None:
                return np.zeros((H, W, 3), dtype=np.uint8)
            return self.latest_frame.copy()

    def get_depth_data(self):
        with self.lock:
            if self.latest_depth_data is None:
                return np.zeros((H, W), dtype=np.uint16), self.depth_scale
            return self.latest_depth_data.copy(), self.depth_scale

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
            port='/dev/ttyUSB0',
            baudrate=115200,
            stopbits=1,
            bytesize=8,
            parity='N',
            timeout=0.2,
            retries=0,
            handle_local_echo=False
        )
        self.client.connect()
        self.current_state = 1.0  # Actual normalized state [0.0 - 1.0]
        self._lock = threading.Lock()
        self._cmd_queue = queue.Queue()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            # 1. Handle commands first
            cmd = None
            try:
                cmd = self._cmd_queue.get_nowait()
            except queue.Empty:
                pass

            if cmd is not None:
                try:
                    self.client.write_register(1, cmd, device_id=1)
                except Exception as e:
                    print(f"\nGripper command write failed: {e}")
            else:
                # 2. Read state if no pending commands
                try:
                    res = self.client.read_input_registers(address=1, count=1, device_id=1)
                    if not res.isError():
                        pos_mm = float(res.registers[0])
                        # Feedback Normalization Formula
                        norm = (pos_mm - GRIPPER_CLOSE_MM) / (GRIPPER_OPEN_MM - GRIPPER_CLOSE_MM)
                        norm_clipped = float(np.clip(norm, 0.0, 1.0))
                        with self._lock:
                            self.current_state = norm_clipped
                except Exception:
                    pass
            time.sleep(0.016)         
    def close(self):
        self._cmd_queue.put(int(GRIPPER_CLOSE_MM))
        

    def open(self):
        self._cmd_queue.put(int(GRIPPER_OPEN_MM))

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


class ai:
    def __init__(self):
        from ultralytics import SAM
        self.sam = SAM('mobile_sam.pt')
        self.sam.to("cuda")
        self.INTRINSIC_PATH = "/media/apicoo-ai/5511010c-3660-41c3-b501-36e739767b6a/ORB_SLAM3/matrix_calib/realsense_flange_louis/camera_intrinsics.json"
        self.EXTRINSIC_PATH = "/media/apicoo-ai/5511010c-3660-41c3-b501-36e739767b6a/ORB_SLAM3/matrix_calib/realsense_flange_louis/eye_in_hand_result.json"
        self.intrinsic_matrix = None
        self.extrinsic_matrix = None
        self.load_matrix()
        self.clicked_point = None

    def set_click(self, x, y):
        self.clicked_point = [x, y]

    def clear_click(self):
        self.clicked_point = None

    def predict(self, image):
        if self.clicked_point is None:
            return None, None, None, None, None

        with torch.no_grad():
            results = self.sam(image, points=[self.clicked_point], labels=[1], verbose=False)

        if not results or not results[0].masks:
            return None, None, None, None, None

        mask_data = results[0].masks.data[0].cpu().numpy()
        mask_data = mask_data.astype(np.uint8)
        mask_data = cv2.resize(mask_data, (image.shape[1], image.shape[0]))
        mask_vis = mask_data * 255

        # --- TIM GOC VA TAM DUA TREN 2 CANH DAI SONG SONG ---
        contours, _ = cv2.findContours(mask_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, None, None, None

        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)

        cx, cy = float(rect[0][0]), float(rect[0][1])

        box_pts = cv2.boxPoints(rect)
        box_pts = np.int64(box_pts)

        edge1 = box_pts[1] - box_pts[0]
        edge2 = box_pts[3] - box_pts[0]

        len1 = np.linalg.norm(edge1)
        len2 = np.linalg.norm(edge2)

        if len1 > len2:
            long_edge_vector = edge1
        else:
            long_edge_vector = edge2

        angle = np.degrees(np.arctan2(long_edge_vector[1], long_edge_vector[0]))

        return cx, cy, angle, mask_vis, contours

    def load_matrix(self):
        import json
        try:
            with open(self.INTRINSIC_PATH, 'r') as f:
                data_int = json.load(f)
                self.intrinsic_matrix = np.array(data_int["camera_matrix"])
            with open(self.EXTRINSIC_PATH, 'r') as f:
                data_ext = json.load(f)
                self.extrinsic_matrix = np.array(data_ext["T_cam_to_tool"])
                # FIX BUGS: Convert units of Extrinsic from Meter to Millimeter!
                self.extrinsic_matrix[:3, 3] *= 1000.0
        except Exception as e:
            print(f"Error loading matrix: {e}")

    def cam_to_3d(self, cx, cy, depth_z):
        if self.intrinsic_matrix is None:
            return None
        fx = self.intrinsic_matrix[0, 0]
        fy = self.intrinsic_matrix[1, 1]
        px = self.intrinsic_matrix[0, 2]
        py = self.intrinsic_matrix[1, 2]

        X = (cx - px) * depth_z / fx
        Y = (cy - py) * depth_z / fy
        Z = depth_z
        return np.array([X, Y, Z, 1.0])

    def get_pick(self, cx, cy, angle_offset_deg, tcp_pose, depth_z):
        # tcp_pose: [x, y, z, rx, ry, rz]
        cam_point = self.cam_to_3d(cx, cy, depth_z)
        if cam_point is None or self.extrinsic_matrix is None:
            return None

        t_base_tcp = np.eye(4)
        t_base_tcp[:3, 3] = np.array(tcp_pose[:3])

        rx, ry, rz = np.radians(tcp_pose[3:])
        R_x = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        R_y = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        R_z = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        t_base_tcp[:3, :3] = R_z @ R_y @ R_x

        p_tcp = self.extrinsic_matrix @ cam_point
        p_base = t_base_tcp @ p_tcp

        r_cam_to_tool = self.extrinsic_matrix[:3, :3]

        theta_rad = np.radians(angle_offset_deg)
        v_cam = np.array([np.cos(theta_rad), np.sin(theta_rad), 0.0])

        v_tool = r_cam_to_tool @ v_cam

        angle_tool_deg = np.degrees(np.arctan2(v_tool[1], v_tool[0]))

        target_rz = tcp_pose[5] - angle_tool_deg - 90

        diff = target_rz - tcp_pose[5]
        while diff > 90.0:
            target_rz -= 180.0
            diff -= 180.0
        while diff <= -90.0:
            target_rz += 180.0
            diff += 180.0

        while target_rz > 180.0:
            target_rz -= 360.0
        while target_rz <= -180.0:
            target_rz += 360.0

        return [p_base[0], p_base[1], p_base[2], tcp_pose[3], tcp_pose[4], target_rz]



# ============================================================
# SLAM BACKGROUND WRITER
# ============================================================
slam_img_queue = queue.Queue()

def slam_image_writer():
    while True:
        try:
            task = slam_img_queue.get(timeout=0.1)
            if task is None:
                break
            ts_str, rgb, depth, save_dir = task
            bgr = rgb
            cv2.imwrite(f"{save_dir}/rgb/{ts_str}.png", bgr)
            cv2.imwrite(f"{save_dir}/depth/{ts_str}.png", depth)
        except queue.Empty:
            pass

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    import datetime
    import csv

    slam_state = {
        "dir": None, "rgb_f": None, "depth_f": None, "assoc_f": None,
        "gripper_f": None, "gripper_w": None, "robot_f": None, "robot_w": None,
        "writer_thread": None
    }
    
    current_gripper_cmd = 1.0  # 1.0 for open, 0.0 for close
    recording = False

    ai_data = {"mask": None, "pose": None, "contours": None, "is_running": True,
               "needs_predict": False, "lock": threading.Lock()}
    robot_moving = {"state": False, "abort": False}

    indy = None
    susgrip = None
    camera1 = None
    ai_thread = None

    try:
        print("Connecting to Indy7 robot...")
        indy = robot()

        print("Connecting to SusGrip gripper...")
        susgrip = gripper()

        print("Initializing camera...")
        camera1 = camera(CAMERA_SERIAL, use_depth=True)

        print("\n" + "=" * 70)
        print("AI + SLAM Data Collector Initialized")
        print("=" * 70)
        print("Use keyboard in the OpenCV window for control:")
        print(f"  o  -> gripper OPEN ({GRIPPER_OPEN_MM:.1f} mm)")
        print(f"  c  -> gripper CLOSE ({GRIPPER_CLOSE_MM:.1f} mm)")
        print("  p  -> PRINT current data snapshot to console")
        print("  r  -> RUN AI predict and move robot")
        print("  [  -> START recording SLAM Dataset")
        print("  ]  -> STOP and SAVE SLAM Dataset")
        print("  \\  -> DISCARD current recording")
        print("  q  -> QUIT")
        print("=" * 70)

        ai_model = ai()

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                current_time = time.time()
                if not hasattr(mouse_callback, "last_click") or current_time - mouse_callback.last_click > 0.5:
                    if x < 960:
                        mouse_callback.last_click = current_time
                        ai_model.set_click(x, y)
                        ai_data["needs_predict"] = True
                        print(f"\n[AI] Received click at ({x}, {y}) on Wrist Camera. Running SAM (1-shot)...")
                    else:
                        print("\n[AI] Click to segment feature is only supported on Wrist Camera!")

        cv2.namedWindow("AI + SLAM Collector")
        cv2.setMouseCallback("AI + SLAM Collector", mouse_callback)

        gui_state = {"image": None, "key": 255, "running": True}

        def continuous_ai_loop():
            print("Starting real-time AI thread...")
            while ai_data["is_running"]:
                try:
                    if not robot_moving["state"] and ai_data["needs_predict"]:
                        ai_data["needs_predict"] = False
                        img = camera1.get_images()
                        current_pose = indy.get_robot_pose()
                        depth_data, depth_scale = camera1.get_depth_data()

                        cx, cy, angle, mask, contours = ai_model.predict(img)

                        with ai_data["lock"]:
                            if cx is not None:
                                h, w = depth_data.shape
                                if mask is not None:
                                    valid_mask = (mask > 0) & (depth_data > 0)
                                    valid_depths = depth_data[valid_mask]
                                    if len(valid_depths) > 0:
                                        z_mm = float(np.median(valid_depths)) * depth_scale * 1000.0
                                    else:
                                        z_mm = 300.0
                                else:
                                    px = int(np.clip(cx, 0, w - 1))
                                    py = int(np.clip(cy, 0, h - 1))
                                    z_mm = depth_data[py, px] * depth_scale * 1000.0
                                    if z_mm <= 0.01:
                                        roi = depth_data[max(0, py - 5):min(h, py + 5), max(0, px - 5):min(w, px + 5)]
                                        valid_depths = roi[roi > 0]
                                        z_mm = np.median(valid_depths) * depth_scale * 1000.0 if len(valid_depths) > 0 else 300.0

                                pick_pose = ai_model.get_pick(cx, cy, angle, current_pose, depth_z=z_mm)
                                ai_data["mask"] = mask
                                ai_data["contours"] = contours
                                ai_data["pose"] = pick_pose
                            else:
                                ai_data["mask"] = None
                                ai_data["contours"] = None
                                ai_data["pose"] = None
                    elif robot_moving["state"]:
                        with ai_data["lock"]:
                            ai_data["mask"] = None
                            ai_data["contours"] = None
                            ai_data["pose"] = None
                        time.sleep(0.5)
                    else:
                        time.sleep(0.05)
                except Exception as e:
                    print(f"[Background AI Thread] Error occurred: {e}")
                    time.sleep(0.5)

        ai_thread = threading.Thread(target=continuous_ai_loop, daemon=True)
        ai_thread.start()

        def command_robot_to_pick():
            with ai_data["lock"]:
                target_pose = ai_data["pose"]

            if target_pose is not None:
                print(f"Commanding robot to move to pose: {target_pose}")
                indy.pick_point = np.array(target_pose) + np.array([12.5, 0, -30, 0, 0, 0])
                print("Pick point has been set!", indy.pick_point)
                #indy.pick_point[3:5] = [0, -179.5]
                try:
                    if robot_moving["abort"]:
                        return
                    indy.move_to_pick()
                    susgrip.open()
                    for i in range(300):
                        if robot_moving["abort"]: return
                        time.sleep(0.01)
                    print("Reached Pick position!")
                    susgrip.close()
                    indy.calculate_points()
                    start_time = time.perf_counter()
                    indy.move_to_place()
                    for i in range(700):
                        if robot_moving["abort"]: return
                        time.sleep(0.01)
                    print(f"Pick to Place time: {time.perf_counter() - start_time}")
                    susgrip.open()
                    time.sleep(0.25)
                    print("Reached Place position!")
                    indy.movel(indy.home, vel_ratio=20, acc_ratio=25)
                    if robot_moving["abort"]: return
                    print("Returned to Home!")
                except Exception as e:
                    print(f"Movement error: {e}")
                finally:
                    robot_moving["state"] = False
            else:
                print("Object not recognized, cannot move!")
                robot_moving["state"] = False

        def real_time_collection_loop():
            nonlocal recording, current_gripper_cmd
            dt = 1.0 / FPS
            frame_count = 0
            fps_real = 0.0
            prev_loop_time = time.perf_counter()
            
            while gui_state["running"]:
                loop_start = time.perf_counter()
                frame_count += 1
                
                # Fetch Data
                q_deg = indy.get_robot_data()
                p_p_tcp = indy.get_robot_pose()
                g_state = susgrip.get_gripper_state()
                wrist_bgr = camera1.get_images()
                depth_data, depth_scale = camera1.get_depth_data()

                # Visuals
                wrist_vis = wrist_bgr.copy()
                status_text = "RECORDING" if recording else "IDLE"
                cv2.putText(wrist_vis, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if recording else (0, 180, 255), 2)
                cv2.putText(wrist_vis, f"FPS: {fps_real:.1f}", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                with ai_data["lock"]:
                    current_contours = ai_data["contours"]
                    current_pose = ai_data["pose"]
                if current_contours is not None:
                    cv2.drawContours(wrist_vis, current_contours, -1, (0, 255, 0), 2)
                if current_pose is not None:
                    pose_text = (f"Pick: X:{current_pose[0]:.1f} Y:{current_pose[1]:.1f} "
                                 f"Z:{current_pose[2]:.1f} Rx:{current_pose[3]:.1f} "
                                 f"Ry:{current_pose[4]:.1f} Rz:{current_pose[5]:.1f}")
                    cv2.putText(wrist_vis, pose_text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                curr_loop_time = time.perf_counter()
                inst_fps = 1.0 / (curr_loop_time - prev_loop_time + 1e-6)
                prev_loop_time = curr_loop_time
                if not hasattr(mouse_callback, "ema_fps"): mouse_callback.ema_fps = inst_fps
                mouse_callback.ema_fps = 0.1 * inst_fps + 0.9 * mouse_callback.ema_fps
                fps_real = mouse_callback.ema_fps

                gui_state["image"] = wrist_vis
                
                key = gui_state["key"]
                gui_state["key"] = 255
                
                if key == ord("q"):
                    gui_state["running"] = False
                    break
                elif key == ord("r"):
                    if robot_moving["abort"]:
                        print("WARNING: Robot is LOCKED (Abort). Press 'd' to unlock before continuing!")
                    elif not robot_moving["state"]:
                        robot_moving["state"] = True
                        threading.Thread(target=command_robot_to_pick, daemon=True).start()
                elif key == ord("s"):
                    print("EMERGENCY STOP! (Abort)")
                    robot_moving["abort"] = True
                    try: indy.stop_motion()
                    except: pass
                elif key == ord("d"):
                    robot_moving["abort"] = False
                    print("Unlocked (Reset Abort). AI execution can resume.")
                    indy.robot.movel(indy.home, vel_ratio=35, acc_ratio=60)
                elif key == ord("o"):
                    susgrip.open()
                    current_gripper_cmd = 1.0
                elif key == ord("c"):
                    susgrip.close()
                    current_gripper_cmd = 0.0
                
                elif key == ord("["):
                    if not recording:
                        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        current_dataset_dir = os.path.join(OUTPUT_DIR, f"dataset_tum_{now_str}")
                        print(f"\n>>> BẮT ĐẦU GHI DATASET: {current_dataset_dir} <<<")
                        os.makedirs(f"{current_dataset_dir}/rgb", exist_ok=True)
                        os.makedirs(f"{current_dataset_dir}/depth", exist_ok=True)
                        
                        slam_state["dir"] = current_dataset_dir
                        slam_state["rgb_f"] = open(f"{current_dataset_dir}/rgb.txt", "w")
                        slam_state["rgb_f"].write("# timestamp filename\n")
                        slam_state["depth_f"] = open(f"{current_dataset_dir}/depth.txt", "w")
                        slam_state["depth_f"].write("# timestamp filename\n")
                        slam_state["assoc_f"] = open(f"{current_dataset_dir}/associations.txt", "w")
                        
                        slam_state["gripper_f"] = open(f"{current_dataset_dir}/gripper_log.csv", "w", newline='')
                        slam_state["gripper_w"] = csv.writer(slam_state["gripper_f"])
                        slam_state["gripper_w"].writerow(["timestamp", "gripper_state", "gripper_cmd"])
                        
                        slam_state["robot_f"] = open(f"{current_dataset_dir}/robot_log.csv", "w", newline='')
                        slam_state["robot_w"] = csv.writer(slam_state["robot_f"])
                        slam_state["robot_w"].writerow(["timestamp", "q1", "q2", "q3", "q4", "q5", "q6", "x", "y", "z", "u", "v", "w"])
                        
                        slam_state["writer_thread"] = threading.Thread(target=slam_image_writer)
                        slam_state["writer_thread"].start()
                        recording = True
                        
                elif key == ord("]"):
                    if recording:
                        recording = False
                        print(f"\n>>> LƯU DATASET: {slam_state['dir']} <<<")
                        slam_img_queue.put(None)
                        if slam_state["writer_thread"]: slam_state["writer_thread"].join()
                        if slam_state["rgb_f"]: slam_state["rgb_f"].close()
                        if slam_state["depth_f"]: slam_state["depth_f"].close()
                        if slam_state["assoc_f"]: slam_state["assoc_f"].close()
                        if slam_state["gripper_f"]: slam_state["gripper_f"].close()
                        if slam_state["robot_f"]: slam_state["robot_f"].close()
                        
                elif key == ord("\\"):
                    if recording:
                        recording = False
                        print(f"\n>>> HỦY DATASET: {slam_state['dir']} <<<")
                        slam_img_queue.put(None)
                        if slam_state["writer_thread"]: slam_state["writer_thread"].join()
                        if slam_state["rgb_f"]: slam_state["rgb_f"].close()
                        if slam_state["depth_f"]: slam_state["depth_f"].close()
                        if slam_state["assoc_f"]: slam_state["assoc_f"].close()
                        if slam_state["gripper_f"]: slam_state["gripper_f"].close()
                        if slam_state["robot_f"]: slam_state["robot_f"].close()
                        import shutil
                        shutil.rmtree(slam_state["dir"], ignore_errors=True)

                if recording:
                    print(f"Recording SLAM Data... Loop FPS: {fps_real:.1f}   ", end='\r')
                    ts_sec = time.time()
                    ts_str = f"{ts_sec:.6f}"
                    
                    slam_state["rgb_f"].write(f"{ts_str} rgb/{ts_str}.png\n")
                    slam_state["depth_f"].write(f"{ts_str} depth/{ts_str}.png\n")
                    slam_state["assoc_f"].write(f"{ts_str} rgb/{ts_str}.png {ts_str} depth/{ts_str}.png\n")
                    
                    slam_state["gripper_w"].writerow([ts_sec, float(g_state), float(current_gripper_cmd)])
                    slam_state["robot_w"].writerow([ts_sec] + list(q_deg) + list(p_p_tcp))
                    
                    # Fix rgb to rgb conversion (wrist_bgr is BGR but our writer expects RGB so we convert or just send it directly if we modify writer)
                    # wait, wrist_bgr is BGR. slam_image_writer currently expects RGB because it does cvtColor(rgb, COLOR_RGB2BGR).
                    # I'll just change slam_image_writer to take BGR directly!
                    slam_img_queue.put((ts_str, wrist_bgr, depth_data, slam_state["dir"]))

                elapsed = time.perf_counter() - loop_start
                if elapsed < dt:
                    sleep_time = dt - elapsed - 0.002
                    if sleep_time > 0: time.sleep(sleep_time)
                    while time.perf_counter() - loop_start < dt: pass

        collection_thread = threading.Thread(target=real_time_collection_loop, daemon=True)
        collection_thread.start()

        cv2.namedWindow("AI + SLAM Collector", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AI + SLAM Collector", 960, 540)
        while gui_state["running"]:
            if gui_state["image"] is not None:
                cv2.imshow("AI + SLAM Collector", gui_state["image"])
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                gui_state["key"] = key
            if key == ord('q'):
                gui_state["running"] = False
                break

    except KeyboardInterrupt:
        gui_state["running"] = False
        print("\nDataset collection interrupted by user.")
    finally:
        ai_data["is_running"] = False
        if recording:
            slam_img_queue.put(None)
            if slam_state["writer_thread"]: slam_state["writer_thread"].join()
        if susgrip is not None:
            try: susgrip.stop()
            except: pass
        if indy is not None:
            try: indy.stop()
            except: pass
        if camera1 is not None:
            try: camera1.stop()
            except: pass
        cv2.destroyAllWindows()
        print("Hardware handles released.")

if __name__ == "__main__":
    main()
