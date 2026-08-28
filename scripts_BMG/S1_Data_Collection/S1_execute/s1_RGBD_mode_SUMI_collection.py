'''
Using SUMI to collect data
Two cameras are used: an Intel RealSense D435i captures RGB-D data for SLAM at 60 FPS,
and a webcam records 20 FPS RGB data for training a diffusion model.
No robot or AI logic is included.
'''
import os
import time
import threading
import queue
import datetime
import csv
import glob
import shutil
import numpy as np
import cv2
import pyrealsense2 as rs
from pymodbus.client import ModbusSerialClient

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# S1_output Directory
OUTPUT_DIR = os.path.join(BASE_DIR, "S1_output")

# Camera Setup
H, W = 540, 960
FPS = 60 
FPS_DATA_TRAINING = 10
CAMERA_SERIAL = "317222074902"

# Gripper Calibration
GRIPPER_OPEN_MM = 90.0
GRIPPER_CLOSE_MM = 10.0

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
            print(f"[Warning] {self.name} is too slow! Elapsed: {elapsed:.4f}s")
        self.last_time = time.perf_counter()

class RealsenseCamera:
    def __init__(self, serial_number, use_depth=True):
        self.serial_number = serial_number
        self.use_depth = use_depth
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(self.serial_number)
        
        # Color: 960x540 @ 60FPS
        self.config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 60)

        if self.use_depth:
            # Depth: 848x480 @ 60FPS
            self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
            self.align = rs.align(rs.stream.color)
        else:
            self.align = None

        profile = self.pipeline.start(self.config)

        if self.use_depth:
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
        else:
            self.depth_scale = 0.001

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

        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        time.sleep(1.0)

    def _poll(self):
        rate = Rate(59, name="realsense")
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

def find_webcam_id():
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

class WebcamStream:
    def __init__(self, target_fps):
        self.cam_id = find_webcam_id()
        self.stream = None
        self.frame = None
        self.frame_id = 0
        self.grabbed = False
        self.stopped = False
        self.lock = threading.Lock()
        
        if self.cam_id is not None:
            self.stream = cv2.VideoCapture(self.cam_id)
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.stream.set(cv2.CAP_PROP_FPS, target_fps)
            (self.grabbed, self.frame) = self.stream.read()
            self.start()

    def start(self):
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        return self
        
    def update(self):
        rate = Rate(12, "webcam")
        while not self.stopped and self.stream is not None:
            if not self.grabbed:
                self.stop()
            else:
                (grabbed, frame) = self.stream.read()
                with self.lock:
                    self.grabbed = grabbed
                    if grabbed:
                        self.frame = frame
                        self.frame_id += 1
            rate.sleep()

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy(), self.frame_id
            return None, -1
            
    def stop(self):
        self.stopped = True
        try:
            if hasattr(self, 'thread'):
                self.thread.join(timeout=0.2)
        except Exception:
            pass
        if self.stream is not None:
            self.stream.release()

class SusGrip:
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
        self.current_state = 1.0  # Normalized [0.0 - 1.0]
        self._lock = threading.Lock()
        self._cmd_queue = queue.Queue()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            rate = Rate(30, "gripper")
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
                try:
                    res = self.client.read_input_registers(address=1, count=1, device_id=1)
                    if not res.isError():
                        pos_mm = float(res.registers[0])
                        norm = (pos_mm - GRIPPER_CLOSE_MM) / (GRIPPER_OPEN_MM - GRIPPER_CLOSE_MM)
                        norm_clipped = float(np.clip(norm, 0.0, 1.0))
                        with self._lock:
                            self.current_state = norm_clipped
                except Exception:
                    pass
            rate.sleep()

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

# ============================================================
# BACKGROUND DATA WRITER
# ============================================================
data_queue = queue.Queue()
def data_writer_thread():
    while True:
        try:
            task = data_queue.get(timeout=0.1)
            if task is None:
                break
            
            ts_str, rgb, depth, web_rgb, save_dir = task
            cv2.imwrite(f"{save_dir}/rgb/{ts_str}.png", rgb)
            cv2.imwrite(f"{save_dir}/depth/{ts_str}.png", depth)
            if web_rgb is not None:
                cv2.imwrite(f"{save_dir}/web_rgb/{ts_str}.png", web_rgb)
        except queue.Empty:
            pass

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    gui_state = {"running": True, "key": 255}
    current_gripper_cmd = 1.0
    recording = False
    
    # State tracking
    dataset_state = {
        "dir": None, "rgb_f": None, "depth_f": None, "web_f": None, "assoc_f": None,
        "gripper_f": None, "gripper_w": None, "writer_thread": None,
        "start_time": 0.0, "frame_count_rs": 0, "frame_count_web": 0,
        "last_web_frame_id": -1, "last_training_save_time": 0.0
    }
    
    susgrip = None
    camera_rs = None
    camera_web = None

    try:
        print("Connecting to SusGrip gripper...")
        susgrip = SusGrip()
        print("Initializing RealSense (RGB-D)...")
        camera_rs = RealsenseCamera(CAMERA_SERIAL, use_depth=True)
        print("Initializing Webcam...")
        camera_web = WebcamStream(target_fps=28)

        print("\n" + "=" * 70)
        print("SUMI Data Collector Initialized")
        print("=" * 70)
        print("Use keyboard in the OpenCV window for control:")
        print(f"  o  -> gripper OPEN ({GRIPPER_OPEN_MM:.1f} mm)")
        print(f"  c  -> gripper CLOSE ({GRIPPER_CLOSE_MM:.1f} mm)")
        print("  [  -> START recording Dataset")
        print("  ]  -> STOP and SAVE Dataset")
        print("  \\  -> DISCARD current recording")
        print("  q  -> QUIT")
        print("=" * 70)

        # UI Window
        cv2.namedWindow("SUMI Data Collector", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("SUMI Data Collector", 960, 540)
        
        dt = 1.0 / FPS
        fps_real = 0.0
        ema_fps = FPS
        prev_loop_time = time.perf_counter()
        
        while gui_state["running"]:
            loop_start = time.perf_counter()
            
            # Fetch Data
            g_state = susgrip.get_gripper_state()
            rs_bgr = camera_rs.get_images()
            depth_data, _ = camera_rs.get_depth_data()
            web_bgr, web_frame_id = camera_web.read()

            # Visuals
            vis = rs_bgr.copy()
            status_text = "RECORDING" if recording else "IDLE"
            color = (0, 0, 255) if recording else (0, 180, 255)
            cv2.putText(vis, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(vis, f"FPS: {fps_real:.1f}", (800, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show small web camera PIP
            if web_bgr is not None:
                web_h, web_w = web_bgr.shape[:2]
                scale = 0.3
                small_web = cv2.resize(web_bgr, (int(web_w*scale), int(web_h*scale)))
                sw_h, sw_w = small_web.shape[:2]
                vis[10:10+sw_h, 960-sw_w-10:960-10] = small_web
                cv2.rectangle(vis, (960-sw_w-10, 10), (960-10, 10+sw_h), (255,255,255), 1)

            cv2.imshow("SUMI Data Collector", vis)
            key = cv2.waitKey(1) & 0xFF
            
            # Key Handling
            if key == ord("q"):
                gui_state["running"] = False
                break
            elif key == ord("o"):
                susgrip.open()
                current_gripper_cmd = 1.0
            elif key == ord("c"):
                susgrip.close()
                current_gripper_cmd = 0.0
                
            elif key == ord("["):
                if not recording:
                    now = datetime.datetime.now()
                    date_folder = now.strftime("Date_%d%m%Y")
                    time_folder = now.strftime("%H%M%S")
                    
                    full_date_dir = os.path.join(OUTPUT_DIR, date_folder)
                    os.makedirs(full_date_dir, exist_ok=True)
                    
                    current_dataset_dir = os.path.join(full_date_dir, f"dataset_{time_folder}")
                    print(f"\n>>> BẮT ĐẦU GHI DATASET: {current_dataset_dir} <<<")
                    
                    os.makedirs(f"{current_dataset_dir}/rgb", exist_ok=True)
                    os.makedirs(f"{current_dataset_dir}/depth", exist_ok=True)
                    os.makedirs(f"{current_dataset_dir}/web_rgb", exist_ok=True)
                    
                    dataset_state["dir"] = current_dataset_dir
                    
                    dataset_state["rgb_f"] = open(f"{current_dataset_dir}/rgb.txt", "w")
                    dataset_state["rgb_f"].write("# timestamp filename\n")
                    
                    dataset_state["depth_f"] = open(f"{current_dataset_dir}/depth.txt", "w")
                    dataset_state["depth_f"].write("# timestamp filename\n")
                    
                    dataset_state["web_f"] = open(f"{current_dataset_dir}/web_rgb.txt", "w")
                    dataset_state["web_f"].write("# timestamp filename\n")
                    
                    dataset_state["assoc_f"] = open(f"{current_dataset_dir}/associations.txt", "w")
                    
                    dataset_state["gripper_f"] = open(f"{current_dataset_dir}/gripper_log.csv", "w", newline='')
                    dataset_state["gripper_w"] = csv.writer(dataset_state["gripper_f"])
                    dataset_state["gripper_w"].writerow(["timestamp", "gripper_state", "gripper_cmd"])
                    
                    dataset_state["start_time"] = time.time()
                    dataset_state["frame_count_rs"] = 0
                    dataset_state["frame_count_web"] = 0
                    dataset_state["last_web_frame_id"] = -1
                    dataset_state["last_training_save_time"] = time.time() - (1.0 / FPS_DATA_TRAINING)
                    
                    dataset_state["writer_thread"] = threading.Thread(target=data_writer_thread)
                    dataset_state["writer_thread"].start()
                    recording = True
                    
            elif key == ord("]"):
                if recording:
                    recording = False
                    
                    # Tính toán thông tin Record
                    duration = time.time() - dataset_state["start_time"]
                    rs_fps_actual = dataset_state["frame_count_rs"] / duration if duration > 0 else 0
                    web_fps_actual = dataset_state["frame_count_web"] / duration if duration > 0 else 0
                    
                    info_path = os.path.join(dataset_state["dir"], "info.txt")
                    with open(info_path, "w") as f_info:
                        f_info.write(f"Record Duration: {duration:.2f} seconds\n")
                        f_info.write(f"Target FPS (Realsense RGB-D): {FPS}\n")
                        f_info.write(f"Total Frames (Realsense): {dataset_state['frame_count_rs']}\n")
                        f_info.write(f"Actual FPS (Realsense): {rs_fps_actual:.2f}\n")
                        f_info.write(f"----------------------------------\n")
                        f_info.write(f"Target FPS (Webcam & Gripper): {FPS_DATA_TRAINING}\n")
                        f_info.write(f"Total Frames (Webcam & Gripper): {dataset_state['frame_count_web']}\n")
                        f_info.write(f"Actual FPS (Webcam & Gripper): {web_fps_actual:.2f}\n")
                    
                    print(f"\n>>> LƯU DATASET THÀNH CÔNG: {dataset_state['dir']} <<<")
                    data_queue.put(None)
                    if dataset_state["writer_thread"]: dataset_state["writer_thread"].join()
                    if dataset_state["rgb_f"]: dataset_state["rgb_f"].close()
                    if dataset_state["depth_f"]: dataset_state["depth_f"].close()
                    if dataset_state["web_f"]: dataset_state["web_f"].close()
                    if dataset_state["assoc_f"]: dataset_state["assoc_f"].close()
                    if dataset_state["gripper_f"]: dataset_state["gripper_f"].close()
                    
            elif key == ord("\\"):
                if recording:
                    recording = False
                    print(f"\n>>> HỦY DATASET: {dataset_state['dir']} <<<")
                    data_queue.put(None)
                    if dataset_state["writer_thread"]: dataset_state["writer_thread"].join()
                    if dataset_state["rgb_f"]: dataset_state["rgb_f"].close()
                    if dataset_state["depth_f"]: dataset_state["depth_f"].close()
                    if dataset_state["web_f"]: dataset_state["web_f"].close()
                    if dataset_state["assoc_f"]: dataset_state["assoc_f"].close()
                    if dataset_state["gripper_f"]: dataset_state["gripper_f"].close()
                    shutil.rmtree(dataset_state["dir"], ignore_errors=True)

            if recording:
                ts_sec = time.time()
                ts_str = f"{ts_sec:.6f}"
                
                # Dữ liệu Realsense lưu ở tốc độ vòng lặp chính
                dataset_state["rgb_f"].write(f"{ts_str} rgb/{ts_str}.png\n")
                dataset_state["depth_f"].write(f"{ts_str} depth/{ts_str}.png\n")
                dataset_state["assoc_f"].write(f"{ts_str} rgb/{ts_str}.png {ts_str} depth/{ts_str}.png\n")
                dataset_state["frame_count_rs"] += 1
                
                # Dữ liệu Training (Webcam & Gripper) lưu đồng bộ ở FPS_DATA_TRAINING
                save_web_bgr = None
                dt_training = 1.0 / FPS_DATA_TRAINING
                if ts_sec - dataset_state["last_training_save_time"] >= dt_training:
                    # Cập nhật bằng += dt_training thay vì = ts_sec để KHÔNG bị cộng dồn sai số thời gian của vòng lặp
                    dataset_state["last_training_save_time"] += dt_training
                    
                    if web_bgr is not None:
                        dataset_state["web_f"].write(f"{ts_str} web_rgb/{ts_str}.png\n")
                        dataset_state["frame_count_web"] += 1
                        save_web_bgr = web_bgr
                    
                    dataset_state["gripper_w"].writerow([ts_sec, float(g_state), float(current_gripper_cmd)])
                
                data_queue.put((ts_str, rs_bgr, depth_data, save_web_bgr, dataset_state["dir"]))

            # Calculate FPS
            curr_loop_time = time.perf_counter()
            inst_fps = 1.0 / (curr_loop_time - prev_loop_time + 1e-6)
            prev_loop_time = curr_loop_time
            ema_fps = 0.1 * inst_fps + 0.9 * ema_fps
            fps_real = ema_fps

            # Maintain 60Hz loop
            elapsed = time.perf_counter() - loop_start
            if elapsed < dt:
                sleep_time = dt - elapsed - 0.002
                if sleep_time > 0: time.sleep(sleep_time)
                while time.perf_counter() - loop_start < dt: pass

    except KeyboardInterrupt:
        print("\nDataset collection interrupted by user.")
    finally:
        gui_state["running"] = False
        if recording:
            data_queue.put(None)
            if dataset_state["writer_thread"]: dataset_state["writer_thread"].join()
        if susgrip is not None:
            try: susgrip.stop()
            except: pass
        if camera_rs is not None:
            try: camera_rs.stop()
            except: pass
        if camera_web is not None:
            try: camera_web.stop()
            except: pass
        cv2.destroyAllWindows()
        print("Hardware handles released.")

if __name__ == "__main__":
    main()
