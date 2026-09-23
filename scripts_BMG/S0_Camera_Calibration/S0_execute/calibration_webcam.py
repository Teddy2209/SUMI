"""
Calibration Node (ROS 2) - Eye-in-Hand
=======================================
Cải tiến độ chính xác:
1. Hỗ trợ nhiều phương pháp giải AX=XB: TSAI, PARK, HORAUD, ANDREFF, DANIILIDIS.
2. Tự động tính toán và so sánh sai số của tất cả các phương pháp.+
3. Chế độ Test/Verify: Click điểm trong VisionNode để xem toạ độ Robot Base tức thời.
4. Tương thích hoàn toàn với vision_node (đã fix unit mismatch).
Đơn vị: Toàn bộ dùng METERS (m) và DEGREES (deg).
"""
import time
import os, sys, json, math, threading, glob
import numpy as np
import cv2
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
import tf2_ros
from geometry_msgs.msg import TransformStamped, Point
from std_msgs.msg import String
from neuromeka import IndyDCP3

# ============================================================
# CẤU HÌNH
# ============================================================
DATA_DIR       = "/home/apicoo-ai/pmg/Training_Test_vision/data_calib_intel_realsense_flange_Louis"
RESULT_FILE    = os.path.join(DATA_DIR, "eye_in_hand_result.json")
INTRINSIC_FILE = os.path.join(DATA_DIR, "camera_intrinsics.json")
BOARD_SIZE     = (11, 8)
SQUARE_SIZE_MM = 10.0
ROBOT_IP       = "192.168.2.100"

# ============================================================
# UTILS
# ============================================================

def print_green(txt): print(f"\033[92m{txt}\033[0m")
def print_red(txt): print(f"\033[91m{txt}\033[0m")

def find_webcam_id():
    """
    Tìm ID của USB Webcam bằng cách loại trừ các camera có tên 'RealSense'.
    Duyệt qua /sys/class/video4linux/video* để đọc tên thiết bị.
    """
    video_paths = glob.glob('/sys/class/video4linux/video*')
    # Sắp xếp để ưu tiên ID nhỏ
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
            
    print("[ERROR] Không tìm thấy Webcam ngoài! Vui lòng kiểm tra cáp cắm.")
    return None

def _mat(lst): return np.array(lst)
def _lst(mat): return mat.tolist()

def euler_from_matrix(R):
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy >= 1e-6:
        x = math.atan2(R[2,1], R[2,2]); y = math.atan2(-R[2,0], sy); z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1]); y = math.atan2(-R[2,0], sy); z = 0.0
    return np.array([x, y, z])

def quaternion_from_euler(ai, aj, ak):
    ai /= 2.0; aj /= 2.0; ak /= 2.0
    ci = math.cos(ai); si = math.sin(ai); cj = math.cos(aj); sj = math.sin(aj)
    ck = math.cos(ak); sk = math.sin(ak)
    cc = ci*ck; cs = ci*sk; sc = si*ck; ss = si*sk
    return np.array([cj*sc-sj*cs, cj*ss+sj*cc, cj*cs-sj*sc, cj*cc+sj*ss])

def _euler_zyx_to_rotmat(rx_d, ry_d, rz_d):
    Rx, Ry, Rz = np.deg2rad(rx_d), np.deg2rad(ry_d), np.deg2rad(rz_d)
    Rmx = np.array([[1,0,0],[0,np.cos(Rx),-np.sin(Rx)],[0,np.sin(Rx),np.cos(Rx)]])
    Rmy = np.array([[np.cos(Ry),0,np.sin(Ry)],[0,1,0],[-np.sin(Ry),0,np.cos(Ry)]])
    Rmz = np.array([[np.cos(Rz),-np.sin(Rz),0],[np.sin(Rz),np.cos(Rz),0],[0,0,1]])
    return Rmz @ Rmy @ Rmx

def _p_to_T(p):
    """[X,Y,Z,Rx,Ry,Rz] mm/deg -> 4x4 homogeneous matrix (m)."""
    p_val = np.array(p).flatten()
    T = np.eye(4)
    T[:3, :3] = _euler_zyx_to_rotmat(p_val[3], p_val[4], p_val[5])
    T[:3,  3] = p_val[:3] / 1000.0
    return T

# ============================================================
# ROS 2 NODE
# ============================================================

# class CalibrationNode(Node):
#     def __init__(self):
#         super().__init__('calibration_node')
#         self.T_cam2tool = None
#         self.robot      = None
#         self.tf_bc      = tf2_ros.StaticTransformBroadcaster(self)

#         try:
#             self.robot = IndyDCP3(robot_ip=ROBOT_IP, index=0)
#             self.get_logger().info(f"[OK] Robot: {ROBOT_IP}")
#         except Exception as e:
#             self.get_logger().warn(f"Robot link failed: {e}")

#         self.coord_pub = self.create_publisher(String, '/pos_robot3d', 10)
#         self.create_subscription(String, '/pos_cam3d', self._on_point, 10)
#         self._load_and_broadcast()

#     def _load_and_broadcast(self):
#         if not os.path.exists(RESULT_FILE): return
#         try:
#             with open(RESULT_FILE) as f:
#                 self.T_cam2tool = _mat(json.load(f)["T_cam_to_tool"])
#             T = self.T_cam2tool
#             t = TransformStamped()
#             t.header.stamp = self.get_clock().now().to_msg()
#             t.header.frame_id, t.child_frame_id = 'robot_tool_frame', 'camera_frame'
#             t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = T[0,3], T[1,3], T[2,3]
#             q = quaternion_from_euler(*euler_from_matrix(T[:3,:3]))
#             t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q[0], q[1], q[2], q[3]
#             self.tf_bc.sendTransform(t)
#             self.get_logger().info(f"Loaded Hand-Eye matrix from {RESULT_FILE}")
#         except: pass

#     def _on_point(self, msg: String):
#         if self.T_cam2tool is None: 
#             self.get_logger().warn("Result file not loaded (Option 3 first?)"); return

#         try:
#             parts = [float(v) for v in msg.data.split(',')]
#             # 6 tham số từ Camera:
#             cx, cy, cz = parts[0], parts[1], parts[2]
#             crx, cry, crz = parts[3], parts[4], parts[5]
#         except Exception as e:
#             self.get_logger().error(f"Message format error: {e}")
#             return

#         # 1. Từ Pose 6D của Camera tạo ra ma trận T_cam_grasp
#         T_cam_grasp = np.eye(4)
#         T_cam_grasp[:3, :3] = _euler_zyx_to_rotmat(crx, cry, crz)
#         T_cam_grasp[:3, 3] = [cx, cy, cz]

#         # 2. Lấy vị trí Mặt bích Robot (Flange) hiện tại
#         T_base_tool_curr = None
#         if self.robot:
#             try: T_base_tool_curr = _p_to_T(self.robot.get_control_data()['p'])
#             except: pass

#         if T_base_tool_curr is not None:
#             # 3. Tính Toạ độ Đích (Grasp Pose) trên hệ Base
#             # T_base_grasp = T_base_tool_curr @ T_tool_cam @ T_cam_grasp
#             # Chú ý: self.T_cam2tool chính là T_tool_cam (mảng kết quả trả về của calibrateHandEye)
#             T_base_grasp = T_base_tool_curr @ self.T_cam2tool @ T_cam_grasp
            
#             # 4. Bù trừ góc RZ của tay kẹp cơ khí (Mech Offset)
#             # Theo người dùng, càng kẹp gắn lệch 57 độ so với hệ mặt bích chuẩn
#             YAW_OFFSET_DEG = 57.0 
#             Rad = math.radians(YAW_OFFSET_DEG)
#             Rz_mech = np.array([
#                 [math.cos(Rad), -math.sin(Rad), 0, 0],
#                 [math.sin(Rad),  math.cos(Rad), 0, 0],
#                 [0,              0,             1, 0],
#                 [0,              0,             0, 1]
#             ])
            
#             T_base_target = T_base_grasp 
#             # Rút ra Tịnh Tiến và Góc U, V, W trên hệ Base
#             P_base = T_base_target[:3, 3]
#             base_rpy = euler_from_matrix(T_base_target[:3, :3])
#             b_rx, b_ry, b_rz = np.degrees(base_rpy)
#             frame = "base"
#         else:
#             # Nếu ko liên kết được Robot, tính tương đối
#             P_base = (self.T_cam2tool @ np.array([cx, cy, cz, 1.0]))[:3]
#             b_rx, b_ry, b_rz = 180.0, 0.0, crz
#             frame = "tool (no FK)"

#         # 5. Chuẩn hoá dữ liệu xuất ra (milimet, W wrap 0-180)
#         P_base_mm = P_base * 1000.0 + [0, -2.5, 7.5] # Offset tinh chỉnh tuỳ ý
#         b_rz_conty = b_rz % 180.0

#         res_str = f"x={P_base_mm[0]:.2f} y={P_base_mm[1]:.2f} z={P_base_mm[2]:.2f} U={b_rx:.1f} V={b_ry:.1f} W={b_rz_conty:.1f} frame:{frame}"
#         self.coord_pub.publish(String(data=res_str))
#         self.get_logger().info(f"CALC TO ROBOT: {res_str}")

#     def destroy_node(self):
#         super().destroy_node()

# ============================================================
# TAKE DATA
# ============================================================

def take_data():
    os.makedirs(os.path.join(DATA_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "poses"),  exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "poses"),  exist_ok=True)
    robot = None
    try: robot = IndyDCP3(robot_ip=ROBOT_IP, index=0)
    except: pass

    cam_id = find_webcam_id()
    if cam_id is None:
        return
        
    #cap = cv2.VideoCapture(cam_id)
    #if not cap.isOpened():
        #print(f"Lỗi: Không thể mở camera tại /dev/video{cam_id}")
        #return
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2) 
    
    # Ép sử dụng định dạng Raw/YUYV (không nén) để đảm bảo chất lượng pixel tốt nhất
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)



    count = sum(1 for f in os.listdir(os.path.join(DATA_DIR,"images")) if f.endswith(".png"))
    print(f"\n[DATA] Next: #{count}. Keys: S=Capture, Q=Quit")

    try:
        while True:
            start= time.time()
            ret_cam, img = cap.read()
            if not ret_cam: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)
            disp = img.copy()
            if ret:
                cv2.drawChessboardCorners(disp, BOARD_SIZE, corners, ret)
                txt, clr = f"OK #{count}", (0,255,0)
            else:
                txt, clr = "BOARD NOT FOUND", (0,0,255)
            cv2.putText(disp, txt, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, clr, 2)
            cv2.imshow("Calibration Capture", disp)
            end= time.time()
            print(f" FPS: {1/(end-start)}")
            k = cv2.waitKey(1) & 0xFF
            if k == ord('s'):
                if not ret: print("Error: No board found."); continue
                cv2.imwrite(os.path.join(DATA_DIR, "images", f"img_{count:03d}.png"), img)
                if robot:
                    pose = _p_to_T(robot.get_control_data()['p'])
                    with open(os.path.join(DATA_DIR, "poses", f"pose_{count:03d}.json"), 'w') as jf:
                        json.dump({"T_base_tool": _lst(pose)}, jf)
                print(f"Captured #{count}"); count += 1
            elif k == ord('q'): break
    finally:
        cap.release(); cv2.destroyAllWindows()

# ============================================================
# CALIBRATION LOGIC
# ============================================================

def intrinsic_calibration():
    img_dir = os.path.join(DATA_DIR, "images")
    if not os.path.exists(img_dir):
        print(f"Lỗi: Không tìm thấy thư mục {img_dir}.")
        return

    imgs = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    if not imgs:
        print("Không có ảnh nào để calib.")
        return

    print(f"\nBắt đầu Intrinsic Calibration với {len(imgs)} ảnh...")
    
    objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
    # LƯU Ý: Ở intrinsic mình có thể dùng mm hoặc m đều được, nhưng dùng mm để tiện vì không liên quan đến pose robot.
    objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE_MM
    
    objpoints = []
    imgpoints = []
    img_shape = None

    for f in imgs:
        img_path = os.path.join(img_dir, f)
        img = cv2.imread(img_path)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]
            
        ret, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                      (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
            
    if not objpoints:
        print("Không tìm thấy bàn cờ trong các ảnh.")
        return

    print(f"Số ảnh hợp lệ được dùng để giải K, D: {len(objpoints)} / {len(imgs)}")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)
    
    print(f"Intrinsic Calibration RMSE Error: {ret:.4f} pixels")
    
    intrinsics_data = {
        "image_size": [img_shape[0], img_shape[1]],
        "camera_matrix": _lst(mtx),
        "dist_coeffs": _lst(dist)
    }
    
    with open(INTRINSIC_FILE, 'w') as f:
        json.dump(intrinsics_data, f, indent=4)
        
    print(f"Đã lưu Intrinsic Calibration (K, D) mới vào {INTRINSIC_FILE}")

def solve_hand_eye():
    if os.path.exists(INTRINSIC_FILE):
        with open(INTRINSIC_FILE) as f: d = json.load(f)
        K, D = _mat(d["camera_matrix"]), _mat(d["dist_coeffs"])
        print(f"Loaded custom intrinsics from {INTRINSIC_FILE}")
    else:
        print("No intrinsic file found. Fetching factory calibration directly from RealSense...")
        try:
            p = rs.pipeline()
            c = rs.config()
            c.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
            prof = p.start(c)
            intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
            D = np.array(intr.coeffs)
            p.stop()
        except Exception as e:
            print("Error connecting to RealSense for intrinsics:", e)
            return

    img_dir, pos_dir = os.path.join(DATA_DIR, "images"), os.path.join(DATA_DIR, "poses")
    imgs = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    poses = sorted([f for f in os.listdir(pos_dir) if f.endswith(".json")])
    
    objp = np.zeros((BOARD_SIZE[0]*BOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2) * (SQUARE_SIZE_MM / 1000.0)
    
    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    for img_f, pose_f in zip(imgs, poses):
        img = cv2.imread(os.path.join(img_dir, img_f))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, crn = cv2.findChessboardCorners(gray, BOARD_SIZE, None)
        if not ret: continue
        crn = cv2.cornerSubPix(gray, crn, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        ok, rvec, tvec = cv2.solvePnP(objp, crn, K, D)
        if not ok: continue
        R, _ = cv2.Rodrigues(rvec)
        R_t2c.append(R); t_t2c.append(tvec.flatten())
        with open(os.path.join(pos_dir, pose_f)) as file:
            T = _mat(json.load(file)["T_base_tool"])
            R_g2b.append(T[:3,:3]); t_g2b.append(T[:3,3])

    if len(R_g2b) < 5: print("Not enough valid pairs."); return

    methods = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS
    }

    print(f"\nComparing {len(methods)} algorithms on {len(R_g2b)} samples...")
    results = {}
    
    for name, code in methods.items():
        try:
            R_c2t, t_c2t = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=code)
            # Evaluate: average distance error between target computed via all poses
            target_pts = []
            for i in range(len(R_g2b)):
                # Target in Base = T_base_tool[i] @ T_cam_to_tool @ T_target_in_cam[i]
                T_c2t = np.eye(4); T_c2t[:3,:3] = R_c2t; T_c2t[:3,3] = t_c2t.flatten()
                T_b2t = np.eye(4); T_b2t[:3,:3] = R_g2b[i]; T_b2t[:3,3] = t_g2b[i]
                T_t2c = np.eye(4); T_t2c[:3,:3] = R_t2c[i]; T_t2c[:3,3] = t_t2c[i]
                T_target_base = T_b2t @ T_c2t @ T_t2c
                target_pts.append(T_target_base[:3, 3])
            
            target_pts = np.array(target_pts)
            mean_pt = np.mean(target_pts, axis=0)
            errors = np.linalg.norm(target_pts - mean_pt, axis=1)
            std_err = np.mean(errors)
            
            results[name] = {"R": R_c2t, "t": t_c2t, "error": std_err}
            print(f"  [{name:10s}] Avg Reprojection Deviation: {std_err*1000.0:.3f} mm")
        except: pass

    # Pick the best
    best_name = min(results, key=lambda k: results[k]["error"])
    best = results[best_name]
    print(f"\nRECOMMENDED: {best_name} (Error: {best['error']*1000.0:.3f} mm)")

    best_T = np.eye(4); best_T[:3,:3] = best["R"]; best_T[:3,3] = best["t"].flatten()
    with open(RESULT_FILE, 'w') as f:
        json.dump({
            "algorithm": best_name,
            "T_cam_to_tool": _lst(best_T),
            "error_m": best["error"]
        }, f, indent=2)
    print(f"Saved best result to {RESULT_FILE}")

# ============================================================
# DIRECT MATH TEST
# ============================================================

def test_math_tracking():
    if not os.path.exists(RESULT_FILE):
        print("Cần chạy bước 3 (Calib Hand-Eye) trước!")
        return
    with open(RESULT_FILE) as f:
        T_cam2tool = _mat(json.load(f)["T_cam_to_tool"])
        
    if not os.path.exists(INTRINSIC_FILE):
        print("Cần chạy bước 2 (Intrinsic) trước!")
        return
        
    with open(INTRINSIC_FILE) as f:
        d = json.load(f)
        K = np.array(d["camera_matrix"])
        intr = rs.intrinsics()
        intr.width = d["image_size"][0]
        intr.height = d["image_size"][1]
        intr.ppx = K[0, 2]
        intr.ppy = K[1, 2]
        intr.fx = K[0, 0]
        intr.fy = K[1, 1]
        intr.model = rs.distortion.brown_conrady
        dist_flat = np.array(d["dist_coeffs"]).flatten().tolist()
        intr.coeffs = dist_flat[:5]
        
    try: 
        robot = IndyDCP3(robot_ip=ROBOT_IP, index=0)
    except Exception as e: 
        print("Robot Error:", e); return
    
    print("Loading SAM...")
    try:
        from ultralytics import SAM
        import torch
        sam_model = SAM('/home/apicoo-ai/pmg/weld_vision_ws/models/mobile_sam.pt')
    except Exception as e: 
        print("SAM Error:", e); return
    
    cam_id = find_webcam_id()
    if cam_id is None:
        return
        
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở camera tại /dev/video{cam_id}")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    click_pt = None
    cached_xyz_world = None
    def on_mouse(event, x, y, flags, param):
        nonlocal click_pt
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pt = [x, y]
            
    cv2.namedWindow("Test Math")
    cv2.setMouseCallback("Test Math", on_mouse)
    
    print("\n[READY] Click vào vật thể để tính toán toạ độ World XYZ.")
    print("Thuật toán: Bắn tia từ pixel đi qua tâm quang học, cắt mặt phẳng Z = 5mm của Base.")
    print("Sau khi click, hãy dùng Teach Pendant để xoay/tịnh tiến robot.")
    print("Nếu toạ độ World XYZ không đổi (hoặc sai lệch nhỏ) thì Matrix đúng!")
    print("Nhấn 'q' để thoát.")
    
    try:
        while True:
            ret_cam, img = cap.read()
            if not ret_cam: continue
            disp = img.copy()
            
            if click_pt is not None:
                res = sam_model(img, points=[click_pt], labels=[1], verbose=False)
                if res and len(res) > 0 and res[0].masks is not None:
                    mask = res[0].masks.data[0].cpu().numpy().astype(np.uint8)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        cnt = max(contours, key=cv2.contourArea)
                        cv2.polylines(disp, [cnt], True, (0,255,0), 2)
                        
                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                        else:
                            cX, cY = click_pt
                            
                        cv2.circle(disp, (cX, cY), 5, (255, 0, 0), -1)

                        pose_data = robot.get_control_data()
                        if pose_data and 'p' in pose_data:
                            T_base_tool = _p_to_T(pose_data['p'])
                            T_base_cam = T_base_tool @ T_cam2tool
                            
                            fx = K[0, 0]; fy = K[1, 1]
                            cx_cam = K[0, 2]; cy_cam = K[1, 2]
                            
                            x_c = (cX - cx_cam) / fx
                            y_c = (cY - cy_cam) / fy
                            V_c = np.array([x_c, y_c, 1.0])
                            
                            O_base = T_base_cam[:3, 3] 
                            D_base = T_base_cam[:3, :3] @ V_c 
                            
                            Z_target = 0.030 # 30mm offset
                            
                            if D_base[2] != 0:
                                t = (Z_target - O_base[2]) / D_base[2]
                                if t > 0:
                                    P_base = O_base + t * D_base
                                    x, y, z = P_base * 1000.0 # m to mm
                                    cached_xyz_world = (x, y, z)
                                    print(f"Webcam(px): {cX}, {cY} ---> World(mm): X={x:.1f} Y={y:.1f} Z={z:.1f}")
                click_pt = None
                
            if cached_xyz_world is not None:
                x, y, z = cached_xyz_world
                cv2.putText(disp, f"World X: {x:.1f} mm", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.putText(disp, f"World Y: {y:.1f} mm", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.putText(disp, f"World Z: {z:.1f} mm", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                
            cv2.imshow("Test Math", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        cap.release(); cv2.destroyAllWindows()

# ============================================================
# MAIN
# ============================================================

def _menu():
    print("\nCalibration Control Panel")
    print("1. Take Data (Images + Poses)")
    print("2. (Auto) Intrinsic Calibration")
    print("3. (Auto) Calculate Hand-Eye (All methods)")
    print("4. Test 3D Coordinate Math directly via SAM2 (Bypass TF)")
    print("q. Exit\n")
    while rclpy.ok():
        c = input("Select: ").strip().lower()
        if c == '1': take_data()
        elif c == '2': intrinsic_calibration()
        elif c == '3': solve_hand_eye()
        elif c == '4': test_math_tracking()
        elif c == 'q': rclpy.shutdown(); break

def main():
    rclpy.init()
    #node = CalibrationNode()
    #threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    _menu()

if __name__ == '__main__': main()