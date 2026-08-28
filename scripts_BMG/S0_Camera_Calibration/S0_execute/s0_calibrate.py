import cv2
import numpy as np
import os
import json
import pyrealsense2 as rs
import glob
from datetime import datetime

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "S0_output", "intrinsics_matrixes_28082026")
RS_IMG_DIR = os.path.join(OUTPUT_DIR, "images", "realsense")
WEB_IMG_DIR = os.path.join(OUTPUT_DIR, "images", "webcam")
FPC_IMG_DIR = os.path.join(OUTPUT_DIR, "images", "fpc")

# Kích thước bàn cờ (Số lượng các điểm giao cắt BÊN TRONG bàn cờ)
# Ví dụ: Bàn cờ 11x8 thì giao cắt là (10, 7)
CHECKERBOARD = (11, 8) 
SQUARE_SIZE = 0.01  # Kích thước 1 ô vuông (mét) - thay đổi theo bàn cờ thực tế

# Đảm bảo thư mục tồn tại
os.makedirs(RS_IMG_DIR, exist_ok=True)
os.makedirs(WEB_IMG_DIR, exist_ok=True)
os.makedirs(FPC_IMG_DIR, exist_ok=True)

def init_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 60)
    pipeline.start(config)
    return pipeline

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

def init_webcam():
    idx = find_webcam_id()
    if idx is None:
        return None
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2) 
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    return cap

def init_fpc():
    print("[INFO] Khởi động FPC Camera (SHUNCCM) tại /dev/video0...")
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2) 
    
    # Chạy ở độ phân giải 640x480 @ 30FPS (Raw/YUYV mặc định)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

def detect_chessboard(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        img_drawn = cv2.drawChessboardCorners(img.copy(), CHECKERBOARD, corners2, ret)
        return True, img_drawn
    return False, img

def take_data_realsense():
    print("\nKhởi động Camera Realsense...")
    try:
        pipeline = init_realsense()
    except Exception as e:
        print(f"Lỗi khởi động camera: {e}")
        return

    img_counter = 0
    existing = glob.glob(os.path.join(RS_IMG_DIR, "*.png"))
    if existing:
        img_counter = max([int(os.path.basename(f).split('_')[1].split('.')[0]) for f in existing]) + 1

    print("\n>>> CHẾ ĐỘ THU THẬP DỮ LIỆU REALSENSE <<<")
    print("- Bấm 's' để LƯU.")
    print("- Bấm 'q' để THOÁT.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame: continue
            img = np.asanyarray(color_frame.get_data())

            ret, drawn = detect_chessboard(img)
            
            cv2.putText(drawn, "REALSENSE: DATA COLLECTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            if ret:
                cv2.putText(drawn, f"READY TO SAVE #{img_counter} (Press 's')", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            else:
                cv2.putText(drawn, "BOARD NOT FOUND", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            cv2.imshow("Realsense (960x540)", drawn)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if ret:
                    path = os.path.join(RS_IMG_DIR, f"rs_{img_counter:04d}.png")
                    cv2.imwrite(path, img) 
                    print(f"✅ Đã lưu ảnh Realsense số {img_counter}")
                    img_counter += 1
                else:
                    print("❌ Chưa nhận diện được bàn cờ!")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def take_data_webcam():
    print("\nKhởi động Webcam...")
    try:
        cap = init_webcam()
        if cap is None: return
    except Exception as e:
        print(f"Lỗi khởi động camera: {e}")
        return

    img_counter = 0
    existing = glob.glob(os.path.join(WEB_IMG_DIR, "*.png"))
    if existing:
        img_counter = max([int(os.path.basename(f).split('_')[1].split('.')[0]) for f in existing]) + 1

    print("\n>>> CHẾ ĐỘ THU THẬP DỮ LIỆU WEBCAM <<<")
    print("- Bấm 's' để LƯU.")
    print("- Bấm 'q' để THOÁT.")

    try:
        while True:
            ret_cam, img = cap.read()
            if not ret_cam: continue

            ret, drawn = detect_chessboard(img)
            
            cv2.putText(drawn, "WEBCAM: DATA COLLECTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            if ret:
                cv2.putText(drawn, f"READY TO SAVE #{img_counter} (Press 's')", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            else:
                cv2.putText(drawn, "BOARD NOT FOUND", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            cv2.imshow("Webcam (640x480)", drawn)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if ret:
                    path = os.path.join(WEB_IMG_DIR, f"web_{img_counter:04d}.png")
                    cv2.imwrite(path, img) 
                    print(f"✅ Đã lưu ảnh Webcam số {img_counter}")
                    img_counter += 1
                else:
                    print("❌ Chưa nhận diện được bàn cờ!")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def take_data_fpc():
    print("\nKhởi động FPC Camera...")
    try:
        cap = init_fpc()
        if cap is None: return
    except Exception as e:
        print(f"Lỗi khởi động camera: {e}")
        return

    img_counter = 0
    existing = glob.glob(os.path.join(FPC_IMG_DIR, "*.png"))
    if existing:
        img_counter = max([int(os.path.basename(f).split('_')[1].split('.')[0]) for f in existing]) + 1

    print("\n>>> CHẾ ĐỘ THU THẬP DỮ LIỆU FPC CAMERA <<<")
    print("- Bấm 's' để LƯU.")
    print("- Bấm 'q' để THOÁT.")

    try:
        while True:
            ret_cam, img = cap.read()
            if not ret_cam: continue

            ret, drawn = detect_chessboard(img)
            
            cv2.putText(drawn, "FPC CAMERA: DATA COLLECTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            if ret:
                cv2.putText(drawn, f"READY TO SAVE #{img_counter} (Press 's')", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            else:
                cv2.putText(drawn, "BOARD NOT FOUND", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            cv2.imshow("FPC Camera (640x480)", drawn)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if ret:
                    path = os.path.join(FPC_IMG_DIR, f"fpc_{img_counter:04d}.png")
                    cv2.imwrite(path, img) 
                    print(f"✅ Đã lưu ảnh FPC Camera số {img_counter}")
                    img_counter += 1
                else:
                    print("❌ Chưa nhận diện được bàn cờ!")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def calibrate_single_camera(image_dir, json_path, target_shape, use_fisheye=False):
    images = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    if len(images) == 0:
        print(f"❌ Không có ảnh nào trong {image_dir}")
        return False

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

    objpoints = [] 
    imgpoints = [] 
    gray_shape = None
    success_count = 0

    print(f"Đang phân tích {len(images)} ảnh từ {os.path.basename(image_dir)}...")
    for fname in images:
        img = cv2.imread(fname)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray_shape is None:
            gray_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            
            if use_fisheye:
                # Mô hình Fisheye đặc biệt nhạy cảm với định dạng mảng:
                # BẮT BUỘC phải là (1, N, 3) và kiểu float64, nếu dùng (-1, 1, 3) hoặc float32 sẽ sinh lỗi norm_u1 > 0
                objpoints.append(objp.reshape(1, -1, 3).astype(np.float64))
                imgpoints.append(corners2.reshape(1, -1, 2).astype(np.float64))
            else:
                objpoints.append(objp)
                imgpoints.append(corners2)
                
            success_count += 1

    if success_count == 0:
        print("❌ Calibrate thất bại: Không nhận diện được bàn cờ trong các ảnh đã lưu.")
        return False

    print(f"Số ảnh hợp lệ được dùng để giải K, D: {success_count} / {len(images)}")
    
    if use_fisheye:
        print("\nĐang tiến hành thuật toán lọc ảnh chống crash (Fisheye InitExtrinsics)...")
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW | cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT | cv2.fisheye.CALIB_FIX_K3 | cv2.fisheye.CALIB_FIX_K4
        
        safe_objpoints = []
        safe_imgpoints = []
        
        for i in range(len(objpoints)):
            test_obj = safe_objpoints + [objpoints[i]]
            test_img = safe_imgpoints + [imgpoints[i]]
            
            if len(test_obj) < 3: # Cần ít nhất 3 ảnh để calib ổn định
                safe_objpoints = test_obj
                safe_imgpoints = test_img
                continue
                
            try:
                K_temp = np.zeros((3, 3), dtype=np.float64)
                K_temp[0, 2] = gray_shape[0] / 2.0  # cx (width/2)
                K_temp[1, 2] = gray_shape[1] / 2.0  # cy (height/2)
                
                D_temp = np.zeros((4, 1), dtype=np.float64)
                cv2.fisheye.calibrate(
                    test_obj, test_img, gray_shape, K_temp, D_temp,
                    flags=flags,
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
                )
                safe_objpoints = test_obj
                safe_imgpoints = test_img
            except Exception as e:
                print(f"-> Đã tự động loại bỏ 1 ảnh gây suy biến ma trận (Ill-conditioned)")
        
        if len(safe_objpoints) < 3:
            print("Lỗi: Không đủ ảnh hợp lệ để Calib Fisheye. Vui lòng chụp thêm!")
            return
            
        print(f"Đã lọc xong! Sử dụng {len(safe_objpoints)} / {len(objpoints)} ảnh an toàn nhất.")
        
        K = np.zeros((3, 3), dtype=np.float64)
        K[0, 2] = gray_shape[0] / 2.0  # cx
        K[1, 2] = gray_shape[1] / 2.0  # cy
        D = np.zeros((4, 1), dtype=np.float64)
        ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
            safe_objpoints, safe_imgpoints, gray_shape, K, D,
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
        
        # Tính toán FOV từ ma trận camera
        fx = mtx[0, 0]
        fy = mtx[1, 1]
        fov_x = np.rad2deg(2 * np.arctan(gray_shape[0] / (2 * fx)))
        fov_y = np.rad2deg(2 * np.arctan(gray_shape[1] / (2 * fy)))
        fov_diag = np.rad2deg(2 * np.arctan(np.sqrt(gray_shape[0]**2 + gray_shape[1]**2) / (2 * fx)))
        print(f"📐 Ước lượng Góc nhìn Fisheye (FOV): Ngang {fov_x:.1f}°, Dọc {fov_y:.1f}°, Chéo {fov_diag:.1f}°")
        
        model_type = "fisheye"
    else:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)
        model_type = "pinhole"

    print(f"[{model_type.upper()}] Calibration RMSE Error: {ret:.4f} pixels")
    
    calib_data = {
        "model": model_type,
        "image_size": [gray_shape[0], gray_shape[1]],
        "camera_matrix": mtx.tolist(),
        "dist_coeffs": dist.tolist(),
        "error_pixels": ret
    }

    with open(json_path, 'w') as f:
        json.dump(calib_data, f, indent=4)
        
    print(f"✅ Đã lưu Intrinsic Calibration mới vào: {json_path}")
    return True

def _menu():
    print("\n" + "="*40)
    print("Multi-Camera Calibration Control Panel")
    print("="*40)
    print("1. Take data Realsense")
    print("2. Take data Webcam")
    print("3. Take data FPC Camera")
    print("4. Calib camera Realsense")
    print("5. Calib camera Webcam")
    print("6. Calib camera FPC Camera")
    print("q. Exit\n")
    try:
        while True:
            c = input("Select: ").strip().lower()
            if c == '1': 
                take_data_realsense()
                print("\n" + "="*40 + "\nControl Panel: 1. Take RS | 2. Take Web | 3. Take FPC | 4. Calib RS | 5. Calib Web | 6. Calib FPC | q. Exit")
            elif c == '2': 
                take_data_webcam()
                print("\n" + "="*40 + "\nControl Panel: 1. Take RS | 2. Take Web | 3. Take FPC | 4. Calib RS | 5. Calib Web | 6. Calib FPC | q. Exit")
            elif c == '3': 
                take_data_fpc()
                print("\n" + "="*40 + "\nControl Panel: 1. Take RS | 2. Take Web | 3. Take FPC | 4. Calib RS | 5. Calib Web | 6. Calib FPC | q. Exit")
            elif c == '4': 
                rs_json = os.path.join(OUTPUT_DIR, "realsense_intrinsics.json")
                print("\n[Calibrating Realsense...]")
                calibrate_single_camera(RS_IMG_DIR, rs_json, (960, 540))
                print("\n" + "="*40 + "\nControl Panel: 1. Take RS | 2. Take Web | 3. Take FPC | 4. Calib RS | 5. Calib Web | 6. Calib FPC | q. Exit")
            elif c == '5': 
                web_json = os.path.join(OUTPUT_DIR, "webcamera_intrinsics.json")
                print("\n[Calibrating Webcam...]")
                calibrate_single_camera(WEB_IMG_DIR, web_json, (640, 480))
                print("\n" + "="*40 + "\nControl Panel: 1. Take RS | 2. Take Web | 3. Take FPC | 4. Calib RS | 5. Calib Web | 6. Calib FPC | q. Exit")
            elif c == '6': 
                fpc_json = os.path.join(OUTPUT_DIR, "fpccamera_intrinsics.json")
                print("\n[Calibrating FPC Camera (Fisheye Model)...]")
                calibrate_single_camera(FPC_IMG_DIR, fpc_json, (640, 480), use_fisheye=True)
                print("\n" + "="*40 + "\nControl Panel: 1. Take RS | 2. Take Web | 3. Take FPC | 4. Calib RS | 5. Calib Web | 6. Calib FPC | q. Exit")
            elif c == 'q': 
                break
    except KeyboardInterrupt:
        print("\nExiting...")

def main():
    _menu()

if __name__ == "__main__":
    main()
