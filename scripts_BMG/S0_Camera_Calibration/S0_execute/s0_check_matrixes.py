import cv2
import numpy as np
import os
import json
import pyrealsense2 as rs
import glob

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "S0_output", "intrinsics_matrixes_28082026")

def load_matrix(json_path):
    if not os.path.exists(json_path):
        print(f"❌ Không tìm thấy file: {json_path}")
        return None, None, None
    with open(json_path, 'r') as f:
        data = json.load(f)
    mtx = np.array(data["camera_matrix"])
    dist = np.array(data["dist_coeffs"])
    model_type = data.get("model", "pinhole")
    return mtx, dist, model_type

def find_webcam_id():
    video_paths = glob.glob('/sys/class/video4linux/video*')
    video_paths.sort(key=lambda x: int(os.path.basename(x).replace('video', '')))
    for path in video_paths:
        try:
            with open(os.path.join(path, 'name'), 'r') as f:
                name = f.read().strip()
                if "RealSense" not in name and "Metadata" not in name:
                    idx = int(os.path.basename(path).replace('video', ''))
                    return idx
        except Exception:
            continue
    return None

def init_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline

def init_webcam():
    idx = find_webcam_id()
    if idx is None:
        return None
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2) 
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

def init_fpc():
    print("[INFO] Khởi động FPC Camera (SHUNCCM) tại /dev/video0...")
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    return cap

def check_camera(cam_type):
    if cam_type == "realsense":
        print("\nKhởi động Realsense...")
        try:
            pipeline = init_realsense()
        except Exception as e:
            print(f"Lỗi: {e}")
            return
        json_path = os.path.join(OUTPUT_DIR, "realsense_intrinsics.json")
    elif cam_type == "fpc_fisheye":
        print("\nKhởi động FPC Camera (Fisheye)...")
        cap = init_fpc()
        if cap is None:
            print("Lỗi: Không tìm thấy FPC Camera.")
            return
        json_path = os.path.join(OUTPUT_DIR, "fpccamera_fisheye_intrinsics.json")
    elif cam_type == "fpc_pinhole":
        print("\nKhởi động FPC Camera (Pinhole)...")
        cap = init_fpc()
        if cap is None:
            print("Lỗi: Không tìm thấy FPC Camera.")
            return
        json_path = os.path.join(OUTPUT_DIR, "fpccamera_pinhole_intrinsics.json")
    else:
        print("\nKhởi động Webcam...")
        cap = init_webcam()
        if cap is None:
            print("Lỗi: Không tìm thấy Webcam.")
            return
        json_path = os.path.join(OUTPUT_DIR, "webcamera_intrinsics.json")

    mtx, dist, model_type = load_matrix(json_path)
    if mtx is None:
        if cam_type == "realsense": pipeline.stop()
        else: cap.release()
        return

    print(f"\n>>> CHẾ ĐỘ KIỂM TRA CALIBRATION [{model_type.upper()}] <<<")
    print("1. Đưa camera lại gần các đường thẳng thực tế (cạnh bàn, mép tường, cái thước...).")
    print("2. Bấm 's' để chụp hình và xem kết quả Nắn thẳng (Undistort).")
    print("3. Bấm 'q' để thoát.")

    try:
        while True:
            # Lấy frame
            if cam_type == "realsense":
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame: continue
                img = np.asanyarray(color_frame.get_data())
            else:
                ret, img = cap.read()
                if not ret: continue

            display = img.copy()
            cv2.putText(display, f"LIVE PREVIEW ({cam_type.upper()})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, "Press 's' to Capture and Undistort", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Live Preview", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Xử lý ảnh Undistort
                print(f"\nĐang xử lý ảnh ({model_type})...")
                h, w = img.shape[:2]
                
                if model_type == "fisheye":
                    # Đổi balance=0.0 để crop hết viền đen và phóng to phần nắn phẳng
                    newcameramtx = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(mtx, dist, (w,h), np.eye(3), balance=1.0)
                    map1, map2 = cv2.fisheye.initUndistortRectifyMap(mtx, dist, np.eye(3), newcameramtx, (w,h), cv2.CV_16SC2)
                    dst = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    dst_cropped = dst # Balance=0.0 đã crop tự động
                else:
                    # KHÔNG DÙNG getOptimalNewCameraMatrix vì thuật toán này sẽ cố dịch chuyển quang tâm (cx, cy) 
                    # để nhét vừa các pixel bị méo, dẫn đến việc khung hình bị lệch hẳn sang 1 bên (viền đen to nhỏ không đều).
                    # Sử dụng trực tiếp ma trận mtx gốc sẽ giữ đúng sự ĐỐI XỨNG tuyệt đối của ống kính.
                    dst = cv2.undistort(img, mtx, dist, None, mtx)
                    dst_cropped = dst

                # Resize lại cho bằng nhau để ghép cho đẹp
                dst_resized = cv2.resize(dst_cropped, (w, h))

                # Vẽ viền đỏ cho dễ phân biệt
                cv2.rectangle(img, (0, 0), (w-1, h-1), (0, 0, 255), 4)
                cv2.rectangle(dst_resized, (0, 0), (w-1, h-1), (0, 255, 0), 4)

                cv2.putText(img, "ORIGINAL (Distorted)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(dst_resized, f"UNDISTORTED ({model_type})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Ghép ảnh ngang
                combined = np.hstack((img, dst_resized))
                
                print(">> Bấm phím bất kỳ (trên cửa sổ ảnh) để đóng và chụp tiếp, hoặc bấm 'q' để thoát.")
                cv2.imshow("Result (Left: Original | Right: Undistorted)", combined)
                
                wait_key = cv2.waitKey(0) & 0xFF
                cv2.destroyWindow("Result (Left: Original | Right: Undistorted)")
                
                if wait_key == ord('q'):
                    break
    finally:
        if cam_type == "realsense":
            pipeline.stop()
        else:
            cap.release()
        cv2.destroyAllWindows()

def _menu():
    print("\n" + "="*40)
    print("Camera Calibration Verification Tool")
    print("="*40)
    print("1. Check Realsense")
    print("2. Check Webcam")
    print("3. Check FPC Camera (Fisheye)")
    print("4. Check FPC Camera (Pinhole)")
    print("q. Quit\n")
    try:
        while True:
            c = input("Select: ").strip().lower()
            if c == '1': 
                check_camera("realsense")
                print("\n" + "="*40 + "\nMenu: 1. RS | 2. Web | 3. FPC (Fisheye) | 4. FPC (Pinhole) | q. Quit")
            elif c == '2': 
                check_camera("webcam")
                print("\n" + "="*40 + "\nMenu: 1. RS | 2. Web | 3. FPC (Fisheye) | 4. FPC (Pinhole) | q. Quit")
            elif c == '3': 
                check_camera("fpc_fisheye")
                print("\n" + "="*40 + "\nMenu: 1. RS | 2. Web | 3. FPC (Fisheye) | 4. FPC (Pinhole) | q. Quit")
            elif c == '4': 
                check_camera("fpc_pinhole")
                print("\n" + "="*40 + "\nMenu: 1. RS | 2. Web | 3. FPC (Fisheye) | 4. FPC (Pinhole) | q. Quit")
            elif c == 'q': 
                break
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    _menu()
