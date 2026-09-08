import cv2
import numpy as np
import os
import json
import glob

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "S0_output", "intrinsics_matrixes_28082026")
FPC_IMG_DIR = os.path.join(OUTPUT_DIR, "images", "fpc_pinhole")

CHECKERBOARD = (11, 8) 
SQUARE_SIZE = 0.01  

os.makedirs(FPC_IMG_DIR, exist_ok=True)

def init_fpc():
    print("[INFO] Khởi động FPC Camera (SHUNCCM) tại /dev/video0...")
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2) 
    
    # Ép sử dụng định dạng Raw/YUYV (không nén) để đảm bảo chất lượng pixel tốt nhất
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    return cap

def detect_chessboard(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD, 
            cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if ret:
        img_drawn = cv2.drawChessboardCorners(img.copy(), CHECKERBOARD, corners, ret)
        return True, img_drawn
    else:
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, 
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            img_drawn = cv2.drawChessboardCorners(img.copy(), CHECKERBOARD, corners2, ret)
            return True, img_drawn
    return False, img

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

def calibrate_fpc_pinhole():
    json_path = os.path.join(OUTPUT_DIR, "fpccamera_pinhole_intrinsics.json")
    images = sorted(glob.glob(os.path.join(FPC_IMG_DIR, '*.png')))
    if len(images) == 0:
        print(f"❌ Không có ảnh nào trong {FPC_IMG_DIR}")
        return False

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

    objpoints = [] 
    imgpoints = [] 
    gray_shape = None
    success_count = 0

    print(f"Đang phân tích {len(images)} ảnh FPC Camera...")
    for fname in images:
        img = cv2.imread(fname)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray_shape is None:
            gray_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if not ret:
            ret, corners_fallback = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners_fallback, (11,11), (-1,-1), criteria)
                
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            success_count += 1

    if success_count == 0:
        print("❌ Calibrate thất bại: Không nhận diện được bàn cờ.")
        return False

    print(f"Số ảnh hợp lệ được dùng: {success_count} / {len(images)}")
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)

    print(f"[PINHOLE] Calibration RMSE Error: {ret:.4f} pixels")
    
    calib_data = {
        "model": "pinhole",
        "image_size": [gray_shape[0], gray_shape[1]],
        "camera_matrix": mtx.tolist(),
        "dist_coeffs": dist.tolist(),
        "error_pixels": ret
    }

    with open(json_path, 'w') as f:
        json.dump(calib_data, f, indent=4)
        
    print(f"✅ Đã lưu Intrinsic Calibration mới vào: {json_path}")
    return True

def main():
    print("\n" + "="*40)
    print("FPC Camera (Pinhole) Calibration Panel")
    print("="*40)
    print("1. Take data FPC Camera")
    print("2. Calibrate FPC Camera (Pinhole)")
    print("q. Exit\n")
    try:
        while True:
            c = input("Select: ").strip().lower()
            if c == '1': 
                take_data_fpc()
                print("\n" + "="*40 + "\n1. Take FPC | 2. Calibrate Pinhole | q. Exit")
            elif c == '2': 
                calibrate_fpc_pinhole()
                print("\n" + "="*40 + "\n1. Take FPC | 2. Calibrate Pinhole | q. Exit")
            elif c == 'q': 
                break
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
