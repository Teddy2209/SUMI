import cv2
import numpy as np
import glob
import os

CHECKERBOARD = (11, 8)
SQUARE_SIZE = 0.015

OUTPUT_DIR = "/media/apicoo-ai/5511010c-3660-41c3-b501-36e739767b6a/SUMI/scripts_BMG/S0_Camera_Calibration/S0_output/intrinsics_matrixes_28082026"
FPC_IMG_DIR = os.path.join(OUTPUT_DIR, "images", "fpc")

images = sorted(glob.glob(os.path.join(FPC_IMG_DIR, '*.png')))

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

valid_images = []
objpoints_all = []
imgpoints_all = []

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        objpoints_all.append(objp.reshape(1, -1, 3).astype(np.float64))
        imgpoints_all.append(corners2.reshape(1, -1, 2).astype(np.float64))
        valid_images.append(fname)

flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW | cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT

safe_objpoints = []
safe_imgpoints = []
safe_images = []

for i in range(len(valid_images)):
    test_obj = safe_objpoints + [objpoints_all[i]]
    test_img = safe_imgpoints + [imgpoints_all[i]]
    
    if len(test_obj) < 2:
        safe_objpoints = test_obj
        safe_imgpoints = test_img
        safe_images.append(valid_images[i])
        continue

    try:
        K_temp = np.zeros((3, 3), dtype=np.float64)
        K_temp[0, 2] = gray.shape[1] / 2.0
        K_temp[1, 2] = gray.shape[0] / 2.0
        D_temp = np.zeros((4, 1), dtype=np.float64)
        cv2.fisheye.calibrate(test_obj, test_img, gray.shape[::-1], K_temp, D_temp, flags=flags | cv2.fisheye.CALIB_FIX_K4 | cv2.fisheye.CALIB_FIX_K3)
        safe_objpoints = test_obj
        safe_imgpoints = test_img
        safe_images.append(valid_images[i])
    except Exception:
        pass

print(f"Filtered down to {len(safe_images)} safe images.")

try:
    K = np.zeros((3, 3), dtype=np.float64)
    K[0, 2] = gray.shape[1] / 2.0
    K[1, 2] = gray.shape[0] / 2.0
    D = np.zeros((4, 1), dtype=np.float64)
    ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
        safe_objpoints, safe_imgpoints, gray.shape[::-1], K, D,
        flags=flags | cv2.fisheye.CALIB_FIX_K4 | cv2.fisheye.CALIB_FIX_K3
    )
    print("RMSE:", ret)
    print("K:", mtx)
    print("D:", dist)
except Exception as e:
    print("Crash:", e)
