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

objpoints = []
imgpoints = []
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        objpoints.append(objp.reshape(1, -1, 3).astype(np.float64))
        imgpoints.append(corners2.reshape(1, -1, 2).astype(np.float64))

print("Testing full calib WITHOUT fix flags (True Fisheye power)...")
K = np.zeros((3, 3), dtype=np.float64)
D = np.zeros((4, 1), dtype=np.float64)

flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW

try:
    ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints, imgpoints, gray.shape[::-1], K, D,
        flags=flags, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
    print("RMSE:", ret)
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    w, h = gray.shape[::-1]
    fov_diag = np.rad2deg(2 * np.arctan(np.sqrt(w**2 + h**2) / (2 * fx)))
    print(f"FOV Diag: {fov_diag:.1f}")
except Exception as e:
    print("CRASH without flags:", e)
