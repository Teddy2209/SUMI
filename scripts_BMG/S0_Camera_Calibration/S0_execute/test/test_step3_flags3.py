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

objpoints_all = []
imgpoints_all = []
valid_images = []
gray_shape = None

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray_shape is None: gray_shape = gray.shape[::-1]
    
    ret, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if not ret:
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            
    if ret:
        objpoints_all.append(objp.reshape(1, -1, 3).astype(np.float64))
        imgpoints_all.append(corners.reshape(1, -1, 2).astype(np.float64))
        valid_images.append(fname)

# Step 1
flags_step1 = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW | cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT | cv2.fisheye.CALIB_FIX_K3 | cv2.fisheye.CALIB_FIX_K4
safe_objpoints = []
safe_imgpoints = []
for i in range(len(objpoints_all)):
    test_obj = safe_objpoints + [objpoints_all[i]]
    test_img = safe_imgpoints + [imgpoints_all[i]]
    if len(test_obj) < 3:
        safe_objpoints = test_obj; safe_imgpoints = test_img
        continue
    try:
        K_temp = np.zeros((3, 3), dtype=np.float64)
        K_temp[0, 2] = gray_shape[0] / 2.0
        K_temp[1, 2] = gray_shape[1] / 2.0
        D_temp = np.zeros((4, 1), dtype=np.float64)
        cv2.fisheye.calibrate(test_obj, test_img, gray_shape, K_temp, D_temp, flags=flags_step1)
        safe_objpoints = test_obj
        safe_imgpoints = test_img
    except:
        pass

K_step1 = np.zeros((3, 3), dtype=np.float64)
K_step1[0, 2] = gray_shape[0] / 2.0; K_step1[1, 2] = gray_shape[1] / 2.0
D_step1 = np.zeros((4, 1), dtype=np.float64)
ret_step1, K_step1, D_step1, rvecs1, tvecs1 = cv2.fisheye.calibrate(
    safe_objpoints, safe_imgpoints, gray_shape, K_step1, D_step1, flags=flags_step1
)

errors = []
for i in range(len(safe_objpoints)):
    proj, _ = cv2.fisheye.projectPoints(safe_objpoints[i], rvecs1[i], tvecs1[i], K_step1, D_step1)
    errors.append(cv2.norm(safe_imgpoints[i], proj, cv2.NORM_L2) / np.sqrt(len(proj[0])))

mean_error = np.mean(errors)
threshold = 1.5 * mean_error
final_objpoints = [safe_objpoints[i] for i in range(len(safe_objpoints)) if errors[i] <= threshold]
final_imgpoints = [safe_imgpoints[i] for i in range(len(safe_imgpoints)) if errors[i] <= threshold]

print("Test E: Use INTRINSIC_GUESS, but DO NOT USE CALIB_RECOMPUTE_EXTRINSIC")
print("We release K3, K4, cx, cy")
try:
    flagsE = cv2.fisheye.CALIB_FIX_SKEW | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
    KE = K_step1.copy()
    DE = D_step1.copy()
    retE, KE, DE, rE, tE = cv2.fisheye.calibrate(final_objpoints, final_imgpoints, gray_shape, KE, DE, flags=flagsE)
    print("Test E SUCCESS! NO CRASH!")
except Exception as e:
    print("Test E CRASH:", e)
