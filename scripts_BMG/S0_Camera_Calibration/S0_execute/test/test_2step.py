import cv2
import numpy as np
import glob
import os

CHECKERBOARD = (11, 8)
SQUARE_SIZE = 0.01

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
    
    # Sử dụng findChessboardCornersSB theo đề xuất
    ret, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if ret:
        objpoints_all.append(objp.reshape(1, -1, 3).astype(np.float64))
        imgpoints_all.append(corners.reshape(1, -1, 2).astype(np.float64))
        valid_images.append(fname)
    else:
        # Fallback to standard
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            objpoints_all.append(objp.reshape(1, -1, 3).astype(np.float64))
            imgpoints_all.append(corners2.reshape(1, -1, 2).astype(np.float64))
            valid_images.append(fname)

gray_shape = gray.shape[::-1]

print(f"Loaded {len(valid_images)} images.")

# Step 1: Lọc crash và lấy K, D base
flags_step1 = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW | cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT | cv2.fisheye.CALIB_FIX_K3 | cv2.fisheye.CALIB_FIX_K4
safe_objpoints = []
safe_imgpoints = []
safe_fnames = []

for i in range(len(objpoints_all)):
    test_obj = safe_objpoints + [objpoints_all[i]]
    test_img = safe_imgpoints + [imgpoints_all[i]]
    
    if len(test_obj) < 3:
        safe_objpoints = test_obj
        safe_imgpoints = test_img
        safe_fnames.append(valid_images[i])
        continue
        
    try:
        K_temp = np.zeros((3, 3), dtype=np.float64)
        K_temp[0, 2] = gray_shape[0] / 2.0
        K_temp[1, 2] = gray_shape[1] / 2.0
        D_temp = np.zeros((4, 1), dtype=np.float64)
        cv2.fisheye.calibrate(test_obj, test_img, gray_shape, K_temp, D_temp, flags=flags_step1)
        safe_objpoints = test_obj
        safe_imgpoints = test_img
        safe_fnames.append(valid_images[i])
    except Exception:
        pass

print(f"Step 1: {len(safe_objpoints)} safe images.")

K_step1 = np.zeros((3, 3), dtype=np.float64)
K_step1[0, 2] = gray_shape[0] / 2.0
K_step1[1, 2] = gray_shape[1] / 2.0
D_step1 = np.zeros((4, 1), dtype=np.float64)

ret_step1, K_step1, D_step1, rvecs_step1, tvecs_step1 = cv2.fisheye.calibrate(
    safe_objpoints, safe_imgpoints, gray_shape, K_step1, D_step1,
    flags=flags_step1,
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
)
print("Step 1 RMSE:", ret_step1)

# Lọc Outlier theo Reprojection Error
errors = []
for i in range(len(safe_objpoints)):
    imgpoints_proj, _ = cv2.fisheye.projectPoints(safe_objpoints[i], rvecs_step1[i], tvecs_step1[i], K_step1, D_step1)
    error = cv2.norm(safe_imgpoints[i], imgpoints_proj, cv2.NORM_L2) / np.sqrt(len(imgpoints_proj[0]))
    errors.append(error)

mean_error = np.mean(errors)
threshold = 1.5 * mean_error
print(f"Mean Error: {mean_error:.3f}, Threshold: {threshold:.3f}")

final_objpoints = []
final_imgpoints = []
for i in range(len(safe_objpoints)):
    if errors[i] <= threshold:
        final_objpoints.append(safe_objpoints[i])
        final_imgpoints.append(safe_imgpoints[i])
    else:
        print(f"Removing outlier: {safe_fnames[i]} (Error: {errors[i]:.3f})")

print(f"Final images: {len(final_objpoints)}")

# Step 2: Calib tinh chỉnh (thả tự do K3, K4, cx, cy)
flags_step2 = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
K_final = K_step1.copy()
D_final = D_step1.copy()

ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
    final_objpoints, final_imgpoints, gray_shape, K_final, D_final,
    flags=flags_step2,
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
)

print("Final RMSE:", ret)
print("K:\n", mtx)
print("D:\n", dist)
