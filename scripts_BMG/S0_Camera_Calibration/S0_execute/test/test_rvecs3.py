import cv2
import numpy as np

# Create valid synthetic data for fisheye
CHECKERBOARD = (11, 8)
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float64)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Create 3 rotated views
rvec1 = np.array([0.1, 0.2, 0.0], dtype=np.float64)
tvec1 = np.array([0.0, 0.0, 5.0], dtype=np.float64)
rvec2 = np.array([-0.1, 0.2, 0.0], dtype=np.float64)
tvec2 = np.array([0.0, 0.0, 5.0], dtype=np.float64)
rvec3 = np.array([0.1, -0.2, 0.0], dtype=np.float64)
tvec3 = np.array([0.0, 0.0, 5.0], dtype=np.float64)

K = np.array([[300, 0, 320], [0, 300, 240], [0, 0, 1]], dtype=np.float64)
D = np.zeros((4, 1), dtype=np.float64)

imgp1, _ = cv2.fisheye.projectPoints(objp[0], rvec1, tvec1, K, D)
imgp2, _ = cv2.fisheye.projectPoints(objp[0], rvec2, tvec2, K, D)
imgp3, _ = cv2.fisheye.projectPoints(objp[0], rvec3, tvec3, K, D)

objpoints = [objp[0].reshape(1, -1, 3)] * 3
imgpoints = [imgp1.reshape(1, -1, 2), imgp2.reshape(1, -1, 2), imgp3.reshape(1, -1, 2)]

flags1 = cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT | cv2.fisheye.CALIB_FIX_K3 | cv2.fisheye.CALIB_FIX_K4
K1 = np.zeros((3, 3), dtype=np.float64)
K1[0, 2] = 320
K1[1, 2] = 240
D1 = np.zeros((4, 1), dtype=np.float64)

ret, K1, D1, rvecs1, tvecs1 = cv2.fisheye.calibrate(objpoints, imgpoints, (640, 480), K1, D1, flags=flags1)
print("Step 1 rvecs type:", type(rvecs1))

# Step 2: Pass rvecs1 and tvecs1 back in without RECOMPUTE_EXTRINSIC
flags2 = cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
K2 = K1.copy()
D2 = D1.copy()
try:
    ret2, K2, D2, rvecs2, tvecs2 = cv2.fisheye.calibrate(
        objpoints, imgpoints, (640, 480), K2, D2, rvecs1, tvecs1, flags=flags2
    )
    print("SUCCESS Step 2 without RECOMPUTE_EXTRINSIC")
except Exception as e:
    print("CRASH Step 2:", e)
