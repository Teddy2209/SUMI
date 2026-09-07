import cv2
import numpy as np

objp = np.zeros((1, 88, 3), np.float64)
imgp = np.zeros((1, 88, 2), np.float64)
imgp[0, :, 0] = np.linspace(0, 100, 88)

K = np.eye(3, dtype=np.float64)
D = np.zeros((4, 1), dtype=np.float64)

# Create 3 images
objpoints = [objp, objp, objp]
imgpoints = [imgp, imgp, imgp]

# Get rvecs and tvecs
flags_step1 = cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT
ret, K1, D1, rvecs, tvecs = cv2.fisheye.calibrate(objpoints, imgpoints, (640, 480), K, D, flags=flags_step1)

print(f"rvecs type: {type(rvecs)}, element shape: {rvecs[0].shape}")

try:
    flags2 = cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
    cv2.fisheye.calibrate(objpoints, imgpoints, (640, 480), K1, D1, rvecs, tvecs, flags=flags2)
    print("SUCCESS without RECOMPUTE_EXTRINSIC")
except Exception as e:
    print("CRASH:", e)
