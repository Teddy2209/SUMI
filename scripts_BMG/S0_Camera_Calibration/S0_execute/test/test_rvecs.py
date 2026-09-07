import cv2
import numpy as np

# Test if we can pass rvecs and tvecs to fisheye.calibrate and avoid crash
objp = np.zeros((1, 88, 3), np.float64)
imgp = np.zeros((1, 88, 2), np.float64)

K = np.eye(3, dtype=np.float64)
D = np.zeros((4, 1), dtype=np.float64)
rvecs = [np.zeros((3, 1), dtype=np.float64)]
tvecs = [np.array([[0.0], [0.0], [1.0]], dtype=np.float64)]

try:
    flags = cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
    # Without CALIB_RECOMPUTE_EXTRINSIC, OpenCV should use the provided rvecs and tvecs
    # In python, rvecs and tvecs can be passed as inputs?
    # Actually cv2.fisheye.calibrate signature in python:
    # (objectPoints, imagePoints, image_size, K, D[, rvecs[, tvecs[, flags[, criteria]]]])
    
    cv2.fisheye.calibrate([objp], [imgp], (640, 480), K, D, rvecs, tvecs, flags=flags)
    print("SUCCESS without RECOMPUTE_EXTRINSIC")
except Exception as e:
    print("CRASH:", e)

try:
    flags2 = cv2.fisheye.CALIB_USE_INTRINSIC_GUESS | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    cv2.fisheye.calibrate([objp], [imgp], (640, 480), K, D, rvecs, tvecs, flags=flags2)
    print("SUCCESS with RECOMPUTE_EXTRINSIC")
except Exception as e:
    print("CRASH with RECOMPUTE:", e)
