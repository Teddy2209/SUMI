import cv2
import numpy as np

# Create valid synthetic data for fisheye
CHECKERBOARD = (11, 8)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp.reshape(1, -1, 3).astype(np.float64)

# Create 3 rotated views (just slightly shifted)
imgp1 = objp[:, :, :2].copy() + 10
imgp2 = objp[:, :, :2].copy() + 20
imgp3 = objp[:, :, :2].copy() + 30

objpoints = [objp, objp, objp]
imgpoints = [imgp1, imgp2, imgp3]

flags1 = cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT | cv2.fisheye.CALIB_FIX_K3 | cv2.fisheye.CALIB_FIX_K4
K1 = np.zeros((3, 3), dtype=np.float64)
K1[0, 2] = 320
K1[1, 2] = 240
D1 = np.zeros((4, 1), dtype=np.float64)

ret, K1, D1, rvecs1, tvecs1 = cv2.fisheye.calibrate(objpoints, imgpoints, (640, 480), K1, D1, flags=flags1)

flags2 = cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
K2 = K1.copy()
D2 = D1.copy()

print("Testing different rvec formats:")

# Tuple of (1, 1, 3) arrays
rvecs_3ch = tuple(r.reshape(1, 1, 3) for r in rvecs1)
tvecs_3ch = tuple(t.reshape(1, 1, 3) for t in tvecs1)
try:
    ret2, K2_new, D2_new, rvecs2, tvecs2 = cv2.fisheye.calibrate(
        objpoints, imgpoints, (640, 480), K2.copy(), D2.copy(), rvecs_3ch, tvecs_3ch, flags=flags2
    )
    print("SUCCESS: Tuple of (1, 1, 3)")
except Exception as e:
    print(f"CRASH: Tuple of (1, 1, 3): {e}")

# Tuple of (1, 3) arrays
rvecs_3ch2 = tuple(r.reshape(1, 3) for r in rvecs1)
tvecs_3ch2 = tuple(t.reshape(1, 3) for t in tvecs1)
try:
    ret2, K2_new, D2_new, rvecs2, tvecs2 = cv2.fisheye.calibrate(
        objpoints, imgpoints, (640, 480), K2.copy(), D2.copy(), rvecs_3ch2, tvecs_3ch2, flags=flags2
    )
    print("SUCCESS: Tuple of (1, 3)")
except Exception as e:
    print(f"CRASH: Tuple of (1, 3): {e}")

# Tuple of (3,) arrays
rvecs_3ch3 = tuple(r.reshape(3) for r in rvecs1)
tvecs_3ch3 = tuple(t.reshape(3) for t in tvecs1)
try:
    ret2, K2_new, D2_new, rvecs2, tvecs2 = cv2.fisheye.calibrate(
        objpoints, imgpoints, (640, 480), K2.copy(), D2.copy(), rvecs_3ch3, tvecs_3ch3, flags=flags2
    )
    print("SUCCESS: Tuple of (3,)")
except Exception as e:
    print(f"CRASH: Tuple of (3,): {e}")
