import cv2
import numpy as np

print("CALIB_USE_INTRINSIC_GUESS:", cv2.fisheye.CALIB_USE_INTRINSIC_GUESS)

# Let's create a minimal test to see if K is updated or ignored
K = np.array([[300.0, 0, 320.0], [0, 300.0, 240.0], [0, 0, 1.0]], dtype=np.float64)
D = np.zeros((4, 1), dtype=np.float64)

objp = np.zeros((1, 88, 3), np.float64)
imgp = np.zeros((1, 88, 2), np.float64)

flags = cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
try:
    cv2.fisheye.calibrate([objp], [imgp], (640, 480), K, D, flags=flags)
except Exception as e:
    print("Crash with positional K, D:", e)

try:
    cv2.fisheye.calibrate([objp], [imgp], (640, 480), K=K, D=D, flags=flags)
except Exception as e:
    print("Crash with kwargs K, D:", e)
