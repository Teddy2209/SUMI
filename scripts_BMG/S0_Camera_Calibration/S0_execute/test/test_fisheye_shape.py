import cv2
import numpy as np

CHECKERBOARD = (9, 6)
SQUARE_SIZE = 0.015

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# create proper imgpoints using projectPoints
K_true = np.array([[300, 0, 320], [0, 300, 240], [0, 0, 1]], dtype=np.float64)
D_true = np.array([[-0.1], [0.01], [-0.001], [0.0]], dtype=np.float64)

objpoints = []
imgpoints = []

# Generate 10 random views
np.random.seed(42)
for i in range(10):
    rvec = np.random.randn(3, 1) * 0.5
    tvec = np.array([[0], [0], [1.0]]) + np.random.randn(3, 1) * 0.2
    
    imgp, _ = cv2.projectPoints(objp, rvec, tvec, K_true, D_true)
    imgp += np.random.randn(*imgp.shape) * 0.1 # add noise
    
    objpoints.append(objp)
    imgpoints.append(imgp.reshape(-1, 2))

# Try 1: (N, 1, 3)
try:
    print("Testing (N, 1, 3)")
    obj_n13 = [o.reshape(-1, 1, 3).astype(np.float64) for o in objpoints]
    img_n12 = [i.reshape(-1, 1, 2).astype(np.float64) for i in imgpoints]
    K = np.zeros((3, 3), np.float64)
    D = np.zeros((4, 1), np.float64)
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    cv2.fisheye.calibrate(obj_n13, img_n12, (640, 480), K, D, flags=flags, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
    print("(N, 1, 3) worked!")
except Exception as e:
    print("Error:", e)

# Try 2: (1, N, 3)
try:
    print("Testing (1, N, 3)")
    obj_1n3 = [o.reshape(1, -1, 3).astype(np.float64) for o in objpoints]
    img_1n2 = [i.reshape(1, -1, 2).astype(np.float64) for i in imgpoints]
    K = np.zeros((3, 3), np.float64)
    D = np.zeros((4, 1), np.float64)
    cv2.fisheye.calibrate(obj_1n3, img_1n2, (640, 480), K, D, flags=flags, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
    print("(1, N, 3) worked!")
except Exception as e:
    print("Error:", e)
