import cv2
import numpy as np
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "S0_output", "intrinsics_matrixes_28082026")
json_path = os.path.join(OUTPUT_DIR, "fpccamera_pinhole_intrinsics.json")

with open(json_path, 'r') as f:
    data = json.load(f)

mtx = np.array(data["camera_matrix"], dtype=np.float64)
dist = np.array(data["dist_coeffs"], dtype=np.float64)
w, h = data["image_size"]

# Load an example image (assuming there's one in FPC_IMG_DIR)
FPC_IMG_DIR = os.path.join(OUTPUT_DIR, "images", "fpc")
img_files = sorted(os.listdir(FPC_IMG_DIR))
if not img_files:
    exit()
img_path = os.path.join(FPC_IMG_DIR, img_files[0])
img = cv2.imread(img_path)

# Method 1: alpha=1.0 (What user sees currently)
newcameramtx_alpha1, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1.0, (w,h))
dst_alpha1 = cv2.undistort(img, mtx, dist, None, newcameramtx_alpha1)
cv2.imwrite("test_alpha1.png", dst_alpha1)

# Method 2: alpha=0.0
newcameramtx_alpha0, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0.0, (w,h))
dst_alpha0 = cv2.undistort(img, mtx, dist, None, newcameramtx_alpha0)
cv2.imwrite("test_alpha0.png", dst_alpha0)

# Method 3: NO newcameramtx (just use original mtx)
dst_original_mtx = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite("test_original_mtx.png", dst_original_mtx)
