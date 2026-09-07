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

print("Original Matrix (mtx):")
print(mtx)

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1.0, (w,h))
print("\nNew Camera Matrix (alpha=1.0):")
print(newcameramtx)

cx_diff = newcameramtx[0, 2] - mtx[0, 2]
cy_diff = newcameramtx[1, 2] - mtx[1, 2]

print(f"\nShift in cx: {cx_diff:.2f} pixels")
print(f"Shift in cy: {cy_diff:.2f} pixels")
