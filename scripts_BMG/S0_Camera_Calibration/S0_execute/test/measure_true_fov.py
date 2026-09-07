import cv2
import numpy as np
import json
import math
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "S0_output", "intrinsics_matrixes_28082026")
json_path = os.path.join(OUTPUT_DIR, "fpccamera_pinhole_intrinsics.json")

if not os.path.exists(json_path):
    print("File không tồn tại.")
    exit(1)

with open(json_path, 'r') as f:
    data = json.load(f)

K = np.array(data["camera_matrix"], dtype=np.float64)
D = np.array(data["dist_coeffs"], dtype=np.float64)
w, h = data["image_size"]

print(f"K:\n{K}")
print(f"D:\n{D}")
print(f"Image Size: {w}x{h}")

# Lấy 4 điểm ở 4 góc của khung hình
points = np.array([
    [[0.0, 0.0]],
    [[float(w - 1), 0.0]],
    [[0.0, float(h - 1)]],
    [[float(w - 1), float(h - 1)]]
], dtype=np.float32)

# cv2.undistortPoints trả về tọa độ chuẩn hóa (normalized coordinates) x, y 
# (nghĩa là đã trừ cx, cy và chia cho fx, fy)
undistorted_pts = cv2.undistortPoints(points, K, D)

for i, pt in enumerate([(0,0), (w-1,0), (0,h-1), (w-1,h-1)]):
    x, y = undistorted_pts[i][0]
    r = math.sqrt(x**2 + y**2)
    theta = math.degrees(math.atan(r))
    dfov = 2 * theta
    print(f"Góc {pt}: x={x:.4f}, y={y:.4f} -> r={r:.4f} -> theta={theta:.2f}° -> DFOV ước tính={dfov:.2f}°")
    
# Tính HFOV (chỉ lấy điểm ở mép trái/phải nằm trên trục ngang cy)
cy = K[1, 2]
h_points = np.array([
    [[0.0, cy]],
    [[float(w - 1), cy]]
], dtype=np.float32)

undistorted_h_pts = cv2.undistortPoints(h_points, K, D)
x_h, y_h = undistorted_h_pts[0][0]
r_h = math.sqrt(x_h**2 + y_h**2)
theta_h = math.degrees(math.atan(r_h))
hfov = 2 * theta_h
print(f"HFOV ước tính (điểm giữa viền ngang): theta={theta_h:.2f}° -> HFOV={hfov:.2f}°")
