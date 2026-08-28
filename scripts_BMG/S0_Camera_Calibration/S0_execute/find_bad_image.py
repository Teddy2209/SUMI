import cv2
import numpy as np
import glob
import os

CHECKERBOARD = (11, 8)
SQUARE_SIZE = 0.015

OUTPUT_DIR = "/media/apicoo-ai/5511010c-3660-41c3-b501-36e739767b6a/SUMI/scripts_BMG/S0_Camera_Calibration/S0_output/intrinsics_matrixes_28082026"
FPC_IMG_DIR = os.path.join(OUTPUT_DIR, "images", "fpc")

images = sorted(glob.glob(os.path.join(FPC_IMG_DIR, '*.png')))

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        
        # Test homography
        H, _ = cv2.findHomography(objp[:, :2], corners2.reshape(-1, 2))
        
        if H is None:
            print(f"-> BAD HOMOGRAPHY (H is None) for {fname}")
            continue
        
        norm_u1 = np.linalg.norm(H[:, 0])
        print(f"{os.path.basename(fname)}: norm_u1 = {norm_u1}")
        if norm_u1 <= 0:
            print(f"-> BAD HOMOGRAPHY (norm_u1=0) for {fname}")
