import cv2
import numpy as np
import matplotlib.pyplot as plt

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)


objpoints = []
imgpoints_left = []
imgpoints_right = []

for i in range(1, 13):
    img_left = cv2.imread('images_left/left%02d.jpg' % i)
    # konwersja do odcieni szarosci
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, (7, 6), None)

    img_rigth = cv2.imread('images_right/right%02d.jpg' % i)
    gray_right = cv2.cvtColor(img_rigth, cv2.COLOR_BGR2GRAY)
    ret_right, corners_rigth = cv2.findChessboardCorners(gray_right, (7, 6), None)

    if ret_left == True and ret_right == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray_left,corners_left, (11,11), (-1,-1), criteria)
        imgpoints_left.append(corners2)

        corners2 = cv2.cornerSubPix(gray_right, corners_rigth, (11, 11), (-1, -1), criteria)
        imgpoints_right.append(corners2)



ret_L, mtx_L, dist_L, rvecs_L, tvecs_L = cv2.calibrateCamera(objpoints, imgpoints_left, gray_left.shape[::-1], None, None)
ret_R, mtx_R, dist_R, rvecs_R, tvecs_R = cv2.calibrateCamera(objpoints, imgpoints_right, gray_right.shape[::-1], None, None)

retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right,mtx_L,dist_L, mtx_R,dist_R, gray_left.shape[::-1])

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2,
                                                                  distCoeffs2, gray_right.shape[::-1], R, T)

map1_L, map2_L = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, gray_left.shape[::-1], cv2.CV_16SC2)
map1_R, map2_R = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, gray_left.shape[::-1], cv2.CV_16SC2)

dst_L = cv2.remap(img_left, map1_L, map2_L, cv2.INTER_LINEAR)
dst_R = cv2.remap(img_rigth, map1_R, map2_R, cv2.INTER_LINEAR)

N, XX, YY = dst_L.shape[::-1]
visRectify = np.zeros((YY,XX * 2,N),np.uint8)
visRectify[:, 0:640: ,:] = dst_L
visRectify[:,640:1280:,:] = dst_R
# Wyrysowanie poziomych linii
for y in range(0, 480, 10):
    cv2.line(visRectify, (0, y), (1280, y), (255, 0, 0))

cv2.imshow('visRectify', visRectify)
cv2.waitKey(0)
