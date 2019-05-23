import cv2
import numpy as np
import matplotlib.pyplot as plt

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
objpoints = []
imgpoints = []

for i in range(1,13):
    img = cv2.imread('/home/agnieszka/PycharmProjects/optflow/images_left/left%02d.jpg' % i)
    # konwersja do odcieni szarosci
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # wyszukiwanie naroznikow na planszyj
    ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
    # jesli znaleniono na obrazie punkty
    if ret == True:
        #dolaczenie wspolrzednych 3D
        objpoints.append(objp)
        # poprawa lokalizacji punktow (podpiskelowo)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        # dolaczenie poprawionych punktow
        imgpoints.append(corners2)
        # wizualizacja wykrytych naroznikow

        cv2.drawChessboardCorners(img, (7,6), corners2, ret)
        cv2.imshow("Corners", img)
        cv2.waitKey(0)


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(ret, mtx, dist, rvecs, tvecs)
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png', dst)
