import cv2
import numpy as np


aloeL = cv2.imread('aloes/aloeL.jpg')
aloeR = cv2.imread('aloes/aloeR.jpg')
imgL = cv2.cvtColor(aloeL, cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(aloeR, cv2.COLOR_BGR2GRAY)

min_disp = 16
num_disp = 112 - min_disp
window_size = 17

stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=window_size)

disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
disp_map = (disp - min_disp) / num_disp

cv2.imshow('disparity', disp_map)
cv2.waitKey()
cv2.destroyAllWindows()
