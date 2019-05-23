import matplotlib.pyplot as plt
import cv2
from Hausdorf import *
from scipy.spatial import distance
import numpy as np
import os

I = cv2.imread('Aegeansea.jpg')
I2 = ~I
I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2HSV)
threshH, im_bwH = cv2.threshold(I2[:, :, 0], 60, 255, cv2.THRESH_BINARY)
threshS, im_bwS = cv2.threshold(I2[:, :, 1], 30, 255, cv2.THRESH_BINARY)
im_bw = im_bwH*im_bwS

contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = [el for el in contours if el.shape[0] > 15 and el.shape[0] < 3000]

# rysowanie konturow
Im = im_bw - im_bw
Im = ~Im
for i, val in enumerate(contours):
    cv2.drawContours(Im, contours, i, (0, 0, 0))

'''
plt.imshow(Im)
plt.show()
'''

I_ref= cv2.imread('imgs/c_astipalea.bmp')
I_ref = 255-I_ref
gray = cv2.cvtColor(I_ref, cv2.COLOR_RGB2GRAY)
thresh, im_bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
contours_ref, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
xy_ref, x, y, cx, cy = normalize(contours_ref)

Hausdorff_min=1000000000

for i in range(0, len(contours)):
    xy_curr, x, y, cx, cy = normalize(contours, i)
    currentHausdorf = Hausdorff(xy_ref, xy_curr)
    if currentHausdorf < Hausdorff_min:
        Hausdorff_min = currentHausdorf
        img_min = i
        cx_min = cx
        cy_min = cy
    print(i)
    cv2.putText(I, str(i), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128))

print(cx_min, cy_min)
cv2.putText(I, 'ASTIPALEA', (int(cx_min), int(cy_min)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))


height, width = I.shape[:2]
scale = 0.5
I = cv2.resize(I, (int(scale*height), int(scale*width)))
cv2.imshow('', I)
cv2.waitKey(0)

