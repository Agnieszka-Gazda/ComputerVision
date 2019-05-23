import matplotlib.pyplot as plt
import cv2
from scipy.spatial import distance
import numpy as np
import os

def normalize(c, i=0):
    c = c[i]
    xy = c[:, 0, :]
    x = c[:, 0, 0]
    y = c[:, 0, 1]
    M = cv2.moments(c, 1)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    x = x - cx
    y = y - cy

    xy[:, 0] = x
    xy[:, 1] = y

    dist = []
    for i in xy:
        for j in xy:
            dist.append(distance.euclidean(i, j))

    max_dist = max(dist)

    x = x / max_dist
    y = y / max_dist

    xy = xy.astype(np.float)
    xy[:, 0] = x
    xy[:, 1] = y

    return xy, x, y, cx, cy


def dH(xy_fig1, xy_fig2):
    dst = []
    for i in xy_fig1:
        dst_point = []
        for j in xy_fig2:
            dst_point.append(distance.euclidean(i, j))
        max_dst_point = min(dst_point)
        dst.append(max_dst_point)
    max_dst = max(dst)
    return max_dst


def Hausdorff(xy_fig1, xy_fig2):
    dH1 = dH(xy_fig1, xy_fig2)
    dH2 = dH(xy_fig2, xy_fig1)
    return max(dH1, dH2)

I = cv2.imread('ithaca_q.bmp')
I = 255-I
gray = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
thresh, im_bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(I, contours, 0, (0, 255, 0))
xy, x, y, cx, cy = normalize(contours)


imgs = os.listdir('imgs')
I_min = cv2.imread('imgs/{}'.format(imgs[0]))
#I_min = cv2.imread('imgs/i_ithaca.bmp')
I_min = 255-I_min
gray = cv2.cvtColor(I_min, cv2.COLOR_RGB2GRAY)
thresh, im_bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
xy_img, x, y, cx, cy = normalize(contours)


Hausdorff_min = Hausdorff(xy, xy_img)

img_min = imgs[0]


for img in os.listdir('imgs'):
    I = cv2.imread('imgs/{}'.format(img))
    I = ~I
    gray = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    thresh, im_bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    xy_img, x, y, cx, cy = normalize(contours)
    currentHausdorf = Hausdorff(xy, xy_img)
    print(img, currentHausdorf)
    if currentHausdorf < Hausdorff_min:
        Hausdorff_min = currentHausdorf
        img_min = img


print(img_min, Hausdorff_min)
