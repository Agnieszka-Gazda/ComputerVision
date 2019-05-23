import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

I = cv2.imread('trybik.jpg')
I = 255-I
gray = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
thresh, im_bw = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
im_cont = ~(im_bw-im_bw)
cv2.drawContours(im_cont, contours, -1, (155, 255, 0))


sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

grad = cv2.sqrt(sobelx*sobelx+sobely*sobely)
max = np.amax(grad)
grad = grad/max

orientation_init = np.arctan2(sobely, sobelx)
orientation_init = np.rad2deg(orientation_init)

M = cv2.moments(im_bw, 1)
cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])

print(cx, cy)

Rtable = [[] for i in range(360)]

'''
plt.figure()
plt.imshow(im_cont)
plt.gray()
plt.show()
'''

for cont in contours:
    for i in cont:
        dist = distance.euclidean(i[0], [cx, cy])
        ang = np.arctan2(i[0][0], i[0][1])
        print(ang)
        ang_grad = int(orientation_init[i[0][0], i[0][1]])
        Rtable[ang_grad].append([dist, ang])


I = cv2.imread('trybiki2.jpg')
gray = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)

print(I.shape)
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

grad = cv2.sqrt(sobelx*sobelx+sobely*sobely)
max = np.amax(grad)
grad = grad/max
orientation = np.arctan2(sobely, sobelx)
orientation = np.rad2deg(orientation)


acc = np.zeros((orientation.shape[0], orientation.shape[1]))
for x in range(I.shape[1]):
    for y in range(I.shape[0]):
        if grad[y][x] > 0.5:
            for val in Rtable[int(orientation[y][x])]:
                fi = val[1]
                r = val[0]
                x1 = -r*np.cos(fi) + x
                y1 = -r*np.sin(fi)+y
                acc[int(x1)][int(y1)] += 1


max_acc = 0
m_x = 0
m_y = 0

points = np.where(acc.max() == acc)
print(points)
plt.figure()
plt.imshow(I)
plt.plot(points[0], points[1], '*', color='r')
plt.show()


