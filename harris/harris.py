import matplotlib.pyplot as plt
import cv2
import scipy.ndimage.filters as filters
import numpy as np
import pm

def find_max(image, size, threshold) : # size - rozmiar maski filtra maksymalnego
    data_max = filters.maximum_filter(image, size)
    maxima = (image == data_max)
    diff = image > threshold
    maxima[diff == 0] = 0
    return np.nonzero(maxima)


def Harris(I, mask_size=5):
    Ix = cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=mask_size)
    Iy = cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=mask_size)
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    Ixx_blur = cv2.GaussianBlur(Ixx, (mask_size, mask_size), 0)
    Iyy_blur = cv2.GaussianBlur(Iyy, (mask_size, mask_size), 0)
    Ixy_blur = cv2.GaussianBlur(Ixy, (mask_size, mask_size), 0)

    det = Ixx_blur * Iyy_blur - Ixy_blur*Ixy_blur
    trace = Ixx_blur + Iyy_blur
    k = 0.05
    H = det - k * trace * trace
    print(H.shape)
    H = cv2.normalize(H, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return H


def show(I, max_I):
    plt.figure()
    plt.imshow(I)
    plt.plot(max_I[1], max_I[0], '*', color='r')
    #plt.show()


def neigh(I, point, size):
    x = point[0]
    y = point[1]
    return I[x-int(size/2):x+int(size/2), y-int(size/2):y+int(size/2)]


def descr(I, pts, size):
    X = I.shape[0]
    Y = I.shape[1]
    pts =list(filter(lambda pt: pt[0] >= size and pt[0] < Y - size and pt[1] >= size and pt[1] < X - size,
                     zip(pts[0], pts[1])))

    l_coordinates = pts
    l_surroundings = []

    for i in pts:
        l_surroundings.append(neigh(I, i, size).flatten())

    return list(zip(l_surroundings, l_coordinates))


def compare(points1, points2, n):
    dissim = []
    for i in points1:
        for j in points2:
            dissim_measure = np.sum(np.square(i[0]-j[0]))
            dissim.append([dissim_measure, i[1], j[1]])

    dissim.sort(key=lambda x: x[0])
    return dissim[0:n]


I = cv2.imread('fontanna1.jpg')
J = cv2.imread('fontanna2.jpg')
I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
J = cv2.cvtColor(J, cv2.COLOR_RGB2GRAY)
I = cv2.normalize(I, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
J = cv2.normalize(J, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

mask_size = 7
H_I = Harris(I, mask_size)
H_I = H_I/np.max(H_I)
max_I = find_max(H_I, mask_size, 0.45)

H_J = Harris(J, mask_size)
H_j = H_J/np.max(H_J)
max_J = find_max(H_J, mask_size, 0.45)

size = 15
points1 = descr(I, max_I, size)
points2 = descr(J, max_J, size)

max_p = compare(points1, points2, 20)
points = []
for i in max_p:
    points.append(([i[1][0], i[1][1]], [i[2][0], i[2][1]]))

print(points)
pm.plot_matches(I, J, points)
plt.gray()


show(I, max_I)
plt.gray()
show(J, max_J)
plt.gray()
plt.show()
