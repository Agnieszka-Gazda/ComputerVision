import numpy as np
import cv2
import matplotlib.pyplot as plt


def of(I, J, u0, v0, W2=1, dY=1, dX=1):

    u = np.zeros((I.shape[0], I.shape[1]))
    v = np.zeros((I.shape[0], I.shape[1]))
    print(I.shape)
    for j in range(2 + W2, I.shape[0] - W2):
        for i in range(2 + W2, I.shape[1] - W2):
            IO = np.float32(I[j - W2:j + W2 + 1, i - W2:i + W2 + 1])
            min_dist = 10000
            for j2 in range(j - dX, j + dX + 1):
                for i2 in range(i - dY, i + dY + 1):
                    j2 = int(j2 + v0[j, i])
                    i2 = int(i2 + u0[j, i])
                    if j2<(J.shape[0]-W2) and i2<(J.shape[1]-W2) and i2>W2 and j2>W2:
                        JO = np.float32(J[j2 - W2:j2 + W2 + 1, i2 - W2: i2 + W2 + 1])
                        dist = np.sum(np.sqrt((np.square(JO - IO))))
                        if dist < min_dist:
                            min_dist = dist
                            u[j, i] = i2 - (i + u0[j, i])
                            v[j, i] = j2 - (j + v0[j, i])
    return u, v


def pyramid(im, max_scale):
    images=[im]
    for k in range(1, max_scale):
        images.append(cv2.resize(images[k-1], (0, 0), fx=0.5, fy=0.5))
    return images


I = cv2.imread('I.jpg')
J = cv2.imread('J.jpg')

I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
J = cv2.cvtColor(J, cv2.COLOR_BGR2GRAY)
diff = cv2.absdiff(I, J)

max_scale = 3
IP = pyramid(I, max_scale)
JP = pyramid(J, max_scale)

u0 = np.zeros(IP[-1].shape, np.float32)
v0 = np.zeros(JP[-1].shape, np.float32)

u, v = of(IP[-1], JP[-1], u0, v0)
for i in range(1, max_scale):
    v=cv2.resize(v, (0,0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    u=cv2.resize(u, (0,0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    u, v = of(IP[-i-1], JP[-1-i], u, v)


cv2.imshow("roznica", diff)
cv2.waitKey(100)

plot = plt.figure()
plt.quiver(u, v)
plt.gca().invert_yaxis()
plt.show(plot)
cv2.waitKey(100)


