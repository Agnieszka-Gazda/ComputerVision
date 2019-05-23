import cv2
import numpy as np
import matplotlib.pyplot as plt

I = cv2.imread('I.jpg')
J = cv2.imread('J.jpg')

I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
J = cv2.cvtColor(J, cv2.COLOR_BGR2GRAY)
diff = cv2.absdiff(I, J)

'''
cv2.imshow("roznica", diff)
cv2.waitKey(100)
'''

W2 = 1
dX = dY = 1

u = np.zeros((I.shape[0], I.shape[1]))
v = np.zeros((I.shape[0], I.shape[1]))
print(I.shape)
for j in range(2*W2, I.shape[0]-2 * W2):
    for i in range(2* W2, I.shape[1]-2 * W2):
        IO = np.float32(I[j-W2:j + W2 + 1, i - W2:i + W2 + 1])
        min_dist = 10000
        for j2 in range(j-dX, j+dX+1):
            for i2 in range(i-dY, i + dY + 1):
                JO = np.float32(J[j2 - W2:j2 + W2 + 1, i2-W2: i2 + W2 + 1])

                dist = np.sum(np.sqrt((np.square(JO - IO))))
                if dist < min_dist:
                    min_dist = dist
                    u[j, i] = i2-i
                    v[j, i] = j2-j

plot = plt.figure()
plt.quiver(v, u)
plt.gca().invert_yaxis()
plt.show(plot)
cv2.waitKey(100)
