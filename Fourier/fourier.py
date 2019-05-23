import matplotlib.pyplot as plt
import cv2
import numpy as np

I = cv2.imread('obrazy_Mellin/wzor.pgm')
small_pattern = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
I = cv2.imread('obrazy_Mellin/domek_r0.pgm')
I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)

pattern = np.zeros(I.shape)
pattern[0:small_pattern.shape[0], 0:small_pattern.shape[1]] = small_pattern

pattern_fft = np.fft.fft2(pattern)
image_fft = np.fft.fft2(I)

R = np.conj(pattern_fft)*image_fft/np.abs(np.conj(pattern_fft)*image_fft)
#R = np.conj(pattern_fft)*image_fft # nie znormalizowany

cor = np.abs(np.fft.ifft2(R))
y, x = np.unravel_index( np.argmax(cor), cor.shape)

translation_matrix = np.float32([[1,0,x],[0,1,y]])  # gdzie dx, dy - wektor przesuniecia
translated_pattern = cv2.warpAffine(pattern, translation_matrix, (pattern.shape[1], pattern.shape[0]))

print(x, y)
plt.imshow(I)
plt.plot(x, y, '*', color='r')
plt.gray()


plt.figure()
plt.imshow(translated_pattern)
plt.gray()
plt.show()