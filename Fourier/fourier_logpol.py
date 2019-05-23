import matplotlib.pyplot as plt
import cv2
import numpy as np


def hanning2D(n):
    h = np.hanning(n)
    return np.sqrt(np.outer(h,h))


def highpassFilter(size):
    rows = np.cos(np.pi*np.matrix([-0.5 + x/(size[0]-1) for x in range (size[0])]))
    cols = np.cos(np.pi*np.matrix([-0.5 + x/(size[1]-1) for x in range(size[1])]))
    X = np.outer(rows,cols)
    return (1.0 - X)*(2.0 - X)


pattern_raw = cv2.imread('obrazy_Mellin/domek_s60.pgm')
pattern_raw = cv2.cvtColor(pattern_raw, cv2.COLOR_RGB2GRAY)

I = cv2.imread('obrazy_Mellin/domek_r30.pgm')
I = cv2.imread('obrazy_Mellin/domek_r60.pgm')
I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)

pattern_hanning = hanning2D(pattern_raw.shape[0])
pattern = pattern_raw*pattern_hanning

#uzupelnienie zerami
small_pattern = pattern
pattern = np.zeros(I.shape)
pattern[0:small_pattern.shape[0], 0:small_pattern.shape[1]] = small_pattern


pattern_fft = np.fft.fft2(pattern)
image_fft = np.fft.fft2(I)
image_fft = np.fft.fftshift(image_fft)
pattern_fft = np.fft.fftshift(pattern_fft)

filter_pattern = highpassFilter(pattern_fft.shape)
filter_image = highpassFilter(image_fft.shape)


filtered_pattern = filter_pattern*np.abs(pattern_fft)
filtered_image = filter_image*np.abs(image_fft)


#logPolar
size = filtered_image.shape
M = filtered_image.shape[0]/np.log(filtered_image.shape[0]//2)
center = (size[0]/2, size[1]/2)
logPolar_image = cv2.logPolar(filtered_image, center, M, flags= cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)

size = filtered_pattern.shape
M = filtered_pattern.shape[0]/np.log(filtered_pattern.shape[0]//2)
center = (size[0]/2, size[1]/2)
logPolar_pattern = cv2.logPolar(filtered_pattern, center, M, flags= cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)


pattern_fft = np.fft.fft2(logPolar_pattern)
image_fft = np.fft.fft2(logPolar_image)
R = np.conj(pattern_fft)*image_fft/np.abs(np.conj(pattern_fft)*image_fft)
cor = np.abs(np.fft.ifft2(R))


angle_coordinates, logr_coordinates = np.unravel_index(np.argmax(cor), cor.shape)

print(logPolar_image.shape)
print(logPolar_pattern.shape)

size_logr = logPolar_pattern.shape[0]

if logr_coordinates >  size_logr//2:
    wyk1 = size_logr-logr_coordinates
else:
    wyk1 = -logr_coordinates

angle_size = cor.shape[0]
A =(angle_coordinates*360)/angle_size
ang1 = -A
ang2 = 180-A

scale = np.exp(wyk1/M)   #gdzie M to parametr funkcji cv2. logPolar, a wykl wyliczamy jako:

#uzupelnienie zerami
small_pattern = pattern_raw
im = np.zeros(I.shape)
im_sh = im.shape
im[int(im_sh[0]/2-small_pattern.shape[0]/2):int(im_sh[0]/2+small_pattern.shape[0]/2),
   int(im_sh[1]/2-small_pattern.shape[1]/2):int(im_sh[1]/2+small_pattern.shape[1]/2)] = small_pattern


centerTrans = (im.shape[0] / 2 - 0.5, im.shape[1]/ 2 - 0.5)
matrix_trans = cv2.getRotationMatrix2D(centerTrans, ang1, scale) # gdzie srodekTrans mozna wyliczyc jako:
rotated_scaled_im1 = cv2.warpAffine(im, matrix_trans, im.shape)
# im to obraz wzorca uzupelniony zerami, ale ze wzorcem umieszczonym na srodku, a nie w lewym, gornym rogu!
rotated_scaled_im2 = cv2.warpAffine(im, matrix_trans, im.shape)


pattern_fft_1 = np.fft.fft2(rotated_scaled_im1)
pattern_fft_2 = np.fft.fft2(rotated_scaled_im2)
J_fft = np.fft.fft2(I)

R1 = np.conj(pattern_fft_1) * J_fft
R1 = R1 / np.abs(R1)
cor1 = np.abs(np.fft.ifft2(R1))

R2 = np.conj(pattern_fft_2) * J_fft
R2 = R2 / np.abs(R2)
cor2 = np.abs(np.fft.ifft2(R2))

if (np.amax(cor1)>np.amax(cor2)):
    cor_new=cor1
    pattern_new = rotated_scaled_im1
else:
    cor_new=cor2
    pattern_new = rotated_scaled_im2

y, x = np.unravel_index(np.argmax(cor_new), cor_new.shape)
if x > I.shape[0] - 5:
    x = x - I.shape[0]

print(x, y)
matrix_trans = np.float32([[1, 0, x], [0, 1, y]]) # gdzie dx, dy - wektor przesuniecia
final_pattern = cv2.warpAffine(pattern_new, matrix_trans, (I.shape[1], I.shape[0]))

plt.figure()
plt.imshow(I)
plt.plot(x, y, '*', color='r')
plt.gray()
plt.figure()
plt.imshow(pattern_new)
plt.gray()
plt.show()

