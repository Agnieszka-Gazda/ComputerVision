import cv2
import numpy as np

f = open('temporalROI.txt', 'r')
line = f.readline()
roi_start, roi_end = line.split()
roi_start=int(roi_start)
roi_end=int(roi_end)
TP_mean=0;
FN_mean=0;
FP_mean=0;

TP_median=0;
FN_median=0;
FP_median=0;

N=60
BUF=np.zeros((240, 360, N), np.uint8)
iN=0


for i in range(roi_start-N, roi_start, 1):
    I = cv2.imread('input/in%06d.jpg' % i)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    BUF[:,:,iN]=I
    iN=iN+1
    if iN==N:
        iN=0

for i in range(roi_start, roi_end, 5):
    if iN==N:
        iN=0

    median = np.uint8(np.median(BUF, 2))
    mean = np.uint8(BUF.mean(2))

    I=cv2.imread('input/in%06d.jpg' % i)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

    BUF[:, :, iN] = I
    iN=iN+1;

    GT=cv2.imread('groundtruth/gt%06d.png' % i)
    GTB = cv2.cvtColor(GT, cv2.COLOR_BGR2GRAY)

    diff_median = cv2.absdiff(I, median)
    diff_mean=cv2.absdiff(I, mean)

    B_mean = cv2.threshold(diff_mean, 15, 255, cv2.THRESH_BINARY)
    I_mean = cv2.erode(B_mean[1], np.ones((4,4),np.uint8), 2)
    I_mean = cv2.dilate(I_mean, np.ones((8,8), np.uint8), 9)
    I_mean = cv2.erode(I_mean, np.ones((3, 3), np.uint8), 1)
    I_mean=cv2.medianBlur(I_mean, 7)
    cv2.imshow('Final mean', I_mean)
    cv2.waitKey(10)

    B_median = cv2.threshold(diff_median, 15, 255, cv2.THRESH_BINARY)
    I_median = cv2.erode(B_median[1], np.ones((4,4),np.uint8), 2)
    I_median = cv2.dilate(I_median, np.ones((8,8), np.uint8), 9)
    I_median = cv2.erode(I_median, np.ones((3, 3), np.uint8), 1)
    I_median=cv2.medianBlur(I_median, 7)
    cv2.imshow('Final median', I_median)
    cv2.waitKey(10)

    if i>roi_start and i<roi_end:
        # 255 - bialy obiekt
        # 0 - czarny tlo
        TP_M = np.logical_and((I_mean == 255), (GTB == 255))
        TP_S = np.sum(TP_M)
        TP_mean = TP_mean + TP_S

        FP_M = np.logical_and((I_mean == 255), (GTB == 0))
        FP_S = np.sum(FP_M)
        FP_mean = FP_mean + FP_S

        FN_M = np.logical_and((I_mean == 0), (GTB == 255))
        FN_S = np.sum(FN_M)
        FN_mean= FN_mean + FN_S

        TP_M = np.logical_and((I_median == 255), (GTB == 255))
        TP_S = np.sum(TP_M)
        TP_median = TP_median + TP_S

        FP_M = np.logical_and((I_median == 255), (GTB == 0))
        FP_S = np.sum(FP_M)
        FP_median = FP_median + FP_S

        FN_M = np.logical_and((I_median == 0), (GTB == 255))
        FN_S = np.sum(FN_M)
        FN_median= FN_median + FN_S


P = int(TP_mean)/(int(TP_mean)+int(FP_mean))
R = int(TP_mean)/(int(TP_mean)+int(FN_mean))
F1_mean = 2*P*R/(P+R)
print(F1_mean)

P = int(TP_median)/(int(TP_median)+int(FP_median))
R = int(TP_median)/(int(TP_median)+int(FN_median))
F1_median = 2*P*R/(P+R)
print(F1_median)