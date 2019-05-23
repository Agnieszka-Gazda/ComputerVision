import cv2
import numpy as np

f = open('temporalROI.txt', 'r')
line = f.readline()
roi_start, roi_end = line.split()
roi_start=int(roi_start)
roi_end=int(roi_end)

BGS = cv2.createBackgroundSubtractorMOG2(100, varThreshold=36)
I = cv2.imread('input/in%06d.jpg' % roi_start)
I2 = cv2.imread('input/in%06d.jpg' % (roi_start+1))
mask = cv2.absdiff(I2,I)

def BG_detection(roi_start, roi_end):

    TP = 0;
    FN = 0;
    FP = 0;

    for i in range(roi_start, roi_end, 5):

        I = cv2.imread('input/in%06d.jpg' % i)
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        B = cv2.threshold(I, 100, 255, cv2.THRESH_BINARY)
        I = cv2.erode(B[1], np.ones((4, 4), np.uint8), 2)
        #I = cv2.dilate(I, np.ones((8, 8), np.uint8), 9)
        I = cv2.erode(I, np.ones((3, 3), np.uint8), 1)
        I = cv2.medianBlur(I, 7)

        BG = BGS.apply(I, mask)
        I=BG

        GT = cv2.imread('groundtruth/gt%06d.png' % i)
        GTB = cv2.cvtColor(GT, cv2.COLOR_BGR2GRAY)
        GTB = cv2.threshold(GTB, 15, 255, cv2.THRESH_BINARY)
        GTB = GTB[1]


        cv2.imshow('Final ', I)
        cv2.waitKey(10)

        if i > roi_start and i < roi_end:
            # 255 - bialy obiekt
            # 0 - czarny tlo
            TP_M = np.logical_and((I == 255), (GTB == 255))
            TP_S = np.sum(TP_M)
            TP = TP + TP_S

            FP_M = np.logical_and((I == 255), (GTB == 0))
            FP_S = np.sum(FP_M)
            FP = FP + FP_S

            FN_M = np.logical_and((I == 0), (GTB == 255))
            FN_S = np.sum(FN_M)
            FN = FN + FN_S

    P = int(TP) / (int(TP) + int(FP))
    R = int(TP) / (int(TP) + int(FN))
    F1 = 2 * P * R / (P + R)
    return F1

print(BG_detection(roi_start, roi_end))


