import cv2
import numpy as np

f = open('temporalROI.txt', 'r')
line = f.readline()
roi_start, roi_end = line.split()
roi_start=int(roi_start)
roi_end=int(roi_end)
TP=0;
FN=0;
FP=0;

I_prev=cv2.imread('input/in%06d.jpg' % 550)
print('input/in%06d.jpg' % 550)
I_prev = cv2.cvtColor(I_prev, cv2.COLOR_BGR2GRAY)

for i in range(550, 1100, 2):
    I_VIS = cv2.imread('input/in%06d.jpg' % i)
    I=cv2.imread('input/in%06d.jpg' % i)
    GT=cv2.imread('groundtruth/gt%06d.png' % i)
    GTB = cv2.cvtColor(GT, cv2.COLOR_BGR2GRAY)

    I=cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    diff=cv2.absdiff(I_prev, I)
    cv2.imshow("I", diff)
    cv2.waitKey(10)
    I_prev=I

    B = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
    cv2.imshow('Bin', B[1])
    cv2.waitKey(10)

    I = cv2.erode(B[1], np.ones((4,4),np.uint8), 2)
    cv2.imshow('Erozja', I)
    cv2.waitKey(10)

    I = cv2.dilate(I, np.ones((8,8), np.uint8), 9)
    I = cv2.erode(I, np.ones((3, 3), np.uint8), 1)
    I=cv2.medianBlur(I, 7)

    cv2.imshow('Final', I)
    cv2.waitKey(10)

    if i>roi_start and i<roi_end:
        # 255 - bialy oviekt
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

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(I)
    cv2.imshow("Labels", np.uint8(labels/stats.shape[0]*255))
    if stats.shape[0]>1:

        tab=stats[1:,4]
        pi=np.argmax(tab)
        pi=pi+1

        cv2.rectangle(I_VIS,(stats[pi,0],stats[pi,1]),(stats[pi,0]+stats[pi,2],stats[pi,1]+stats[pi,3]),(255,0,0),2)
        cv2.putText(I_VIS, "%f" % stats[pi, 4], (stats[pi, 0], stats[pi, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 0))
        cv2.putText(I_VIS, "%d" % pi, (np.int(centroids[pi, 0]), np.int(centroids[pi, 1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0))
    cv2.imshow('sss', I_VIS)


P = int(TP)/(int(TP)+int(FP))
R = int(TP)/(int(TP)+int(FN))
F1 = 2*P*R/(P+R)
print(F1)
