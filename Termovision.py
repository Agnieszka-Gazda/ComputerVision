import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
cap = cv2.VideoCapture('vid1_IR.avi')
d = 30

def remove_rec(recs):
    for i in recs:
        for j in recs:
            if i != j:
                if (i[0] == j[0] and j[2] == i[2]) or (i[1] == j[1] and i[3] == j[3]):
                    if i in recs:
                        recs.remove(i)
                if i[0] in range(j[0], j[1]) and i[2] in range(j[2], j[3]) or \
                        i[1] in range(j[0], j[1]) and i[3] in range(j[2], j[3]) or \
                        i[0] in range(j[0], j[1]) and i[3] in range(j[2], j[3]) or \
                        i[1] in range(j[0], j[1]) and i[2] in range(j[2], j[3]):
                    if i in recs:
                        recs.remove(i)
    return recs


while (cap.isOpened()):
    ret, frame = cap.read()
    G = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    B = cv2.threshold(G, 40, 255, cv2.THRESH_BINARY)
    I = cv2.erode(B[1], np.ones((12, 12), np.uint8), 10)
    I = cv2.dilate(I, np.ones((12, 12), np.uint8), 10)
    I = cv2.erode(I, np.ones((3, 3), np.uint8), 1)
    I = cv2.medianBlur(I, 7)
    I_ref = copy.deepcopy(I)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # przerwanie petli powcisnieciu klawisza ’q’
        break

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(I)
    recs = []
    new_recs = list()
    for pi in range(1, stats.shape[0]):
        if stats.shape[0] > 0:
            x = stats[pi, 0]
            y = stats[pi, 1]
            w = stats[pi, 2]
            h = stats[pi, 3]
            x1 = x
            x2 = x + w
            y1 = y
            y2 = y + h

            recs.append([x1, x2, y1, y2])

            for i in recs:
                if (i[1]>(x1-d) and i[1]<(x2+d) and i[3]>(y1-d) and i[3]<(y2+d)) or \
                        (i[0] > (x1-d) and i[0] < (x2+d) and i[2] > (y1-d) and i[2] < (y2+d)) or \
                        (i[1]>(x1-d) and i[1]<(x2+d) and i[2]>(y1-d) and i[2]<(y2+d)) or \
                        (i[0]>(x1-d) and i[0]<(x2+d) and i[3]>(y1-d) and i[3]<(y2+d)):

                    x1 = min([x1, i[0], x2, i[1]])
                    x2 = max([x2, i[1], x1, i[0]])
                    y1 = min([y1, i[2], y2, i[3]])
                    y2 = max([y2, i[3], y1, i[2]])

                    new_recs.append([x1, x2, y1, y2])

            recs = recs + new_recs

            recs = remove_rec(recs)
            recs = remove_rec(recs)
            recs = remove_rec(recs)

    for i in recs:
        cv2.rectangle(I, (i[0], i[2]), (i[1], i[3]), (200, 0, 0), 2)

    cv2.imshow('', I)
    cv2.waitKey(10)

cap.release()
