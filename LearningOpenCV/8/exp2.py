import numpy as np
import cv2

cap = cv2.VideoCapture(0)

mog = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()
    fgmask = mog.apply(frame)
    cv2.imshow('frame',fgmask)
    if cv2.waitKey(int(3000)) & 0xff:
        break

cap.release()
cv2.destroyAllWindows()