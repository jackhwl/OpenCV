import cv2
import sys
import numpy as np

imgpath = sys.argv[1]
img = cv2.imread(imgpath)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create() #cv2.xfeatures2d.SIFT_create()
keypoints, descriptor = sift.detectAndCompute(gray,None)

#img = cv2.drawKeypoints(image=img, outImage=img, keypoints = keypoints, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINT, color = (51, 163, 236))

img2 = img.copy()
for marker in keypoints:
	img2 = cv2.drawMarker(img2, tuple(int(i) for i in marker.pt), color=(51, 163, 236))

cv2.imshow('sift_keypoints', img2)
while (True):
    if cv2.waitKey(int(1000 / 12)) & 0xff == ord("q"):
        break
cv2.destroyAllWindows()