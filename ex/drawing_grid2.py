import numpy as np
import cv2

canvas = np.zeros((300, 300, 3), dtype = "uint8")

red = (0, 0, 255)
lag = 100

for row in range(0, 300 // lag):
    for col in range(0, 300 // lag):
        if row%2 != col%2:
            print(row, col, lag*col, lag*row, lag*col+lag, lag*row+lag)
            cv2.rectangle(canvas, (lag*col, lag*row), (lag * col + lag, lag * row + lag), red, -1)

(centerX, centerY) = (canvas.shape[1] // 2, canvas.shape[0] // 2)
#green = (0, 255, 0)

#cv2.circle(canvas, (centerX, centerY), 50, green, -1)

cv2.imshow("Canvas", canvas)

cv2.waitKey(0)