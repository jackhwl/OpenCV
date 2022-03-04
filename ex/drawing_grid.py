import numpy as np
import cv2

canvas = np.zeros((300, 300, 3), dtype = "uint8")

red = (0, 0, 255)
lag = 50

for row in range(0, 300 // lag):
    for col in range(0, (300 + lag) // (2 * lag) ):
        x = (lag if row % 2 == 0 else 0) + col * lag * 2
        y = row * lag
        print(x, y, x+lag, y+lag) 
        if x < 300 and y < 300:
            cv2.rectangle(canvas, (x, y), (x+lag, y+lag), red, -1)

(centerX, centerY) = (canvas.shape[1] // 2, canvas.shape[0] // 2)
green = (0, 255, 0)

cv2.circle(canvas, (centerX, centerY), 50, green, -1)

cv2.imshow("Canvas", canvas)

cv2.waitKey(0)