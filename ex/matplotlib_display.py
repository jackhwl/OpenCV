import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
image = cv2.imread("../images/chelsea-the-cat.jpeg")
plt.axis("off")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()