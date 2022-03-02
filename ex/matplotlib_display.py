import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread("../images/chelsea-the-cat.jpeg")
plt.axis("off")
plt.imshow(image)
plt.show()