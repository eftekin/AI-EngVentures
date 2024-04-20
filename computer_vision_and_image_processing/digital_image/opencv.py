import cv2
import matplotlib.pyplot as plt

image = cv2.imread("../../assets/darth_vader.jpg")

print("Type of image:", type(image))
print("Shape of image:", image.shape)


plt.imshow(image)

new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # to change color space

plt.imshow(new_image)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("../../../assets/darth_vader_gray_cv.jpg", image_gray)
