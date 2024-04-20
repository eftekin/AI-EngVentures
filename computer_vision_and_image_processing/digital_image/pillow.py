import matplotlib.pyplot as plt
from PIL import Image, ImageOps

image = Image.open("../../assets/darth_vader.jpg")

plt.imshow(image)

print("Format of the image:", image.format)
print("Size of the image:", image.size)
print("Mode of the image:", image.mode)

image_gray = ImageOps.grayscale(image)
image_gray.show(title="darth_vader_gray")

print("Mode of the gray image:", image_gray.mode)  # L means luminance

red, green, blue = image.split()
red.show(title="darth_vader_red")
green.show(title="darth_vader_green")
