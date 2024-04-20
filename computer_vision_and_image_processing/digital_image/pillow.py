import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps


def get_concat_h(im1, im2):
    dst = Image.new("RGB", (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


image = Image.open("../../assets/darth_vader.jpg")

plt.imshow(image)
plt.show()
plt.clf()

print("Format of the image:", image.format)
print("Size of the image:", image.size)
print("Mode of the image:", image.mode)

image_gray = ImageOps.grayscale(image)
image_gray.show(title="darth_vader_gray")

print("Mode of the gray image:", image_gray.mode)  # L means luminance

red, green, blue = image.split()

get_concat_h(image, red).show()
get_concat_h(image, green).show()
get_concat_h(image, blue).show()


# get darth vader without blue
blue_darth_vader = image
blue_array = np.array(blue_darth_vader)
blue_array[:, :, 2] = 0

plt.figure(figsize=(10, 10))
plt.imshow(blue_array)
plt.show()
plt.clf()
