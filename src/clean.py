from __future__ import absolute_import, division, print_function, unicode_literals
import cv2
import numpy as np
from PIL import Image, ImageFilter
import PIL
import matplotlib.pyplot as plt

raw_img = cv2.imread("../images/twitch/smash/https:++static-cdn.jtvnw.net+previews-ttv+live_user_t5ace-440x248.jpg")

# You may need to convert the color.
img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
im_pil = Image.fromarray(img)

# For reversing the operation:
im_np = np.asarray(im_pil)

print(im_np.shape)

# plt.figure()
# plt.imshow(im_np)
# plt.colorbar()
# plt.grid(False)
# plt.show()

mn_img = imageprepare("../images/twitch/smash/https:++static-cdn.jtvnw.net+previews-ttv+live_user_t5ace-440x248.jpg")
print(mn_img)

plt.figure()
plt.imshow(mn_img)
plt.colorbar()
plt.grid(False)
plt.show()


def imageprepare(argv):
    im = PIL.Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = PIL.Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels#28 28

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), PIL.Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), PIL.Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas
    tv = list(newImage.getdata())  # get pixel values
    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return tva
