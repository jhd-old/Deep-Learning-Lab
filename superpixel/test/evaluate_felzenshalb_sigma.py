#####
# Test superpixel algorithm on image
####


import matplotlib.pyplot as plt
import numpy as np

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

from skimage.io import imread

from test.util import *

#t = convert_kitti_to_superpixel("C:/Users/helge/Desktop/KITTI", KittiCamera.color_left, "")

img = imread("images/0000000100.png")

img = img_as_float(img)


segments_fz_50 = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
segments_fz_100 = felzenszwalb(img, scale=100, sigma=1.0, min_size=50)
segments_fz_150 = felzenszwalb(img, scale=100, sigma=1.5, min_size=50)
segments_fz_200 = felzenszwalb(img, scale=100, sigma=2.0, min_size=50)

fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

ax[0, 0].imshow(mark_boundaries(img, segments_fz_50))
ax[0, 0].set_title("Sigma 0.5")
ax[0, 1].imshow(mark_boundaries(img, segments_fz_100))
ax[0, 1].set_title("Sigma 1.0")
ax[1, 0].imshow(mark_boundaries(img, segments_fz_150))
ax[1, 0].set_title("Sigma 1.5")
ax[1, 1].imshow(mark_boundaries(img, segments_fz_200))
ax[1, 1].set_title("Sigma 2.0")

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()