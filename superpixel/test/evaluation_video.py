

#####
# Test superpixel algorithm on images
####

import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.util import img_as_float

from skimage.io import imread

folder = r'C:/Users/helge/Desktop/2011_09_26_drive_0005_sync/2011_09_26/2011_09_26_drive_0005_sync/image_02/data'

imgs = []

imgs_paths = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

for image_path in imgs_paths:
    imgs.append(img_as_float(imread(os.path.join(folder, image_path))))

for i, img in enumerate(imgs):
    segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    segments_slic = slic(img, n_segments=250, compactness=10, sigma=1)
    segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
    gradient = sobel(rgb2gray(img))
    segments_watershed = watershed(gradient, markers=250, compactness=0.001)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    ax[0, 0].imshow(segments_fz)
    ax[0, 0].set_title("Felzenszwalbs's method")
    ax[0, 1].imshow(segments_slic)
    ax[0, 1].set_title('SLIC')
    ax[1, 0].imshow(segments_quick)
    ax[1, 0].set_title('Quickshift')
    ax[1, 1].imshow(segments_watershed)
    ax[1, 1].set_title('Compact watershed')

    for a in ax.ravel():
        a.set_axis_off()

    plt.tight_layout()

    name = "eval/" + str(i+100) + ".png"
    plt.savefig(name, bbox_inches='tight')

    # free memory
    plt.clf()
    plt.close(fig)

    print("calculated " + str(i+1) + "/" + str(len(imgs)) + " frames")

# convert images to video
image_folder = 'eval'
video_name = 'eval.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
fps = 10
video = cv2.VideoWriter(video_name, 0, fps, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

print("FINISHED!")
