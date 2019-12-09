import numpy as np
from matplotlib import pyplot as plt
import cv2


def surface_normal(path):
    d_im = cv2.imread(path, 0)
    d_im = d_im.astype("float64")

    zy, zx = np.gradient(d_im)
    # You may also consider using Sobel to get a joint Gaussian smoothing and differentation
    # to reduce noise
    #zx = cv2.Sobel(d_im, cv2.CV_64F, 1, 0, ksize=5)
    #zy = cv2.Sobel(d_im, cv2.CV_64F, 0, 1, ksize=5)

    normal = np.dstack((-zx, -zy, np.ones_like(d_im)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    # offset and rescale values to be in 0-255
    normal += 1
    normal /= 2
    normal *= 255

    cv2.imwrite("normal3Cat.png", normal[:, :, ::-1])