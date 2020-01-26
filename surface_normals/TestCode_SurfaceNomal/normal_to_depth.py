
import numpy as np
import cv2
from sklearn import preprocessing

def normal_to_depth(K, d_im, normal):

    h, w = d_im.shape
    depth = np.zeros(d_im.shape)
    K_inv = np.linalg.pinv(K)
    for x in range(0, h):
        for y in range(0, w):
            pixel = np.array([x, y, 1])
            pt_3d = K_inv.dot(pixel)
            depth[x, y] = 1/(normal[x, y, :].dot(pt_3d))

    return depth


def depth_to_disp(K, depth):
    h, w = depth.shape
    focallength = K[1, 1] * w
    baseline = 0.54
    disp = (baseline*focallength) / (depth + 1e-8)

    return disp