import numpy as np
from matplotlib import pyplot as plt
import cv2


def surface_normal(path):
    d_im = cv2.imread(path)
    d_im = d_im.astype("float64")

    normals = np.array(d_im, dtype="float64")
    h,w,d = d_im.shape
    for i in range(1,w-1):
        for j in range(1,h-1):
            t = np.array([i,j-1,d_im[j-1,i,0]],dtype="float64")
            f = np.array([i-1,j,d_im[j,i-1,0]],dtype="float64")
            c = np.array([i,j,d_im[j,i,0]] , dtype = "float64")
            d = np.cross(f-c,t-c)
            n = d / np.sqrt((np.sum(d**2)))
            normals[j,i,:] = n

    cv2.imwrite("normal2Kitti.jpg", normals*255)