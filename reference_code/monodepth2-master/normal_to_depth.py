import numpy as np
import torch


def normal_to_depth(K, d_im, normal):

    # Konvertierung in Numpy
    K = K.numpy()
    normal = normal.numpy()

    # Funktion Normal to depth
    h = d_im[0]
    w = d_im[1]

    depth = np.zeros((h, w))
    K_inv = np.linalg.pinv(K)
    for x in range(0, h):
        for y in range(0, w):
            pixel = np.array([x, y, 1])
            pt_3d = K_inv.dot(pixel)
            depth[x, y] = 1/(normal[x, y, :].dot(pt_3d))

    # Konvertierung in Torch tensor, nicht nötig da die funtionen aufeinander übergeben
    # depth = torch.from_numpy(depth_np)
    return depth


def depth_to_disp(K, depth):
    # Konvertierung in Numpy
    K = K.numpy()
    # depth = depth.numpy()

    # Funktion depth to disp
    h, w = depth.shape
    focallength = K[1, 1] * w
    baseline = 0.54
    disp_np = (baseline*focallength) / (depth + 1e-8)

    # Konvertierung in Torch tensor
    disp = torch.from_numpy(disp_np)
    return disp