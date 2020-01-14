import numpy as np
import torch
from numba import jit, prange


def normal_to_depth(K_inv, d_im, normal, optimized=False):
    """
    Converts normal vectors to depth.

    :param K_inv: inverted intrinsic matrix (Torch tensor)
    :param d_im: image dimensions (list(height, width))
    :param normal: normals vector (Torch tensor)
    :param optimized: using numba package for loop optimization
    :return: depth matrix
    :rtype: Torch tensor
    """

    if optimized:
        # use numba optimization
        # first convert all tensors to numpy
        K_inv = K_inv.cpu().detach().numpy()
        normal = normal.cpu().detach().numpy()

        depth = torch.from_numpy(optimized_loops(K_inv, d_im, normal))
    else:
        # use standard loop

        K_inv.cuda().float()
        normal.cuda()
        #print(normal[11, :, :, :])
        K_inv = K_inv[0, 0:3, 0:3]

        scale = normal.shape[0]

        h = d_im[0]
        w = d_im[1]

        depth = torch.empty(scale, h, w).cuda()
        #print(depth.size())
        # K_inv = np.linalg.pinv(K)

        for n in range(11, scale + 1):
            for x in range(0, h):
                for y in range(0, w):
                    pixel = torch.tensor([x, y, 1]).float().cuda()
                    pt_3d = torch.mm(K_inv, pixel).cuda()
                    vec_values = normal[n, :, x, y]
                    normal_vec = torch.tensor([vec_values[0], vec_values[1], vec_values[2]]).view(1, 3)
                    normal_vec = normal_vec.cuda()
                    depth[n, x, y] = float(1) / (torch.dot(normal_vec, pt_3d).to(dtype=torch.float).item())
        #print(depth)

    return depth


@jit(nopython=True, nogil=True, parallel=True)
def optimized_loops(K_inv, d_im, normal):
    """
    running normal to depth conversion in optimized loops.

    :param K_inv: inverted intrinsic matrix (Torch tensor)
    :param d_im: image dimensions (list(height, width))
    :param normal: normals vector (Torch tensor)
    :return: depth matrix
    :rtype: numpy nd-array
    """
    # use numba package to optimize the following for loops:
    # https://numba.pydata.org/numba-doc/dev/index.html

    K_inv = K_inv[0, 0:3, 0:3]

    # batch size should be first in normal vector
    scale = normal.shape[0]
    h2 = normal.shape[2]
    w2 = normal.shape[3]

    h = d_im[0]
    w = d_im[1]

    depth = np.zeros((scale, h, w))

    # K_inv = np.linalg.pinv(K)

    # using numba to parallelize following loops.
    # numba needs prange instead of numpy's range

    for n in prange(scale):
        for x in prange(h):
            for y in prange(w):

                # pixel values need to be transposed
                pixel = np.array([x, y, 1]).reshape((-1, 1))

                # dot product with 3x3 (k_inv) and 1x3 --> results to 3x1 array
                pt_3d = K_inv * pixel

                # get the normal values for current size and pixel
                vec_values = normal[n, :, x, y]

                # create 1x3 array for the current normal vector (x , y, z)
                normal_vec = np.array([vec_values[0], vec_values[1], vec_values[2]])

                # calculate final depth (scalar value) for current size and pixel
                depth[n, x, y] = 1 / np.dot(normal_vec, pt_3d)

    return depth


def depth_to_disp(K, depth):
    """

    :param K:
    :param depth:
    :return:
    """

    batch, h, w = depth.size()
    disp = torch.empty(batch, h, w).cuda()
    K = K[0, 0:3, 0:3]
    focallength = K[1, 1] * w
    baseline = 0.54
    for n in range(0, batch):
        disp[n, :, :] = (baseline*focallength) / (depth[n, :, :] + 1e-8)
    print(disp[11, :, :])

    return disp