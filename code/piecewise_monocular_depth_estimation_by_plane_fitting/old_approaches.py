import numpy as np
import torch
from numba import jit, prange


#################
# OLD APPROACHES, not used ATM
#################

def normal_to_depth(K_inv, normal, optimized=False):
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

        # depth = torch.from_numpy(optimized_loops(K_inv, d_im, normal))
        depth = torch.from_numpy(optimized_loops(K_inv, normal))
    else:
        # use standard loop
        K_inv.cuda().float()
        normal.cuda()
        # print(normal[11, :, :, :])
        K_inv = K_inv[0, 0:3, 0:3]

        scale = normal.shape[0]

        # h = d_im[0]
        # w = d_im[1]
        h = normal.shape[2]
        w = normal.shape[3]
        print('w = ', w, 'h = ', h, 'scale = ', scale)
        # we will need following shape: [batchsize, channel, h, w]
        depth = torch.empty(scale, 1, h, w).cuda()
        # K_inv = np.linalg.pinv(K)

        for n in range(scale):
            for x in range(w):
                for y in range(h):
                    pixel = torch.tensor([x, y, 1]).float().view(3, 1).cuda()

                    # matrix multiplication with 3x3 (k_inv) and 3x1 --> results to 3x1 array
                    pt_3d = torch.mm(K_inv, pixel).cuda()

                    # get the normal values for current size and pixel
                    vec_values = normal[n, :, y, x]
                    # print('x = ', x, 'y = ', y)

                    # create 1x3 array for the current normal vector (x , y, z)
                    normal_vec = torch.tensor([vec_values[0], vec_values[1], vec_values[2]]).view(1, 3)
                    normal_vec = normal_vec.cuda()

                    # calculate final depth (scalar value) for current size and pixel
                    # do matrix multiplication of 1x3 * 3x1 --> results in 1x1
                    depth[n, 0, y, x] = float(1) / (torch.mm(normal_vec, pt_3d).to(dtype=torch.float).item())

    return depth


##################
# NOT USED ATM
##################

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

    # get batch size, height and width
    batch, channel, h, w = depth.size()

    # init an empty 4D tensor and put it onto gpu
    disp = torch.empty(batch, 1, h, w).cuda()

    # intrinsic matrix reduced to 3x3
    K = K[0, 0:3, 0:3]

    # calculate focal length
    focallength = K[1, 1] * w

    # distance between stereo cameras. Taken from kitti dataset.
    baseline = 0.54

    for n in range(0, batch):
        disp[n, 0, :, :] = (baseline * focallength) / (depth[n, 0, :, :] + 1e-8)

    return disp
