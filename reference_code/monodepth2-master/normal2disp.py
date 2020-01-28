import numpy as np
import torch
from torch import nn


def normal_to_depth(inv_K, normal, min_depth, max_depth):
    """
    Converts normal vectors to depth.

    :param K_inv: inverted intrinsic matrix (Torch tensor)
    :param normal: normals vector (Torch tensor)
    :return: depth matrix
    :rtype: Torch tensor
    :todo: continue implementing and change to use this method in trainer.py
    """

    batch_size = normal.shape[0]
    h = normal.shape[2]
    w = normal.shape[3]

    inv_K = inv_K[0, 0:3, 0:3].cuda().float()

    # init meshgrid for image height and width
    meshgrid = np.meshgrid(range(w), range(h), indexing='xy')

    # use numpy stack to create an ndarray out of the meshgrid
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)

    # convert numpy nd-array to a torch parameter (~tensor)
    id_coords = nn.Parameter(torch.from_numpy(id_coords),
                             requires_grad=False).cuda()

    # add one dimension to coordinates
    pix_coords = torch.stack([id_coords[0].view(-1), id_coords[1].view(-1)], dim=0)

    # unsqueeze of 1xn or nx1 tensor basically just changes there orientation
    pix_coords = torch.unsqueeze(pix_coords, 0)

    # repeat pixel coordinates for every batch size.
    pix_coords = pix_coords.repeat(batch_size, 1, 1)

    # init tensor with ones to be able to change the coordinates (2D) to homogenous coordinates (3D)
    ones = nn.Parameter(torch.ones(batch_size, 1, h * w),
                        requires_grad=False).cuda()

    # concatinate ones to 2D points to get 3D homogenous form.
    pix_coords = nn.Parameter(torch.cat([pix_coords, ones], 1),
                              requires_grad=False)

    # matrix multiplication with 3x3 (k_inv) and
    cam_points = torch.matmul(inv_K, pix_coords).cuda()  # entspricht K^1*pixel

    # convert numpy nd-array to a torch parameter (~tensor)
    normal_vec = nn.Parameter(normal,
                              requires_grad=False).cuda().float()

    # get batch size
    batch = normal_vec.shape[0]

    # do some magic
    normal_vectors = torch.stack([normal_vec[:, 0].view(batch, -1), normal_vec[:, 1].view(batch, -1),
                                  normal_vec[:, 2].view(batch, -1)], dim=1)

    # do dot product by multiplication of both matrices (x_r=x*x', y_r=y*y', z_r=z*z') and sum (d=x_r+y_r+z_r)
    depth = (normal_vectors * cam_points).sum(1)

    # depth = 1 / values
    # matrix**(-1) should solve this
    depth = depth.pow(-1)

    # add dimension one to change (batch size, h*w) to (batch size, 1, h*w)
    depth = torch.unsqueeze(depth, dim=1)

    # unstack to retrieve one depth matrix per batch (batch size, 1, h, w)
    depth = depth.view(batch_size, 1, h, w)

    # normalization from https://codereview.stackexchange.com/questions/185785/scale-numpy-array-to-certain-range
    depth = (depth - (depth.max() + depth.min()) / 2) / (depth.max() - depth.min())
    depth = depth * (max_depth - min_depth) + (max_depth + min_depth) / 2

    return depth


def depth_to_disp(K, depth):
    """
    Converts depth to disparity.

    :param K: intrinsic matrix
    :param depth: depth matrix (batch size, 1, h, w)
    :return:
    """

    # get batch size, height and width
    batch, channel, h, w = depth.shape

    # init an empty 4D tensor and put it onto gpu
    disp = torch.empty(batch, 1, h, w).cuda()

    # intrinsic matrix reduced to 3x3
    K = K[0, 0:3, 0:3].cuda().float()

    # calculate focal length
    focal_length = K[1, 1] * w

    # distance between stereo cameras. Taken from kitti dataset.
    baseline = 0.54

    for n in range(0, batch):
        disp[n, 0, :, :] = (baseline * focal_length) / (depth[n, 0, :, :] + 1e-8)

    return disp
