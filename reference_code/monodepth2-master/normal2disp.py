import numpy as np
import torch
from torch import nn


def normal_to_depth(inv_K, d_im, normal):
    """

    :param inv_K:
    :param d_im:
    :param normal:
    :return:
    :todo: continue implementing and change to use this method in trainer.py
    """

    batch_size = normal.shape[0]
    h = d_im[0]
    w = d_im[1]

    inv_K = inv_K.cuda().float()[0, 0:3, 0:3]

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
    cam_points = torch.matmul(inv_K, pix_coords) #entspricht K^1*pixel


    # depth = 1/(normal_vec)T * cam_points

    return cam_points

def normal_to_dips(depth):

    return disp