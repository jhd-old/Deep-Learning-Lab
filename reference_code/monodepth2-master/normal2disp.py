import numpy as np
import torch


def get_pixelgrid(b, h, w):
    grid_h = torch.linspace(0.0, w - 1, w).view(1, 1, 1, w).expand(b, 1, h, w)
    grid_v = torch.linspace(0.0, h - 1, h).view(1, 1, h, 1).expand(b, 1, h, w)

    ones = torch.ones_like(grid_h)
    pixelgrid = torch.cat((grid_h, grid_v, ones), dim=1).float().requires_grad_(False).cuda()

    return pixelgrid


def normals_to_disp3(inv_K, normal):

    ## get size of tensor
    batch_size, _, h, w = normal.size()
    inv_K = inv_K[0, 0:3, 0:3].cpu()

    # get 3D homogenous pixel coordinate
    pixelgrid = get_pixelgrid(batch_size, h, w)
    pix_coords = pixelgrid.view(batch_size, 3, -1).cpu()

    # matrix multiplication with 3x3 (k_inv)
    cam_points = torch.matmul(inv_K, pix_coords).view(batch_size, -1, h, w)

    # get disparity
    disp = (normal * cam_points).sum(dim=1, keepdim=True)

    # added clamping according to Jun's suggestion
    disp = torch.clamp(disp, 0.00001, 1.0)

    return disp


def normals_to_disp2(inv_K, normal):

    # batch_size = normal.shape[0]
    # h = normal.shape[2]
    # w = normal.shape[3]
    batch_size, _, h, w = normal.size()

    # inv_K = inv_K[0, 0:3, 0:3].cuda().float()
    inv_K = inv_K[0, 0:3, 0:3]

    # init meshgrid for image height and width
    meshgrid = np.meshgrid(range(w), range(h), indexing='xy')

    # use numpy stack to create an ndarray out of the meshgrid
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)

    # convert numpy nd-array to a torch parameter (~tensor)
    # id_coords = torch.tensor(torch.from_numpy(id_coords), requires_grad=False).cuda()
    id_coords = torch.from_numpy(id_coords).cuda().requires_grad_(False)

    # add one dimension to coordinates
    pix_coords = torch.stack([id_coords[0].view(-1), id_coords[1].view(-1)], dim=0)

    # unsqueeze of 1xn or nx1 tensor basically just changes there orientation
    pix_coords = torch.unsqueeze(pix_coords, 0)

    # repeat pixel coordinates for every batch size.
    pix_coords = pix_coords.repeat(batch_size, 1, 1)

    # init tensor with ones to be able to change the coordinates (2D) to homogenous coordinates (3D)
    ones = torch.ones((batch_size, 1, h * w), requires_grad=False).cuda()

    # concatinate ones to 2D points to get 3D homogenous form.
    # pix_coords = torch.tensor(torch.cat([pix_coords, ones], 1), requires_grad=False)
    pix_coords = torch.cat([pix_coords, ones], 1)

    # matrix multiplication with 3x3 (k_inv) and
    cam_points = torch.matmul(inv_K, pix_coords).cuda()  # equals K^1*pixel

    # convert numpy nd-array to a torch parameter (~tensor)
    # normal_vec = torch.tensor(normal, requires_grad=False).cuda().float()
    normal_vec = normal

    # get batch size
    batch = normal_vec.shape[0]

    # do some magic
    normal_vectors = torch.stack([normal_vec[:, 0].view(batch, -1), normal_vec[:, 1].view(batch, -1),
                                  normal_vec[:, 2].view(batch, -1)], dim=1)

    # do dot product by multiplication of both matrices (x_r=x*x', y_r=y*y', z_r=z*z') and sum (d=x_r+y_r+z_r)
    # IMPORTANT: ASSUME HERE THAT DISPARITY is this value
    # TODO: check if will still want to use this
    # disp = (normal_vectors * cam_points).sum(1)
    disp = (normal_vectors * cam_points).sum(1, keepdim=True).view(batch, -1, h, w)

    # added clamping according to Jun's suggestion
    disp = torch.clamp(disp, 0.00001, 1.0)

    return disp


def normals_to_disp(inv_K, normal):
    batch_size = normal.shape[0]
    h = normal.shape[2]
    w = normal.shape[3]

    inv_K = inv_K[0, 0:3, 0:3].cuda().float()

    # init meshgrid for image height and width
    meshgrid = np.meshgrid(range(w), range(h), indexing='xy')

    # use numpy stack to create an ndarray out of the meshgrid
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)

    # convert numpy nd-array to a torch parameter (~tensor)
    id_coords = torch.tensor(torch.from_numpy(id_coords), requires_grad=False).cuda()

    # add one dimension to coordinates
    pix_coords = torch.stack([id_coords[0].view(-1), id_coords[1].view(-1)], dim=0)

    # unsqueeze of 1xn or nx1 tensor basically just changes there orientation
    pix_coords = torch.unsqueeze(pix_coords, 0)

    # repeat pixel coordinates for every batch size.
    pix_coords = pix_coords.repeat(batch_size, 1, 1)

    # init tensor with ones to be able to change the coordinates (2D) to homogenous coordinates (3D)
    ones = torch.ones((batch_size, 1, h * w), requires_grad=False).cuda()

    # concatinate ones to 2D points to get 3D homogenous form.
    pix_coords = torch.tensor(torch.cat([pix_coords, ones], 1), requires_grad=False)

    # matrix multiplication with 3x3 (k_inv) and
    cam_points = torch.matmul(inv_K, pix_coords).cuda()  # equals K^1*pixel

    # convert numpy nd-array to a torch parameter (~tensor)
    normal_vec = torch.tensor(normal, requires_grad=False).cuda().float()

    # get batch size
    batch = normal_vec.shape[0]

    # do some magic
    normal_vectors = torch.stack([normal_vec[:, 0].view(batch, -1), normal_vec[:, 1].view(batch, -1),
                                  normal_vec[:, 2].view(batch, -1)], dim=1)

    # do dot product by multiplication of both matrices (x_r=x*x', y_r=y*y', z_r=z*z') and sum (d=x_r+y_r+z_r)
    # IMPORTANT: ASSUME HERE THAT DISPARITY is this value
    # TODO: check if will still want to use this
    disp = (normal_vectors * cam_points).sum(1, keepdim=True).view(batch, -1, h, w)

    # added clamping according to Jun's suggestion
    disp = torch.clamp(disp, 0.00001, 1.0)

    return disp


def normal_to_depth(inv_K, normal, min_depth, max_depth):
    """
    Converts normal vectors to depth.

    :param K_inv: inverted intrinsic matrix (Torch tensor)
    :param normal: normals vector (Torch tensor)
    :param min_depth: minimum range for normalization
    :param max_depth: maximum range for normalization
    :return: depth matrix
    :rtype: Torch tensor
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
    id_coords = torch.tensor(torch.from_numpy(id_coords), requires_grad=False).cuda()

    # add one dimension to coordinates
    pix_coords = torch.stack([id_coords[0].view(-1), id_coords[1].view(-1)], dim=0)

    # unsqueeze of 1xn or nx1 tensor basically just changes there orientation
    pix_coords = torch.unsqueeze(pix_coords, 0)

    # repeat pixel coordinates for every batch size.
    pix_coords = pix_coords.repeat(batch_size, 1, 1)

    # init tensor with ones to be able to change the coordinates (2D) to homogenous coordinates (3D)
    ones = torch.ones((batch_size, 1, h * w), requires_grad=False).cuda()

    # concatinate ones to 2D points to get 3D homogenous form.
    pix_coords = torch.tensor(torch.cat([pix_coords, ones], 1), requires_grad=False)

    # matrix multiplication with 3x3 (k_inv) and
    cam_points = torch.matmul(inv_K, pix_coords).cuda()  # equals K^1*pixel

    # convert numpy nd-array to a torch parameter (~tensor)
    normal_vec = torch.tensor(normal, requires_grad=False).cuda().float()

    # get batch size
    batch = normal_vec.shape[0]

    # do some magic
    normal_vectors = torch.stack([normal_vec[:, 0].view(batch, -1), normal_vec[:, 1].view(batch, -1),
                                  normal_vec[:, 2].view(batch, -1)], dim=1)

    # do dot product by multiplication of both matrices (x_r=x*x', y_r=y*y', z_r=z*z') and sum (d=x_r+y_r+z_r)
    # IMPORTANT: ASSUME HERE THAT DISPARITY is this value
    # TODO: check if will still want to use this
    disp = (normal_vectors * cam_points).sum(1)

    # normalize disparity
    normalized_disp = scale_min_max(disp, out_range=(min_depth, max_depth))

    # depth = 1 / values
    # matrix**(-1) should solve this
    depth = normalized_disp.pow(-1)

    # add dimension one to change (batch size, h*w) to (batch size, 1, h*w)
    depth = torch.unsqueeze(depth, dim=1)

    # unstack to retrieve one depth matrix per batch (batch size, 1, h, w)
    depth = depth.view(batch_size, 1, h, w)

    return depth


def scale_min_max(arr, out_range=(-1, 1), axis=None):
    """
    Normalize arr or matrix between given range.
    normalization from https://codereview.stackexchange.com/questions/185785/scale-numpy-array-to-certain-range

    :param arr: array or matrix to normalize
    :param out_range: range to use for min and max values
    :param axis: axis to normalize on
    :return: in given range scaled arr/matrix
    """

    domain = np.min(arr, axis), np.max(arr, axis)
    y = (arr - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


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
