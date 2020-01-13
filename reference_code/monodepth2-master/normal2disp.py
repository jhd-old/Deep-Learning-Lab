import numpy as np
import torch


def normal_to_depth(normal, inv_K, d_im):

    batch_size = torch.tensor(normal[:, 0, 0, 0].size())
    height = torch.tensor(d_im[0])
    width = torch.tensor(d_im[1])

    meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = nn.Parameter(torch.from_numpy(id_coords),
                                  requires_grad=False)

    ones = nn.Parameter(torch.ones(batch_size, 1, height * width),
                             requires_grad=False)

    pix_coords = torch.unsqueeze(torch.stack(
        [id_coords[0].view(-1), id_coords[1].view(-1)], 0), 0)
    pix_coords = pix_coords.repeat(batch_size, 1, 1)
    pix_coords = nn.Parameter(torch.cat([pix_coords, ones], 1),
                                   requires_grad=False)

    cam_points = torch.matmul(inv_K[:, :3, :3], pix_coords) #entspricht K^1*pixel

    # depth = 1/(normal_vec)T * cam_points

    return depth

def normal_to_dips(depth)

    return disp