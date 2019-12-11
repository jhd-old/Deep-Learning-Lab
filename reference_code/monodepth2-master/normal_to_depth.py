import numpy as np
import torch


def normal_to_depth(K_inv, d_im, normal):

    K_inv.cuda().float()
    normal.cuda()
    print(normal[11, :, :, :])
    K_inv = K_inv[0, 0:3, 0:3]

    batch_size = torch.tensor(normal[:, 0, 0, 0].size())

    h = torch.tensor(d_im[0])
    w = torch.tensor(d_im[1])

    depth = torch.empty(batch_size, h, w).cuda()
    print(depth.size())
    #K_inv = np.linalg.pinv(K)
    for n in range(11, batch_size):
        for x in range(0, h):
            for y in range(0, w):
                pixel = torch.tensor([[x], [y], [1]]).float().cuda()
                pt_3d = torch.mm(K_inv, pixel).cuda()
                vec_values = normal[n, :, x, y]
                normal_vec = torch.tensor([vec_values[0], vec_values[1], vec_values[2]]).view(1, 3)
                normal_vec = normal_vec.cuda()
                depth[n, x, y] = 1/(torch.mm(normal_vec, pt_3d))
    print(depth)

    return depth

def depth_to_disp(K, depth):

    batch, h, w = depth.size()
    disp = torch.empty(batch, h, w).cuda()
    K = K[0, 0:3, 0:3]
    focallength = K[1, 1] * w
    baseline = 0.54
    for n in range(0, batch):
        disp[n, :, :] = (baseline*focallength) / (depth[n, :, :] + 1e-8)
    print(disp[11, :, :])

    return disp