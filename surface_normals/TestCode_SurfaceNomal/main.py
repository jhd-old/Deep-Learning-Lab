import cv2
# import torch
import numpy as np
import surface_normal_3
import normal_to_depth
import torch
import normal2disp


#depth =
import surface_normal_2

print(cv2.__version__)
#print(dlib.__version__)

K = np.array([[70, 0, 50], [0, 70, 40], [0, 0, 1]])
K_inv = np.linalg.pinv(K)


if __name__ == "__main__":
    # surface_normal_2.surface_normal('data/test_img.png')
    normal, d_im = surface_normal_3.surface_normal('data/test_img.png')

    depth = normal_to_depth.normal_to_depth(K, d_im, normal)
    disp = normal_to_depth.depth_to_disp(K, depth)

    cv2.imwrite("normal2depth.png", depth * 10255)
    cv2.imwrite("normal2disp.png", disp * 10255)

    ######################
    # NEW CODE
    ######################

    # generate a tensor with the surface normals
    normal_np = torch.from_numpy(normal)

    # get normal_tensor into the shape for normal2depth. -> normal_tensor(batch_size, ch, h, w)

    w, h = d_im.shape
    # generate empty tensor
    normal_tensor = torch.zeros((1, 3, w, h))

    # unbind the input tensor to the three channels
    normal_np = torch.unbind(normal_np, dim=2)

    # rebuild the normal vector tensor to match the right shape
    normal_tensor[0, 0, :, :] = normal_np[0]
    normal_tensor[0, 1, :, :] = normal_np[1]
    normal_tensor[0, 2, :, :] = normal_np[2]

    # generate depth and disparity
    new_depth = normal2disp.normal_to_depth(K_inv, normal_tensor)
    new_disp = normal2disp.depth_to_disp(K, new_depth)

    # remove not used dimensions
    new_depth = torch.squeeze(new_depth)
    new_disp = torch.squeeze(new_disp)
    # turn back into numpy array
    new_depth = new_depth.numpy()
    new_disp = new_disp.numpy()

    # save the image to a file
    cv2.imwrite("normal2depth_fastFunction.png", new_depth * 10255)
    cv2.imwrite("normal2disp_fastFunction.png", new_disp * 10255)