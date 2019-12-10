import cv2
import numpy as np
import surface_normal_3
import normal_to_depth


#depth =
import surface_normal_2

print(cv2.__version__)
#print(dlib.__version__)

K = np.array([[70, 0, 50], [0, 70, 40], [0, 0, 1]])


if __name__ == "__main__":
    #surface_normal_2.surface_normal('data/test_img.png')
    normal, d_im = surface_normal_3.surface_normal('data/test_img.png')
    depth = normal_to_depth.normal_to_depth(K, d_im, normal)
    disp = normal_to_depth.depth_to_disp(K, depth)

    cv2.imwrite("normal2depth.png", disp*255)