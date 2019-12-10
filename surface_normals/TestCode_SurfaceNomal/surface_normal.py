import numpy as np
from matplotlib import pyplot as plt
import cv2



def surface_normal():

    img = cv2.imread('data/test_img.png',0)
    normal = np.zeros(img.shape, np.uint32)
    height, width = img.shape
    direction = np.zeros([3, 1],  dtype=np.float64)

    for x in range(121, width):
        for y in range(87, height):
            dzdx = (img[x+1][y]-img[x-1][y])/2
            dzdy = (img[x][y+1] - img[x][y-1])/2

            direction[0] = -dzdx
            direction[1] = -dzdx
            direction[2] = 1.0

            n = direction/np.linalg.norm(direction)
            normal[x][y] = n[0:1]

    plt.imshow(img, interpolation='bicubic')
    plt.imshow(normal, interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

    return None


#Mat depth = img of type CV_32FC1
#Mat normals(depth.size(), CV_32FC3);

#for(int x = 0; x < depth.rows; ++x)
#{
#    for(int y = 0; y < depth.cols; ++y)
#    {
#
#        float dzdx = (depth.at<float>(x+1, y) - depth.at<float>(x-1, y)) / 2.0;
#        float dzdy = (depth.at<float>(x, y+1) - depth.at<float>(x, y-1)) / 2.0;
#
#        Vec3f d(-dzdx, -dzdy, 1.0f);
#        Vec3f n = normalize(d);
#
#        normals.at<Vec3f>(x, y) = n;
#    }
#}

#imshow("depth", depth / 255);
#imshow("normals", normals);