import cv2
import dlib
import surface_normal


#depth =
import surface_normal_2

print(cv2.__version__)
print(dlib.__version__)

if __name__ == "__main__":
    #surface_normal_2.surface_normal('data/test_img.png')
    surface_normal_2.surface_normal('data/test_img_monodepth2.jpg')