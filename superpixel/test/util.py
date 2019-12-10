import numpy as np
from enum import IntEnum
import os
from skimage.io import imread, imsave
from skimage.util import img_as_float
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed


def save_superpixel_in_archive(file, data, compressed=True):
    """
    Save superpixel information in a numpy archive

    :param file: path to save to
    :param data: data to save. List of numpy arrays
    :param compressed: compression on off
    """

    if compressed:
        np.savez_compressed(file, data)

    else:
        np.savez(file, data)


def load_superpixel_from_archive(file):
    """
    loads superpixel data from an numpy archive

    :param file: file to load
    :return: list of superpixel data
    """

    archive = np.load(file)
    data = []

    if isinstance(archive, np.lib.io.NpzFile):

        for file in archive.files:
            data.append(file)
    else:
        raise TypeError("Wrong data type!")

    return data


def convert_kitti_to_superpixel(path_to_raw_data, camera, save_path="", superpixel_method=None):
    """

    :param path_to_raw_data:
    :param camera:
    :param save_path:
    :param superpixel_method:
    :return:
    """

    camera_specific_naming = "image_0" + str(camera.value)

    # search in kitti directory for camera folders
    for (root, dirs, files) in os.walk(path_to_raw_data, topdown=True):
        for dir in dirs:
            if camera_specific_naming in dir:

                imgs = []

                img_folder = os.path.join(os.path.join(root, dir), "data")

                img_paths = [f for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f))]

                for image_path in img_paths:
                    img = img_as_float(imread(image_path))

                    sup_img = calc_superpixel(img, superpixel_method)

                    save_img(sup_img, image_path)


def save_img(img, path):
    """
    save image to given path
    :param img:
    :param path:
    :return:
    """

    imsave(path, img)

def avg_img_in_superpixel(img, superpixel):
    """

    :param img:
    :param superpixel:
    :return:
    """

    min_ind = np.min(superpixel)
    max_ind = np.max(superpixel)

    for i in range(min_ind, max_ind+1):
        np.where(img is i) = 

def calc_superpixel(img, method=None):
    """

    :param img:
    :param method:
    :return:
    """

    sup = felzenszwalb(img, scale=200, sigma=1.4, min_size=60)

    return sup


class KittiCamera(IntEnum):
    """
    Camera to use
    """

    gray_left = 0
    gray_right = 1
    color_left = 2
    color_right = 3

