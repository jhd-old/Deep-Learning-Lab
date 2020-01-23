import numpy as np
from enum import IntEnum
import os
from skimage.io import imread, imsave
from skimage.util import img_as_float
from skimage.color import label2rgb
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from datasets.kitti_dataset import KITTIRAWDataset
from datasets.mono_dataset import pil_loader
from PIL import Image
from matplotlib import pyplot as plt
from pathlib import Path
import multiprocessing as mp


def save_superpixel_in_archive(file_path, data, compressed=True):
    """
    Save superpixel information in a numpy archive

    :param file_path: path to save to
    :param data: data to save. List of numpy arrays
    :param compressed: compression on off
    """

    if compressed:
        np.savez_compressed(file_path, data)

    else:
        np.savez(file_path, data)


def load_superpixel_from_archive(file_path):
    """
    loads superpixel data from an numpy archive

    :param file_path: file to load
    :return: list of superpixel data
    """

    archive = np.load(file_path)
    data = []

    if isinstance(archive, np.lib.io.NpzFile):

        for file_path in archive.files:
            data.append(file_path)
    else:
        raise TypeError("Wrong data type!")

    return data


def convert_rgb_to_superpixel(dataset_path, paths, superpixel_method=None, superpixel_arguments=[],
                              img_ext='.jpg', path_insert="super_", save_to_same_folder=True):
    """

    :param path_to_raw_data:
    :param camera:
    :param save_path:
    :param superpixel_method:
    :return:

    :TODO: finish implementation
    """
    pool = mp.Pool(mp.cpu_count())

    print("Starting multiprocessing pool on " + str(mp.cpu_count()) + " kernels.")

    results = [pool.apply(convert_func, args=(dataset_path, path, superpixel_method, superpixel_arguments, img_ext,
                                              path_insert, save_to_same_folder)) for path in paths]

    pool.close()
    pool.join()

    print("Pool closed. Finished! Converted " + str(results.count(True)) + "/" + str(len(results)) + " succesfully!")


def convert_func(dataset_path, path=None, superpixel_method=None, superpixel_arguments=[], img_ext='.jpg',
                 path_insert="super_", save_to_same_folder=True):

    # get image path
    # if none, converts all images in dataset
    line = path.split()
    folder = line[0]

    if len(line) == 3:
        frame_index = int(line[1])
    else:
        frame_index = 0

    if len(line) == 3:
        side = line[2]
    else:
        side = None

    side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    img_path_1 = "{:010d}{}".format(frame_index, img_ext)
    img_path = os.path.join(dataset_path, folder, "image_0{}".format(side_map[side]), "data", img_path_1)

    # change path to new folder
    if not save_to_same_folder:
        save_sup_path = img_path.replace("image", str(path_insert) + "image")
    else:
        save_sup_path = img_path

    # change file type to none, numpy will add .npy automatically
    save_sup_path = save_sup_path.replace(img_ext, "")
    save_sup_path = save_sup_path.replace("/", "\\")

    # check if already converted
    if not os.path.isfile(save_sup_path):
        # get folder name of the current image to retrieve saving path

        # load image
        img = pil_loader(img_path)

        # convert image to numpy
        img = np.array(img)

        # calculate superpixel
        sup = calc_superpixel(img, superpixel_method, superpixel_arguments)

        # create directory to be save
        Path(save_sup_path).parent.mkdir(parents=True, exist_ok=True)

        # save superpixel in numpy archive
        np.save(save_sup_path, sup)
        print("Converted image to superpixel.")

        return True if os.path.isfile(save_sup_path) else False

    else:
        return False


def calc_superpixel(img, method="fz", args=[]):
    """
    Calculates superpixels from given image.

    :param img:
    :param method:
    :param args:
    :return:
    """
    if method == "fz":
        if args is not None:
            if len(args) is 3:
                scale = args[0]
                sigma = args[1]
                min_size = args[2]
                sup = felzenszwalb(img, scale=int(scale), sigma=sigma, min_size=int(min_size))
            else:
                sup = felzenszwalb(img)
        else:
            sup = felzenszwalb(img)

    elif method == "slic":
        if args is not None:
            if len(args) is 3:
                num_seg = args[0]
                comp = args[1]
                sig = args[2]
                sup = slic(img, n_segments=int(num_seg), compactness=comp, sigma=sig)
            else:
                sup = slic(img)
        else:
            sup = slic(img)

    elif method == "watershed":
        raise NotImplementedError

    elif method == "quickshift":
        raise NotImplementedError

    else:
        raise TypeError("Given method not valid!")

    return sup


def avg_image(image, label):
    """
    Average an image for each given superpixel area.

    :param image: image to average
    :param label: superpixel labels
    :return: averaged image
    :rtype: numpy nd-array
    """

    avg_image = label2rgb(label, image, kind='avg')

    return avg_image


class KittiCamera(IntEnum):
    """
    Camera to use
    """

    gray_left = 0
    gray_right = 1
    color_left = 2
    color_right = 3


if __name__ == "__main__":

    data_set = ""
    method = "fz"

    convert_rgb_to_superpixel()

