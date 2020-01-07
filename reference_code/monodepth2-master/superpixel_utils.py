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


def convert_rgb_to_superpixel(dataset_path, paths, superpixel_method=None, superpixel_arguments=[], img_ext='.jpg', path_insert="super_"):
    """

    :param path_to_raw_data:
    :param camera:
    :param save_path:
    :param superpixel_method:
    :return:

    :TODO: finish implementation
    """

    for i, path in enumerate(paths):

        # get image path

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

        save_image_path = img_path.replace("image", str(path_insert) + "image")
        save_image_path = save_image_path.replace("/", "\\")
        # get folder name of the current image to retrieve saving path

        # load image
        img = pil_loader(img_path)

        # convert image to numpy
        img = np.array(img)

        # calculate superpixel
        sup = calc_superpixel(img, superpixel_method, superpixel_arguments)

        # convert image to superpixel image
        sup_img = avg_image(img, sup)

        # convert numpy img back to PIL Image
        sup_img = Image.fromarray(sup_img)

        # create directory to be save
        Path(save_image_path).parent.mkdir(parents=True, exist_ok=True)
        sup_img.save(save_image_path)

        print("Converted " + str(i+1) + "/" + str(len(paths)) + " images to superpixel.")


def calc_superpixel(img, method="fz", args=[]):
    """

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

    avg_image = label2rgb(label, image, kind='avg')

    plt.imshow(avg_image)
    plt.show()

    return avg_image

def mean_image(image, label):

    sli_1d = np.reshape(label, -1)
    uni = np.unique(sli_1d)

    num_ch = image.shape[2]

    for channel in range(num_ch):
        img_chan = image[:, :, channel]

        for i in uni:
            ma = np.ma.masked_where(label != i, img_chan)
            val = np.mean(ma)

            img_chan[ma] = val

        image[:, :, channel] = img_chan


    return image


class KittiCamera(IntEnum):
    """
    Camera to use
    """

    gray_left = 0
    gray_right = 1
    color_left = 2
    color_right = 3
