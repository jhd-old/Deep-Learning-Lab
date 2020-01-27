from __future__ import absolute_import, division, print_function

import multiprocessing as mp
import os
from enum import IntEnum
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import label2rgb
from skimage.segmentation import felzenszwalb, slic

from datasets.mono_dataset import pil_loader


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
                              img_ext='.jpg', path_insert="super_", num_channel=4):
    """

    :param path_to_raw_data:
    :param camera:
    :param save_path:
    :param superpixel_method:
    :return:
    """

    if superpixel_arguments is None:
        print("Using {} method with default arguments!".format(superpixel_method))
        superpixel_arguments = []

    elif None in superpixel_arguments:
        print("Using {} method with wrong arguments. At least one of them was None!".format(superpixel_method))

    elif len(superpixel_arguments) is 3:
        print("Using {} method with {}, {}, {} as arguments!".format(superpixel_method, superpixel_arguments[0],
                                                                     superpixel_arguments[1],
                                                                     superpixel_arguments[2]))

    else:
        raise NotImplementedError("Length of given superpixel arguments is not implemented!")
    pool = mp.Pool(mp.cpu_count())

    print("Starting multiprocessing pool on " + str(mp.cpu_count()) + " kernels.")

    results = [pool.apply(convert_func, args=(dataset_path, path, superpixel_method, superpixel_arguments, img_ext,
                                              path_insert, num_channel)) for path in paths]

    pool.close()
    pool.join()

    print("Pool closed. Finished!")
    print(
        "Converted " + str(results.count(ConversionState.converted) + results.count(ConversionState.already_converted))
        + "/" + str(len(results)) + " images to superpixel!")
    print(str(results.count(ConversionState.already_converted)) + "/" + str(len(results)) + " were already calculated.")
    print(str(results.count(ConversionState.converted)) + "/" + str(len(results)) + " have been calculated.")
    print("Failed to calculate" + str(results.count(ConversionState.failed_to_convert)) + "/" + str(len(results))
          + " images.")


def convert_func(dataset_path, path=None, superpixel_method=None, superpixel_arguments=[], img_ext='.jpg',
                 path_insert="super_", num_channel=4):
    # TODO: change this back
    # for now force to only save as 1 channel with indices
    num_channel = 4

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

    if num_channel is 4:
        # save superpixel to same folder
        save_sup_path = img_path

        # change file type to none, numpy will add .npy automatically
        save_sup_path = save_sup_path.replace(img_ext, "")

        # add identifier for superpixel method and arguments
        superpixel_ident = str(superpixel_method)

        for a in superpixel_arguments:
            superpixel_ident += str(a)

        save_sup_path += superpixel_ident

    elif num_channel is 3 or num_channel is 6:
        # save superpixel to seperate folder
        save_sup_path = img_path.replace("image", str(path_insert) + "image")

        # add identifier for superpixel method and arguments
        superpixel_ident = str(superpixel_method)

        for a in superpixel_arguments:
            superpixel_ident += str(a)
        save_sup_path.replace(img_ext, superpixel_ident + img_ext)

    else:
        raise NotImplementedError("Currently not supported!")

    # replace backslashes
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

        if num_channel is 4:
            # save superpixel in numpy archive

            print("Try to save numpy array to " + str(os.path.abspath(save_sup_path)))
            np.save(os.path.abspath(save_sup_path), sup)

        elif num_channel is 3 or num_channel is 6:
            # save as image

            # average over superpixel area
            sup_img = avg_image(img, sup)

            # convert numpy img back to PIL Image
            sup_img = Image.fromarray(sup_img)

            # save image
            sup_img.save(os.path.abspath(save_sup_path))

        state = ConversionState.converted if os.path.isfile(save_sup_path) else ConversionState.failed_to_convert

        # delete this
        if state == ConversionState.failed_to_convert:
            raise IOError("Superpixel couldn't be saved at the following path: " + str(os.path.abspath(save_sup_path)))

        return state

    else:
        return ConversionState.already_converted


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

    image = label2rgb(label, image, kind='avg')

    return image


class KittiCamera(IntEnum):
    """
    Camera to use
    """

    gray_left = 0
    gray_right = 1
    color_left = 2
    color_right = 3


class ConversionState(IntEnum):
    """
    State of the conversion
    """

    converted = 0
    already_converted = 1
    failed_to_convert = 2
