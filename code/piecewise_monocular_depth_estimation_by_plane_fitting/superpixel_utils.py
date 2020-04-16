from __future__ import absolute_import, division, print_function

import multiprocessing as mp
import os
from enum import IntEnum
from pathlib import PurePath

import numpy as np
from PIL import Image
from skimage.color import label2rgb
from skimage.segmentation import felzenszwalb, slic
from options import MonodepthOptions

####################
# UTILS TO USE FOR SUPERPIXEL
###################


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def load_superpixel_data(path, method, arguments, img_ext):
    """
    load superpixel data from file. Returns int32 numpy matrix. Multiple fallbacks to be sure superpixel data will be
    returned.

    :param path: path to load file from.
    :param method: superpixel method to use
    :param arguments: arguments for superpixel method (parameters)
    :param img_ext: image file extension
    :return: superpixel data
    :rtype: int32 numpy array
    """

    # check if file exists
    # it really should, but just to be safe
    try:
        # saved superpixel for key "x"
        if os.path.isfile(path):
            super_file = np.load(path)
        else:
            raise IOError("No file at given path: " + (str(path)))
    except:

        try:
            # try to use posix
            # pure path will use unix or windows correct path depending on detected system
            pos_path = PurePath(path.replace("/", "\\"))

            pos_path = pos_path.as_posix()

            super_file = np.load(pos_path)
        except:
            # do the necessary thing: create superpixel information
            try:
                print("Error while loading superpixel. Take fallback and calculate online!")
                superpixel_label = convert_single_rgb_to_superpixel(path, img_ext, method, arguments)
                superpixel_label = superpixel_label.astype(np.int32)
                super_file = None
            except:
                raise IOError("Error while loading superpixel path at " + str(path) +
                              ". Tried to calculate online, but failed for image:" + str(path))
    try:
        if super_file is not None:
            superpixel_label = super_file["x"].astype(np.int32)
    except:
        raise IOError("Error while reading data of superpixel!")

    return superpixel_label


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
    Converts RGB image to superpixel data for given superpixel method and parameters. Can convert multiple images at
    once. Saves superpixel data to RGB image folder with path_insert as pre.

    :param dataset_path: path to KITTI dataset
    :param paths: path of the images to convert
    :param superpixel_method: superpixel method to use
    :param superpixel_arguments: arguments for superpixel method (parameters)
    :param img_ext: image file extension
    :param path_insert: naming for superpixel file.
    :param num_channel: number of input channels, which will be used
    """

    n_prints = 500
    converted = 0
    already_converted = 0

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

    for path in paths:
        state = convert_func(dataset_path, path, superpixel_method, superpixel_arguments, img_ext, path_insert,
                             num_channel)

        if state == ConversionState.already_converted:
            already_converted += 1
        elif state == ConversionState.converted:
            converted += 1
        else:
            raise IOError("Unknown conversion state!")

        if (already_converted + converted) % n_prints == 0:
            print("Converted {}/{} images to superpixel! {} were already converted.".format(converted,
                                                                                            len(paths),
                                                                                            already_converted))


def convert_rgb_to_superpixel_multiprocess(dataset_path, paths, superpixel_method=None, superpixel_arguments=[],
                              img_ext='.jpg', path_insert="super_", num_channel=4):
    """
    Converts RGB image to superpixel data for given superpixel method and parameters USING multiprocessing.
    Can convert multiple images at once.
    Saves superpixel data to RGB image folder with path_insert as pre.

    :param dataset_path: path to KITTI dataset
    :param paths: path of the images to convert
    :param superpixel_method: superpixel method to use
    :param superpixel_arguments: arguments for superpixel method (parameters)
    :param img_ext: image file extension
    :param path_insert: naming for superpixel file.
    :param num_channel: number of input channels, which will be used
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
    """
        Converts RGB image to superpixel data for given superpixel method and parameters USING multiprocessing.
        Can convert multiple images at once.
        Saves superpixel data to RGB image folder with path_insert as pre.

        :param dataset_path: path to KITTI dataset
        :param paths: path of the images to convert
        :param superpixel_method: superpixel method to use
        :param superpixel_arguments: arguments for superpixel method (parameters)
        :param img_ext: image file extension
        :param path_insert: naming for superpixel file.
        :param num_channel: number of input channels, which will be used
        """

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

        # change file type to none, numpy will add .npz automatically
        save_sup_path = save_sup_path.replace(img_ext, "")

        # add identifier for superpixel method and arguments
        superpixel_ident = str(superpixel_method)

        for a in superpixel_arguments:
            # replace . with _
            a = str(a).replace(".", "_")
            superpixel_ident += a

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
    save_sup_path = (save_sup_path + ".npz").replace("/", "\\")

    # check if already converted
    if not os.path.isfile(save_sup_path):

        # get folder name of the current image to retrieve saving path

        # load image
        img = pil_loader(img_path)

        # convert image to numpy
        img = np.array(img)

        # calculate superpixel
        sup = calc_superpixel(img, superpixel_method, superpixel_arguments)

        # force superpixel to be unsigned 16bit integer
        sup = sup.astype(np.uint16)

        if num_channel is 4:
            # save superpixel in numpy archive

            save_sup_path = PurePath(save_sup_path).as_posix()

            # save superpixel information as uint16 in a compressed numpy archive
            np.savez_compressed(save_sup_path, x=sup)

        elif num_channel is 3 or num_channel is 6:
            # save as image

            # average over superpixel area
            sup_img = avg_image(img, sup)

            # convert numpy img back to PIL Image
            sup_img = Image.fromarray(sup_img)

            # save image
            sup_img.save(save_sup_path)

        state = ConversionState.converted if os.path.isfile(save_sup_path) else ConversionState.failed_to_convert

        if state == ConversionState.failed_to_convert:
            raise IOError("Superpixel couldn't be saved at the following path: " + str(save_sup_path))

        return state

    else:
        # check if its valid
        if np.load(save_sup_path)["x"] is not None:
            return ConversionState.already_converted
        else:
            # get folder name of the current image to retrieve saving path

            # load image
            img = pil_loader(img_path)

            # convert image to numpy
            img = np.array(img)

            # calculate superpixel
            sup = calc_superpixel(img, superpixel_method, superpixel_arguments)

            # force superpixel to be unsigned 16bit integer
            sup = sup.astype(np.uint16)

            # save superpixel in numpy archive

            save_sup_path = PurePath(save_sup_path).as_posix()

            # save superpixel information as uint16 in a compressed numpy archive
            np.savez_compressed(save_sup_path, x=sup)

            state = ConversionState.converted if os.path.isfile(save_sup_path) else ConversionState.failed_to_convert

            if state == ConversionState.failed_to_convert:
                raise IOError("Superpixel couldn't be saved at the following path: " + str(save_sup_path))

            print("Repaired superpixel data!")

            return state


def convert_single_rgb_to_superpixel(superpixel_path, img_ext='jpg', superpixel_method='fz', superpixel_arguments=None):

    # get folder name of the current image to retrieve saving path

    superpixel_ident_idx = superpixel_path.find(superpixel_method)

    img_path = superpixel_path[:superpixel_ident_idx] + str(img_ext)

    # load image
    img = pil_loader(img_path)

    # convert image to numpy
    img = np.array(img)

    # calculate superpixel
    sup = calc_superpixel(img, superpixel_method, superpixel_arguments)

    # force superpixel to be unsigned 16bit integer
    sup = sup.astype(np.uint16)

    # save superpixel information as uint16 in a compressed numpy archive
    np.savez_compressed(superpixel_path, x=sup)

    return sup


def convert_all_in_folder(folder, superpixel_method='fz', superpixel_arguments=[], img_ext='.jpg'):
    """
    Convert all images in given folder to superpixel.

    :param folder:
    :param superpixel_method:
    :param superpixel_arguments:
    :return:
    """

    converted = 0
    already_converted = 0

    # print steps
    n_prints = 500

    if not os.path.isdir(folder):
        raise IOError("Given folder is no folder")
    else:

        for dirpath, dirnames, filenames in os.walk(folder):
            all_images = [f for f in filenames if f.endswith(img_ext)]

            for filename in all_images:

                image_path = os.path.join(dirpath, filename)

                # change file type to none, numpy will add .npz automatically
                save_sup_path = image_path.replace(img_ext, "")

                # add identifier for superpixel method and arguments
                superpixel_ident = str(superpixel_method)

                for a in superpixel_arguments:
                    # replace . with _
                    a = str(a).replace(".", "_")
                    superpixel_ident += a

                save_sup_path += superpixel_ident + '.npz'

                if not os.path.isfile(save_sup_path):

                    # load image
                    img = pil_loader(image_path)

                    # convert image to numpy
                    img = np.array(img)

                    # calculate superpixel
                    sup = calc_superpixel(img, superpixel_method, superpixel_arguments)

                    # force superpixel to be unsigned 16bit integer
                    sup = sup.astype(np.uint16)

                    np.savez_compressed(save_sup_path, x=sup)

                    if not os.path.isfile(save_sup_path):
                        raise IOError("Couldnt save superpixel at following path: {}!".format(save_sup_path))
                    else:
                        converted += 1

                else:
                    already_converted += 1
                    
                if converted % n_prints == 0 or already_converted % n_prints == 0:
                    print("Converted {}/{} images to superpixel! {} were already converted.".format(converted,
                                                                                                    len(all_images),
                          already_converted))

    return converted


def calc_superpixel(img, s_method="fz", s_args=[]):
    """
    Calculates superpixels from given image.

    :param img: image to calculate superpixel for
    :param s_method: superpixel method to use
    :param s_args: arguments (parameters) for superpixel method
    :return: return superpixel
    """
    if s_method == "fz":
        if s_args is not None:
            if len(s_args) is 3:
                scale = s_args[0]
                sigma = s_args[1]
                min_size = s_args[2]
                sup = felzenszwalb(img, scale=int(scale), sigma=sigma, min_size=int(min_size))
            else:
                sup = felzenszwalb(img)
        else:
            sup = felzenszwalb(img)

    elif s_method == "slic":
        if s_args is not None:
            if len(s_args) is 3:
                num_seg = s_args[0]
                comp = s_args[1]
                sig = s_args[2]
                sup = slic(img, n_segments=int(num_seg), compactness=comp, sigma=sig)
            else:
                sup = slic(img)
        else:
            sup = slic(img)

    elif s_method == "watershed":
        raise NotImplementedError

    elif s_method == "quickshift":
        raise NotImplementedError

    else:
        raise TypeError("Given s_method not valid!")

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


class ConversionState(IntEnum):
    """
    State of the conversion
    """

    converted = 0
    already_converted = 1
    failed_to_convert = 2


options = MonodepthOptions()
opts = options.parse()

if __name__ == "__main__":

    method = opts.superpixel_method
    args = opts.superpixel_arguments

    folder = opts.data_path
    print("Start converting images at {}".format(folder))
    print("Using {} method and {} as arguments".format(method, args))
    num_con = convert_all_in_folder(folder, method, args)
    print("Converted {} images to superpixel!".format(num_con))
