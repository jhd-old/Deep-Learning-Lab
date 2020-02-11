# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

# from __future__ import absolute_import, division, print_function

import argparse
import glob
import os

import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import torch
from torchvision import transforms

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist

from superpixel_utils import load_superpixel_data, avg_image


def parse_args_custom():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--model_name_t', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "custom",
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    parser.add_argument('model_path', type=str, help='path to custom model')
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.model_name_t is not None, \
        "You must specify the --model_name_t parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.model_name_t == "custom":
        # use OURS
        model_path = os.path.join("log", args.model_path)

    else:
        # use mododepth pretrained ones
        download_model_if_doesnt_exist(args.model_name_t)
        model_path = os.path.join("models", args.model_name_t)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    # load more information
    input_channels = loaded_dict_enc['input_channels']
    sup_method = loaded_dict_enc['superpixel_method']
    sup_args = loaded_dict_enc['superpixel_arguments']

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        output_directory = args.image_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image_pil = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image_pil).unsqueeze(0)

            input_image = input_image.to(device)

            if input_channels is 3:
                inp = input_image

            elif input_channels is 4:
                sup = load_superpixel(image_path, sup_method, sup_args, args.ext)
                inp = torch.cat((input_image, sup), dim=1)

            elif input_channels is 6:
                sup = load_superpixel(image_path, sup_method, sup_args, args.ext, img=input_image_pil)
                inp = torch.cat((input_image, sup), dim=1)

            else:
                raise NotImplementedError("Given channel size is not implemented!")

            # PREDICTION
            features = encoder(inp)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
            np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved prediction to {}".format(
                idx + 1, len(paths), name_dest_im))

    print('-> Done!')


def load_superpixel(image_path, superpixel_method, superpixel_arguments, img_ext, img=None):
    """

    :param image_path:
    :param superpixel_method:
    :param superpixel_arguments:
    :param img_ext:
    :param img:
    :return:
    """

    superpixel_ident = str(superpixel_method)

    for a in superpixel_arguments:
        # replace . with _
        a = str(a).replace(".", "_")

        superpixel_ident += a

    path = image_path.replace(img_ext, superpixel_ident + ".npz")

    super_label = load_superpixel_data(path, superpixel_method, superpixel_arguments, img_ext)

    # channel is 3
    if img is not None:
        # need to convert image from pil to numpy first
        img_np = np.array(img)

        super_img = avg_image(img_np, super_label)

        # convert label to pillow image
        super_img = transforms.ToPILImage()(super_img)

        # be sure to delete img_np
        del img_np

        sup = super_img
    else:

        # add an empty dimension for channel
        super_label = np.expand_dims(super_label, axis=2)

        # convert label to pillow image
        # since there is no 16bit support, we need to use 32bit:
        # mode I: (32-bit signed integer pixels)
        super_label = transforms.ToPILImage(mode='I')(super_label)

        sup = super_label

    return sup


if __name__ == '__main__':
    print("Starting test simple...")
    args = parse_args_custom()
    test_simple(args)
