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
import torch.nn as nn
import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
import normal_to_depth as nd
from superpixel_utils import load_superpixel_data, avg_image
import cv2

# MOD

def parse_args_custom():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument("--decoder",
                             type=str,
                             help="standart or normal_vector",
                             default="standart",
                             choices=["standart", "normal_vector"])

    parser.add_argument("--input_channels",
                             type=int,
                             help="Number of input channels",
                             choices=[3, 4, 6],
                             default=3)
    parser.add_argument("--superpixel_method",
                             type=str,
                             help="method to use for superpixel calculation",
                             choices=["fz", "slic"],
                             default="fz")

    # additional arguments for superpixel calculation
    # 1. for fz: scale=int(scale), sigma=sigma, min_size=int(min_size)
    # 2. for slic: n_segments=int(num_seg), compactness=comp, sigma=sig
    parser.add_argument("--superpixel_arguments",
                             nargs="+",
                             type=float,
                             help="additional arguments for superpixel methods",
                             default=[120, 0.8, 80])

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--model_name_t', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "custom",
                            "direct",
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    parser.add_argument('--model_path', type=str, help='path to custom model')
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

    elif args.model_name_t == "direct":
        model_path = ""

    else:
        # use mododepth pretrained ones
        download_model_if_doesnt_exist(args.model_name_t)
        model_path = os.path.join("models", args.model_name_t)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # load more information
    input_channels = args.input_channels
    print("Using {} channel input".format(input_channels))
    sup_method = args.superpixel_method
    sup_args = args.superpixel_arguments

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False, num_input_channels=input_channels)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']

    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}

    encoder.load_state_dict(filtered_dict_enc)

    encoder.to(device)
    encoder.eval()

    if args.decoder == "normal_vector":
        print("   Loading normal decoder")
        depth_decoder = networks.NormalDecoder(
            num_ch_enc=encoder.num_ch_enc, scales=range(4))
    else:
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
                sup = sup.resize((feed_width, feed_height), pil.LANCZOS)
                sup = transforms.ToTensor()(sup).float().unsqueeze(0)
                inp = torch.cat((input_image, sup), dim=1)

            elif input_channels is 6:
                sup = load_superpixel(image_path, sup_method, sup_args, args.ext, img=input_image_pil)
                inp = torch.cat((input_image, sup), dim=1)

            else:
                raise NotImplementedError("Given channel size is not implemented!")

            # PREDICTION
            features = encoder(inp)
            outputs = depth_decoder(features)

            if args.decoder == "normal_vector":

                normal_vec = outputs[("normal_vec", 0)]

                K = np.array([[0.58, 0, 0.5, 0],
                              [0, 1.92, 0.5, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]], dtype=np.float32)

                K[0, :] *= feed_width
                K[1, :] *= feed_height

                inv_K = np.linalg.pinv(K)
                inv_K = np.expand_dims(inv_K, axis=0)
                inv_K = torch.from_numpy(inv_K)

                disp = nd.normals_to_disp3(inv_K, normal_vec, cuda=False)
                # print("new depth tensor shape", depth.shape)

                # Save normal_vector as numpy
                output_name = os.path.splitext(os.path.basename(image_path))[0]
                name_vector_map_npy = os.path.join(output_directory, "{}_vector_map.npy".format(output_name))
                np.save(name_vector_map_npy, normal_vec.cpu().numpy())

                # generate the normal_vec image
                name_vector_map_img = os.path.join(output_directory, "{}_vector_map.png".format(output_name))

                # unpack the vectors into x and y coordina..
                zx = normal_vec[0, :, :, 0]
                zy = normal_vec[0, :, :, 1]

                # stack them, norm them
                normal = np.dstack((-zx, -zy, np.ones_like(normal_vec[0, :, :, 3])))
                n = np.linalg.norm(normal, axis=2)
                normal[:, :, 0] /= n
                normal[:, :, 1] /= n
                normal[:, :, 2] /= n

                # offset and rescale values to be in 0-255
                normal += 1
                normal /= 2
                normal *= 255

                # save the image into the same folder
                cv2.imwrite(name_vector_map_img, normal[:, :, ::-1])

                outputs[("disp", 0)] = disp
            else:

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
