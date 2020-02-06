# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import json
import time

import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import datasets
import networks
# import normal_to_depth as nd
from layers import disp_to_depth
import normal2disp as nd
from kitti_utils import *
from superpixel_utils import convert_rgb_to_superpixel
from layers import *
from utils import *
from skimage.segmentation import find_boundaries


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        if self.opt.dataset == "kitti_superpixel":
            # use a four channel resnet encoder
            self.models["encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained", num_input_channels=4)
        else:
            self.models["encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained")

        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        if self.opt.decoder == "normal_vector":
            self.models["depth"] = networks.NormalDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales)
        else:
            # use standard monodepth2 depth decoder
            self.models["depth"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales)

        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "kitti_superpixel": datasets.SuperpixelDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        #
        # Get images that should be used for selected data split
        #
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        # Check if superpixel dataset is used and create superpixel image
        if "superpixel" in self.opt.dataset or self.opt.superpixel_mask_loss_binary or self.opt.normal_loss or \
                self.opt.superpixel_mask_loss_continuous:

            # force dataset to be kitti superpixel dataset
            self.opt.dataset = "kitti_superpixel"
            self.dataset = datasets_dict[self.opt.dataset]

            # get number of channels to use for superpixel
            # 4 channel will use numpy array with superpixel indices
            # 3 channel will use only image averaged over superpixel area
            # 6 channel will use normal image + image averaged over superpixel area

            num_sup_channels = self.opt.input_channels
            print("Using {} channel input.".format(num_sup_channels))

            if self.opt.no_superpixel_check:
                # dont check if superpixel information is correct
                print("Warning: Skip checking superpixel information.")

            else:
                print("Start converting training images to superpixel.")
                convert_rgb_to_superpixel(self.opt.data_path, train_filenames, self.opt.superpixel_method,
                                          self.opt.superpixel_arguments, img_ext=img_ext, num_channel=num_sup_channels)

                print("Start converting validation images to superpixel.")
                convert_rgb_to_superpixel(self.opt.data_path, val_filenames, self.opt.superpixel_method,
                                          self.opt.superpixel_arguments, img_ext=img_ext, num_channel=num_sup_channels)

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, self.opt, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, self.opt, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":

            if self.opt.dataset == "kitti_superpixel":
                raise NotImplementedError("Using superpixel and shared encoder is not implemented yet!")

            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder

            if self.opt.dataset == "kitti_superpixel":

                if self.opt.input_channels is 4:
                    # use four channel encoder
                    # concat superpixel to the image
                    image = inputs["color_aug", 0, 0]
                    superpixel = inputs["super_label", 0, 0]

                    # need to concat at dimension 1
                    # because it is: (batchsize, channels, h, w)
                    inp = torch.cat((image, superpixel), dim=1)

                elif self.opt.input_channels is 3:
                    # use only superpixel 3 channel input
                    inp = inputs["super_img_aug", 0, 0]

                elif self.opt.input_channels is 6:
                    # use 3 channel superpixel and 3 channel standard rgb image
                    image = inputs["color_aug", 0, 0]
                    superpixel = inputs["super_img_aug", 0, 0]
                    inp = torch.cat((image, superpixel), dim=1)

                else:
                    raise NotImplementedError("Given input channel number is not implemented yet!")

                features = self.models["encoder"](inp)

            else:
                # use standard 3 channel
                features = self.models["encoder"](inputs["color_aug", 0, 0])

            outputs = self.models["depth"](features)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        # changed method to Transform Normal "outputs" to former disparity outputs and return new outputs 
        outputs = self.generate_images_pred(inputs, outputs)

        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        ###
        # edit by Jan

        for scale in self.opt.scales:

            # check which type of decoder is used
            if self.opt.decoder == "normal_vector":

                # tranform normals to depth to disp
                normal_vec = outputs[("normal_vec", scale)]

                K_inv = inputs[("inv_K", scale)]

                disp = nd.normals_to_disp3(K_inv, normal_vec)

                outputs[("disp", scale)] = disp

            else:
                disp = outputs[("disp", scale)]

            if self.opt.v1_multiscale:

                source_scale = scale

            else:
                try:
                    disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                except:
                    raise IOError("Error with shape" + str(disp.shape))

                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]
        # added return
        return outputs

    def compute_normals_loss(self, superpixel, normals):

        """
        compute the loss with superpixel information.
        Forces normal vectors in one superpixel to be equal.

        :param superpixel: superpixel labels
        :param normals: vector normals
        :return: normals loss

        :TODO: check implementation and input shapes
        """

        # init normals loss
        normals_loss = 0

        batch_size = normals.shape[0]
        # shape of normals should be batchsize, 3, h, w

        print("Normals shape in normals loss: {}".format(normals.shape))
        print("Superpixel shape in normals loss: {}".format(superpixel.shape))

        superpixel_indices = torch.unique_consecutive(superpixel)
        zeros = torch.zeros_like(normals)

        for idx in superpixel_indices:
            superpixel_list = torch.where(superpixel == idx, normals, zeros)
            torch.std

        for b in range(batch_size):

            # get indices
            superpixel_indices = torch.unique_consecutive(superpixel[b])

            # convert torch superpixel to numpy
            superpixel_np = superpixel[b].detach().numpy()

            # get pixel for each superpixel area
            superpixel_list = [np.where(superpixel_np == i) for i in superpixel_indices]

            # convert torch normals to numpy
            normals_np = normals[b].detach().numpy()

            # get all normals pixel values per superpixel area
            normals_per_superpixel = [normals_np[idx] for idx in superpixel_list]

            for normals in normals_per_superpixel:
                # calculate standard deviation for each area
                # calculate first for each channel of current area, then sum for current area
                normals_loss += np.sum(np.std(normals, axis=0))

        return normals_loss

    def get_superpixel_mask_loss_binary(self, disp, img,  superpixel , threshold = 0.045):
        """
        compute the loss with superpixel information.
        Takes the superpixel boundaries to mask the gradient of the disparity.
        Assumes same Height and Width  aka. scale of disp and superpixel and img

        :param superpixel: superpixel indices/labels
        :param disp: disparity output from network
        :param img: color output from network
        :param threshold: gradient threshold
        :return: superpixel loss

        :TODO: finish implementation
        """

        # calculate gradient from disp in x/y - direction 
        # lenght is reduced by 1 
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        # calculate normal image gradient
        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True).squeeze(0).squeeze(0)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True).squeeze(0).squeeze(0)

        # Calculate boundaries from SP-labels. Subpixel would be possible too. Would return double sized image.
        # https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/boundaries.py#L48
        # boundaries = find_boundaries(superpixel, mode='outer')

        # calculate x and y boundaries seperatly

        # transform superpixel segments to tensor
        # in shape (1,H,W)
        labels = torch.tensor(superpixel).cuda().float()

        # calculate gradient of the labels in x/y - direction & remove one dim
        boundaries_x = torch.abs(labels[:,:, :-1] - labels[:,:, 1:]).squeeze(0)
        boundaries_y = torch.abs(labels[:,:-1, :] - labels[:,1:, :]).squeeze(0)

        # array with ones
        # ones_x = torch.ones(boundaries_x.shape).cuda().float()
        # ones_y = torch.ones(boundaries_y.shape).cuda().float()
        # zeros_x = torch.zeros_like(boundaries_x).cuda().float()
        # zeros_y = torch.zeros_like(boundaries_y).cuda().float()

        zero = torch.tensor([0]).cuda().float()
        one = torch.tensor([1]).cuda().float()

        # if inside SP --> 1, if at SP edge --> 0
        boundaries_x = torch.where(boundaries_x == 0,one,zero)
        boundaries_y = torch.where(boundaries_y == 0,one,zero)


        # if the image gradient on an SP-edge is low --> its likely the same plane
        # if same plane ---> remove SP edge, so set value to 1
        boundaries_x = torch.where((boundaries_x == 0) & (grad_img_x < threshold), one, boundaries_x)
        boundaries_y = torch.where((boundaries_y == 0) & (grad_img_y < threshold), one, boundaries_y)

        # transform to tensor with shape (1,1,h,w)
        boundaries_x = boundaries_x.unsqueeze(0).unsqueeze(0)
        boundaries_y = boundaries_y.unsqueeze(0).unsqueeze(0)
        
        grad_disp_x *= boundaries_x
        grad_disp_y *= boundaries_y

        return grad_disp_x.mean() + grad_disp_y.mean()

    def get_superpixel_mask_loss_continuous(self, disp, img, superpixel):

        """
        compute the loss with superpixel information.
        Takes the superpixel boundaries to mask the gradient of the disparity.
        Assumes same Height and Width  aka. scale of disp and superpixel and img

        :param superpixel: superpixel indices/labels
        :param disp: disparity output from network
        :param img: color output from network
        :param threshold: gradient threshold
        :return: superpixel loss

        :TODO: finish implementation
        """

        # calculate gradient from disp in x/y - direction
        # lenght is reduced by 1
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        # calculate normal image gradient
        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True).squeeze(0).squeeze(0)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True).squeeze(0).squeeze(0)

        # Calculate boundaries from SP-labels. Subpixel would be possible too. Would return double sized image.
        # https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/boundaries.py#L48
        # boundaries = find_boundaries(superpixel, mode='outer')

        # calculate x and y boundaries seperatly

        # transform superpixel segments to tensor
        # in shape (1,H,W)
        labels = torch.tensor(superpixel).cuda().float()

        # calculate gradient of the labels in x/y - direction & remove one dim
        boundaries_x = torch.abs(labels[:, :, :-1] - labels[:, :, 1:]).squeeze(0)
        boundaries_y = torch.abs(labels[:, :-1, :] - labels[:, 1:, :]).squeeze(0)

        zero = torch.tensor([0]).cuda().float()
        one = torch.tensor([1]).cuda().float()

        # if inside SP --> 1, if at SP edge --> 0
        # if the image gradient on an SP-edge is low --> its likely the same plane
        # if same plane ---> set value to continuous value from smooth loss

        boundaries_x = torch.where(boundaries_x == 0, one, torch.exp(-grad_img_x))
        boundaries_y = torch.where(boundaries_y == 0, one, torch.exp(-grad_img_y))

        # transform to tensor with shape (1,1,h,w)
        boundaries_x = boundaries_x.unsqueeze(0).unsqueeze(0)
        boundaries_y = boundaries_y.unsqueeze(0).unsqueeze(0)

        grad_disp_x *= boundaries_x
        grad_disp_y *= boundaries_y

        return grad_disp_x.mean() + grad_disp_y.mean()

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """

        """
        Some explanation for the following lines:
        
        options:
        
        1. v1_multiscale: 
            True: the reprojection error will be calculated between same scales
            False: the reprojection error will be calculated between current scale and scale=0
            
        2. frame_ids:
            -1: last frame
            0: current frame
            1: next frame
        
        3. automasking:
            True: calculate identity reprojection loss
            False: dont calculate identity reprojection loss
        
        4. Reprojection loss:
            reprojection between current scale and frame id to target scale and id=0
            
        4. Identity reprojection loss:
        
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            # not used per default
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            # USED PER DEFAULT
            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            # not used per default
            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            # not used per default
            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            # USED PER DEFAULT
            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            # USED PER DEFAULT
            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                        idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)

            if self.opt.superpixel_mask_loss_binary:
                # get superpixel label data for current scale
                superpixel = inputs["super_label", 0, scale]
                smooth_loss = self.get_superpixel_mask_loss_binary(disp, color, superpixel)

            elif self.opt.superpixel_mask_loss_continuous:
                # get superpixel label data for current scale
                superpixel = inputs["super_label", 0, scale]
                smooth_loss = self.get_superpixel_mask_loss_continuous(disp, color, superpixel)

            else:
                smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

            # add addtional superpixel loss if selected
            if self.opt.normal_loss:

                # need to be sure that superpixel dataset is used
                if self.opt.dataset == "kitti_superpixel":

                    superpixel = inputs[("super_label", 0, scale)]
                    normals = outputs[("normal_vec", 0)]
                    loss += self.opt.normals_smoothness * self.compute_normals_loss(superpixel, normals) / (
                                2 ** scale)

                else:
                    print("Warning: superpixel loss cant be used, because superpixel dataset isn't selected!")

            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
                                     self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
