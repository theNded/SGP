import os, sys

file_path = os.path.abspath(__file__)
project_path = os.path.dirname(os.path.dirname(file_path))
sys.path.append(project_path)

caps_path = os.path.join(project_path, 'ext', 'caps')
sys.path.append(caps_path)

from ext.caps.CAPS.caps_model import CAPSModel
from ext.caps.utils import cycle

from dataset.megadepth_train import DatasetMegaDepthTrain
from dataset.megadepth_test import DatasetMegaDepthTest
from dataset.megadepth_sgp import DatasetMegaDepthSGP
from geometry.image import *

from geometry.common import rotation_error, angular_translation_error

from tensorboardX import SummaryWriter
import configargparse

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

import utils
import collections
from tqdm import tqdm
import dataloader.data_utils as data_utils

rand = np.random.RandomState(234)


class CAPSConfigParser(configargparse.ArgParser):
    def __init__(self):
        super().__init__(default_config_files=[
            os.path.join(os.path.dirname(__file__), 'caps_train_config.yml')
        ],
                         conflict_handler='resolve')

        ## path options
        self.add('--datadir', type=str, help='the dataset directory')
        self.add("--logdir",
                 type=str,
                 default='caps_logs',
                 help='dir of tensorboard logs')
        self.add("--outdir",
                 type=str,
                 default='caps_outputs',
                 help='dir of output e.g., ckpts')
        self.add(
            "--ckpt_path",
            type=str,
            default='',
            help='specific checkpoint path to load the model from, '
            'if not specified, automatically reload from most recent checkpoints'
        )
        self.add('--pseudo_label_dir',
                 type=str,
                 default='caps_pseudo_label',
                 help='the pseudo-gt directory storing pairs and F matrices')
        self.add(
            '--label_dir',
            type=str,
            default='',
            help=
            'the gt directory storing pairs and F matrices. Reserved for pose test set.'
        )

        # SGP options
        self.add('--scenes',
                 nargs='+',
                 help='scenes used for training/testing')
        self.add('--inlier_ratio_thr', type=float, default=0.001)
        self.add('--num_matches_thr', type=int, default=100)
        self.add('--sample_rate',
                 type=float,
                 default=1,
                 help='rate of samples from the huge megadepth dataset')
        self.add('--num_kpts',
                 type=int,
                 default=10000,
                 help='number of key points detected during teaching')
        self.add('--match_ratio_test',
                 type=bool,
                 default=True,
                 help='performs ratio test in feature matching')
        self.add('--match_ratio_thr',
                 type=float,
                 default=0.75,
                 help='ratio between best and second best matchings')
        self.add('--ransac_thr',
                 type=float,
                 default=1e-3,
                 help='RANSAC threshold in estimating essential matrices')

        self.add(
            '--restart_meta_iter',
            type=int,
            default=-1,
            help='start of teacher-student iterations. -1 indicates bootstrap')
        self.add('--max_meta_iters',
                 type=int,
                 default=2,
                 help='number of teacher-student iterations')
        self.add('--finetune',
                 action='store_true',
                 help='train from previous checkpoint during SGP.')

        ## general options
        self.add("--exp_name", type=str, help='experiment name')
        self.add('--n_iters',
                 type=int,
                 default=100,
                 help='max number of training iterations')
        self.add("--save_interval",
                 type=int,
                 default=100,
                 help='frequency of weight ckpt saving')
        self.add('--phase',
                 type=str,
                 default='train',
                 choices=['train', 'val', 'test'])

        # data options
        self.add('--workers',
                 type=int,
                 help='number of data loading workers',
                 default=8)
        self.add('--num_pts',
                 type=int,
                 default=500,
                 help='num of points trained in each pair')
        self.add('--train_kp',
                 type=str,
                 default='mixed',
                 help='sift/random/mixed')
        self.add('--prune_kp',
                 type=int,
                 default=1,
                 help='if prune non-matchable keypoints')

        # training options
        self.add('--batch_size', type=int, default=2, help='input batch size')
        self.add('--lr', type=float, default=1e-4, help='base learning rate')
        self.add(
            "--lrate_decay_steps",
            type=int,
            default=80000,
            help=
            'decay learning rate by a factor every specified number of steps')
        self.add(
            "--lrate_decay_factor",
            type=float,
            default=0.5,
            help=
            'decay learning rate by a factor every specified number of steps')

        ## model options
        self.add(
            '--backbone',
            type=str,
            default='resnet50',
            help=
            'backbone for feature representation extraction. supported: resent'
        )
        self.add(
            '--pretrained',
            type=int,
            default=1,
            help='if use ImageNet pretrained weights to initialize the network'
        )
        self.add('--coarse_feat_dim',
                 type=int,
                 default=128,
                 help='the feature dimension for coarse level features')
        self.add('--fine_feat_dim',
                 type=int,
                 default=128,
                 help='the feature dimension for fine level features')
        self.add(
            '--prob_from',
            type=str,
            default='correlation',
            help=
            'compute prob by softmax(correlation score), or softmax(-distance),'
            'options: correlation|distance')
        self.add(
            '--window_size',
            type=float,
            default=0.125,
            help='the size of the window, w.r.t image width at the fine level')
        self.add('--use_nn',
                 type=int,
                 default=1,
                 help='if use nearest neighbor in the coarse level')

        ## loss function options
        self.add('--std',
                 type=int,
                 default=1,
                 help='reweight loss using the standard deviation')
        self.add('--w_epipolar_coarse',
                 type=float,
                 default=1,
                 help='coarse level epipolar loss weight')
        self.add('--w_epipolar_fine',
                 type=float,
                 default=1,
                 help='fine level epipolar loss weight')
        self.add('--w_cycle_coarse',
                 type=float,
                 default=0.1,
                 help='coarse level cycle consistency loss weight')
        self.add('--w_cycle_fine',
                 type=float,
                 default=0.1,
                 help='fine level cycle consistency loss weight')
        self.add('--w_std',
                 type=float,
                 default=0,
                 help='the weight for the loss on std')
        self.add(
            '--th_cycle',
            type=float,
            default=0.025,
            help=
            'if the distance (normalized scale) from the prediction to epipolar line > this th, '
            'do not add the cycle consistency loss')
        self.add(
            '--th_epipolar',
            type=float,
            default=0.5,
            help=
            'if the distance (normalized scale) from the prediction to epipolar line > this th, '
            'do not add the epipolar loss')

        ## logging options
        self.add('--log_scalar_interval',
                 type=int,
                 default=20,
                 help='print interval')
        self.add('--log_img_interval',
                 type=int,
                 default=500,
                 help='log image interval')

        ## eval options
        self.add('--extract_img_dir',
                 type=str,
                 help='the directory of images to extract features')
        self.add('--extract_out_dir',
                 type=str,
                 help='the directory of images to extract features')

    def get_config(self):
        config = self.parse_args()
        return config


def my_collate(batch):
    ''' Puts each data field into a tensor with outer dimension batch size '''
    batch = list(filter(lambda b: b is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class DatasetMegaDepthAdaptor(Dataset):
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config

        if config.phase == 'train':
            # augment during training
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=1,
                                       contrast=1,
                                       saturation=1,
                                       hue=0.4),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225)),
            ])
        self.phase = config.phase

    def __getitem__(self, idx):
        pass

    def __len__(self):
        return len(self.dataset)


# For vanilla train & test
class DatasetMegaDepthTrainAdaptor(DatasetMegaDepthAdaptor):
    def __init__(self, dataset, config):
        super(DatasetMegaDepthTrainAdaptor, self).__init__(dataset, config)

    def __getitem__(self, idx):
        im_src, im_dst, cam_src, cam_dst, _ = self.dataset[idx]
        h, w = im_src.shape[:2]

        im1_ori = torch.from_numpy(im_src)
        im2_ori = torch.from_numpy(im_dst)

        im1_tensor = self.transform(im_src)
        im2_tensor = self.transform(im_dst)

        coord1 = data_utils.generate_query_kpts(im_src, self.config.train_kp,
                                                10 * self.config.num_pts, h, w)

        # if no keypoints are detected
        if len(coord1) == 0:
            return None

        # prune query keypoints that are not likely to have correspondence in the other image
        coord1 = utils.random_choice(coord1, self.config.num_pts)
        coord1 = torch.from_numpy(coord1).float()

        K_src, T_src = cam_src
        K_dst, T_dst = cam_dst

        T_src2dst = torch.from_numpy(T_dst.dot(np.linalg.inv(T_src)))
        F = compute_fundamental_from_poses(K_src, K_dst, T_src, T_dst)
        F = torch.from_numpy(F).float() / (F[-1, -1] + 1e-16)

        out = {
            'im1_ori': im1_ori,
            'im2_ori': im2_ori,
            'intrinsic1': K_src,
            'intrinsic2': K_dst,

            # Additional, for training
            'im1': im1_tensor,
            'im2': im2_tensor,
            'coord1': coord1,
            'F': F,
            'pose': T_src2dst
        }

        return out


# For SGP train
class DatasetMegaDepthSGPAdaptor(DatasetMegaDepthAdaptor):
    def __init__(self, dataset, config):
        super(DatasetMegaDepthSGPAdaptor, self).__init__(dataset, config)

    def __getitem__(self, idx):
        im1, im2, K_src, K_dst, F = self.dataset[idx]
        h, w = im1.shape[:2]

        im1_ori, im2_ori = torch.from_numpy(im1), torch.from_numpy(im2)

        im1_tensor = self.transform(im1)
        im2_tensor = self.transform(im2)

        coord1 = data_utils.generate_query_kpts(im1, self.config.train_kp,
                                                10 * self.config.num_pts, h, w)

        # if no keypoints are detected
        if len(coord1) == 0:
            return None

        # prune query keypoints that are not likely to have correspondence in the other image
        coord1 = utils.random_choice(coord1, self.config.num_pts)
        coord1 = torch.from_numpy(coord1).float()

        F = torch.from_numpy(F).float() / (F[-1, -1] + 1e-16)

        out = {
            'im1_ori': im1_ori,
            'im2_ori': im2_ori,
            'intrinsic1': K_src,
            'intrinsic2': K_dst,

            # Additional, for training
            'im1': im1_tensor,
            'im2': im2_tensor,
            'coord1': coord1,
            'F': F,

            # Pose is required in the base but not used in CAPSModel
            'pose': np.eye(4)
        }

        return out


def align(im_src, im_dst, K_src, K_dst, detector, feature, model, config):
    kpts_src = detect_keypoints(im_src, detector, num_kpts=config.num_kpts)
    kpts_dst = detect_keypoints(im_dst, detector, num_kpts=config.num_kpts)

    # Too few keypoints
    if len(kpts_src) < 5 or len(kpts_dst) < 5:
        return np.eye(3), np.eye(3), np.ones((3)), [], [], [], np.zeros((0))

    feats_src = extract_feats(im_src, kpts_src, feature, model)
    feats_dst = extract_feats(im_dst, kpts_dst, feature, model)
    matches = match_feats(feats_src, feats_dst, feature,
                          config.match_ratio_test, config.match_ratio_thr)
    num_matches = len(matches)

    # Too few matches
    if num_matches <= 5:  # 5-pts method
        return np.eye(3), np.eye(3), np.ones(
            (3)), kpts_src, kpts_dst, [], np.zeros((len(matches)))

    E, mask, R, t = estimate_essential(kpts_src,
                                       kpts_dst,
                                       matches,
                                       K_src,
                                       K_dst,
                                       th=config.ransac_thr)
    F = np.linalg.inv(K_dst).T.dot(E).dot(np.linalg.inv(K_src))
    F = F / (F[-1, -1] + 1e-16)

    return F, R, t, kpts_src, kpts_dst, matches, mask


def caps_train(dataset, config):
    # save a copy for the current config in out_folder
    out_folder = os.path.join(config.outdir, config.exp_name)
    os.makedirs(out_folder, exist_ok=True)
    f = os.path.join(out_folder, 'config.txt')
    with open(f, 'w') as file:
        for arg in vars(config):
            attr = getattr(config, arg)
            file.write('{} = {}\n'.format(arg, attr))

    # tensorboard writer
    tb_log_dir = os.path.join(config.logdir, config.exp_name)
    print('tensorboard log files are stored in {}'.format(tb_log_dir))
    writer = SummaryWriter(tb_log_dir)

    # megadepth data loader
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=config.batch_size,
                                             shuffle=True,
                                             num_workers=config.workers,
                                             collate_fn=my_collate)

    model = CAPSModel(config)

    start_step = model.start_step
    dataloader_iter = iter(cycle(dataloader))
    for step in range(start_step + 1, start_step + config.n_iters + 1):
        data = next(dataloader_iter)
        if data is None:
            continue

        model.set_input(data)
        model.optimize_parameters()
        model.write_summary(writer, step)
        if step % config.save_interval == 0 and step > 0:
            model.save_model(step)


def caps_test(dataset, config):
    model = CAPSModel(config)

    r_errs = []
    t_errs = []

    for data in tqdm(dataset):
        im_src, im_dst, cam_src, cam_dst, _ = data

        K_src, T_src = cam_src
        K_dst, T_dst = cam_dst
        T_src2dst_gt = T_dst.dot(np.linalg.inv(T_src))

        F, R, t, kpts_src, kpts_dst, matches, mask = align(
            im_src, im_dst, K_src, K_dst, 'sift', 'caps', model, config)

        r_err = rotation_error(R, T_src2dst_gt[:3, :3])
        t_err = angular_translation_error(t, T_src2dst_gt[:3, 3])
        r_errs.append(r_err)
        t_errs.append(t_err)

        if config.debug:
            im = draw_matches(kpts_src, kpts_dst, matches, im_src, im_dst, F,
                              mask)
            cv2.imshow('matches', im)
            cv2.waitKey(-1)

    return np.array(r_errs), np.array(t_errs)
