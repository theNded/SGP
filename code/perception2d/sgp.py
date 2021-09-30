import os, sys

file_path = os.path.abspath(__file__)
project_path = os.path.dirname(os.path.dirname(file_path))
sys.path.append(project_path)

caps_path = os.path.join(project_path, 'ext', 'caps')
sys.path.append(caps_path)

import cv2
import torch

from sgp_base import SGPBase
from dataset.megadepth_sgp import DatasetMegaDepthSGP
from perception2d.adaptor import CAPSConfigParser, DatasetMegaDepthSGPAdaptor, CAPSModel, caps_train, caps_test, align
from geometry.image import *


class SGP2DFundamental(SGPBase):
    def __init__(self):
        super(SGP2DFundamental, self).__init__()

    # override
    def perception_bootstrap(self, src_data, dst_data, src_info, dst_info,
                             config):
        F, R, t, kpts_src, kpts_dst, matches, mask = align(
            src_data, dst_data, src_info, dst_info, 'sift', 'sift', None,
            config)

        if config.debug:
            im = draw_matches(kpts_src, kpts_dst, matches, src_data, dst_data,
                              F, mask)
            cv2.imshow('matches', im)
            cv2.waitKey(-1)

        return F, (mask.sum(), len(matches))

    # override
    def perception(self, src_data, dst_data, src_info, dst_info, model,
                   config):
        F, R, t, kpts_src, kpts_dst, matches, mask = align(
            src_data, dst_data, src_info, dst_info, 'sift', 'caps', model,
            config)

        if config.debug:
            im = draw_matches(kpts_src, kpts_dst, matches, src_data, dst_data,
                              F, mask)
            cv2.imshow('matches', im)
            cv2.waitKey(-1)

        return F, (mask.sum(), len(matches))

    # override
    def train_adaptor(self, sgp_dataset, config):
        caps_train(sgp_dataset, config)

    def run(self, config):
        base_outdir = config.outdir
        base_logdir = config.logdir
        base_pseudo_label_dir = config.pseudo_label_dir

        pseudo_label_path_bs = os.path.join(base_pseudo_label_dir, 'bs')

        if config.restart_meta_iter < 0:
            # Only sample a subset for teaching.
            teach_dataset = DatasetMegaDepthSGP(config.datadir,
                                           config.scenes,
                                           pseudo_label_path_bs,
                                           'teaching',
                                           inlier_ratio_thr=config.inlier_ratio_thr,
                                           num_matches_thr=config.num_matches_thr,
                                           sample_rate=config.sample_rate)
            print('Dataset size: {}'.format(len(teach_dataset)))
            sgp.teach_bootstrap(teach_dataset, config)

            learn_dataset = DatasetMegaDepthSGPAdaptor(
                DatasetMegaDepthSGP(config.datadir,
                               config.scenes,
                               pseudo_label_path_bs,
                               'learning',
                               inlier_ratio_thr=config.inlier_ratio_thr,
                               num_matches_thr=config.num_matches_thr,
                               sample_rate=1), config)
            config.outdir = os.path.join(base_outdir, 'bs')
            config.logdir = os.path.join(base_logdir, 'bs')
            sgp.learn(learn_dataset, config)

        config.match_ratio_test = False
        start_meta_iter = max(config.restart_meta_iter, 0)
        for i in range(start_meta_iter, config.max_meta_iters):
            pseudo_label_path_i = os.path.join(base_pseudo_label_dir,
                                               '{:02d}'.format(i))
            teach_dataset = DatasetMegaDepthSGP(config.datadir,
                                           config.scenes,
                                           pseudo_label_path_i,
                                           'teaching',
                                           inlier_ratio_thr=config.inlier_ratio_thr,
                                           num_matches_thr=config.num_matches_thr,
                                           sample_rate=config.sample_rate)
            model = CAPSModel(config)
            sgp.teach(teach_dataset, model, config)

            learn_dataset = DatasetMegaDepthSGPAdaptor(
                DatasetMegaDepthSGP(config.datadir,
                               config.scenes,
                               pseudo_label_path_i,
                               'learning',
                               inlier_ratio_thr=config.inlier_ratio_thr,
                               num_matches_thr=config.num_matches_thr,
                               sample_rate=1), config)

            if not config.finetune:
                config.outdir = os.path.join(base_outdir, '{:02d}'.format(i))
                config.logdir = os.path.join(base_logdir, '{:02d}'.format(i))
            sgp.learn(learn_dataset, config)


if __name__ == '__main__':
    parser = CAPSConfigParser()
    parser.add(
        '--config',
        is_config_file=True,
        default=os.path.join(os.path.dirname(__file__),
                             'config_sgp_sample.yml'),
        help='YAML config file path. Please refer to caps_config.yml as a '
        'reference. It overrides the default config file, but will be '
        'overridden by other command line inputs.')
    parser.add('--debug', action='store_true')
    config = parser.get_config()

    sgp = SGP2DFundamental()
    sgp.run(config)
