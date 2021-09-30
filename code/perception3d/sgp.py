import sys, os

file_path = os.path.abspath(__file__)
project_path = os.path.dirname(os.path.dirname(file_path))
sys.path.append(project_path)

fcgf_path = os.path.join(project_path, 'ext', 'FCGF')
sys.path.append(fcgf_path)

from tqdm import tqdm

import torch
import numpy as np
import open3d as o3d

from sgp_base import SGPBase
from dataset.threedmatch_sgp import Dataset3DMatchSGP
from perception3d.adaptor import DatasetFCGFAdaptor, FCGFConfigParser, load_fcgf_model, fcgf_train, reload_config, register


class SGP3DRegistration(SGPBase):
    def __init__(self):
        super(SGP3DRegistration, self).__init__()

    # override
    def perception_bootstrap(self, src_data, dst_data, src_info, dst_info,
                             config):
        T, fitness = register(src_data, dst_data, 'FPFH', 'RANSAC', None,
                              config)
        if config.debug:
            src_data.paint_uniform_color([1, 0, 0])
            dst_data.paint_uniform_color([0, 1, 0])
            o3d.visualization.draw([src_data.transform(T), dst_data])
        return T, fitness

    # override
    def perception(self, src_data, dst_data, src_info, dst_info, model,
                   config):
        T, fitness = register(src_data, dst_data, 'FCGF', 'RANSAC', model,
                              config)
        if config.debug:
            src_data.paint_uniform_color([1, 0, 0])
            dst_data.paint_uniform_color([0, 1, 0])
            o3d.visualization.draw([src_data.transform(T), dst_data])
        return T, fitness

    # override
    def train_adaptor(self, sgp_dataset, config):
        fcgf_train(sgp_dataset, config)

    def run(self, config):
        epochs = config.max_epoch
        base_pseudo_label_dir = config.pseudo_label_dir
        base_outdir = config.out_dir

        # Bootstrap
        if config.restart_meta_iter < 0:
            pseudo_label_path_bs = os.path.join(base_pseudo_label_dir, 'bs')
            teach_dataset = Dataset3DMatchSGP(config.dataset_path,
                                              config.scenes,
                                              pseudo_label_path_bs, 'teaching',
                                              config.overlap_thr)
            # We need mutual filter for less reliable FPFH
            config.mutual_filter = True
            sgp.teach_bootstrap(teach_dataset, config)

            learn_dataset = DatasetFCGFAdaptor(
                Dataset3DMatchSGP(config.dataset_path, config.scenes,
                                  pseudo_label_path_bs, 'learning',
                                  config.overlap_thr), config)
            config.out_dir = os.path.join(base_outdir, 'bs')
            sgp.learn(learn_dataset, config)

        # Loop
        start_meta_iter = max(config.restart_meta_iter, 0)
        for i in range(start_meta_iter, config.meta_iters):
            pseudo_label_path_i = os.path.join(config.pseudo_label_dir,
                                               '{:02d}'.format(i))
            teach_dataset = Dataset3DMatchSGP(config.dataset_path,
                                              config.scenes,
                                              pseudo_label_path_i, 'teaching',
                                              config.overlap_thr)

            # No mutual filter results in better FCGF teaching
            config.mutual_filter = False
            model = load_fcgf_model(config)
            sgp.teach(teach_dataset, model, config)

            learn_dataset = DatasetFCGFAdaptor(
                Dataset3DMatchSGP(config.dataset_path, config.scenes,
                                  pseudo_label_path_i, 'learning',
                                  config.overlap_thr), config)

            # There is a bug in FCGF finetuning.
            # Suppose previous epochs are [1, n],
            # then finetuning will be [n, n+n] (double counting n), instead of [n+1, n+n]
            # To address this without changing the original repo, we need to reduce max epochs by 1.
            # The actual finetuning iters will be correct, while FCGF's output will be slightly different.
            if config.finetune:
                config.resume_dir = config.out_dir
                config = reload_config(config)
                config.max_epoch += (epochs - 1)
            else:
                config.out_dir = os.path.join(base_outdir, '{:02d}'.format(i))

            sgp.learn(learn_dataset, config)


if __name__ == '__main__':
    parser = FCGFConfigParser()
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

    sgp = SGP3DRegistration()
    sgp.run(config)
