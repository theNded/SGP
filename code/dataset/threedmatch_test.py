import os, sys

file_path = os.path.abspath(__file__)
project_path = os.path.dirname(os.path.dirname(file_path))
sys.path.append(project_path)

import glob

import open3d as o3d
import numpy as np

from dataset.base import DatasetBase


class Dataset3DMatchTest(DatasetBase):
    def __init__(self, root, scenes):
        super(Dataset3DMatchTest, self).__init__(root, scenes)

    # override
    def parse_scene(self, root, scene):
        scene_path = os.path.join(root, scene)

        l = len(scene_path)
        fnames = sorted(
            glob.glob(os.path.join(scene_path, '*.ply')),
            key=lambda fname: int(fname.split('.')[0].split('_')[-1]))
        fnames = [fname[l + 1:] for fname in fnames]

        # Load gt
        scene_gt_path = os.path.join(root, scene + '-evaluation')
        gt_fname = os.path.join(scene_gt_path, 'gt.log')
        with open(gt_fname, 'r') as f:
            pair_content = f.readlines()

        pairs = []
        binary_info = []

        # For a 3DMatch test dataset,
        # binary_info is the gt label: src to dst transformation.
        for i in range(0, len(pair_content), 5):
            lst = pair_content[i].strip().split('\t')
            src_idx = int(lst[0])
            dst_idx = int(lst[1])

            res = map(lambda x: np.fromstring(x.strip(), sep='\t'),
                      pair_content[i+1:i+5])
            T_src2dst = np.stack(list(res))
            pairs.append((src_idx, dst_idx))
            binary_info.append(np.linalg.inv(T_src2dst))

        return {
            'folder': scene,
            'fnames': fnames,
            'pairs': pairs,
            'unary_info': [None for i in range(len(fnames))],
            'binary_info': binary_info
        }

    # override
    def load_data(self, folder, fname):
        fname = os.path.join(self.root, folder, fname)
        return o3d.io.read_point_cloud(fname)
