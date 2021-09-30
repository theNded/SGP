import os, sys

file_path = os.path.abspath(__file__)
project_path = os.path.dirname(os.path.dirname(file_path))
sys.path.append(project_path)

import glob

import open3d as o3d
import numpy as np

from dataset.base import DatasetBase
from geometry.pointcloud import make_o3d_pointcloud

class Dataset3DMatchTrain(DatasetBase):
    def __init__(self, root, scenes, overlap_thr=0.3):
        self.overlap_thr = overlap_thr
        super(Dataset3DMatchTrain, self).__init__(root, scenes)

    # override
    def parse_scene(self, root, scene):
        scene_path = os.path.join(root, scene)

        l = len(scene_path)
        fnames = sorted(glob.glob(os.path.join(scene_path, '*.npz')))
        fnames = [fname[l + 1:] for fname in fnames]

        # Load overlaps.txt
        pair_fname = os.path.join(scene_path, 'overlaps.txt')
        with open(pair_fname, 'r') as f:
            pair_content = f.readlines()

        pairs = []
        binary_info = []

        # For a preprocessed 3DMatch training dataset,
        # binary_info is the gt label: pre-calibrated identity matrix.
        for line in pair_content:
            lst = line.strip().split(' ')
            src_idx = int(lst[0].split('.')[0].split('_')[-1])
            dst_idx = int(lst[1].split('.')[0].split('_')[-1])
            overlap = float(lst[2])

            if overlap >= self.overlap_thr:
                pairs.append((src_idx, dst_idx))
                binary_info.append(np.eye(4))

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
        return make_o3d_pointcloud(np.load(fname)['pcd'])
