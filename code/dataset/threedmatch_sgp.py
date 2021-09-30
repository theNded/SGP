import os, sys

file_path = os.path.abspath(__file__)
project_path = os.path.dirname(os.path.dirname(file_path))
sys.path.append(project_path)

import glob

import open3d as o3d
import numpy as np
from tqdm import tqdm

from dataset.base import DatasetBase
from geometry.pointcloud import make_o3d_pointcloud, extract_feats, match_feats, solve, refine
PSEUDO_LABEL_FNAME = 'pseudo-label.log'


class Dataset3DMatchSGP(DatasetBase):
    '''
    During teaching: labels are written to a separate directory
    During learning: it acts like the train, with labels in a separate directory
    '''
    def __init__(self, data_root, scenes, label_root, mode, overlap_thr=0.3):
        self.label_root = label_root
        self.overlap_thr = overlap_thr

        if not os.path.exists(label_root):
            print(
                'label root {} does not exist, entering teaching mode.'.format(
                    label_root))
            self.mode = 'teaching'
            os.makedirs(label_root, exist_ok=True)
        elif mode == 'teaching':
            print('label root {} will be overwritten to enter teaching mode'.
                  format(label_root))
            self.mode = 'teaching'
        else:
            print('label root {} exists, entering learning mode.'.format(
                label_root))
            self.mode = 'learning'

        super(Dataset3DMatchSGP, self).__init__(data_root, scenes)

    # override
    def parse_scene(self, root, scene):
        if self.mode == 'teaching':
            return self._parse_scene_teaching(root, scene)
        elif self.mode == 'learning':
            return self._parse_scene_learning(root, scene)
        else:
            print('Unsupported mode, abort')
            exit()

    # override
    def load_data(self, folder, fname):
        fname = os.path.join(self.root, folder, fname)
        return make_o3d_pointcloud(np.load(fname)['pcd'])

    def write_pseudo_label(self, idx, label, overlap):
        scene_idx = self.scene_idx_map[idx]
        pair_idx = self.pair_idx_map[idx]

        # Access actual data
        scene = self.scenes[scene_idx]
        i, j = scene['pairs'][pair_idx]
        folder = scene['folder']

        label_file = os.path.join(self.label_root, folder, PSEUDO_LABEL_FNAME)
        with open(label_file, 'a') as f:
            f.write('{} {} {} '.format(i, j, overlap))
            label_str = ' '.join(map(str, label.flatten()))
            f.write(label_str)
            f.write('\n')

    def _parse_scene_teaching(self, root, scene):
        # Generate pseudo labels
        label_path = os.path.join(self.label_root, scene)
        os.makedirs(label_path, exist_ok=True)
        label_file = os.path.join(label_path, PSEUDO_LABEL_FNAME)

        if os.path.exists(label_file):
            os.remove(label_file)
        with open(label_file, 'w') as f:
            pass

        # Load actual data
        scene_path = os.path.join(root, scene)

        # Load filenames
        l = len(scene_path)
        fnames = sorted(glob.glob(os.path.join(scene_path, '*.npz')))
        fnames = [fname[l + 1:] for fname in fnames]

        # Load overlaps.txt
        pair_fname = os.path.join(scene_path, 'overlaps.txt')
        with open(pair_fname, 'r') as f:
            pair_content = f.readlines()

        pairs = []
        binary_info = []

        # For a 3DMatch dataset for teaching,
        # binary_info is (optional) for filtering: overlap
        for line in pair_content:
            lst = line.strip().split(' ')
            src_idx = int(lst[0].split('.')[0].split('_')[-1])
            dst_idx = int(lst[1].split('.')[0].split('_')[-1])
            overlap = float(lst[2])

            if overlap >= self.overlap_thr:
                pairs.append((src_idx, dst_idx))
                binary_info.append(overlap)

        return {
            'folder': scene,
            'fnames': fnames,
            'pairs': pairs,
            'unary_info': [None for i in range(len(fnames))],
            'binary_info': binary_info
        }

    '''
    Pseudo-Labels not available. Generate paths for writing to them later.
    '''

    def _parse_scene_learning(self, root, scene):
        # Load pseudo labels
        label_path = os.path.join(self.label_root, scene, PSEUDO_LABEL_FNAME)
        if not os.path.exists(label_path):
            raise Exception('{} not found', label_path)

        # Load actual data
        scene_path = os.path.join(root, scene)

        # Load filenames
        l = len(scene_path)
        fnames = sorted(glob.glob(os.path.join(scene_path, '*.npz')))
        fnames = [fname[l + 1:] for fname in fnames]

        # Load overlaps.txt
        with open(label_path, 'r') as f:
            pair_content = f.readlines()

        pairs = []
        binary_info = []

        # For a 3DMatch dataset for learning,
        # binary_info is the pseudo label: src to dst transformation.
        for line in pair_content:
            lst = line.strip().split(' ')
            src_idx = int(lst[0].split('.')[0].split('_')[-1])
            dst_idx = int(lst[1].split('.')[0].split('_')[-1])
            overlap = float(lst[2])
            T_data = list(map(float, lst[3:]))
            T = np.array(T_data).reshape((4, 4))

            if overlap >= self.overlap_thr:
                pairs.append((src_idx, dst_idx))
                binary_info.append(T)

        return {
            'folder': scene,
            'fnames': fnames,
            'pairs': pairs,
            'unary_info': [None for i in range(len(fnames))],
            'binary_info': binary_info
        }
