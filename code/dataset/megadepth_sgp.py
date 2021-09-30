import os, sys

file_path = os.path.abspath(__file__)
project_path = os.path.dirname(os.path.dirname(file_path))
sys.path.append(project_path)

import cv2
import numpy as np
import open3d as o3d

from dataset.base import DatasetBase
from geometry.image import skew, detect_keypoints, extract_feats, match_feats, estimate_essential, draw_matches

from tqdm import tqdm

PSEUDO_LABEL_FNAME = 'pseudo-label.log'


# Train and test sets are identical for CAPS
class DatasetMegaDepthSGP(DatasetBase):
    def __init__(self,
                 data_root,
                 scenes,
                 label_root,
                 mode,
                 inlier_ratio_thr=0.3,
                 num_matches_thr=100,
                 sample_rate=0.2):
        self.label_root = label_root
        self.inlier_ratio_thr = inlier_ratio_thr
        self.num_matches_thr = num_matches_thr
        self.sample_rate = sample_rate

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

        super(DatasetMegaDepthSGP, self).__init__(data_root, scenes)

    # override
    def parse_scene(self, root, scene):
        if self.mode == 'teaching':
            return self._parse_scene_teaching(root, scene)
        elif self.mode == 'learning':
            return self._parse_scene_learning(root, scene)
        else:
            print('Unsupported mode, abort')
            exit()

    def write_pseudo_label(self, idx, label, info):
        scene_idx = self.scene_idx_map[idx]
        pair_idx = self.pair_idx_map[idx]

        # Access actual data
        scene = self.scenes[scene_idx]
        i, j = scene['pairs'][pair_idx]
        folder = scene['folder']

        num_inliers, num_matches = info
        label_file = os.path.join(self.label_root, folder, PSEUDO_LABEL_FNAME)
        with open(label_file, 'a') as f:
            f.write('{} {} {} {} '.format(i, j, num_inliers, num_matches))
            label_str = ' '.join(map(str, label.flatten()))
            f.write(label_str)
            f.write('\n')

    def _deterministic_shuffle_(self, seq):
        import random
        random.Random(15213).shuffle(seq)

    def _parse_scene_teaching(self, root, scene):
        # Generate pseudo labels
        label_path = os.path.join(self.label_root, scene)
        os.makedirs(label_path, exist_ok=True)
        label_file = os.path.join(label_path, PSEUDO_LABEL_FNAME)

        if os.path.exists(label_file):
            os.remove(label_file)
        with open(label_file, 'w') as f:
            pass

        scene_path = os.path.join(root, scene)

        fnames = os.listdir(os.path.join(scene_path, 'images'))
        fnames_map = {fname: i for i, fname in enumerate(fnames)}

        cam_fname = os.path.join(scene_path, 'img_cam.txt')
        with open(cam_fname, 'r') as f:
            cam_content = f.readlines()

        cnt = 0
        intrinsics = np.zeros((len(fnames), 3, 3))
        extrinsics = np.zeros((len(fnames), 4, 4))
        for line in cam_content:
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                lst = line.split()
                fname = lst[0]
                idx = fnames_map[fname]

                fx, fy = float(lst[3]), float(lst[4])
                cx, cy = float(lst[5]), float(lst[6])
                intrinsics[idx] = np.array([fx, 0, cx, 0, fy, cy, 0, 0,
                                            1]).reshape((3, 3))
                cnt += 1

        assert cnt == len(fnames)

        # Load pairs.txt
        pair_fname = os.path.join(scene_path, 'pairs.txt')
        with open(pair_fname, 'r') as f:
            pair_content = f.readlines()

        pairs = []
        for line in pair_content:
            lst = line.strip().split(' ')
            src_fname = lst[0]
            dst_fname = lst[1]

            src_idx = fnames_map[src_fname]
            dst_idx = fnames_map[dst_fname]
            pairs.append((src_idx, dst_idx))

        pairs_cnt = len(pairs)
        idx_selection = np.arange(pairs_cnt)
        self._deterministic_shuffle_(idx_selection)
        idx_selection = idx_selection[:int(self.sample_rate *
                                           pairs_cnt)].astype(int)

        return {
            'folder': scene,
            'fnames': fnames,
            'pairs': np.asarray(pairs)[idx_selection],
            'unary_info': intrinsics,
            'binary_info': [None for i in range(len(pairs))]
        }

    def _parse_scene_learning(self, root, scene):
        # Load pseudo labels
        label_path = os.path.join(self.label_root, scene, PSEUDO_LABEL_FNAME)
        if not os.path.exists(label_path):
            raise Exception('{} not found', label_path)

        scene_path = os.path.join(root, scene)

        fnames = os.listdir(os.path.join(scene_path, 'images'))
        fnames_map = {fname: i for i, fname in enumerate(fnames)}

        cam_fname = os.path.join(scene_path, 'img_cam.txt')
        with open(cam_fname, 'r') as f:
            cam_content = f.readlines()

        cnt = 0
        intrinsics = np.zeros((len(fnames), 3, 3))
        for line in cam_content:
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                lst = line.split()
                fname = lst[0]
                idx = fnames_map[fname]

                fx, fy = float(lst[3]), float(lst[4])
                cx, cy = float(lst[5]), float(lst[6])

                intrinsics[idx] = np.array([fx, 0, cx, 0, fy, cy, 0, 0,
                                            1]).reshape((3, 3))
                cnt += 1

        assert cnt == len(fnames)

        with open(label_path, 'r') as f:
            pair_content = f.readlines()

        pairs = []
        binary_info = []

        for line in pair_content:
            lst = line.strip().split(' ')
            src_idx = int(lst[0])
            dst_idx = int(lst[1])

            num_inliers = float(lst[2])
            num_matches = float(lst[3])

            F_data = list(map(float, lst[4:]))
            F = np.array(F_data).reshape((3, 3))

            if num_matches >= self.num_matches_thr \
               and (num_inliers / num_matches) >= self.inlier_ratio_thr:
                pairs.append((src_idx, dst_idx))
                binary_info.append(F)

        return {
            'folder': scene,
            'fnames': fnames,
            'pairs': pairs,
            'unary_info': intrinsics,
            'binary_info': binary_info
        }

    # override
    def load_data(self, folder, fname):
        fname = os.path.join(self.root, folder, 'images', fname)
        return cv2.imread(fname)

    # override
    def collect_scenes(self, root, scenes):
        scene_collection = []

        for scene in scenes:
            scene_path = os.path.join(root, scene)
            subdirs = os.listdir(scene_path)
            for subdir in subdirs:
                if subdir.startswith('dense') and \
                   os.path.isdir(
                        os.path.join(scene_path, subdir)):
                    scene_dict = self.parse_scene(
                        root, os.path.join(scene, subdir, 'aligned'))
                    scene_collection.append(scene_dict)

        return scene_collection
