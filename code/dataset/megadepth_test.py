import os, sys

file_path = os.path.abspath(__file__)
project_path = os.path.dirname(os.path.dirname(file_path))
sys.path.append(project_path)

import cv2
import numpy as np
import open3d as o3d

from dataset.base import DatasetBase
from geometry.image import compute_fundamental_from_poses, detect_keypoints, extract_feats, match_feats, estimate_essential, draw_matches


# Train and test sets are identical for CAPS
class DatasetMegaDepthTest(DatasetBase):
    def __init__(self, data_root, scenes, label_root):
        self.data_root = data_root
        super(DatasetMegaDepthTest, self).__init__(label_root, scenes)

    # override
    def parse_scene(self, label_root, scene):
        scene_path = os.path.join(label_root, scene)

        # Load cameras
        cam_fname = os.path.join(scene_path, 'img_cam.txt')
        with open(cam_fname, 'r') as f:
            cam_content = f.readlines()

        cnt = 0
        fnames = []
        fnames_map = {}
        intrinsics = []
        extrinsics = []
        for i, line in enumerate(cam_content):
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                lst = line.split()
                seq = lst[0]
                fname = lst[1]

                fx, fy = float(lst[4]), float(lst[5])
                cx, cy = float(lst[6]), float(lst[7])

                R = np.array(lst[8:17]).reshape((3, 3))
                t = np.array(lst[17:20])
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t

                fnames.append(
                    os.path.join(self.data_root, seq, 'dense', 'aligned',
                                 'images', fname))
                fnames_map[fname] = i
                intrinsics.append(
                    np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape((3, 3)))
                extrinsics.append(T)

        # Load pairs.txt
        pair_fname = os.path.join(scene_path, 'pairs.txt')
        with open(pair_fname, 'r') as f:
            pair_content = f.readlines()

        pairs = []
        for line in pair_content:
            lst = line.strip().split(' ')
            seq = lst[0]
            src_fname = lst[1]
            dst_fname = lst[2]

            src_idx = fnames_map[src_fname]
            dst_idx = fnames_map[dst_fname]
            pairs.append((src_idx, dst_idx))

        return {
            'folder': scene,
            'fnames': fnames,
            'pairs': pairs,
            'unary_info': [(K, T) for K, T in zip(intrinsics, extrinsics)],
            'binary_info': [None for i in range(len(pairs))]
        }

    # override
    def load_data(self, folder, fname):
        return cv2.imread(fname)
