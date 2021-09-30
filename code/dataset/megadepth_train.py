import os, sys

file_path = os.path.abspath(__file__)
project_path = os.path.dirname(os.path.dirname(file_path))
sys.path.append(project_path)

import cv2
import numpy as np
import open3d as o3d

from dataset.base import DatasetBase
from geometry.image import compute_fundamental_from_poses, detect_keypoints, extract_feats, match_feats, estimate_essential, draw_matches


class DatasetMegaDepthTrain(DatasetBase):
    def __init__(self, root, scenes):
        super(DatasetMegaDepthTrain, self).__init__(root, scenes)

    # override
    def parse_scene(self, root, scene):
        scene_path = os.path.join(root, scene)

        fnames = os.listdir(os.path.join(scene_path, 'images'))
        fnames_map = {fname: i for i, fname in enumerate(fnames)}

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

                R = np.array(lst[7:16]).reshape((3, 3))
                t = np.array(lst[16:19])
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t

                intrinsics[idx] = np.array([fx, 0, cx, 0, fy, cy, 0, 0,
                                            1]).reshape((3, 3))
                extrinsics[idx] = T
                cnt += 1

        assert cnt == len(fnames)

        return {
            'folder': scene,
            'fnames': fnames,
            'pairs': pairs,
            'unary_info': [(K, T) for K, T in zip(intrinsics, extrinsics)],
            'binary_info': [None for i in range(len(pairs))]
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
