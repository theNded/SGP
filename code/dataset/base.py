import os
import numpy as np


class DatasetBase:
    def __init__(self, root, scenes):
        self.root = root
        self.scenes = self.collect_scenes(root, scenes)

        scene_ids = []
        pair_ids = []

        for i, scene in enumerate(self.scenes):
            num_pairs = len(scene['pairs'])
            scene_ids.append(np.ones((num_pairs), dtype=np.int) * i)
            pair_ids.append(np.arange(0, num_pairs, dtype=np.int))

        self.scene_idx_map = np.concatenate(scene_ids)
        self.pair_idx_map = np.concatenate(pair_ids)

    def __len__(self):
        return len(self.scene_idx_map)

    def __getitem__(self, idx):
        # Use the LUT
        scene_idx = self.scene_idx_map[idx]
        pair_idx = self.pair_idx_map[idx]

        # Access actual data
        scene = self.scenes[scene_idx]
        folder = scene['folder']

        i, j = scene['pairs'][pair_idx]
        fname_src = scene['fnames'][i]
        fname_dst = scene['fnames'][j]

        data_src = self.load_data(folder, fname_src)
        data_dst = self.load_data(folder, fname_dst)

        # Optional. Could be None
        info_src = scene['unary_info'][i]
        info_dst = scene['unary_info'][j]
        info_pair = scene['binary_info'][pair_idx]

        return data_src, data_dst, info_src, info_dst, info_pair

    # NOTE: override in inheritance
    def parse_scene(self, root, scene):
        return {
            'folder': scene,
            'fnames': [],
            'pairs': [],
            'unary_info': [],
            'binary_info': []
        }

    # NOTE: override in inheritance
    def load_data(self, folder, fname):
        return os.path.join(folder, fname)

    # NOTE: optionally override in inheritance, if a scene includes more than 1 subset
    def collect_scenes(self, root, scenes):
        return [self.parse_scene(root, scene) for scene in scenes]
