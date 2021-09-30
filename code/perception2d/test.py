import os, sys

file_path = os.path.abspath(__file__)
project_path = os.path.dirname(os.path.dirname(file_path))
sys.path.append(project_path)

caps_path = os.path.join(project_path, 'ext', 'caps')
sys.path.append(caps_path)

from dataset.megadepth_test import DatasetMegaDepthTest
from perception2d.adaptor import CAPSConfigParser, caps_test

import numpy as np

if __name__ == '__main__':
    parser = CAPSConfigParser()
    parser.add(
        '--config',
        is_config_file=True,
        default=os.path.join(os.path.dirname(__file__), 'config_test.yml'),
        help='YAML config file path. Please refer to caps_config.yml as a '
        'reference. It overrides the default config file, but will be '
        'overridden by other command line inputs.')
    parser.add('--debug', action='store_true')
    parser.add('--output', type=str, default='caps_test_result.npz')
    config = parser.get_config()

    # Note: for testing, our own interface would suffices.
    config.match_ratio_test = False
    dataset = DatasetMegaDepthTest(config.datadir, config.scenes, config.label_dir)
    r_errs, t_errs = caps_test(dataset, config)

    rot_recall = (r_errs < 10.0)
    angular_trans_recall = (t_errs < 10.0)
    print('Rotation Recall: {}/{} = {}'.format(
        rot_recall.sum(), len(rot_recall),
        float(rot_recall.sum()) / len(rot_recall)))
    print('Translation Recall: {}/{} = {}'.format(
        angular_trans_recall.sum(), len(angular_trans_recall),
        float(angular_trans_recall.sum()) / len(angular_trans_recall)))

    np.savez(config.output, rotation_errs=r_errs, translation_errs=t_errs)
