import sys, os

file_path = os.path.abspath(__file__)
project_path = os.path.dirname(os.path.dirname(file_path))
sys.path.append(project_path)

fcgf_path = os.path.join(project_path, 'ext', 'FCGF')
sys.path.append(fcgf_path)

from dataset.threedmatch_test import Dataset3DMatchTest
from perception3d.adaptor import FCGFConfigParser, fcgf_test

import numpy as np

if __name__ == '__main__':
    parser = FCGFConfigParser()
    parser.add(
        '--config',
        is_config_file=True,
        default=os.path.join(os.path.dirname(__file__), 'config_test.yml'),
        help='YAML config file path. Please refer to caps_config.yml as a '
        'reference. It overrides the default config file, but will be '
        'overridden by other command line inputs.')
    parser.add('--debug', action='store_true')
    parser.add('--output', type=str, default='fcgf_test_result.npz')
    config = parser.get_config()

    dataset = Dataset3DMatchTest(config.dataset_path, config.scenes)
    r_errs, t_errs = fcgf_test(dataset, config)

    recall = (r_errs < 15.0) * (t_errs < 0.3)
    print('Recall: {}/{} = {}'.format(recall.sum(), len(recall),
          float(recall.sum()) / len(recall)))

    np.savez(config.output, rotation_errs=r_errs, translation_errs=t_errs)
