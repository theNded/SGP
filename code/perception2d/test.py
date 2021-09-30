import os, sys

file_path = os.path.abspath(__file__)
project_path = os.path.dirname(os.path.dirname(file_path))
sys.path.append(project_path)

caps_path = os.path.join(project_path, 'ext', 'caps')
sys.path.append(caps_path)

from dataset.caps_train_test import DatasetCAPS
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
    config = parser.get_config()

    # Note: for testing, our own interface would suffices.
    dataset = DatasetCAPS(config.datadir, config.scenes)
    caps_test(dataset, config)

