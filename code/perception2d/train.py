import os, sys

file_path = os.path.abspath(__file__)
project_path = os.path.dirname(os.path.dirname(file_path))
sys.path.append(project_path)

caps_path = os.path.join(project_path, 'ext', 'caps')
sys.path.append(caps_path)

from dataset.megadepth_train import DatasetMegaDepthTrain
from perception2d.adaptor import CAPSConfigParser, DatasetMegaDepthTrainAdaptor, caps_train

if __name__ == '__main__':
    parser = CAPSConfigParser()
    parser.add(
        '--config',
        is_config_file=True,
        default=os.path.join(os.path.dirname(__file__), 'config_train.yml'),
        help='YAML config file path. Please refer to caps_config.yml as a '
        'reference. It overrides the default config file, but will be '
        'overridden by other command line inputs.')
    config = parser.get_config()

    # Note: for training, we need to wrap up with an adaptor to provide a consistent interface.
    dataset = DatasetMegaDepthTrainAdaptor(
        DatasetMegaDepthTrain(config.datadir, config.scenes), config)
    caps_train(dataset, config)
