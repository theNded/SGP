import sys, os

file_path = os.path.abspath(__file__)
project_path = os.path.dirname(os.path.dirname(file_path))
sys.path.append(project_path)

fcgf_path = os.path.join(project_path, 'ext', 'FCGF')
sys.path.append(fcgf_path)

from dataset.threedmatch_train import Dataset3DMatchTrain
from perception3d.adaptor import DatasetFCGFAdaptor, FCGFConfigParser, fcgf_train

if __name__ == '__main__':
    parser = FCGFConfigParser()
    parser.add(
        '--config',
        is_config_file=True,
        default=os.path.join(os.path.dirname(__file__), 'config_train.yml'),
        help='YAML config file path. Please refer to caps_config.yml as a '
        'reference. It overrides the default config file, but will be '
        'overridden by other command line inputs.')
    config = parser.get_config()

    dataset = DatasetFCGFAdaptor(
        Dataset3DMatchTrain(config.dataset_path, config.scenes, config.overlap_thr), config)
    fcgf_train(dataset, config)
