import sys, os

file_path = os.path.abspath(__file__)
project_path = os.path.dirname(os.path.dirname(file_path))
sys.path.append(project_path)

fcgf_path = os.path.join(project_path, 'ext', 'FCGF')
sys.path.append(fcgf_path)

import torch
from easydict import EasyDict as edict

from ext.FCGF.lib.data_loaders import *
from ext.FCGF.lib.trainer import *
from ext.FCGF.model import load_model

from dataset.threedmatch_train import Dataset3DMatchTrain
from dataset.threedmatch_test import Dataset3DMatchTest
from dataset.threedmatch_sgp import Dataset3DMatchSGP

from geometry.pointcloud import *
from geometry.common import rotation_error, translation_error

from tqdm import tqdm

import configargparse


def reload_config(config):
    dconfig = vars(config)

    if config.resume_dir:
        resume_config = json.load(open(config.resume_dir + '/config.json',
                                       'r'))
        for k in dconfig:
            if k not in ['resume_dir'] and k in resume_config:
                dconfig[k] = resume_config[k]
        dconfig['resume'] = resume_config['out_dir'] + '/checkpoint.pth'

    return edict(dconfig)


class FCGFConfigParser(configargparse.ArgParser):
    def __init__(self):
        super().__init__(default_config_files=[
            os.path.join(os.path.dirname(__file__), 'fcgf_config.yml')
        ],
                         conflict_handler='resolve')

        # Mainly used params
        self.add('--dataset_path',
                 type=str,
                 default="/home/wei/Workspace/data/threedmatch_reorg")
        self.add('--scenes',
                 nargs='+',
                 help='scenes used for training/testing')
        self.add('--out_dir',
                 type=str,
                 default='fcgf_outputs',
                 help='outputs containing summary and checkpoints')

        self.add(
            '--restart_meta_iter',
            type=int,
            default=-1,
            help='Restart of teacher-student iterations. -1 indicates bootstrap'
        )
        self.add('--meta_iters',
                 type=int,
                 default=2,
                 help='number of teacher-student iterations')
        self.add('--finetune',
                 action='store_true',
                 help='train from previous checkpoint during SGP.')

        self.add('--pseudo_label_dir',
                 type=str,
                 help='the pseudo-gt directory storing pairs and T matrices')

        self.add('--overlap_thr',
                 type=float,
                 default=0.3,
                 help='overlap threshold to filter outlier pairs')
        self.add('--voxel_size', type=float, default=0.05)
        self.add('--mutual_filter', type=bool, default=False)
        self.add('--ransac_iters', type=int, default=10000)
        self.add('--confidence', type=float, default=0.9999)

        # Other core configs from the FCGF repo.
        # See https://github.com/chrischoy/FCGF/blob/master/config.py
        self.add('--trainer',
                 type=str,
                 default='HardestContrastiveLossTrainer')
        self.add('--save_freq_epoch', type=int, default=1)
        self.add('--batch_size', type=int, default=4)
        self.add('--val_batch_size', type=int, default=1)

        # Hard negative mining
        self.add('--use_hard_negative', type=bool, default=True)
        self.add('--hard_negative_sample_ratio', type=int, default=0.05)
        self.add('--hard_negative_max_num', type=int, default=3000)
        self.add('--num_pos_per_batch', type=int, default=1024)
        self.add('--num_hn_samples_per_batch', type=int, default=256)

        # Metric learning loss
        self.add('--neg_thresh', type=float, default=1.4)
        self.add('--pos_thresh', type=float, default=0.1)
        self.add('--neg_weight', type=float, default=1)

        # Data augmentation
        self.add('--use_random_scale', type=bool, default=False)
        self.add('--min_scale', type=float, default=0.8)
        self.add('--max_scale', type=float, default=1.2)
        self.add('--use_random_rotation', type=bool, default=True)
        self.add('--rotation_range', type=float, default=360)

        # Data loader configs
        self.add('--train_phase', type=str, default="train")
        self.add('--val_phase', type=str, default="val")
        self.add('--test_phase', type=str, default="test")

        self.add('--stat_freq', type=int, default=40)
        self.add('--test_valid', type=bool, default=True)
        self.add('--val_max_iter', type=int, default=400)
        self.add('--val_epoch_freq', type=int, default=1)
        self.add('--positive_pair_search_voxel_size_multiplier',
                 type=float,
                 default=1.5)

        self.add('--hit_ratio_thresh', type=float, default=0.1)

        # Triplets
        self.add('--triplet_num_pos', type=int, default=256)
        self.add('--triplet_num_hn', type=int, default=512)
        self.add('--triplet_num_rand', type=int, default=1024)

        # Network specific configurations
        self.add('--model', type=str, default='ResUNetBN2C')
        self.add('--model_n_out',
                 type=int,
                 default=32,
                 help='Feature dimension')
        self.add('--conv1_kernel_size', type=int, default=5)
        self.add('--normalize_feature', type=bool, default=True)
        self.add('--dist_type', type=str, default='L2')
        self.add('--best_val_metric', type=str, default='feat_match_ratio')

        # Optimizer arguments
        self.add('--optimizer', type=str, default='SGD')
        self.add('--max_epoch', type=int, default=100)
        self.add('--lr', type=float, default=1e-1)
        self.add('--momentum', type=float, default=0.8)
        self.add('--sgd_momentum', type=float, default=0.9)
        self.add('--sgd_dampening', type=float, default=0.1)
        self.add('--adam_beta1', type=float, default=0.9)
        self.add('--adam_beta2', type=float, default=0.999)
        self.add('--weight_decay', type=float, default=1e-4)
        self.add('--iter_size',
                 type=int,
                 default=1,
                 help='accumulate gradient')
        self.add('--bn_momentum', type=float, default=0.05)
        self.add('--exp_gamma', type=float, default=0.99)
        self.add('--scheduler', type=str, default='ExpLR')

        self.add('--use_gpu', type=bool, default=True)
        self.add('--weights', type=str, default=None)
        self.add('--weights_dir', type=str, default=None)
        self.add('--resume', type=str, default=None)
        self.add('--resume_dir', type=str, default=None)
        self.add('--train_num_thread', type=int, default=2)
        self.add('--val_num_thread', type=int, default=1)
        self.add('--test_num_thread', type=int, default=2)
        self.add('--fast_validation', type=bool, default=False)
        self.add(
            '--nn_max_n',
            type=int,
            default=500,
            help=
            'The maximum number of features to find nearest neighbors in batch'
        )

    def get_config(self):
        config = self.parse_args()
        config.device = 'cuda' if config.use_gpu else 'cpu'

        return reload_config(config)


def get_trainer(trainer):
    if trainer == 'ContrastiveLossTrainer':
        return ContrastiveLossTrainer
    elif trainer == 'HardestContrastiveLossTrainer':
        return HardestContrastiveLossTrainer
    elif trainer == 'TripletLossTrainer':
        return TripletLossTrainer
    elif trainer == 'HardestTripletLossTrainer':
        return HardestTripletLossTrainer
    else:
        raise ValueError(f'Trainer {trainer} not found')


class DatasetFCGFAdaptor(torch.utils.data.Dataset):
    '''
    Wrapper dataset for our data format and FCGF's sample format
    '''
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.randg = np.random.RandomState()
        self.config = config

    def reset_seed(self, seed):
        self.randg.seed(seed)

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        pcd0, pcd1, _, _, trans = self.dataset[idx]

        xyz0 = np.asarray(pcd0.points)
        xyz1 = np.asarray(pcd1.points)

        # Data augmentation
        T0 = sample_random_trans(xyz0, self.randg, 360)
        T1 = sample_random_trans(xyz1, self.randg, 360)
        trans = T1 @ trans @ np.linalg.inv(T0)

        xyz0 = self.apply_transform(xyz0, T0)
        xyz1 = self.apply_transform(xyz1, T1)

        # Voxelization after random transformation
        voxel_size = 0.05
        _, sel0 = ME.utils.sparse_quantize(xyz0,
                                           return_index=True,
                                           quantization_size=voxel_size)
        _, sel1 = ME.utils.sparse_quantize(xyz1,
                                           return_index=True,
                                           quantization_size=voxel_size)
        xyz0 = xyz0[sel0]
        xyz1 = xyz1[sel1]

        # Make point clouds using voxelized points
        pcd0 = make_o3d_pointcloud(xyz0)
        pcd1 = make_o3d_pointcloud(xyz1)
        matches = get_matching_indices(pcd0, pcd1, trans, voxel_size * 2)

        # Dummy features
        feats0 = np.ones((xyz0.shape[0], 1))
        feats1 = np.ones((xyz1.shape[0], 1))

        # Coordinates
        coords0 = np.floor(xyz0 / voxel_size)
        coords1 = np.floor(xyz1 / voxel_size)

        return (xyz0, xyz1, coords0, coords1, feats0, feats1, matches, trans)


def load_fcgf_model(config):
    resume_ckpt_path = config.resume
    input_ckpt_path = config.weights
    out_ckpt_path = os.path.join(config.out_dir, 'checkpoint.pth')

    if resume_ckpt_path is not None and os.path.isfile(resume_ckpt_path):
        ckpt_path = resume_ckpt_path
    elif input_ckpt_path is not None and os.path.isfile(input_ckpt_path):
        ckpt_path = input_ckpt_path
    elif out_ckpt_path is not None and os.path.isfile(out_ckpt_path):
        ckpt_path = out_ckpt_path
    else:
        raise NotImplementedError('checkpoint not found, abort')

    print(f'load FCGF from checkpoint {ckpt_path}.')
    checkpoint = torch.load(ckpt_path)
    ckpt_cfg = checkpoint['config']

    Model = load_model(ckpt_cfg['model'])
    model = Model(in_channels=1,
                  out_channels=ckpt_cfg['model_n_out'],
                  bn_momentum=ckpt_cfg['bn_momentum'],
                  normalize_feature=ckpt_cfg['normalize_feature'],
                  conv1_kernel_size=ckpt_cfg['conv1_kernel_size'],
                  D=3)
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(config.device)


def register(pcd_src, pcd_dst, feature, solver, model, config):
    pcd_src, feat_src = extract_feats(pcd_src, feature, config.voxel_size,
                                      model)
    pcd_dst, feat_dst = extract_feats(pcd_dst, feature, config.voxel_size,
                                      model)
    corrs = match_feats(feat_src, feat_dst, mutual_filter=config.mutual_filter)

    if len(corrs) < 10:
        print('Too few corres ({}), abort'.format(len(corrs)))
        return np.eye(4), 0

    T, fitness = solve(pcd_src, pcd_dst, corrs, solver,
                       config.voxel_size * 1.4, config.ransac_iters,
                       config.confidence)
    if fitness > 1e-6:
        T, fitness = refine(pcd_src, pcd_dst, T, config.voxel_size * 1.4)

    return T, fitness


def fcgf_train(dataset, config):
    ch = logging.StreamHandler(sys.stdout)
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%m/%d %H:%M:%S',
                        handlers=[ch])

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    logging.basicConfig(level=logging.INFO, format="")

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=8,
                                             shuffle=True,
                                             num_workers=8,
                                             collate_fn=collate_pair_fn,
                                             pin_memory=False,
                                             drop_last=True)
    Trainer = get_trainer(config.trainer)
    trainer = Trainer(config=config, data_loader=dataloader)
    trainer.train()


def fcgf_test(dataset, config):
    model = load_fcgf_model(config)

    r_errs = []
    t_errs = []

    for data in tqdm(dataset):
        pcd_src, pcd_dst, _, _, T_gt = data

        T, fitness = register(pcd_src, pcd_dst, 'FCGF', 'RANSAC', model,
                              config)
        r_err = rotation_error(T[:3, :3], T_gt[:3, :3])
        t_err = translation_error(T[:3, 3], T_gt[:3, 3])
        r_errs.append(r_err)
        t_errs.append(t_err)

        if config.debug:
            pcd_src.paint_uniform_color([1, 0, 0])
            pcd_dst.paint_uniform_color([0, 1, 0])
            o3d.visualization.draw_geometries([pcd_src.transform(T), pcd_dst])

    return np.array(r_errs), np.array(t_errs)
