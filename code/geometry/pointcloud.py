import open3d as o3d
import numpy as np
import torch

import MinkowskiEngine as ME
from scipy.spatial import cKDTree


def make_o3d_pointcloud(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def extract_feats(pcd, feature_type, voxel_size, model=None):
    xyz = np.asarray(pcd.points)
    _, sel = ME.utils.sparse_quantize(xyz,
                                      return_index=True,
                                      quantization_size=voxel_size)
    xyz = xyz[sel]
    pcd = make_o3d_pointcloud(xyz)

    if feature_type == 'FPFH':
        radius_normal = voxel_size * 2
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal,
                                                 max_nn=30))
        radius_feat = voxel_size * 5
        feat = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feat,
                                                 max_nn=100))
        # (N, 33)
        return pcd, feat.data.T

    elif feature_type == 'FCGF':
        DEVICE = torch.device('cuda')
        coords = ME.utils.batched_coordinates(
            [torch.floor(torch.from_numpy(xyz) / voxel_size).int()]).to(DEVICE)

        feats = torch.ones(coords.size(0), 1).to(DEVICE)
        sinput = ME.SparseTensor(feats, coordinates=coords)  # .to(DEVICE)

        # (N, 32)
        return pcd, model(sinput).F.detach().cpu().numpy()

    else:
        raise NotImplementedError(
            'Unimplemented feature type {}'.format(feature_type))


def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
    feat1tree = cKDTree(feat1)
    dists, nn_inds = feat1tree.query(feat0, k=knn, n_jobs=-1)
    if return_distance:
        return nn_inds, dists
    else:
        return nn_inds


def match_feats(feat_src, feat_dst, mutual_filter=True, k=1):
    if not mutual_filter:
        nns01 = find_knn_cpu(feat_src, feat_dst, knn=1, return_distance=False)
        corres01_idx0 = np.arange(len(nns01)).squeeze()
        corres01_idx1 = nns01.squeeze()
        return np.stack((corres01_idx0, corres01_idx1)).T
    else:
        # for each feat in src, find its k=1 nearest neighbours
        nns01 = find_knn_cpu(feat_src, feat_dst, knn=1, return_distance=False)
        # for each feat in dst, find its k nearest neighbours
        nns10 = find_knn_cpu(feat_dst, feat_src, knn=k, return_distance=False)
        # find corrs
        num_feats = len(nns01)
        corres01 = []
        if k == 1:
            for i in range(num_feats):
                if i == nns10[nns01[i]]:
                    corres01.append([i, nns01[i]])
        else:
            for i in range(num_feats):
                if i in nns10[nns01[i]]:
                    corres01.append([i, nns01[i]])
        # print(
        #     f'Before mutual filter: {num_feats}, after mutual_filter with k={k}: {len(corres01)}.'
        # )

        # Fallback if mutual filter is too aggressive
        if len(corres01) < 10:
            nns01 = find_knn_cpu(feat_src,
                                 feat_dst,
                                 knn=1,
                                 return_distance=False)
            corres01_idx0 = np.arange(len(nns01)).squeeze()
            corres01_idx1 = nns01.squeeze()
            return np.stack((corres01_idx0, corres01_idx1)).T

        return np.asarray(corres01)


def weighted_procrustes(A, B, weights=None):
    num_pts = A.shape[1]
    if weights is None:
        weights = np.ones(num_pts)

    # compute weighted center
    A_center = A @ weights / np.sum(weights)
    B_center = B @ weights / np.sum(weights)

    # compute relative positions
    A_ref = A - A_center[:, np.newaxis]
    B_ref = B - B_center[:, np.newaxis]

    # compute rotation
    M = B_ref @ np.diag(weights) @ A_ref.T
    U, _, Vh = np.linalg.svd(M)
    S = np.identity(3)
    S[-1, -1] = np.linalg.det(U) * np.linalg.det(Vh)
    R = U @ S @ Vh

    # compute translation
    t = B_center - R @ A_center

    return R, t


def solve(src, dst, corres, solver_type, distance_thr, ransac_iters,
          confidence):
    if solver_type.startswith('RANSAC'):
        corres = o3d.utility.Vector2iVector(corres)

        result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            src, dst, corres, distance_thr,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(
                False), 3, [],
            o3d.pipelines.registration.RANSACConvergenceCriteria(
                ransac_iters, confidence))

        return result.transformation, result.fitness

    else:
        raise NotImplementedError(
            'Unimplemented solver type {}'.format(solver_type))


def refine(src, dst, ransac_T, distance_thr):
    result = o3d.pipelines.registration.registration_icp(
        src, dst, distance_thr, ransac_T,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    icp_T = result.transformation
    icp_fitness = result.fitness

    fitness = icp_fitness * np.minimum(
        1.0,
        float(len(dst.points)) / float(len(src.points)))

    return icp_T, fitness
