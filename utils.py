import numpy as np
import open3d as o3d
import torch
import yaml


def load_yaml_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def compute_diameter(pcd: o3d.geometry.PointCloud) -> float:
    pts = np.asarray(pcd.points)
    dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    return dists.max()


def normalize_and_center(pcd: o3d.geometry.PointCloud, scale: float):
    pts = np.asarray(pcd.points)
    pts /= scale
    pts -= pts.mean(axis=0)
    pcd.points = o3d.utility.Vector3dVector(pts)


def extract_features(pcd_norm, model, device):
    pts = torch.from_numpy(np.asarray(pcd_norm.points, dtype=np.float32)).to(device)
    offsets = torch.tensor([0, pts.shape[0]], dtype=torch.long, device=device)
    with torch.no_grad():
        feats = model.compute_feats_in_batch_variable_length(pts, offsets)
    return feats.cpu().numpy()


def register_one(pcd_q_reg, feats_q, pcd_t_reg, feats_t, ransac_threshold, icp_threshold):
    """Run RANSAC then ICP for one query/target pair. Returns (result_ransac, result_icp)."""
    dim = feats_q.shape[1]

    feat_q_o3d = o3d.pipelines.registration.Feature()
    feat_q_o3d.resize(dim, len(pcd_q_reg.points))
    feat_q_o3d.data = feats_q.T

    feat_t_o3d = o3d.pipelines.registration.Feature()
    feat_t_o3d.resize(dim, len(pcd_t_reg.points))
    feat_t_o3d.data = feats_t.T

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=pcd_q_reg,
        target=pcd_t_reg,
        source_feature=feat_q_o3d,
        target_feature=feat_t_o3d,
        mutual_filter=True,
        max_correspondence_distance=ransac_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(ransac_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=10000, confidence=0.999),
    )

    result_icp = o3d.pipelines.registration.registration_icp(
        source=pcd_q_reg,
        target=pcd_t_reg,
        max_correspondence_distance=icp_threshold,
        init=result.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=2000
        ),
    )

    return result, result_icp
