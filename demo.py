#!/usr/bin/env python3
"""
Load query and target point clouds, extract dGeDi features,
run RANSAC + ICP registration, and save PCA-coloured PLYs + registered results.
"""

import os
import glob
import argparse
import numpy as np
import open3d as o3d
import torch
from sklearn.decomposition import PCA

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from core.dgedi_distilled import dgedi
from utils import load_yaml_config, compute_diameter, normalize_and_center, extract_features, register_one


def main(args):
    # ----------------------------------------------------------------
    # 1) Load model
    # ----------------------------------------------------------------
    cfg = load_yaml_config(args.config)
    mode_cfg = cfg[args.mode]
    model_cfg = dict(mode_cfg['model_config'])
    model_cfg['weights_path'] = mode_cfg['weights_path']
    print(f"Mode: {args.mode}")

    device = torch.device(args.device)
    model = dgedi({'query': model_cfg, 'target': model_cfg, 'device': args.device})

    # ----------------------------------------------------------------
    # 2) Load and prepare query (done once, reused for every target)
    # ----------------------------------------------------------------
    pcd_q_orig = o3d.io.read_point_cloud(args.query_pcd)
    print(f"Query points (original): {len(pcd_q_orig.points)}")

    pcd_q = pcd_q_orig.farthest_point_down_sample(num_samples=min(6000, len(pcd_q_orig.points)))
    print(f"Query points (after FPS): {len(pcd_q.points)}")

    diam_q = compute_diameter(pcd_q)
    print(f"Query diameter: {diam_q:.6f}")

    pcd_q_feat = o3d.geometry.PointCloud(pcd_q)
    normalize_and_center(pcd_q_feat, diam_q)

    pcd_q_reg = o3d.geometry.PointCloud(pcd_q)
    normalize_and_center(pcd_q_reg, diam_q)
    pcd_q_reg.colors = o3d.utility.Vector3dVector(
        np.tile([1.0, 0.0, 0.0], (len(pcd_q_reg.points), 1))
    )

    # ----------------------------------------------------------------
    # 3) Extract query features + fit PCA (done once)
    # ----------------------------------------------------------------
    print("Extracting query features...")
    feats_q = extract_features(pcd_q_feat, model, device)
    print(f"Query features shape: {feats_q.shape}")

    pca = PCA(n_components=3)
    feats_q_3d = pca.fit_transform(feats_q)
    mins_q, maxs_q = feats_q_3d.min(axis=0), feats_q_3d.max(axis=0)

    os.makedirs(args.out_dir, exist_ok=True)
    base_q = os.path.splitext(os.path.basename(args.query_pcd))[0]

    rgb_q = np.clip((feats_q_3d - mins_q) / (maxs_q - mins_q), 0, 1)
    pcd_q_pca = o3d.geometry.PointCloud(pcd_q_feat)
    pcd_q_pca.colors = o3d.utility.Vector3dVector(rgb_q)
    out_q_pca = os.path.join(args.out_dir, f"{base_q}_PCA.ply")
    o3d.io.write_point_cloud(out_q_pca, pcd_q_pca, write_ascii=True)
    print(f"Query PCA PLY saved: {out_q_pca}")

    # ----------------------------------------------------------------
    # 4) Collect target PCDs
    # ----------------------------------------------------------------
    target_pattern = os.path.join(args.target_dir, "target_pcd_?????.ply")
    all_targets = sorted(glob.glob(target_pattern))

    if not all_targets:
        print(f"No target PCDs found matching: {target_pattern}")
        return

    if args.target_ids is not None:
        selected = []
        for tid in args.target_ids:
            fname = os.path.join(args.target_dir, f"target_pcd_{tid:05d}.ply")
            if os.path.isfile(fname):
                selected.append(fname)
            else:
                print(f"Warning: {fname} not found, skipping.")
        all_targets = selected

    if not all_targets:
        print("No valid targets to process.")
        return

    print(f"\nProcessing {len(all_targets)} target(s) in [{args.mode}] mode.")

    # ----------------------------------------------------------------
    # 5) Loop over targets
    # ----------------------------------------------------------------
    iterator = tqdm(all_targets, desc="Targets", unit="target") if TQDM_AVAILABLE else all_targets

    for target_path in iterator:
        base_t = os.path.splitext(os.path.basename(target_path))[0]

        if not TQDM_AVAILABLE:
            print(f"\n--- {base_t} ---")

        pcd_t_orig = o3d.io.read_point_cloud(target_path)
        pcd_t_orig.points = o3d.utility.Vector3dVector(np.asarray(pcd_t_orig.points) * 1000.0)

        # outlier removal (needed if depth is noisy, but can be skipped for clean PCDs)
        # pcd_t_orig, _ = pcd_t_orig.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)


        pcd_t = pcd_t_orig.farthest_point_down_sample(num_samples=min(6000, len(pcd_t_orig.points)))

        pcd_t_feat = o3d.geometry.PointCloud(pcd_t)
        normalize_and_center(pcd_t_feat, diam_q)

        pcd_t_reg = o3d.geometry.PointCloud(pcd_t)
        normalize_and_center(pcd_t_reg, diam_q)
        pcd_t_reg.colors = o3d.utility.Vector3dVector(
            np.tile([0.0, 0.0, 1.0], (len(pcd_t_reg.points), 1))
        )

        feats_t = extract_features(pcd_t_feat, model, device)

        result, result_icp = register_one(
            pcd_q_reg, feats_q, pcd_t_reg, feats_t,
            args.ransac_threshold, args.icp_threshold,
        )

        pcd_q_icp_t = o3d.geometry.PointCloud(pcd_q_reg)
        pcd_q_icp_t.transform(result_icp.transformation)
        out_reg_icp = os.path.join(args.out_dir, f"{base_q}_to_{base_t}_registered_RANSAC_ICP.ply")
        o3d.io.write_point_cloud(out_reg_icp, pcd_q_icp_t + pcd_t_reg, write_ascii=True)

        feats_t_3d = pca.transform(feats_t)
        rgb_t = np.clip((feats_t_3d - mins_q) / (maxs_q - mins_q), 0, 1)
        pcd_t_feat.colors = o3d.utility.Vector3dVector(rgb_t)
        out_t_pca = os.path.join(args.out_dir, f"{base_t}_PCA.ply")
        o3d.io.write_point_cloud(out_t_pca, pcd_t_feat, write_ascii=True)

    print(f"\nDone. Results saved to: {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dGeDi: PCA feature visualisation + RANSAC/ICP registration.")
    parser.add_argument("--query_pcd", default='pcds/query_pcd/obj_000001.ply')
    parser.add_argument("--target_dir", default='pcds/target_pcd')
    parser.add_argument("--target_ids", type=int, nargs='+', default=None,
                        help="Target IDs to process (e.g. --target_ids 0 1). Omit for all.")
    parser.add_argument("--config", default='config_dgedi.yaml')
    parser.add_argument("--mode", choices=["single_scale", "multi_scale"], default="multi_scale")
    parser.add_argument("--out_dir", default="./dgedi_pca_out")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ransac_threshold", type=float, default=0.03)
    parser.add_argument("--icp_threshold", type=float, default=0.03)

    args = parser.parse_args()
    main(args)
