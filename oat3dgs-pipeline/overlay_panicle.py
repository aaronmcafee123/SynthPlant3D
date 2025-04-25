#!/usr/bin/env python3
"""
overlay_panicle.py

Overlay a single panicle point cloud on the full 3DGS point cloud for visualization.

Usage:
    python overlay_panicle.py \
        --full_ply      sparse/0/ply/points3D.ply \
        --onehot_npz    points_onehot.npz \
        --panicle_id    1

This will paint the full point cloud light gray and the selected panicle bright red.

Dependencies:
    pip install open3d numpy
"""
import argparse
import numpy as np
import open3d as o3d

def parse_args():
    p = argparse.ArgumentParser(description="Overlay a masked panicle on the full point cloud.")
    p.add_argument("--full_ply",   required=True, help="Path to full 3DGS PLY file")
    p.add_argument("--onehot_npz", required=True, help="NPZ file with 'positions' and 'onehot'")
    p.add_argument("--panicle_id", type=int, default=0, help="ID of panicle to highlight (0-indexed)")
    return p.parse_args()


def main():
    args = parse_args()

    # Load full point cloud
    full_pcd = o3d.io.read_point_cloud(args.full_ply)
    full_pcd.paint_uniform_color([0.8, 0.8, 0.8])  # light gray

    # Load one-hot data
    data   = np.load(args.onehot_npz)
    pts    = data["positions"]  # (N,3)
    onehot = data["onehot"]     # (N, M+1)
    labels = onehot.argmax(axis=1)

    # Extract panicle points
    mask_pts = (labels == args.panicle_id)
    panicle_pts = pts[mask_pts]
    panicle_pcd = o3d.geometry.PointCloud()
    panicle_pcd.points = o3d.utility.Vector3dVector(panicle_pts)
    panicle_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # bright red

    # Visualize overlay
    o3d.visualization.draw_geometries(
        [full_pcd, panicle_pcd],
        window_name=f"Panicle {args.panicle_id} Overlay",
        width=1024, height=768,
        point_show_normal=False
    )

if __name__ == '__main__':
    main()
