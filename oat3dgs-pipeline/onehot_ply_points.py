#!/usr/bin/env python3
"""
onehot_ply_points.py

One-hot encode PLY points by panicle instance with mask cleaning and vote thresholding,
and optional spatial DBSCAN refinement. Handles complex mask filenames.

Usage:
    python onehot_ply_points.py \
      --ply_path        points3D.ply \
      --poses_json      poses.json \
      --intrinsics_json intrinsics.json \
      --mask_dir        masks/ \
      --output_npz      points_onehot.npz \
      --output_ply      instances_colored.ply \
      [--min_votes      3] \
      [--dbscan_eps     0.02] \
      [--dbscan_min     10]

Dependencies:
    pip install open3d numpy opencv-python pillow scikit-learn
"""
import argparse
import json
import numpy as np
import open3d as o3d
import cv2
from pathlib import Path
from PIL import Image
from sklearn.cluster import DBSCAN


def parse_args():
    p = argparse.ArgumentParser(description="One-hot encode PLY points with mask votes and refinement")
    p.add_argument('--ply_path',        type=Path, required=True, help='Input PLY file')
    p.add_argument('--poses_json',      type=Path, required=True, help='Camera extrinsics JSON')
    p.add_argument('--intrinsics_json', type=Path, required=True, help='Camera intrinsics JSON')
    p.add_argument('--mask_dir',        type=Path, required=True, help='Masks directory')
    p.add_argument('--output_npz',      type=Path, required=True, help='Output NPZ for positions+onehot')
    p.add_argument('--output_ply',      type=Path, help='Optional colored PLY output')
    p.add_argument('--min_votes',       type=int, default=6, help='Minimum view votes to assign')
    p.add_argument('--dbscan_eps',      type=float, default=0.1, help='DBSCAN epsilon for refinement (0 to skip)')
    p.add_argument('--dbscan_min',      type=int, default=1, help='DBSCAN min_samples for refinement')
    return p.parse_args()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def project(points, K, P):
    N = points.shape[0]
    pts_h = np.hstack([points, np.ones((N,1))])
    cam = (P @ pts_h.T).T
    uvw = (K @ cam[:,:3].T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    return np.round(uv).astype(int)


def clean_mask(mask, min_area=500, kernel_size=5):
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(out, [cnt], -1, 255, -1)
    return out


def main():
    args = parse_args()

    # Load point cloud
    pcd = o3d.io.read_point_cloud(str(args.ply_path))
    points = np.asarray(pcd.points)
    N = points.shape[0]
    print(f"Loaded {N} points from {args.ply_path}")

    # Load camera parameters
    poses = {Path(p['filename']).stem: np.array(p['pose']) for p in load_json(args.poses_json)}
    Ks    = {Path(i['filename']).stem: np.array(i['K'])     for i in load_json(args.intrinsics_json)}

    # Build mask map
    mask_map = {}
    for mpath in args.mask_dir.glob('*.png'):
        stem = mpath.stem
        parts = stem.rsplit('_', 1)
        if len(parts) != 2:
            continue
        prefix, inst_str = parts
        try:
            inst = int(inst_str)
        except ValueError:
            continue
        # match prefix to view key
        view = next((v for v in poses if prefix.startswith(v)), None)
        if view:
            mask_map.setdefault(view, []).append((inst, mpath))
    print(f"Collected masks for {len(mask_map)} views")

    # Unique instances
    inst_ids = sorted({inst for masks in mask_map.values() for inst,_ in masks})
    M = len(inst_ids)
    inst2idx = {inst:i for i,inst in enumerate(inst_ids)}
    print(f"Found {M} unique instances")

    # Voting
    votes = np.zeros((N, M), dtype=int)
    for view, masks in mask_map.items():
        if view not in poses or view not in Ks:
            print(f"[WARN] Missing parameters for view '{view}'")
            continue
        uv = project(points, Ks[view], poses[view])
        for inst, mpath in masks:
            mask = np.array(Image.open(mpath).convert('L'))
            mask = clean_mask(mask)
            h, w = mask.shape
            idx = inst2idx[inst]
            xs, ys = uv[:,0], uv[:,1]
            valid = (xs>=0)&(xs<w)&(ys>=0)&(ys<h)
            sel = np.where(valid)[0]
            hits = mask[ys[sel], xs[sel]] > 0
            votes[sel[hits], idx] += 1
        print(f"Processed view '{view}'")

    # Determine labels with thresholding
    max_votes = votes.max(axis=1)
    assigned = np.argmax(votes, axis=1)
    assigned[max_votes < args.min_votes] = M  # background index

    # Build one-hot (N x (M+1))
    labels = np.where(max_votes >= args.min_votes, assigned, M)
    onehot = np.zeros((N, M+1), dtype=int)
    onehot[np.arange(N), labels] = 1

    # Optional DBSCAN refinement
    if args.dbscan_eps > 0 and args.dbscan_min > 0:
        print("Refining with DBSCAN...")
        for j in range(M):
            mask_pts = np.where(labels == j)[0]
            if len(mask_pts) < args.dbscan_min:
                onehot[mask_pts, j] = 0
                onehot[mask_pts, M] = 1
                continue
            pts = points[mask_pts]
            cl = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min).fit(pts)
            if len(cl.labels_) == 0:
                continue
            main_cluster = np.argmax(np.bincount(cl.labels_[cl.labels_>=0]))
            drop = mask_pts[cl.labels_ != main_cluster]
            onehot[drop, j] = 0
            onehot[drop, M] = 1

    # Save outputs
    np.savez(args.output_npz, positions=points, onehot=onehot)
    print(f"Saved one-hot to {args.output_npz}")

    # Colored PLY output
    if args.output_ply:
        import numpy as _np
        _np.random.seed(0)
        cols = (_np.random.rand(M+1,3) * 255).astype(int)
        point_colors = cols[labels] / 255.0
        pcd.colors = o3d.utility.Vector3dVector(point_colors)
        o3d.io.write_point_cloud(str(args.output_ply), pcd)
        print(f"Wrote colored PLY to {args.output_ply}")

if __name__ == '__main__':
    main()
