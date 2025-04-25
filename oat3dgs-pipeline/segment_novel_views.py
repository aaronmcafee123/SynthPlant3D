#!/usr/bin/env python3
"""
segment_novel_views.py

Run YOLO → SAM segmentation on a folder of novel‐view renderings.

Usage:
    python segment_novel_views.py \
      --image_dir       renders/ \
      --detector_model  runs/train/oat_experiment/weights/best.pt \
      --sam_checkpoint  sam_vit_h.pth \
      --model_type      vit_h \
      --output_dir      masks_renders/ \
      --boxes_json      boxes_renders.json \
      --conf_thresh     0.3

This will:
 1. Detect panicle bounding boxes in each render,
 2. Save them to `boxes_renders.json`,
 3. Run SAM inside each box,
 4. Write crisp `<viewname>_<inst>.png` masks to `masks_renders/`.
"""
import os
import json
import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

def parse_args():
    p = argparse.ArgumentParser(description="Segment novel-view renders with YOLO+SAM")
    p.add_argument("--image_dir",      type=str, required=True, help="Folder of rendered images")
    p.add_argument("--detector_model", type=str, required=True, help="YOLOv8 weights (.pt)")
    p.add_argument("--sam_checkpoint", type=str, required=True, help="SAM checkpoint (.pth)")
    p.add_argument("--model_type",     type=str, default="vit_h", choices=["vit_h","vit_l","vit_b"])
    p.add_argument("--output_dir",     type=str, required=True, help="Where to save masks")
    p.add_argument("--boxes_json",     type=str, required=True, help="Where to dump detections JSON")
    p.add_argument("--conf_thresh",    type=float, default=0.3, help="YOLO confidence threshold")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Load YOLOv8
    print("[INFO] Loading YOLO model...")
    model = YOLO(args.detector_model)
    model.to(device)

    # Gather rendered images
    images = sorted([f for f in os.listdir(args.image_dir)
                     if f.lower().endswith((".png",".jpg"))])
    print(f"[INFO] Found {len(images)} rendered views in {args.image_dir}")

    # 1) Detection
    boxes_dict = {}
    for fname in tqdm(images, desc="Detecting boxes"):
        path = os.path.join(args.image_dir, fname)
        res  = model.predict(source=path, conf=args.conf_thresh, device=device, verbose=False)
        dets = res[0].boxes.xyxy.cpu().numpy() if len(res)>0 else np.zeros((0,4))
        boxes_dict[fname] = dets.astype(int).tolist()
    with open(args.boxes_json, "w") as f:
        json.dump(boxes_dict, f, indent=2)
    print(f"[INFO] Saved detection boxes → {args.boxes_json}")

    # 2) Load SAM
    print("[INFO] Loading SAM checkpoint...")
    sam       = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)

    # 3) SAM segmentation
    for fname, boxes in tqdm(boxes_dict.items(), desc="Segmenting masks"):
        img    = cv2.imread(os.path.join(args.image_dir, fname))
        rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb)
        for idx, box in enumerate(boxes):
            box_np = np.array(box, dtype=int)[None,:]
            masks, _, _ = predictor.predict(box=box_np, multimask_output=False)
            mask = (masks[0].astype(np.uint8) * 255)
            out_name = f"{os.path.splitext(fname)[0]}_{idx:03d}.png"
            cv2.imwrite(os.path.join(args.output_dir, out_name), mask)
    print(f"[INFO] Wrote {len(images)}× masks to {args.output_dir}")

if __name__ == "__main__":
    main()
