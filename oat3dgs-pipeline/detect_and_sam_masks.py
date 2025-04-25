#!/usr/bin/env python3
"""
detect_and_sam_masks_debug.py

DEBUG version of detect_and_sam_masks.py with verbose logging and device checks

Usage:
    python detect_and_sam_masks_debug.py \
        --image_dir images/ \
        --detector_model runs/train/oat_experiment/weights/best.pt \
        --sam_checkpoint sam_vit_h.pth \
        --model_type vit_h \
        --output_dir masks/ \
        --boxes_json boxes_debug.json

Dependencies:
    pip install ultralytics segment-anything opencv-python numpy tqdm torch
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
    parser = argparse.ArgumentParser(description="Debug: Detect panicles and segment with SAM, with verbose logs.")
    parser.add_argument("--image_dir",      type=str, required=True, help="Directory of input images")
    parser.add_argument("--detector_model", type=str, required=True, help="Path to YOLO model weights (.pt)")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="Path to SAM checkpoint (.pth)")
    parser.add_argument("--model_type",     type=str, default="vit_h", choices=["vit_h","vit_l","vit_b"], help="SAM model type")
    parser.add_argument("--output_dir",     type=str, required=True, help="Directory to save masks")
    parser.add_argument("--boxes_json",     type=str, required=True, help="Path to save detection boxes JSON")
    parser.add_argument("--conf_thresh",    type=float, default=0.3, help="Detection confidence threshold")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Load YOLO detector
    print(f"[INFO] Loading YOLO model from {args.detector_model}")
    model = YOLO(args.detector_model)
    model.to(device)
    model_conf = args.conf_thresh

    # Gather images
    image_files = sorted([f for f in os.listdir(args.image_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    print(f"[INFO] Found {len(image_files)} images in {args.image_dir}")

    # Detection phase
    boxes_dict = {}
    print("[INFO] Starting detection phase...")
    for idx, fname in enumerate(image_files, 1):
        print(f"[DETECT] ({idx}/{len(image_files)}) Processing {fname}")
        img_path = os.path.join(args.image_dir, fname)
        try:
            results = model.predict(source=img_path, conf=model_conf, device=device, verbose=False)
            dets = results[0].boxes.xyxy.cpu().numpy() if results and len(results) > 0 else np.zeros((0,4))
        except Exception as e:
            print(f"[ERROR] YOLO detection failed on {fname}: {e}")
            dets = np.zeros((0,4))
        boxes = dets.astype(int).tolist()
        boxes_dict[fname] = boxes
    with open(args.boxes_json, 'w') as f:
        json.dump(boxes_dict, f, indent=2)
    print(f"[INFO] Detection done. Boxes saved to {args.boxes_json}")

    # Initialize SAM
    print(f"[INFO] Loading SAM model {args.model_type} checkpoint from {args.sam_checkpoint}")
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)

    # Segmentation phase
    print("[INFO] Starting segmentation phase...")
    for idx, (fname, boxes) in enumerate(boxes_dict.items(), 1):
        print(f"[SEG] ({idx}/{len(boxes_dict)}) Segmenting {fname}, {len(boxes)} boxes")
        img_path = os.path.join(args.image_dir, fname)
        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARN] Could not read image {fname}, skipping.")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)
        for bidx, box in enumerate(boxes, 1):
            print(f"    [MASK] Box {bidx}/{len(boxes)}: {box}")
            box_arr = np.array(box, dtype=int)[None, :]
            try:
                masks, scores, _ = predictor.predict(box=box_arr, multimask_output=False)
            except Exception as e:
                print(f"    [ERROR] SAM predict failed on box {box}: {e}")
                continue
            mask = (masks[0].astype(np.uint8) * 255)
            out_name = f"{os.path.splitext(fname)[0]}_{bidx:03d}.png"
            out_path = os.path.join(args.output_dir, out_name)
            cv2.imwrite(out_path, mask)
            print(f"    [SAVED] {out_name}")
    print(f"[INFO] Segmentation done. Masks saved to {args.output_dir}")

if __name__ == '__main__':
    main()
