#!/usr/bin/env python3
"""
train_yolo_oat.py

Train a YOLOv8 model on your oat panicle dataset.

Usage:
    python train_yolo_oat.py \
        --data_dir path/to/oat_dataset \
        --model yolov8n.pt \
        --epochs 50 \
        --img_size 640 \
        --batch_size 16 \
        --project runs/train \
        --name oat_experiment

Dataset structure (inside --data_dir):
    images/
        train/
        val/
    labels/
        train/    # YOLO txt labels, one class (panicle)
        val/

After running, trained weights and logs will be in:
    runs/train/oat_experiment

Dependencies:
    pip install ultralytics pyyaml
"""
import argparse
import os
import yaml
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on oat panicle dataset")
    parser.add_argument("--data_dir",    type=str, required=True,
                        help="Root of dataset with images/ and labels/ subfolders")
    parser.add_argument("--model",       type=str, default="yolov8n.pt",
                        help="Pretrained YOLOv8 model (e.g. yolov8n.pt)")
    parser.add_argument("--epochs",      type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--img_size",    type=int, default=640,
                        help="Image size for training (pixels)")
    parser.add_argument("--batch_size",  type=int, default=16,
                        help="Batch size per GPU")
    parser.add_argument("--project",     type=str, default="runs/train",
                        help="Root directory for logs and weights")
    parser.add_argument("--name",        type=str, default="oat_experiment",
                        help="Name for this training run")
    return parser.parse_args()


def main():
    args = parse_args()
    # Create data.yaml for YOLO
    data_yaml = {
        'path': args.data_dir,
        'train': 'images/train',
        'val':   'images/val',
        'nc':    1,
        'names': ['panicle']
    }
    yaml_path = os.path.join(args.data_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)
    print(f"Wrote data config to {yaml_path}")

    # Load YOLO model and train
    model = YOLO(args.model)
    model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        project=args.project,
        name=args.name
    )

if __name__ == '__main__':
    main()
