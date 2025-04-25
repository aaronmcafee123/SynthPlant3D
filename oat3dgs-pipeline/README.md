# Oat3DGS Pipeline

A complete end-to-end toolkit for 3D reconstruction, instance segmentation, and phenotyping of oat panicles using 3D Gaussian Splatting.

## Repository Layout

```
oat3dgs-pipeline/
├── dataset/                # Train/val images & YOLO labels
├── sam_masks/              # SAM-generated masks for original views
├── masks_renders/          # SAM masks for novel-view renders
├── renders/                # Gaussian-splat novel-view RGB renderings
├── train_yolo.py           # Train YOLOv8 detector on oat images
├── masks_to_yolo_labels.py # Convert masks → YOLO labels
├── detect_and_sam_masks.py # YOLO → SAM segmentation on original images
├── onehot_ply_points.py    # Paint and one-hot encode 3D points by panicle
├── overlay_panicle.py      # Visualize segmented oat panicle vs the orginal 3DGS particles
├── segment_novel_views.py  # YOLO → SAM on rendered novel-view images
├── render_3dgs_views.py    # Render novel views of the 3DGS model
├── requirements.txt        # Python dependencies
├── README.md               # This guide
└── best.pt                 # Trained YOLOv8 panicle detector
```

**Install SAM and download checkpoint**:
   ```bash
   pip install segment-anything
   # Download large (ViT-H) model (~2.6 GB):
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O sam_vit_h.pth
   # Or smaller (ViT-B) model (~430 MB):
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O sam_vit_b.pth
   ```

##  Usage


### 1. Train YOLOv8 panicle detector
```bash
python train_yolo.py \
  --data_dir dataset \
  --model yolov8n.pt \
  --epochs 50 \
  --img_size 640 \
  --batch_size 16 \
  --project runs/train \
  --name oat_experiment
```

### 2. Generate SAM masks for original views
```bash
python detect_and_sam_masks.py \
  --image_dir    dataset/images/train \
  --detector_model runs/train/oat_experiment/weights/best.pt \
  --sam_checkpoint sam_vit_h.pth \
  --model_type   vit_h \
  --output_dir   sam_masks \
  --boxes_json   boxes.json
```

### 3. One-hot encode your 3D PLY
```bash
python onehot_ply_points.py \
  --ply_path      oat_splat.ply \
  --poses_json    poses.json \
  --intrinsics_json intrinsics.json \
  --mask_dir      sam_masks \
  --output_npz    points_onehot.npz \
  --output_ply    instances_colored.ply \
  --min_votes     5      \
  --dbscan_eps    0.02   \
  --dbscan_min    10
```

### 4. Segment novel-view renders
```bash
python segment_novel_views.py \
  --image_dir       renders \
  --detector_model  runs/train/oat_experiment/weights/best.pt \
  --sam_checkpoint  sam_vit_h.pth \
  --model_type      vit_h \
  --output_dir      masks_renders \
  --boxes_json      boxes_renders.json \
  --conf_thresh     0.3
```
### 5. Visualize segmented oat panicle vs the orginal 3DGS particles

python overlay_panicle.py \
        --full_ply      oat_splat.ply \
        --onehot_npz    points_onehot.npz \
        --panicle_id    1

