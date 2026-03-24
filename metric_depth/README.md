# Metric Depth Only

If this folder is being copied into another project, read `CODEX_MERGE_NOTE.md` first.

This repository is reduced to one task: run metric depth inference on `metric_depth/photos/1.jpg` with the checkpoint in `metric_depth/checkpoints`.
The full inference pipeline is now contained in a single file: `metric_depth/infer_photo.py`.

## Install

```powershell
pip install -r requirements.txt
```

## Checkpoint

`.pth` weight files are ignored by git.
Place the required checkpoint at `metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth` before running the script.

`.pt` YOLO weight files are also ignored by git.
If you use object detection, place a YOLO11 model such as `metric_depth/checkpoints/yolo11n.pt` in the checkpoints directory.

## Run

```powershell
python metric_depth\infer_photo.py
```

## Cup Detection + Depth

```powershell
python metric_depth\detect_cup_depth.py
```

## Detect Everything + Depth

```powershell
python metric_depth\detect_everything_depth.py
```

The detect-everything script automatically measures every YOLO11 detection in the image.

Default behavior:

- Read `metric_depth/photos/1.jpg`
- Compress the image resolution once before inference
- Load `metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth`
- Save the depth image to `metric_depth/outputs`
- Save the raw depth matrix to `metric_depth/outputs/*.npy`
- Print a sampled preview depth matrix in the terminal
- The cup-detection script prints only one sentence like `水杯在距离摄像头0.8129米处。`
- The detect-everything script prints one distance line per detected object
- Detection outputs are saved to `metric_depth/outputs/detect/<num>/`
- Saved files include the full depth image, the YOLO detection image, a simplified full-image depth matrix text file, one proportional boxed-region depth matrix text file per detection, and a distance summary text file
