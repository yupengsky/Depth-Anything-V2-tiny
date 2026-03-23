# Metric Depth Only

This repository is reduced to one task: run metric depth inference on `metric_depth/photos/1.jpg` with the checkpoint in `metric_depth/checkpoints`.

## Install

```powershell
pip install -r requirements.txt
```

## Checkpoint

`.pth` weight files are ignored by git.
Place the required checkpoint at `metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth` before running the script.

## Run

```powershell
python metric_depth\infer_photo.py
```

Default behavior:

- Read `metric_depth/photos/1.jpg`
- Compress the image resolution once before inference
- Load `metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth`
- Save the depth image to `metric_depth/outputs`
- Save the raw depth matrix to `metric_depth/outputs/*.npy`
- Print a sampled preview depth matrix in the terminal
