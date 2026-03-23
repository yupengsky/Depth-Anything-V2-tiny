import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

logging.getLogger("dinov2").setLevel(logging.ERROR)

from depth_anything_v2.dpt import DepthAnythingV2


MODEL_CONFIG = {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]}
PREVIEW_ROWS = 6
PREVIEW_COLS = 6

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_PATH = BASE_DIR / "photos" / "1.jpg"
DEFAULT_CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "depth_anything_v2_metric_hypersim_vitl.pth"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model(checkpoint_path: Path, max_depth: float, device: str) -> DepthAnythingV2:
    model = DepthAnythingV2(**{**MODEL_CONFIG, "max_depth": max_depth})
    state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def compress_image(raw_image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    height, width = raw_image.shape[:2]
    longest_side = max(height, width)

    if longest_side <= max_side:
        return raw_image, 1.0

    scale = max_side / float(longest_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    compressed = cv2.resize(raw_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return compressed, scale


def depth_to_vis(depth: np.ndarray) -> np.ndarray:
    depth_min = float(depth.min())
    depth_max = float(depth.max())

    if depth_max == depth_min:
        return np.zeros_like(depth, dtype=np.uint8)

    return ((depth - depth_min) / (depth_max - depth_min) * 255.0).astype(np.uint8)


def preview_depth_matrix(depth: np.ndarray, rows: int = PREVIEW_ROWS, cols: int = PREVIEW_COLS) -> np.ndarray:
    row_indices = np.linspace(0, depth.shape[0] - 1, num=min(rows, depth.shape[0]), dtype=int)
    col_indices = np.linspace(0, depth.shape[1] - 1, num=min(cols, depth.shape[1]), dtype=int)
    return depth[np.ix_(row_indices, col_indices)]


def parse_args():
    parser = argparse.ArgumentParser(description="Infer metric depth for a single photo.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH, help="Input image path.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH, help="Metric depth checkpoint path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for output depth files.")
    parser.add_argument("--input-size", type=int, default=518, help="Inference input size.")
    parser.add_argument("--max-depth", type=float, default=20.0, help="Maximum metric depth in meters.")
    parser.add_argument("--compress-max-side", type=int, default=1024, help="Resize the input image once before inference.")
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = args.image.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    raw_image = cv2.imread(str(image_path))
    if raw_image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    compressed_image, resize_scale = compress_image(raw_image, args.compress_max_side)
    device = get_device()
    model = build_model(checkpoint_path, args.max_depth, device)
    depth = model.infer_image(compressed_image, args.input_size)

    depth_vis = depth_to_vis(depth)
    preview = preview_depth_matrix(depth)
    vis_output_path = output_dir / f"{image_path.stem}_depth.png"
    raw_output_path = output_dir / f"{image_path.stem}_depth.npy"

    cv2.imwrite(str(vis_output_path), depth_vis)
    np.save(str(raw_output_path), depth)

    print(f"Device: {device}")
    print(f"Input image: {image_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Original image shape: {raw_image.shape[:2]}")
    print(f"Compressed image shape: {compressed_image.shape[:2]}")
    print(f"Resize scale: {resize_scale:.4f}")
    print(f"Depth image saved to: {vis_output_path}")
    print(f"Raw depth matrix saved to: {raw_output_path}")
    print(f"Depth shape: {depth.shape}")
    np.set_printoptions(precision=4, suppress=True, linewidth=160)
    print("Depth matrix preview (meters):")
    print(preview)


if __name__ == "__main__":
    main()
