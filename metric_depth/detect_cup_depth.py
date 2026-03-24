import argparse
from pathlib import Path

from detect_everything_depth import DEFAULT_CONFIDENCE, DEFAULT_YOLO_MODEL, run_detection_for_target
from infer_photo import DEFAULT_CHECKPOINT_PATH, DEFAULT_IMAGE_PATH, DEFAULT_OUTPUT_DIR, INPUT_SIZE, MAX_DEPTH


TARGET_CLASS_NAME = "cup"
TARGET_LABEL_TEXT = "\u6c34\u676f"


def parse_args():
    parser = argparse.ArgumentParser(description="Detect a cup with YOLO11 and estimate its distance.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH, help="Input image path.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH, help="Metric depth checkpoint path.")
    parser.add_argument("--yolo-model", type=Path, default=DEFAULT_YOLO_MODEL, help="YOLO11 model path, e.g. yolo11n.pt.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for output files.")
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE, help="Metric depth input size.")
    parser.add_argument("--max-depth", type=float, default=MAX_DEPTH, help="Maximum metric depth in meters.")
    parser.add_argument("--compress-max-side", type=int, default=1024, help="Resize the input image once before detection and depth inference.")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE, help="YOLO confidence threshold.")
    parser.add_argument(
        "--no-upscale-small-image",
        action="store_true",
        help="Keep small compressed images near their current size instead of enlarging them back toward the depth encoder input size.",
    )
    parser.add_argument("--no-save", action="store_true", help="Print the detected distance without writing output files.")
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_detection_for_target(
        TARGET_CLASS_NAME,
        label_text=TARGET_LABEL_TEXT,
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        yolo_model_path=args.yolo_model,
        output_dir=args.output_dir,
        input_size=args.input_size,
        max_depth=args.max_depth,
        compress_max_side=args.compress_max_side,
        upscale_small_images=not args.no_upscale_small_image,
        conf_threshold=args.conf,
        save_outputs=not args.no_save,
    )
    print(result["sentence"])


if __name__ == "__main__":
    main()
