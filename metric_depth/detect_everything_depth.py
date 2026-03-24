import argparse
import os
import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from infer_photo import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_IMAGE_PATH,
    DEFAULT_OUTPUT_DIR,
    INPUT_SIZE,
    MAX_DEPTH,
    depth_to_vis,
    infer_depth_from_path,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_YOLO_MODEL = BASE_DIR / "checkpoints" / "yolo11n.pt"
LOCAL_YOLO_CONFIG_DIR = BASE_DIR / ".ultralytics"
DEFAULT_CONFIDENCE = 0.25
MAX_MATRIX_ROWS = 100
MAX_MATRIX_COLS = 100


def parse_args():
    parser = argparse.ArgumentParser(description="Detect everything YOLO11 can recognize and estimate distance.")
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
    parser.add_argument("--no-save", action="store_true", help="Print detections without writing output files.")
    return parser.parse_args()


@lru_cache(maxsize=None)
def load_yolo_model(model_path):
    LOCAL_YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(LOCAL_YOLO_CONFIG_DIR)
    os.environ["YOLO_VERBOSE"] = "False"

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ultralytics is not installed. Install it with `pip install ultralytics` or `pip install -r requirements.txt`."
        ) from exc

    return YOLO(str(Path(model_path)))


def normalize_class_name(name):
    return str(name).strip().lower()


def get_model_class_names(model):
    names = model.names
    if isinstance(names, dict):
        ordered = [names[index] for index in sorted(names)]
    else:
        ordered = list(names)
    return [normalize_class_name(name) for name in ordered]


def detect_all_objects(model, image, conf_threshold):
    results = model.predict(source=image, conf=conf_threshold, verbose=False)
    result = results[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        raise ValueError("YOLO11 found no objects in the image.")

    names = result.names
    detections = []

    for index, class_id_tensor in enumerate(boxes.cls):
        class_id = int(class_id_tensor.item())
        class_name = normalize_class_name(names[class_id])
        xyxy = boxes.xyxy[index].detach().cpu().numpy()
        confidence = float(boxes.conf[index].item())
        detections.append(
            {
                "class_name": class_name,
                "confidence": confidence,
                "xyxy": xyxy,
            }
        )

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return detections


def clamp_box(xyxy, image_shape):
    height, width = image_shape[:2]
    x1, y1, x2, y2 = xyxy
    left = max(0, min(width - 1, int(np.floor(x1))))
    top = max(0, min(height - 1, int(np.floor(y1))))
    right = max(left + 1, min(width, int(np.ceil(x2))))
    bottom = max(top + 1, min(height, int(np.ceil(y2))))
    return left, top, right, bottom


def compute_trimmed_median_mean(matrix, count=5):
    flat = matrix.reshape(-1)
    if flat.size == 0:
        raise ValueError("The detected object depth matrix is empty.")

    median = float(np.median(flat))
    nearest_count = min(count, flat.size)
    nearest_indices = np.argsort(np.abs(flat - median))[:nearest_count]
    nearest_values = flat[nearest_indices]
    return float(np.mean(nearest_values))


def create_run_dir(output_dir):
    detect_root = output_dir / "detect"
    detect_root.mkdir(parents=True, exist_ok=True)

    while True:
        run_ids = [int(path.name) for path in detect_root.iterdir() if path.is_dir() and path.name.isdigit()]
        run_id = max(run_ids, default=0) + 1
        run_dir = detect_root / str(run_id)
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue


def simplify_depth_matrix(matrix, max_rows=MAX_MATRIX_ROWS, max_cols=MAX_MATRIX_COLS):
    height, width = matrix.shape[:2]
    scale = min(max_rows / height, max_cols / width, 1.0)
    simplified_height = max(1, int(round(height * scale)))
    simplified_width = max(1, int(round(width * scale)))
    return cv2.resize(matrix.astype(np.float32), (simplified_width, simplified_height), interpolation=cv2.INTER_AREA)


def scale_box(box, source_shape, target_shape):
    left, top, right, bottom = box
    source_height, source_width = source_shape[:2]
    target_height, target_width = target_shape[:2]
    scale_x = target_width / source_width
    scale_y = target_height / source_height

    scaled_left = max(0, min(target_width - 1, int(np.floor(left * scale_x))))
    scaled_top = max(0, min(target_height - 1, int(np.floor(top * scale_y))))
    scaled_right = max(scaled_left + 1, min(target_width, int(np.ceil(right * scale_x))))
    scaled_bottom = max(scaled_top + 1, min(target_height, int(np.ceil(bottom * scale_y))))
    return scaled_left, scaled_top, scaled_right, scaled_bottom


def format_matrix(matrix):
    rows = ["[" + " ".join(f"{value:7.4f}" for value in row) + "]" for row in matrix]
    return "\n".join(rows) + "\n"


def save_text(path, text):
    path.write_text(text, encoding="utf-8")


def slugify_name(name):
    slug = re.sub(r"[^0-9a-zA-Z_-]+", "_", name.strip()).strip("_")
    return slug or "object"


def build_detection_record(detection, image_shape, depth, simplified_depth):
    box = clamp_box(detection["xyxy"], image_shape)
    left, top, right, bottom = box
    object_depth_matrix = depth[top:bottom, left:right]
    distance_value = compute_trimmed_median_mean(object_depth_matrix)

    simplified_box = scale_box(box, depth.shape, simplified_depth.shape)
    simple_left, simple_top, simple_right, simple_bottom = simplified_box
    simplified_object_depth = simplified_depth[simple_top:simple_bottom, simple_left:simple_right]

    return {
        **detection,
        "box": box,
        "distance": distance_value,
        "simplified_box": simplified_box,
        "simplified_depth_matrix": simplified_object_depth,
    }


def draw_detections(image, detections):
    annotated = image.copy()

    for index, detection in enumerate(detections, start=1):
        left, top, right, bottom = detection["box"]
        label = f"{index}.{detection['class_name']} {detection['confidence']:.2f} {detection['distance']:.2f}m"
        cv2.rectangle(annotated, (left, top), (right - 1, bottom - 1), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            label,
            (left, max(24, top - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated


def build_console_line(index, class_name, distance_value):
    return f"{index}. {class_name}\u5728\u8ddd\u79bb\u6444\u50cf\u5934{distance_value:.4f}\u7c73\u5904\u3002"


def build_summary_line(index, detection):
    left, top, right, bottom = detection["box"]
    return (
        f"{index}. class={detection['class_name']} "
        f"confidence={detection['confidence']:.4f} "
        f"distance_m={detection['distance']:.4f} "
        f"box=({left},{top},{right},{bottom})"
    )


def prepare_detection_context(
    image_path,
    checkpoint_path,
    yolo_model_path,
    *,
    input_size,
    max_depth,
    compress_max_side,
    upscale_small_images,
    conf_threshold,
):
    image_path = Path(image_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    yolo_model_path = Path(yolo_model_path).resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not yolo_model_path.is_file():
        raise FileNotFoundError(f"YOLO model not found: {yolo_model_path}")

    yolo_model = load_yolo_model(yolo_model_path)
    inference = infer_depth_from_path(
        image_path,
        checkpoint_path,
        input_size=input_size,
        max_depth=max_depth,
        compress_max_side=compress_max_side,
        upscale_small_images=upscale_small_images,
    )
    compressed_image = inference["compressed_image"]
    depth = inference["depth"]
    raw_detections = detect_all_objects(yolo_model, compressed_image, conf_threshold)
    simplified_depth = simplify_depth_matrix(depth)
    detections = [
        build_detection_record(detection, compressed_image.shape, depth, simplified_depth) for detection in raw_detections
    ]

    return {
        "compressed_image": compressed_image,
        "depth": depth,
        "simplified_depth": simplified_depth,
        "detections": detections,
        "class_names": get_model_class_names(yolo_model),
    }


def run_detection_for_target(
    target_class,
    *,
    label_text=None,
    image_path=DEFAULT_IMAGE_PATH,
    checkpoint_path=DEFAULT_CHECKPOINT_PATH,
    yolo_model_path=DEFAULT_YOLO_MODEL,
    output_dir=DEFAULT_OUTPUT_DIR,
    input_size=INPUT_SIZE,
    max_depth=MAX_DEPTH,
    compress_max_side=1024,
    upscale_small_images=True,
    conf_threshold=DEFAULT_CONFIDENCE,
    save_outputs=True,
):
    normalized_target = normalize_class_name(target_class)
    output_dir = Path(output_dir).resolve()
    context = prepare_detection_context(
        image_path,
        checkpoint_path,
        yolo_model_path,
        input_size=input_size,
        max_depth=max_depth,
        compress_max_side=compress_max_side,
        upscale_small_images=upscale_small_images,
        conf_threshold=conf_threshold,
    )

    candidates = [detection for detection in context["detections"] if detection["class_name"] == normalized_target]
    if not candidates:
        raise ValueError(f"YOLO11 found objects, but none of them were class '{normalized_target}'.")

    detection = max(candidates, key=lambda item: item["confidence"])
    sentence_target = label_text or normalized_target
    sentence = f"{sentence_target}\u5728\u8ddd\u79bb\u6444\u50cf\u5934{detection['distance']:.4f}\u7c73\u5904\u3002"
    run_dir = None

    if save_outputs:
        run_dir = create_run_dir(output_dir)
        depth_image_path = run_dir / "depth.png"
        yolo_image_path = run_dir / "yolo_detection.png"
        full_matrix_path = run_dir / "full_depth_matrix.txt"
        object_matrix_path = run_dir / f"{slugify_name(normalized_target)}_depth_matrix.txt"
        distance_path = run_dir / "distance.txt"

        cv2.imwrite(str(depth_image_path), depth_to_vis(context["depth"]))
        cv2.imwrite(str(yolo_image_path), draw_detections(context["compressed_image"], [detection]))
        save_text(full_matrix_path, format_matrix(context["simplified_depth"]))
        save_text(object_matrix_path, format_matrix(detection["simplified_depth_matrix"]))
        save_text(distance_path, sentence + "\n")

    return {
        "sentence": sentence,
        "run_dir": run_dir,
        "distance": detection["distance"],
        "class_names": context["class_names"],
    }


def run_detect_everything(
    *,
    image_path=DEFAULT_IMAGE_PATH,
    checkpoint_path=DEFAULT_CHECKPOINT_PATH,
    yolo_model_path=DEFAULT_YOLO_MODEL,
    output_dir=DEFAULT_OUTPUT_DIR,
    input_size=INPUT_SIZE,
    max_depth=MAX_DEPTH,
    compress_max_side=1024,
    upscale_small_images=True,
    conf_threshold=DEFAULT_CONFIDENCE,
    save_outputs=True,
):
    output_dir = Path(output_dir).resolve()
    context = prepare_detection_context(
        image_path,
        checkpoint_path,
        yolo_model_path,
        input_size=input_size,
        max_depth=max_depth,
        compress_max_side=compress_max_side,
        upscale_small_images=upscale_small_images,
        conf_threshold=conf_threshold,
    )
    run_dir = None
    if save_outputs:
        run_dir = create_run_dir(output_dir)
        depth_image_path = run_dir / "depth.png"
        yolo_image_path = run_dir / "yolo_detection.png"
        full_matrix_path = run_dir / "full_depth_matrix.txt"
        distance_path = run_dir / "distance.txt"

        cv2.imwrite(str(depth_image_path), depth_to_vis(context["depth"]))
        cv2.imwrite(str(yolo_image_path), draw_detections(context["compressed_image"], context["detections"]))
        save_text(full_matrix_path, format_matrix(context["simplified_depth"]))

    console_lines = []
    summary_lines = []
    for index, detection in enumerate(context["detections"], start=1):
        console_lines.append(build_console_line(index, detection["class_name"], detection["distance"]))
        summary_lines.append(build_summary_line(index, detection))
        if save_outputs:
            matrix_name = f"{index:02d}_{slugify_name(detection['class_name'])}_depth_matrix.txt"
            matrix_path = run_dir / matrix_name
            save_text(matrix_path, format_matrix(detection["simplified_depth_matrix"]))

    if save_outputs:
        save_text(distance_path, "\n".join(summary_lines) + "\n")

    return {
        "console_lines": console_lines,
        "summary_lines": summary_lines,
        "run_dir": run_dir,
        "detections": context["detections"],
        "class_names": context["class_names"],
    }


def main():
    args = parse_args()
    result = run_detect_everything(
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

    for line in result["console_lines"]:
        print(line)


if __name__ == "__main__":
    main()
