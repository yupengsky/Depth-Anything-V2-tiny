# To The Next Codex

This `metric_depth` folder is intended to be copied as a self-contained depth-estimation module into another project.
Please read this note before refactoring or merging it.

## What This Folder Does

This folder contains a local metric-depth pipeline plus optional YOLO-based object detection:

- `infer_photo.py`
  - Core metric depth implementation.
  - Contains the model definition, checkpoint loading, image preprocessing, and depth inference helpers.
  - Reusable entry points:
    - `infer_depth_from_image(...)`
    - `infer_depth_from_path(...)`
- `detect_everything_depth.py`
  - Runs YOLO detection on the image, runs metric depth, and assigns a distance to each detected object.
  - Reusable entry points:
    - `prepare_detection_context(...)`
    - `run_detect_everything(...)`
    - `run_detection_for_target(...)`
- `detect_cup_depth.py`
  - Thin wrapper around `run_detection_for_target(...)` for the `cup` class.

## Important Runtime Assumptions

The current code was written as a folder-local script bundle, not as a polished installable package.
It works today because the scripts are executed from inside or adjacent to `metric_depth`.

Current assumptions:

- Weights are loaded from local files under `metric_depth/checkpoints/`.
- Sample images are under `metric_depth/photos/`.
- Generated artifacts are written under `metric_depth/outputs/`.
- Ultralytics writes local settings under `metric_depth/.ultralytics/`.
- `detect_everything_depth.py` and `detect_cup_depth.py` currently use sibling imports such as `from infer_photo import ...`.
  - This is script-style import behavior.
  - If this folder is merged into a package-based application, convert these imports to package-safe imports.

## Recommended Merge Strategy

Please keep the algorithm behavior stable first, and only then adapt structure.

Suggested order:

1. Treat `infer_photo.py` as the source of truth for depth inference.
2. Convert `metric_depth` into a proper package if the host project expects package imports.
   - Usually this means adding `__init__.py`.
   - Then replace sibling imports with either relative imports or project-level absolute imports.
3. Move path configuration out of hardcoded defaults if the host project already has its own asset/config system.
   - The current defaults are based on `BASE_DIR = Path(__file__).resolve().parent`.
4. Decide whether the host project actually needs file outputs.
   - If not, call the Python functions directly and pass `save_outputs=False` where supported.
   - The detection scripts now support `--no-save`, and the internal APIs also support `save_outputs=False`.
5. Keep checkpoint paths explicit.
   - Do not silently download weights unless the host project already has a managed model-download flow.
6. Preserve the existing inference shortcuts unless there is a reason to remove them.
   - There is in-process model caching for the depth model.
   - YOLO loading is also cached in-process.
   - There is a `--no-upscale-small-image` option to avoid re-enlarging already compressed images.

## Integration Risks To Check

### 1. Python 3.14 dependency compatibility

This folder depends on:

- `numpy`
- `opencv-python`
- `torch`
- `ultralytics`

Before spending time on code refactors, verify that the target Python 3.14 environment can actually install and run the required `torch` and `ultralytics` stack.
If Python 3.14 package support is incomplete in the target environment, do not rewrite the model code as a first reaction.
Instead, prefer one of these paths:

- keep this module in a supported Python runtime;
- isolate it as a service/subprocess;
- or use a project-approved version pin strategy.

### 2. Import model

If the target project imports modules by package path, the current script-style imports will likely break.
Fix imports before deeper refactors.

### 3. Files that are not core source

These directories are artifacts, not core logic:

- `outputs/`
- `__pycache__/`
- `.ultralytics/`

They do not need to be preserved unless the host project explicitly wants historical outputs or local Ultralytics settings.

### 4. Large binary assets

The main depth checkpoint is large.
Do not assume it belongs in source control.
The host project may need a different model-storage convention.

## Current Performance Notes

This implementation is functional on CPU, but the main bottleneck is the metric depth model, especially with the large checkpoint.

Existing speed-related controls:

- `--compress-max-side`
- `--input-size`
- `--no-upscale-small-image`
- `--no-save`

If the host project only needs distances or in-memory results, prefer the internal function calls with output saving disabled.

## What Not To Change First

Unless the host project explicitly requires it, do not start by:

- replacing the depth model architecture;
- changing the depth aggregation logic for object distance;
- introducing network downloads for checkpoints;
- deleting the current reusable function entry points.

First make it importable and configurable.
Then integrate it with the host project's calling conventions.

## Minimal Merge Goal

The minimal successful merge is:

- the host project can import this module without path hacks;
- checkpoint paths come from host-project config or explicit arguments;
- the host project can call depth inference in memory;
- the host project can optionally call detection-plus-depth without writing files.

## If You Need A Safe First Refactor

If you are the next Codex and need a conservative first pass, do this:

1. Add package-safe imports.
2. Add `__init__.py` if the host project expects a package.
3. Centralize path/config injection.
4. Keep public function names stable.
5. Only after that, adapt CLI behavior or output layout.

## Human Intent

The human plans to drop this entire folder into another Python 3.14 project and wants you to merge it with minimal surprises.
Favor low-risk structural integration over algorithm rewrites.
