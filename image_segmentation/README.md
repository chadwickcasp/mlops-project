# Image segmentation sandbox (Mobile SAM)

Interactive evaluation of Ultralytics Mobile SAM on your own test photos. This code is isolated from other project work (object detection training, mobile apps) but exports a small API (`load_model`, `segment_with_points`) you can reuse when wiring tap-to-segment on a device.

## Problem solved

Qualitative review of segment-anything–style prompts (multi-click foreground/background) on images under `data/test_images/`, with saved overlays and click metadata under `outputs/image_segmentation/`.

## Setup

1. Python 3.10+ recommended (3.13 used in development).
2. Create a venv and install:

```bash
pip install -r requirements-image-segmentation.txt
```

3. Place weights at **`models/image_segmentation/mobile_sam.pt`**.  
   The filename must end with `mobile_sam.pt` so Ultralytics selects the Mobile-SAM architecture. Weights are available from Ultralytics (Mobile SAM).

4. Put photos in **`data/test_images/`**.

## Run

From the **repository root**:

```bash
python scripts/evaluate_image_segmentation.py
```

- **Left-click**: foreground point  
- **Right-click**: background point  
- **b**: toggle box mode (click-drag-release to add a box prompt)  
- **s**: save PNG + JSON to `outputs/image_segmentation/`  
- **r**: reset clicks for the current image  
- **n** / **p**: next / previous image  
- **d**: delete last point  
- **backspace**: delete last box  
- **q**: quit  

Each save writes `<image_stem>_<utc_timestamp>.png` and a sidecar `.json` with points, labels, paths, and library versions.

The bottom “tray” shows a strip of your most recent saved overlays so you can quickly confirm what got written without leaving the app.

The overlay also shows how many prompt points you’ve added. Boxes count as **4 prompt points** each. If total prompt points exceed **48**, a large warning appears.

### Environment overrides (optional)

| Variable | Purpose |
|----------|---------|
| `IMAGE_SEGMENTATION_WEIGHTS` | Override weights path |
| `IMAGE_SEGMENTATION_TEST_DIR` | Override test image directory |
| `IMAGE_SEGMENTATION_OUTPUT_DIR` | Override output directory |
| `IMAGE_SEGMENTATION_IMGSZ` | Inference size (default `1024`) |
| `IMAGE_SEGMENTATION_DEVICE` | e.g. `cpu`, `0`, `mps` |
| `IMAGE_SEGMENTATION_CONF` | Mask score threshold (default `0.0` for SAM) |

## Headless / SSH

Matplotlib needs a display for the interactive window. On a headless machine, use X forwarding or run locally on a machine with a GUI.

## Chosen approach

- **Ultralytics `SAM`** with `points` and `labels` (`1` foreground, `0` background`) for parity with future mobile “tap” prompts.
- **OpenCV** for I/O and markers; **matplotlib** for the simple event loop.

## Validation

- Imports and helpers: `python -c "from image_segmentation.inference import load_model"` (from repo root).
- Full check: run the script on real images and confirm masks and saved JSON match your clicks.

## Limitations

- No training, dataset prep, or export-to-ONNX/TFLite in this milestone.
- Segmentation quality depends on Mobile SAM and your prompts.

## Deferred (not in this milestone)

- Minimal **Android** demo on device (e.g. Google Pixel 9 Pro) using camera images.
- **Labeling backend** feeding a separate object-detection (e.g. YOLO) training loop.
- On-device inference validation is a later acceptance step.

## Interview line

*“I sandboxed Mobile SAM behind a reusable `segment_with_points` wrapper and a matplotlib click tool that logs prompts and overlays for qualitative review—same API shape I’d use for tap-to-segment on mobile later.”*
