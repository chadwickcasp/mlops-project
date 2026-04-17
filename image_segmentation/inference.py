"""Load Mobile SAM (Ultralytics) and run multi-point prompted segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ultralytics import SAM


def load_model(weights_path: str | Path) -> "SAM":
    """Load SAM weights. Path must end with a known suffix (e.g. ``mobile_sam.pt``)."""
    from ultralytics import SAM

    path = Path(weights_path)
    if not path.is_file():
        msg = (
            f"Weights not found: {path.resolve()}\n"
            "Place Mobile SAM weights at models/image_segmentation/mobile_sam.pt "
            "(filename must end with mobile_sam.pt for Ultralytics)."
        )
        raise FileNotFoundError(msg)

    return SAM(str(path))


def segment_with_prompts(
    model: "SAM",
    image_bgr: np.ndarray,
    points: list[tuple[float, float]],
    labels: list[int],
    bboxes_xyxy: list[tuple[float, float, float, float]] | None = None,
    *,
    imgsz: int = 1024,
    conf: float = 0.0,
    device: str | int | None = None,
) -> np.ndarray | None:
    """
    Run prompted segmentation using point and/or bounding box prompts.

    Args:
        model: The Mobile SAM model instance.
        image_bgr: Input image as a BGR NumPy array.
        points: List of (x, y) point coordinates, in image (pixel) coordinates.
        labels: List of integer labels for each point: 1 = foreground, 0 = background.
        bboxes_xyxy: Optional list of bounding boxes, where each box is a tuple (x1, y1, x2, y2)
            corresponding to top-left and bottom-right pixel corners in (x, y) order.
            When provided, the function assigns each point to a box (if inside, or to
            the nearest box if outside) and refines the mask for each box accordingly.
        imgsz: Inference resolution.
        conf: Confidence threshold for the model.
        device: Device identifier for inference (e.g., "cpu", "cuda:0", etc.).

    Returns:
        A boolean (H, W) mask aligned with image_bgr, or None if no prompts are provided.

    If both points and bounding boxes are provided, each box will be segmented using the
    points inside it and points outside all boxes are assigned to the nearest box for refinement.
    If only points are given, a single mask is produced using all points.
    """
    if len(points) != len(labels):
        raise ValueError("points and labels must have the same length")
    if not points and not bboxes_xyxy:
        return None

    def _predict_mask(**kwargs) -> np.ndarray:
        """Run prediction and return a (H,W) bool mask (union over returned masks)."""
        if device is not None:
            kwargs["device"] = device
        kwargs.setdefault("imgsz", imgsz)
        kwargs.setdefault("conf", conf)
        kwargs.setdefault("verbose", False)

        results = model.predict(image_bgr, **kwargs)
        if not results:
            h, w = image_bgr.shape[:2]
            return np.zeros((h, w), dtype=bool)

        masks = results[0].masks
        if masks is None or masks.data is None or len(masks.data) == 0:
            h, w = image_bgr.shape[:2]
            return np.zeros((h, w), dtype=bool)

        data = masks.data
        if hasattr(data, "cpu"):
            arr = data.cpu().numpy()
        else:
            arr = np.asarray(data)

        # arr is expected (N, H, W) or (H, W)
        if arr.ndim == 2:
            return arr.astype(bool) if arr.dtype != bool else arr
        if arr.ndim == 3:
            union = np.any(arr.astype(bool), axis=0)
            return union
        raise ValueError(f"Unexpected masks array shape: {arr.shape}")

    # If we have boxes, refine each box with:
    # - points inside that box
    # - points outside all boxes, assigned to the nearest box by distance-to-rectangle (based on x/y limits)
    if bboxes_xyxy:
        boxes = [(float(x1), float(y1), float(x2), float(y2)) for x1, y1, x2, y2 in bboxes_xyxy]

        def _inside(p: tuple[float, float], b: tuple[float, float, float, float]) -> bool:
            x, y = p
            x1, y1, x2, y2 = b
            return (x1 <= x <= x2) and (y1 <= y <= y2)

        def _dist_to_box(p: tuple[float, float], b: tuple[float, float, float, float]) -> float:
            # Distance from point to rectangle (0 if inside), derived from x/y limits.
            x, y = p
            x1, y1, x2, y2 = b
            xc = min(max(x, x1), x2)
            yc = min(max(y, y1), y2)
            dx = x - xc
            dy = y - yc
            return float(np.hypot(dx, dy))

        pts_f = [(float(x), float(y)) for x, y in points]
        labs_i = [int(i) for i in labels]

        inside_idxs_per_box: list[list[int]] = [[] for _ in boxes]
        outside_idxs: list[int] = []

        for j, p in enumerate(pts_f):
            any_inside = False
            for bi, b in enumerate(boxes):
                if _inside(p, b):
                    inside_idxs_per_box[bi].append(j)
                    any_inside = True
            if not any_inside:
                outside_idxs.append(j)

        assigned_outside_per_box: list[list[int]] = [[] for _ in boxes]
        for j in outside_idxs:
            pt = pts_f[j]
            best_i = min(range(len(boxes)), key=lambda bi: _dist_to_box(pt, boxes[bi]))
            assigned_outside_per_box[best_i].append(j)

        out_mask = None
        for bi, b in enumerate(boxes):
            # Build per-box points: inside + assigned-outside
            idxs = inside_idxs_per_box[bi] + assigned_outside_per_box[bi]
            if idxs:
                p_box = [[pts_f[k][0], pts_f[k][1]] for k in idxs]
                l_box = [labs_i[k] for k in idxs]
                m = _predict_mask(bboxes=[[b[0], b[1], b[2], b[3]]], points=[p_box], labels=[l_box])
            else:
                m = _predict_mask(bboxes=[[b[0], b[1], b[2], b[3]]])
            out_mask = m if out_mask is None else (out_mask | m)

        assert out_mask is not None
        return out_mask

    # No boxes: point-only prompt (single object with multi-point labels)
    pts = [[float(x), float(y)] for x, y in points]
    labs = [int(i) for i in labels]
    return _predict_mask(points=[pts], labels=[labs]) if points else None


def segment_with_points(
    model: "SAM",
    image_bgr: np.ndarray,
    points: list[tuple[float, float]],
    labels: list[int],
    *,
    imgsz: int = 1024,
    conf: float = 0.0,
    device: str | int | None = None,
) -> np.ndarray | None:
    """Backwards-compatible wrapper for point-only prompts."""
    return segment_with_prompts(
        model,
        image_bgr,
        points,
        labels,
        bboxes_xyxy=None,
        imgsz=imgsz,
        conf=conf,
        device=device,
    )
