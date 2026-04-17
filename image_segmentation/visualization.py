"""Draw segmentation overlays and click markers on BGR images."""

from __future__ import annotations

from pathlib import Path

import cv2  # type: ignore[import-untyped]
import numpy as np

# OpenCV's Python bindings are dynamic; static analyzers often flag valid members.
# pylint: disable=no-member


def overlay_mask_bgr(
    image_bgr: np.ndarray,
    mask_bool: np.ndarray,
    *,
    color_bgr: tuple[int, int, int] = (0, 255, 128),
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a semi-transparent color over ``mask_bool`` regions."""
    if mask_bool.shape[:2] != image_bgr.shape[:2]:
        raise ValueError("mask shape must match image height and width")
    out = image_bgr.astype(np.float32).copy()
    layer = np.zeros_like(out)
    layer[mask_bool] = color_bgr
    m = mask_bool.astype(np.float32)[..., None]
    blended = out * (1.0 - alpha * m) + layer * (alpha * m)
    return np.clip(blended, 0, 255).astype(np.uint8)


def draw_click_markers(
    image_bgr: np.ndarray,
    points: list[tuple[float, float]],
    labels: list[int],
    *,
    radius: int = 10,
    thickness: int = 3,
    annotate_last: bool = True,
) -> np.ndarray:
    """Draw positive (green) and negative (red) click markers."""
    vis = image_bgr.copy()
    for (x, y), lab in zip(points, labels, strict=True):
        cx, cy = int(round(x)), int(round(y))
        col = (0, 255, 0) if lab == 1 else (0, 0, 255)
        cv2.circle(vis, (cx, cy), radius, col, thickness, lineType=cv2.LINE_AA)
        cv2.circle(vis, (cx, cy), 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    if annotate_last and points:
        lx, ly = points[-1]
        cx, cy = int(round(lx)), int(round(ly))
        cv2.putText(
            vis,
            "last pt",
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.05,
            (0, 255, 0),
            3,
            lineType=cv2.LINE_AA,
        )
    return vis


def draw_boxes(
    image_bgr: np.ndarray,
    boxes_xyxy: list[tuple[float, float, float, float]],
    *,
    color_bgr: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
    annotate_last: bool = True,
) -> np.ndarray:
    """Draw thin wireframe rectangles for positive box prompts."""
    vis = image_bgr.copy()
    for x1, y1, x2, y2 in boxes_xyxy:
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(vis, p1, p2, color_bgr, thickness, lineType=cv2.LINE_AA)
    if annotate_last and boxes_xyxy:
        x1, y1, x2, y2 = boxes_xyxy[-1]
        p = (int(round(x1)) + 6, int(round(y1)) + 18)
        cv2.putText(
            vis,
            "last box",
            p,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.05,
            color_bgr,
            3,
            lineType=cv2.LINE_AA,
        )
    return vis


def compose_eval_view(
    image_bgr: np.ndarray,
    mask_bool: np.ndarray | None,
    points: list[tuple[float, float]],
    labels: list[int],
    boxes_xyxy: list[tuple[float, float, float, float]] | None = None,
) -> np.ndarray:
    """Overlay mask (if any) and draw all click markers."""
    base = image_bgr
    if mask_bool is not None and mask_bool.any():
        base = overlay_mask_bgr(image_bgr, mask_bool)
    if boxes_xyxy:
        base = draw_boxes(base, boxes_xyxy)
    return draw_click_markers(base, points, labels)


def save_bgr_png(path: str | Path, image_bgr: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image_bgr):
        raise OSError(f"Failed to write image: {path}")
