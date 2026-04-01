"""Discover test images and write evaluation artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_test_images(directory: Path) -> list[Path]:
    """Sorted list of image files under ``directory`` (non-recursive)."""
    if not directory.is_dir():
        return []
    out: list[Path] = []
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            out.append(p)
    return sorted(out)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_run_metadata(
    *,
    source_path: Path,
    weights_path: Path,
    points: list[tuple[float, float]],
    labels: list[int],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Sidecar content for one saved evaluation."""
    meta: dict[str, Any] = {
        "source_image": str(source_path.resolve()),
        "weights_path": str(weights_path.resolve()),
        "points": [[float(x), float(y)] for x, y in points],
        "labels": [int(l) for l in labels],
        "label_meaning": {"1": "foreground", "0": "background"},
        "saved_at_utc": utc_timestamp(),
    }
    try:
        import importlib.metadata

        meta["ultralytics_version"] = importlib.metadata.version("ultralytics")
    except (ImportError, ValueError):
        meta["ultralytics_version"] = None
    try:
        import torch

        meta["torch_version"] = torch.__version__
    except ImportError:
        meta["torch_version"] = None
    if extra:
        meta.update(extra)
    return meta
