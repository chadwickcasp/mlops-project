"""Defaults for interactive Mobile SAM evaluation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent


def _env_path(key: str, default: Path) -> Path:
    val = os.environ.get(key)
    return Path(val).expanduser() if val else default


@dataclass
class SegmentationEvalConfig:
    """Paths and inference knobs; override via env where noted."""

    repo_root: Path = field(default_factory=lambda: _REPO_ROOT)
    weights_path: Path = field(
        default_factory=lambda: _env_path(
            "IMAGE_SEGMENTATION_WEIGHTS",
            _REPO_ROOT / "models" / "image_segmentation" / "mobile_sam.pt",
        )
    )
    test_images_dir: Path = field(
        default_factory=lambda: _env_path(
            "IMAGE_SEGMENTATION_TEST_DIR",
            _REPO_ROOT / "data" / "test_images",
        )
    )
    output_dir: Path = field(
        default_factory=lambda: _env_path(
            "IMAGE_SEGMENTATION_OUTPUT_DIR",
            _REPO_ROOT / "outputs" / "image_segmentation",
        )
    )
    imgsz: int = field(
        default_factory=lambda: int(os.environ.get("IMAGE_SEGMENTATION_IMGSZ", "1024"))
    )
    device: str | None = field(
        default_factory=lambda: os.environ.get("IMAGE_SEGMENTATION_DEVICE") or None
    )
    conf: float = field(
        default_factory=lambda: float(os.environ.get("IMAGE_SEGMENTATION_CONF", "0.0"))
    )


def default_config() -> SegmentationEvalConfig:
    return SegmentationEvalConfig()
